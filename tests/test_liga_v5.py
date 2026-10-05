"""Pruebas del simulador de ligas de la Fase 3 (reglas oficiales, rivales sintéticos, Regla Maestra noche a noche)."""
import copy

import numpy as np
import pandas as pd
import pytest

from src import liga_v5 as L
from src.regla_maestra import MANTENER
from tests.helpers import make_cfg

CFG = make_cfg()
CAL = L.calendario_concurso(CFG)
NOMBRES = ["TSLA", "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META"]
SIGMA = {"TSLA": 0.005, "NVDA": 0.03, "AAPL": 0.012, "MSFT": 0.012, "AMZN": 0.015, "GOOGL": 0.014, "META": 0.02}
S = 70                                                                                               # inicio de la ventana en el panel de juguete


def panel_toy(T=140, seed=0, sig=None):
    sig = sig or SIGMA
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=T)
    r = np.column_stack([rng.normal(0.0005, sig[n], T) for n in NOMBRES])
    C = 100 * np.cumprod(1 + r, axis=0)
    previo = np.vstack([np.full((1, len(NOMBRES)), 100.0), C[:-1]])
    O = previo * (1 + rng.normal(0, 0.002, C.shape))
    ret = pd.DataFrame(C).pct_change().to_numpy().copy()                                              # copia: pandas 3 devuelve vistas de sólo lectura
    sigma = pd.DataFrame(ret).shift(1).rolling(20).std().to_numpy().copy()
    verde = np.full(C.shape, "VERDE", dtype=object)
    return L.PanelLiga(idx, list(NOMBRES), O, C, list(NOMBRES), sigma, verde, verde.copy(), C.copy(), np.ones(C.shape), np.zeros(C.shape),
                       np.full(C.shape, 1000.0), ret)


def rivales_constantes(valor, n=200):
    """Todos los rivales llevan la misma rentabilidad cada sesión (objetivo de los cortes = ese valor)."""
    return np.full((n, CAL["n_ses"]), valor)


def costos():
    v = CFG["costos_v5"]
    return v["comision_pct"] + v["spread_mgc"], np.cumsum(np.isin(np.arange(CAL["n_ses"]), CAL["ops"])) * L.COSTO_OP_MICRO


# =============================== calendario oficial ===============================
def test_el_calendario_traduce_las_fechas_oficiales_a_sesiones():
    assert [k["sesion"] for k in CAL["cortes"]] == [4, 9, 14, 19] and [k["top_pct"] for k in CAL["cortes"]] == [50, 33, 10, 10]
    assert CAL["final"] == 24 and CAL["n_ses"] == 25 and CAL["semana_final_desde"] == 20
    assert CAL["noches"] == [3, 8, 13, 15, 16, 17, 18, 20, 21, 22, 23]
    assert len(CAL["ops"]) == 19 and CAL["ops"][:4] == [1, 2, 3, 5]                                   # semana 1: compra inicial + 3 micro-compras
    assert L.proximo_corte(3, CAL) == (4, 50.0) and L.proximo_corte(15, CAL) == (19, 10.0) and L.proximo_corte(20, CAL) == (24, None)


# =============================== rivales ===============================
def test_los_pesos_de_los_rivales_suman_uno_y_respetan_la_concentracion():
    rng = np.random.default_rng(1)
    w = L.pesos_rivales(2000, 50, rng, 0.30, [0, 1], peso_conc_min=0.70)
    assert np.allclose(w.sum(axis=1), 1.0)
    conc = w[:600]
    assert (conc[:, :2].max(axis=1) >= 0.70 - 1e-12).all() and (conc[:, :2].max(axis=1) <= 1.0).all()
    assert abs((conc[:, 0] > 0).mean() - 0.5) < 0.08                                                   # mitad TSLA, mitad NVDA
    resto = w[600:]
    k = (resto > 0).sum(axis=1)
    assert k.min() >= 2 and k.max() <= 4 and np.allclose(resto.max(axis=1) * k, 1.0)                    # pesos iguales entre 2 y 4 activos


def test_peor_caso_los_concentrados_ponen_el_100_por_ciento():
    w = L.pesos_rivales(1000, 30, np.random.default_rng(2), 0.30, [0, 1], peso_conc_min=1.0)
    conc = w[:300]
    assert (conc.max(axis=1) == 1.0).all() and ((conc[:, 0] == 1.0) | (conc[:, 1] == 1.0)).all()


def test_los_pesos_son_reproducibles_con_la_misma_semilla():
    a = L.pesos_rivales(100, 20, np.random.default_rng(3), 0.3, [0, 1])
    b = L.pesos_rivales(100, 20, np.random.default_rng(3), 0.3, [0, 1])
    assert np.array_equal(a, b)


def test_recorrido_de_un_rival_compra_a_la_apertura_y_marca_al_cierre_con_costos():
    O0 = np.array([100.0, 50.0])
    C = np.array([[101.0, 49.0], [110.0, 48.0]])
    w = np.array([[1.0, 0.0], [0.5, 0.5]])
    r = L.recorridos_rivales(w, O0, C, 0.005, np.array([0.0, 0.001]))
    assert r[0] == pytest.approx([0.01 - 0.005, 0.10 - 0.005 - 0.001])
    assert r[1, 0] == pytest.approx(0.5 * 0.01 + 0.5 * (-0.02) - 0.005)


def test_un_activo_sin_precios_en_la_ventana_se_descarta_y_se_renormaliza():
    O0 = np.array([100.0, np.nan])
    C = np.array([[110.0, np.nan]])
    r = L.recorridos_rivales(np.array([[0.5, 0.5], [0.0, 1.0]]), O0, C, 0.0, np.array([0.0]))
    assert r[0, 0] == pytest.approx(0.10) and np.isnan(r[1, 0])                                         # el 2.º sólo tenía el activo inexistente


# =============================== cortes ===============================
def test_cortes_con_campo_ordenado_pasa_quien_esta_dentro_del_top():
    N = 100
    rivales = np.tile(np.linspace(0.0, 0.99, N)[:, None], (1, CAL["n_ses"]))                           # rival i rinde i/100 siempre
    mis = {"alto": np.full(CAL["n_ses"], 0.95), "medio": np.full(CAL["n_ses"], 0.60), "bajo": np.full(CAL["n_ses"], 0.30)}
    res, vivos = L.jugar_cortes(rivales, mis, CAL, "campo")
    assert res["alto"]["pasa"] == [True] * 4 and res["alto"]["vivo"]
    assert res["medio"]["pasa"] == [True, False, False, False] and not res["medio"]["vivo"]            # 60 % está en el top 50 % pero no en el top 33 %
    assert res["bajo"]["pasa"] == [False] * 4
    assert np.isnan(res["bajo"]["q_final"]) and res["alto"]["q_final"] < 0.1
    assert vivos.sum() <= 10                                                                           # al final sólo queda ≈ el top 10 %


def test_quien_pierde_un_corte_no_puede_volver_aunque_luego_rinda_mas():
    N = 100
    r = np.tile(np.linspace(0.0, 0.99, N)[:, None], (1, CAL["n_ses"]))
    mio = np.full(CAL["n_ses"], 0.95)
    mio[:5] = 0.10                                                                                     # en el corte 1 voy mal; después voy primero
    res, _ = L.jugar_cortes(r, {"yo": mio}, CAL, "campo")
    assert res["yo"]["pasa"] == [False, False, False, False] and not res["yo"]["vivo"]


def test_base_supervivientes_es_mas_exigente_que_base_campo():
    N = 1000
    rivales = np.tile(np.linspace(0.0, 0.999, N)[:, None], (1, CAL["n_ses"]))
    mis = {"yo": np.full(CAL["n_ses"], 0.93)}
    campo, _ = L.jugar_cortes(rivales, mis, CAL, "campo")
    sup, _ = L.jugar_cortes(rivales, mis, CAL, "supervivientes")
    assert campo["yo"]["pasa"] == [True] * 4                                                           # 93 % está en el top 10 % del campo total
    assert sup["yo"]["pasa"] != [True] * 4                                                             # pero no en el top 10 % del top 10 % del top 33 %…


def test_vivos_por_noche_nunca_resucita_a_un_eliminado():
    rng = np.random.default_rng(4)
    R = rng.normal(0, 0.05, (500, CAL["n_ses"])).cumsum(axis=1)
    v = L.vivos_por_noche(R, CAL)
    assert v.shape == (CAL["n_ses"], 500) and v[0].all()
    assert (np.diff(v.sum(axis=1)) <= 0).all()                                                         # el número de vivos sólo baja
    assert (v[1:] <= v[:-1]).all()                                                                     # y nadie vuelve a entrar
    assert v[5].sum() < v[4].sum() and v[4].sum() == v[0].sum()                                        # el primer corte (sesión 4) se aplica a partir de la noche 5


def test_probabilidades_finales_binomial_y_los_que_no_sobreviven_aportan_cero():
    q = np.array([0.01, 0.01, 0.5])
    vivo = np.array([True, False, True])
    p1, top = L.probabilidades_finales(q, vivo, 100)
    assert p1 == pytest.approx((0.99 ** 100 + 0 + 0.5 ** 100) / 3)
    assert 0 <= top <= 1 and top == pytest.approx((1.0 - 0.0 + 0.0) / 3, abs=0.4)                      # sólo el primero suma (≈ 1: con N=100 casi siempre < 30 por delante)
    assert L.probabilidades_finales(np.array([0.01]), np.array([False]), 5000) == (0.0, 0.0)
    assert L.probabilidades_finales(np.array([0.001]), np.array([True]), 1000)[1] > 0.99               # N·q = 1 por delante en promedio: casi seguro top 30


# =============================== mi estrategia ===============================
def test_mantener_replica_la_formula_de_97_por_ciento_con_costos():
    p = panel_toy()
    ce, ca = costos()
    r = L.simular_estrategia(p, S, CFG, CAL, rivales_constantes(0.0), np.ones((25, 200), bool), "TSLA", False, ce, ca)
    j = p.col("TSLA")
    esperado = 0.03 + 0.97 * (1 - ce) * p.C[S: S + 25, j] / p.O[S, j] - 1 - ca
    assert np.allclose(r["camino"], esperado) and r["cambios"] == 0 and r["eventos"] == []


def test_un_cambio_se_ejecuta_a_la_apertura_siguiente_y_cobra_el_1_3_por_ciento(monkeypatch):
    p = panel_toy()
    ce, ca = costos()
    monkeypatch.setattr(L, "_decidir", lambda *a, **k: ("CAMBIAR", "NVDA") if a[2] == 3 else None)       # a[2] es la noche t
    r = L.simular_estrategia(p, S, CFG, CAL, rivales_constantes(0.0), np.ones((25, 200), bool), "TSLA", True, ce, ca)
    jt, jn = p.col("TSLA"), p.col("NVDA")
    unidades = 0.97 * (1 - ce) / p.O[S, jt]
    valor = unidades * p.O[S + 4, jt] * (1 - CFG["costos_v5"]["costo_cambio"])                         # se vende a la apertura de la sesión 4 (no a la del cierre de la 3)
    nuevas = valor / p.O[S + 4, jn]
    assert r["eventos"] == [dict(t=4, tipo="cambio", de="TSLA", a="NVDA")] and r["cambios"] == 1
    assert r["camino"][3] == pytest.approx(0.03 + unidades * p.C[S + 3, jt] - 1 - ca[3])                # hasta la noche de la orden sigo en TSLA
    assert r["camino"][4] == pytest.approx(0.03 + nuevas * p.C[S + 4, jn] - 1 - ca[4])
    assert r["camino"][24] == pytest.approx(0.03 + nuevas * p.C[S + 24, jn] - 1 - ca[24])


def test_vender_todo_deja_efectivo_y_el_recorrido_ya_no_sigue_al_mercado(monkeypatch):
    p = panel_toy()
    ce, ca = costos()
    monkeypatch.setattr(L, "_decidir", lambda *a, **k: ("VENDER", "") if a[2] == 23 else None)
    r = L.simular_estrategia(p, S, CFG, CAL, rivales_constantes(0.0), np.ones((25, 200), bool), "TSLA", True, ce, ca)
    j = p.col("TSLA")
    unidades = 0.97 * (1 - ce) / p.O[S, j]
    cash = 0.03 + unidades * p.O[S + 24, j] * (1 - (CFG["costos_v5"]["slippage_mgc"] + CFG["costos_v5"]["comision_pct"]))
    assert r["vendio"] is True and r["camino"][24] == pytest.approx(cash - 1 - ca[24]) and r["eventos"][0]["tipo"] == "venta_total"


def test_la_estrategia_a_no_decide_nunca(monkeypatch):
    p = panel_toy()
    ce, ca = costos()
    monkeypatch.setattr(L, "_decidir", lambda *a, **k: pytest.fail("la estrategia (a) no debe llamar a la Regla Maestra"))
    L.simular_estrategia(p, S, CFG, CAL, rivales_constantes(0.0), np.ones((25, 200), bool), "TSLA", False, ce, ca)


# =============================== decisión: Regla Maestra con información del momento ===============================
def decision(p, t=3, mia=0.0, campo=0.30, cfg=CFG, activo="TSLA", usados=0, cambio_hoy=False, rivales=None):
    R = rivales if rivales is not None else rivales_constantes(campo)
    return L._decidir(p, S, t, 25, cfg, CAL, R, np.ones((25, R.shape[0]), bool), activo, 100.0, 100.0 * (1 + mia), mia, usados, cambio_hoy, p.fechas[S: S + 25])


def test_si_voy_muy_atras_y_hay_un_activo_mucho_mas_volatil_la_regla_cambia():
    p = panel_toy()
    assert decision(p, mia=0.0, campo=0.30) == ("CAMBIAR", "NVDA")                                     # g ≈ 31; NVDA se mueve 6× lo que TSLA


def test_si_voy_por_encima_del_objetivo_se_mantiene():
    assert decision(panel_toy(), mia=0.40, campo=0.30) is None                                         # g ≤ 0


def test_no_se_cambia_dos_veces_el_mismo_dia_ni_pasados_los_4_cambios():
    p = panel_toy()
    assert decision(p, cambio_hoy=True) is None and decision(p, usados=4) is None


def test_el_activo_actual_nunca_es_su_propio_candidato():
    assert decision(panel_toy(), activo="NVDA", mia=0.0, campo=0.30) != ("CAMBIAR", "NVDA")


def test_las_decisiones_no_miran_al_futuro():
    """Se altera TODO lo posterior a la noche t (precios, volúmenes, σ, rivales): la decisión debe ser idéntica."""
    p = panel_toy()
    base = decision(p, t=3, mia=0.0, campo=0.30)
    assert base == ("CAMBIAR", "NVDA")                                                                 # la prueba sólo vale si la decisión no es trivial
    q = copy.deepcopy(p)
    g = S + 3
    rng = np.random.default_rng(99)
    for arr in (q.O, q.C):
        arr[g + 1:] = arr[g + 1:] * rng.uniform(0.2, 5, arr[g + 1:].shape)
    q.sigma[g + 1:] = rng.uniform(0, 1, q.sigma[g + 1:].shape)
    q.ret[g + 1:] = rng.normal(0, 0.5, q.ret[g + 1:].shape)
    q.close_usd[g + 1:] = q.close_usd[g + 1:] * 3
    R = rivales_constantes(0.30)
    R[:, 4:] = rng.normal(0, 1, R[:, 4:].shape)                                                         # el futuro de los rivales también cambia
    assert decision(q, t=3, mia=0.0, rivales=R) == base
    # y con otro futuro distinto, a otra noche, lo mismo:
    r2 = decision(p, t=16, mia=0.0, campo=0.30)
    q2 = copy.deepcopy(p)
    q2.C[S + 17:] *= 7
    q2.sigma[S + 17:] = 5.0
    assert decision(q2, t=16, mia=0.0, campo=0.30) == r2


def test_en_modo_negro_el_listón_baja_y_con_cambio_por_negro_apagado_no_se_aplica():
    p = panel_toy()
    p.sigma[:, :] = 0.003                                                                              # σ fijas: todos menos volátiles que TSLA…
    p.sigma[:, 0], p.sigma[:, 1] = 0.005, 0.0052                                                       # …salvo NVDA, apenas más volátil (ratio 1,04)
    p.close_usd[:, 0] = np.r_[np.full(S + 1, 100.0), np.full(len(p.fechas) - S - 1, 80.0)]            # TSLA cae 20 % el día 1 de la ventana…
    p.color_dia[S + 1, 0] = "ROJO"                                                                      # …día ROJO sin recuperación al cierre del día 2
    on, off = copy.deepcopy(CFG), copy.deepcopy(CFG)
    on["regla_maestra"]["cambio_por_negro"], off["regla_maestra"]["cambio_por_negro"] = True, False
    assert L._color_propio(p, S, S + 2, 0, CFG) == "NEGRO"
    assert decision(p, t=3, mia=-0.02, campo=0.0, cfg=on) == ("CAMBIAR", "NVDA")                         # NEGRO: múltiplo (g+1,3)/(g+1,5) < 1: cualquiera con MEE parecido sirve
    assert decision(p, t=3, mia=-0.02, campo=0.0, cfg=off) is None                                      # sin cambio_por_negro: 1,04 < (g+1,3)/g → mantener


def test_los_episodios_de_mi_activo_empiezan_en_la_ventana_no_antes():
    p = panel_toy()
    p.close_usd[:, 0] = np.r_[np.full(S - 10, 100.0), np.full(len(p.fechas) - S + 10, 70.0)]            # caída de 30 ANTES de la ventana
    p.color_dia[S - 10, 0] = "ROJO"
    assert p.color_dia[S - 10, 0] == "ROJO" and L._color_propio(p, S, S + 3, 0, CFG) == "VERDE"        # el sistema en vivo arranca sin memoria el 5-oct


def test_un_candidato_en_rojo_no_entra_al_banco():
    p = panel_toy()
    p.color_dia[S + 3, 1] = "ROJO"                                                                      # NVDA en ROJO hoy
    assert decision(p, mia=0.0, campo=0.30) != ("CAMBIAR", "NVDA")


def test_el_corte_manana_convierte_el_rojo_de_un_candidato_en_negro():
    p = panel_toy()
    p.color_dia[S + 3, 1] = "ROJO"                                                                      # noche 3: el corte es mañana (sesión 4)
    assert decision(p, t=3, mia=0.0, campo=0.30) != ("CAMBIAR", "NVDA")
    q = panel_toy()
    q.color_dia[S + 8, 1] = "ROJO"
    assert decision(q, t=8, mia=0.0, campo=0.30) != ("CAMBIAR", "NVDA")                                 # noche 8: corte mañana (sesión 9)


# =============================== integración ===============================
def test_correr_liga_en_un_panel_de_juguete_entrega_filas_por_ventana_y_estrategia():
    p = panel_toy(T=140)
    df = L.correr_liga(p, CFG, str(p.fechas[S].date()), "TSLA", 300, 0.30, 5, "campo", paso=7, peso_conc_min=0.70)
    assert set(df["estrategia"]) == {"a", "b", "c"} and df["inicio"].nunique() >= 5
    assert df.groupby("estrategia").size().nunique() == 1
    a = df[df.estrategia == "a"]
    assert (a["cambios"] == 0).all() and (a["pasa2"] <= a["pasa1"]).all() and (a["pasa4"] <= a["pasa3"]).all() and (a["vivo_final"] == a["pasa4"]).all()
    assert (a["q_final"].dropna().between(0, 1)).all() and a["q_final"].isna().equals(~a["vivo_final"])


def test_resumen_liga_probabilidades_acumuladas_y_en_rango():
    p = panel_toy(T=140)
    df = L.correr_liga(p, CFG, str(p.fechas[S].date()), "TSLA", 300, 0.30, 5, "campo", paso=5)
    r = L.resumen_liga(df, CFG, [1000, 5000]).set_index("estrategia")
    for k in "abc":
        f = r.loc[k]
        assert 1 >= f.pasa1 >= f.pasa2 >= f.pasa3 >= f.pasa4 >= 0
        assert f.pasa1_lo <= f.pasa1 <= f.pasa1_hi and 0 <= f.p1_5000 <= f.top30_5000 <= 1
    assert r.loc["b", "pct_con_cambio"] >= 0 and MANTENER == "MANTENER"


# =============================== ¿y si la base fuera otra acción? ===============================
def test_correr_bases_comparte_los_rivales_y_omite_activos_sin_precios():
    p = panel_toy(T=140)
    p.C[:, 2] = np.nan                                                                                 # AAPL sin precios: no se puede usar de base
    p.O[:, 2] = np.nan
    df = L.correr_bases(p, CFG, str(p.fechas[S].date()), ["TSLA", "NVDA", "AAPL"], 300, 0.30, 5, "campo", paso=9)
    assert set(df["base"]) == {"TSLA", "NVDA"} and df.groupby("base").size().nunique() == 1
    a = df[df.base == "TSLA"].set_index("inicio")
    solo = L.correr_liga(p, CFG, str(p.fechas[S].date()), "TSLA", 300, 0.30, 5, "campo", paso=9, estrategias=("a",))
    assert np.allclose(a["ret_final"].to_numpy(), solo.set_index("inicio")["ret_final"].to_numpy())       # igual que la estrategia (a): mismos rivales y mismas reglas
    assert (a["pasa2"] <= a["pasa1"]).all() and (a["vivo_final"] == a["pasa4"]).all()


def test_resumen_bases_ordena_por_probabilidad_de_llegar_al_final():
    p = panel_toy(T=140)
    df = L.correr_bases(p, CFG, str(p.fechas[S].date()), ["TSLA", "NVDA", "META"], 300, 0.0, 5, "campo", paso=5)
    r = L.resumen_bases(df, 5000)
    assert list(r.columns[:2]) == ["base", "ventanas"] and (r["pasa4"].diff().dropna() <= 1e-12).all() and set(r["base"]) == {"TSLA", "NVDA", "META"}
    assert ((r[["pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30", "pct_pierde"]] >= 0) & (r[["pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30", "pct_pierde"]] <= 1)).all().all()


def test_el_azar_esperado_depende_de_como_se_mide_el_top_x():
    from report import validacion as RV
    assert RV.azar_esperado([50, 33, 10, 10], "campo") == pytest.approx([0.5, 0.33, 0.10, 0.10])
    assert RV.azar_esperado([50, 33, 10, 10], "supervivientes") == pytest.approx([0.5, 0.165, 0.0165, 0.00165])


def test_tabla_bases_incluye_la_base_actual_aunque_no_este_arriba():
    from report import validacion as RV
    r = pd.DataFrame({"base": list("ABCDEFG") + ["TSLA"], "ventanas": 10, **{k: np.linspace(0.8, 0.1, 8) for k in ("pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30", "pct_pierde")},
                      "ret_mediana": 0.0, "ret_p10": -0.1, "ret_p90": 0.1})
    t = RV.tabla_bases(r, "TSLA", n=3)
    assert [x["base"] for x in t] == ["A", "B", "C", "TSLA"] and t[-1]["es_base"] and t[-1]["puesto"] == 8 and sum(x["es_base"] for x in t) == 1
