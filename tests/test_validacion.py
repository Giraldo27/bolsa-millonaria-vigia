"""Pruebas de la Fase 3 (estudio de eventos): la versión vectorizada debe coincidir con el semáforo real, sin mirar al futuro."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import validacion as V
from src.semaforo import AMARILLO, AZUL, NEGRO, ROJO, VERDE, EntradaSemaforo, evaluar
from tests.helpers import make_cfg

CFG = make_cfg()
SHOCKS = (150, 230, 300, 360)


def mercado_sintetico(n=420, seed=1, con_shocks=True):
    """Acción con beta ≈ 1,2 contra un índice; cuatro caídas propias de −9 % con volumen ×3 y hueco de apertura de −4 %."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    rn = rng.normal(0.0004, 0.01, n)
    r = 1.2 * rn + rng.normal(0, 0.012, n)
    vol = rng.integers(800_000, 1_200_000, n).astype(float)
    gap = rng.normal(0, 0.003, n)
    if con_shocks:
        for k in SHOCKS:
            r[k], gap[k] = -0.09, -0.04
            vol[k] *= 3
    close = 100 * np.cumprod(1 + r)
    prev = np.r_[100.0, close[:-1]]
    abre = prev * (1 + gap)
    nclose = 10_000 * np.cumprod(1 + rn)
    nprev = np.r_[10_000.0, nclose[:-1]]
    px = pd.DataFrame(dict(Open=abre, High=np.maximum(abre, close), Low=np.minimum(abre, close), Close=close, Volume=vol), index=idx)
    nd = pd.DataFrame(dict(Open=nprev, Close=nclose), index=idx)
    return px, nd


def entrada(px, nd, i):
    """El mismo día i visto por el sistema en vivo: sólo cierres hasta ayer, y el precio/volumen de hoy."""
    return EntradaSemaforo(
        ticker="X", fecha=px.index[i].date(), precio=float(px["Close"].iloc[i]), cierre_previo=float(px["Close"].iloc[i - 1]),
        cierres=px["Close"].iloc[:i], volumenes=px["Volume"].iloc[:i], volumen_dia=float(px["Volume"].iloc[i]), ndx_cierres=nd["Close"].iloc[:i],
        ndx_precio=float(nd["Close"].iloc[i]), ndx_cierre_previo=float(nd["Close"].iloc[i - 1]), noticias=None, base_noticias_diaria=None,
        ahora_utc=dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc), es_cierre=True, corte_manana=False, episodio=None,
        precio_apertura=float(px["Open"].iloc[i]))


# =============================== equivalencia con el sistema real ===============================
def test_las_metricas_vectorizadas_coinciden_con_el_semaforo_real_dia_por_dia():
    px, nd = mercado_sintetico()
    m = V.metricas_vectorizadas(px, nd, CFG)
    color = V.clasificar_vectorizado(m, CFG)
    comparados = 0
    for i in range(V.WARMUP, len(px)):
        real = evaluar(entrada(px, nd, i), CFG, lambda t: 0.0)
        fila = m.iloc[i]
        for nombre, valor in (("z", real.z), ("z_res", real.z_res), ("beta", real.beta), ("vol_rel", real.vol_rel), ("resid5", real.resid5),
                              ("sigma20", real.sigma20), ("sigma_res", real.sigma_res), ("umbral5", real.umbral5)):
            assert fila[nombre] == pytest.approx(valor, rel=1e-6, abs=1e-9), (i, nombre)
        assert color.iloc[i] == real.color, (i, color.iloc[i], real.color, real.motivo)
        comparados += 1
    assert comparados > 300


def test_el_proxy_de_noticia_negativa_es_el_mismo_que_usa_el_sistema():
    px, nd = mercado_sintetico()
    m = V.metricas_vectorizadas(px, nd, CFG)
    for k in SHOCKS:                                                                            # caída de −9 %, volumen ×3, hueco −4 %: proxy activo
        assert bool(m["negativa"].iloc[k]) and evaluar(entrada(px, nd, k), CFG, lambda t: 0.0).proxy_noticias
    assert V.clasificar_vectorizado(m, CFG).iloc[list(SHOCKS)].tolist() == [ROJO] * 4
    assert int(m["negativa"].iloc[V.WARMUP:].sum()) >= 4


def test_el_semaforo_vectorizado_ve_todos_los_colores_en_una_serie_realista():
    px, nd = mercado_sintetico(n=1500, seed=7)
    cd = V.clasificar_vectorizado(V.metricas_vectorizadas(px, nd, CFG), CFG).iloc[V.WARMUP:]
    assert set(cd.unique()) == {VERDE, AZUL, AMARILLO, ROJO}
    assert (cd == VERDE).mean() > 0.9                                                           # la gran mayoría de los días es normal


def test_sin_mirar_al_futuro_las_metricas_de_un_dia_no_cambian_si_se_borra_lo_posterior():
    px, nd = mercado_sintetico()
    completo = V.metricas_vectorizadas(px, nd, CFG)
    for i in (120, 200, 301):
        parcial = V.metricas_vectorizadas(px.iloc[: i + 1], nd.iloc[: i + 1], CFG)
        pd.testing.assert_series_equal(completo.iloc[i], parcial.iloc[i], check_names=False)
    alterado = px.copy()
    alterado.iloc[250:, :] = alterado.iloc[250:, :] * 3                                         # el futuro cambia por completo…
    otro = V.metricas_vectorizadas(alterado, nd, CFG)
    pd.testing.assert_series_equal(completo.iloc[240], otro.iloc[240], check_names=False)       # …y el día 240 no se entera


# =============================== retornos futuros ===============================
def test_los_retornos_futuros_parten_de_la_apertura_del_dia_siguiente():
    px, nd = mercado_sintetico()
    beta = pd.Series(1.5, index=px.index)
    f = V.retornos_futuros(px, nd, beta, [1, 5])
    t = 100
    assert f["R1"].iloc[t] == pytest.approx(px["Close"].iloc[t + 1] / px["Open"].iloc[t + 1] - 1)
    assert f["R5"].iloc[t] == pytest.approx(px["Close"].iloc[t + 5] / px["Open"].iloc[t + 1] - 1)
    n5 = nd["Close"].iloc[t + 5] / nd["Open"].iloc[t + 1] - 1
    assert f["AR5"].iloc[t] == pytest.approx(f["R5"].iloc[t] - 1.5 * n5)
    assert f["R5"].iloc[-5:].isna().all() and f["R1"].iloc[-1:].isna().all()                    # sin datos del futuro no se inventa nada


def test_el_retorno_futuro_ignora_el_cierre_de_la_propia_senal():
    px, nd = mercado_sintetico()
    beta = pd.Series(1.0, index=px.index)
    base = V.retornos_futuros(px, nd, beta, [5]).iloc[100]
    px2 = px.copy()
    px2.loc[px.index[100], ["Close", "High", "Low"]] *= 0.5                                     # el cierre del día de la señal cambia…
    assert V.retornos_futuros(px2, nd, beta, [5]).iloc[100]["R5"] == pytest.approx(base["R5"])  # …pero la entrada es la apertura siguiente


# =============================== episodios ROJO → NEGRO ===============================
def serie_episodio(cierres, colores, vols=None):
    idx = pd.bdate_range("2024-01-02", periods=len(cierres))
    px = pd.DataFrame(dict(Close=cierres), index=idx)
    return px, pd.Series(colores, index=idx, dtype=object), pd.Series(vols if vols is not None else [1.0] * len(cierres), index=idx)


def test_negro_se_declara_al_cierre_del_dia_2_si_no_recupero_el_50_por_ciento():
    # día 0 normal (100) · día 1 ROJO cierra 90 (caída 10) · día 2 cierra 92 (recuperó 20 % < 50 %) → NEGRO · día 3 cierra 94 · día 4 cierra 96 (60 %) → sale
    px, cd, vr = serie_episodio([100, 90, 92, 94, 96, 97], [VERDE, ROJO, VERDE, VERDE, VERDE, VERDE])
    e = V.episodios(px, cd, vr, CFG)
    assert e["nuevo_rojo"].tolist() == [False, True, False, False, False, False]
    assert e["nuevo_negro"].tolist() == [False, False, True, False, False, False]
    assert e["color"].tolist() == [VERDE, ROJO, NEGRO, NEGRO, VERDE, VERDE]
    assert e["en_negro"].tolist() == [False, False, True, True, False, False]


def test_si_recupera_el_50_por_ciento_el_dia_2_no_hay_negro():
    px, cd, vr = serie_episodio([100, 90, 96, 97], [VERDE, ROJO, VERDE, VERDE])
    e = V.episodios(px, cd, vr, CFG)
    assert e["nuevo_rojo"].sum() == 1 and e["nuevo_negro"].sum() == 0 and NEGRO not in e["color"].tolist()


def test_variante_volumen_dia2_exige_que_la_noticia_siga_viva():
    px, cd, vr = serie_episodio([100, 90, 92, 93], [VERDE, ROJO, VERDE, VERDE], vols=[1.0, 3.0, 0.9, 1.0])
    assert V.episodios(px, cd, vr, CFG, "siempre")["nuevo_negro"].sum() == 1
    assert V.episodios(px, cd, vr, CFG, "volumen_dia2")["nuevo_negro"].sum() == 0               # volumen del día 2 < 1,5×: la noticia ya se enfrió


def test_un_segundo_rojo_dentro_del_episodio_no_abre_otro_episodio():
    px, cd, vr = serie_episodio([100, 90, 85, 86], [VERDE, ROJO, ROJO, VERDE])
    e = V.episodios(px, cd, vr, CFG)
    assert e["nuevo_rojo"].sum() == 1


# =============================== bootstrap por bloques ===============================
def test_bootstrap_detecta_un_efecto_claro_y_no_inventa_uno_inexistente():
    rng = np.random.default_rng(3)
    bloques = np.repeat(np.arange(60), 5).astype(str)
    claro = rng.normal(0.03, 0.02, 300)
    media, lo, hi, p = V.bootstrap_por_bloques(claro, bloques, 3000, 1)
    assert lo < media < hi and lo > 0 and p < 0.01
    ruido = rng.normal(0.0, 0.02, 300)
    m2, lo2, hi2, p2 = V.bootstrap_por_bloques(ruido, bloques, 3000, 1)
    assert lo2 < 0 < hi2 and p2 > 0.05


def test_bootstrap_por_bloques_es_mas_conservador_que_remuestrear_datos_sueltos():
    rng = np.random.default_rng(5)
    nivel_bloque = np.repeat(rng.normal(0, 0.03, 40), 10)                                      # cada mes tiene su propio "clima" (crisis, rally…)
    vals = nivel_bloque + rng.normal(0, 0.005, 400)
    bloques = np.repeat(np.arange(40), 10).astype(str)
    _, lo_b, hi_b, _ = V.bootstrap_por_bloques(vals, bloques, 3000, 2)
    _, lo_i, hi_i, _ = V.bootstrap_por_bloques(vals, np.arange(400).astype(str), 3000, 2)
    assert (hi_b - lo_b) > 2 * (hi_i - lo_i)


def test_bootstrap_es_reproducible_con_la_misma_semilla():
    v = np.random.default_rng(0).normal(0.01, 0.02, 100)
    b = np.repeat(np.arange(20), 5).astype(str)
    assert V.bootstrap_por_bloques(v, b, 1000, 9) == V.bootstrap_por_bloques(v, b, 1000, 9)


# =============================== tablas y criterio ===============================
def panel_sintetico():
    px, nd = mercado_sintetico(n=700, seed=11)
    a = V.estudio_activo("TSLA", px, nd, CFG, "2023-06-01")
    px2, nd2 = mercado_sintetico(n=700, seed=12)
    b = V.estudio_activo("KO", px2, nd2, CFG, "2023-06-01")
    return pd.concat([a, b])


def test_estudio_activo_trae_metricas_colores_y_retornos_futuros():
    p = panel_sintetico()
    assert {"color_dia", "color", "nuevo_rojo", "nuevo_negro", "z_res", "vol_rel", "AR1", "AR5", "AR10", "AR20", "R20"} <= set(p.columns)
    assert p["color_dia"].dropna().isin([VERDE, AZUL, AMARILLO, ROJO]).all()
    assert p["ticker"].nunique() == 2 and p.index.min() >= pd.Timestamp("2023-06-01")


def test_tabla_eventos_resume_por_grupo_clase_horizonte_y_calcula_el_valor_de_salir_neto():
    p = panel_sintetico()
    t = V.tabla_eventos(p, CFG, {"grandes": ["TSLA"], "resto": ["KO"]})
    assert set(t["grupo"]) == {"grandes", "resto"} and set(t["clase"]) == set(V.CLASES) and set(t["medida"]) == {"AR", "ARx", "R"}
    assert len(t) == 2 * 5 * 4 * 3
    ar = t[(t["medida"] == "AR") & (t["n"] > 0)]
    assert np.allclose(ar["salir_neto"], -ar["media"] - CFG["costos_v5"]["costo_cambio"])
    base = t[(t["clase"] == VERDE) & (t["medida"] == "AR") & (t["h"] == 5) & (t["grupo"] == "resto")].iloc[0]
    assert base["n"] > 300 and base["ic_lo"] <= base["media"] <= base["ic_hi"]


def test_el_exceso_resta_la_deriva_normal_de_cada_accion():
    p = V.agregar_exceso(panel_sintetico(), [5])
    for _, g in p.groupby("ticker"):
        assert g["AR5x"].mean() == pytest.approx(0.0, abs=1e-12)


def tabla_falsa(h5, h10, n=40):
    filas = []
    for h, (media, lo, hi) in ((5, h5), (10, h10)):
        filas.append(dict(grupo="grandes", clase=NEGRO, h=h, medida="AR", n=n, media=media, ic_lo=lo, ic_hi=hi, p=0.01, salir_neto=-media - 0.013))
    return pd.DataFrame(filas)


def test_criterio_negro_exige_deriva_a_la_baja_en_todos_los_horizontes():
    assert V.criterio_negro(tabla_falsa((-0.03, -0.05, -0.01), (-0.04, -0.07, -0.01)), CFG)["deriva"] is True
    assert V.criterio_negro(tabla_falsa((-0.03, -0.05, -0.01), (-0.01, -0.05, 0.03)), CFG)["deriva"] is False          # a 10 sesiones el IC cruza 0
    assert V.criterio_negro(tabla_falsa((0.01, -0.02, 0.04), (0.02, -0.02, 0.05)), CFG)["deriva"] is False              # sube después de NEGRO
    pocos = V.criterio_negro(tabla_falsa((-0.03, -0.05, -0.01), (-0.04, -0.07, -0.01), n=5), CFG)
    assert pocos["deriva"] is False and pocos["datos_suficientes"] is False                                              # n < n_minimo: no se concluye


def test_criterio_negro_distingue_deriva_estadistica_de_ganancia_neta_de_costos():
    c = V.criterio_negro(tabla_falsa((-0.005, -0.009, -0.001), (-0.006, -0.01, -0.002)), CFG)
    assert c["deriva"] is True and c["rentable_neto"] is False                                  # cae algo, pero menos que el 1,3 % que cuesta cambiar
    c2 = V.criterio_negro(tabla_falsa((-0.03, -0.05, -0.01), (-0.04, -0.07, -0.01)), CFG)
    assert c2["rentable_neto"] is True
