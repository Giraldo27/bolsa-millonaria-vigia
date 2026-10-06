"""Pruebas de la Fase 2: sentimiento, semáforo (5 clases + episodio NEGRO), MEE, banco de relevo y Regla Maestra."""
import datetime as dt
import math

import numpy as np
import pandas as pd
import pytest

from src.data_sources import Noticia
from src.regla_maestra import (CAMBIAR, MANTENER, SIN_DATOS, VENDER_TODO, ContextoRegla, g_de, multiplo_requerido, regla_maestra)
from src.relevo import Candidato, calcular_mee, construir_banco, reporte_antes_del_corte
from src.semaforo import (AMARILLO, AZUL, NEGRO, ROJO, VERDE, EntradaSemaforo, analizar_noticias, avanzar_episodio, calcular_metricas,
                          clasificar_base, contar_sesiones, evaluar, perfil_volumen, proyectar_volumen)
from src.sentimiento import crear_puntuador, palabras_clave, puntuar_vader
from tests.helpers import make_cfg

CFG = make_cfg()
KW = CFG["semaforo"]["palabras_negativas"]
AHORA = dt.datetime(2026, 10, 14, 15, 0, tzinfo=dt.timezone.utc)


# =============================== sentimiento ===============================
def test_palabras_clave_palabra_completa_y_sin_mayusculas():
    assert palabras_clave("Tesla faces SEC probe over deliveries", KW) == ["deliveries", "probe", "sec"]
    assert palabras_clave("Second quarter secures growth", KW) == []              # 'second' / 'secures' no son 'sec'
    assert palabras_clave("Analyst DOWNGRADES Tesla after recall", KW) == ["downgrades", "recall"]


def test_vader_distingue_tono():
    assert puntuar_vader("Tesla stock plunges after fraud investigation and lawsuit") < -0.3
    assert puntuar_vader("Tesla beats estimates with record profits and strong growth") > 0.3


def test_finbert_cae_a_vader_si_no_esta_instalado():
    cfg = make_cfg()
    cfg["sentimiento"]["motor"] = "finbert"
    f, motor = crear_puntuador(cfg)
    assert motor in ("vader", "finbert") and isinstance(f("Tesla plunges"), float)


# =============================== utilidades de prueba ===============================
def serie(n=120, seed=3, beta=1.2, sd_mercado=0.01, sd_ruido=0.004):
    rng = np.random.default_rng(seed)
    rn = rng.normal(0, sd_mercado, n)
    ra = beta * rn + rng.normal(0, sd_ruido, n)
    idx = pd.bdate_range(end="2026-10-13", periods=n)
    return (pd.Series(100 * np.cumprod(1 + ra), index=idx), pd.Series(1000 * np.cumprod(1 + rn), index=idx),
            pd.Series(rng.uniform(0.9e6, 1.1e6, n), index=idx))


def entrada(z=0.0, z_res=0.0, vol_rel=1.0, noticias=None, es_cierre=False, corte_manana=False, episodio=None, fecha=dt.date(2026, 10, 14),
            ajustar_cierres=None, precio_fijo=None):
    c, n, v = serie()
    if ajustar_cierres is not None:
        c = ajustar_cierres(c)
    base = EntradaSemaforo("TSLA", fecha, 100.0, float(c.iloc[-1]), c, v, 1.0e6, n, None, float(n.iloc[-1]), noticias, 10.0, AHORA,
                           es_cierre, corte_manana, episodio)
    m = calcular_metricas(EntradaSemaforo(**{**base.__dict__, "precio": base.cierre_previo, "ndx_precio": base.ndx_cierre_previo}), CFG)
    r = z * m["sigma20"]
    rn = (r - z_res * m["sigma_res"]) / m["beta"]
    base.precio = float(c.iloc[-1]) * (1 + r) if precio_fijo is None else precio_fijo
    base.ndx_precio = float(n.iloc[-1]) * (1 + rn)
    base.volumen_dia = vol_rel * float(v.tail(20).mean())
    return base


def noticia(titulo, horas=2, ticker="TSLA"):
    ts = (AHORA - dt.timedelta(hours=horas)).isoformat()
    return Noticia(ts, titulo, "", "Reuters", "http://x", "finnhub", ticker)


NEG = [noticia("Tesla recalls vehicles amid federal probe and lawsuit", 3), noticia("Tesla stock plunges after downgrade", 5)]
POS = [noticia("Tesla beats estimates with record profits and strong growth", 3)]


def color(e):
    return evaluar(e, CFG, puntuar_vader).color


# =============================== clases del semáforo ===============================
def test_metricas_z_zres_y_vol_rel_coinciden_con_lo_pedido():
    e = entrada(z=-2.5, z_res=-3.0, vol_rel=2.0)
    m = calcular_metricas(e, CFG)
    assert m["z"] == pytest.approx(-2.5, abs=1e-6) and m["z_res"] == pytest.approx(-3.0, abs=1e-6) and m["vol_rel"] == pytest.approx(2.0)
    assert 1.0 < m["beta"] < 1.4 and m["sigma_res"] < m["sigma20"]


def test_verde_si_z_mayor_que_menos_2():
    assert color(entrada(z=-1.9, z_res=-1.9)) == VERDE and color(entrada(z=1.0, z_res=1.0)) == VERDE


def test_azul_si_la_caida_la_explica_el_mercado():
    assert color(entrada(z=-2.5, z_res=-1.0, vol_rel=2.0, noticias=NEG)) == AZUL


def test_amarillo_sin_noticia_negativa():
    assert color(entrada(z=-2.5, z_res=-3.0, vol_rel=2.0, noticias=POS)) == AMARILLO
    assert color(entrada(z=-2.5, z_res=-3.0, vol_rel=2.0, noticias=[])) == AMARILLO


def test_amarillo_con_noticia_negativa_pero_volumen_bajo():
    assert color(entrada(z=-2.5, z_res=-3.0, vol_rel=1.2, noticias=NEG)) == AMARILLO


def test_rojo_con_noticia_negativa_y_volumen():
    r = evaluar(entrada(z=-2.5, z_res=-3.0, vol_rel=1.6, noticias=NEG), CFG, puntuar_vader)
    assert r.color == ROJO and r.noticia_negativa and "noticia negativa" in r.motivo and r.claves


def test_zona_gris_entre_menos_2_y_menos_1_5_se_trata_como_configurado():
    e = entrada(z=-2.4, z_res=-1.8, vol_rel=2.0, noticias=NEG)
    assert color(e) == AMARILLO
    cfg = make_cfg()
    cfg["semaforo"]["zona_gris"] = "AZUL"
    assert evaluar(e, cfg, puntuar_vader).color == AZUL


def test_umbrales_exactos():
    # se usan márgenes mínimos para evitar el error de redondeo de los flotantes en el borde exacto
    assert color(entrada(z=-2.01, z_res=-1.49, vol_rel=2.0, noticias=NEG)) == AZUL        # z_res > -1,5
    assert color(entrada(z=-2.01, z_res=-1.51, vol_rel=2.0, noticias=NEG)) == AMARILLO    # entre -2 y -1,5: zona gris
    assert color(entrada(z=-2.01, z_res=-2.01, vol_rel=1.51, noticias=NEG)) == ROJO       # z_res <= -2 y vol_rel >= 1,5
    assert color(entrada(z=-2.01, z_res=-2.01, vol_rel=1.49, noticias=NEG)) == AMARILLO
    assert color(entrada(z=-1.99, z_res=-3.0, vol_rel=3.0, noticias=NEG)) == VERDE        # z > -2: normal


def test_rojo_por_residual_acumulado_de_5_dias_con_noticias_negativas():
    def deprimir(c):                                                    # los últimos 4 días caen por su cuenta (no por el mercado)
        c = c.copy()
        c.iloc[-4:] = c.iloc[-5] * np.cumprod([0.98] * 4)
        return c
    e = entrada(z=-0.5, z_res=-0.5, vol_rel=1.0, noticias=NEG, ajustar_cierres=deprimir)
    r = evaluar(e, CFG, puntuar_vader)
    assert r.resid5 <= r.umbral5 and r.color == ROJO and "acumulado" in r.motivo
    assert evaluar(entrada(z=-0.5, z_res=-0.5, noticias=POS, ajustar_cierres=deprimir), CFG, puntuar_vader).color == VERDE   # sin noticia negativa no


def test_sin_datos_de_noticias_nunca_da_rojo_y_lo_avisa():
    r = evaluar(entrada(z=-2.5, z_res=-3.0, vol_rel=3.0, noticias=None), CFG, puntuar_vader)
    assert r.color == AMARILLO and r.sin_noticias and any("sin datos de noticias" in a for a in r.advertencias)


def test_volumen_no_disponible_impide_rojo():
    e = entrada(z=-2.5, z_res=-3.0, noticias=NEG)
    e.volumen_dia = None
    r = evaluar(e, CFG, puntuar_vader)
    assert r.color == AMARILLO and r.vol_rel is None


# =============================== noticias ===============================
def analiza(noticias, base=10.0):
    return analizar_noticias(noticias, base, AHORA, CFG, puntuar_vader)


def test_noticias_solo_cuentan_las_ultimas_24_horas():
    r = analiza([noticia("Tesla recalls cars", horas=30), noticia("Normal day", horas=2)])
    assert r["n24"] == 1


def test_palabra_clave_en_titular_positivo_no_cuenta_como_negativa():
    r = analiza([noticia("Tesla beats expectations with record deliveries and strong growth", 2)])
    assert r["negativa"] is False


def test_titular_positivo_con_palabra_anuladora_no_cuenta_aunque_vader_lo_vea_neutro():
    for t in ("Gary Black Flags Waymo Is Leaving TSLA Behind On Autonomy Even As Tesla Beats Delivery Est",
              "Tesla tops delivery estimates", "Analyst upgrades Tesla after delivery record"):
        r = analiza([noticia(t, 2)])
        assert r["negativa"] is False, t
    assert analiza([noticia("Tesla delivery numbers miss expectations", 2)])["negativa"] is True


def test_palabra_clave_en_titular_no_positivo_si_cuenta():
    assert analiza([noticia("Tesla delivery numbers disappoint", 2)])["negativa"] is True


def test_sentimiento_medio_bajo_es_noticia_negativa_sin_palabras_clave():
    r = analiza([noticia("Stock crashes as investors panic and sell off in terrible day", 2),
                 noticia("Shares tumble amid awful sentiment and fear", 3)])
    assert r["sentimiento"] < -0.3 and r["negativa"]


def test_ratio_de_titulares_y_claves_ordenadas():
    r = analiza(NEG + POS, base=1.0)
    assert r["n24"] == 3 and r["ratio"] == pytest.approx(3.0)
    assert r["claves"][0]["palabras"] and len(r["claves"]) <= 3


def test_sin_titulares_en_24h_no_hay_noticia_negativa():
    r = analiza([])
    assert r["negativa"] is False and r["disponible"] is True and r["n24"] == 0 and analiza(None)["disponible"] is False


# =============================== respaldos de noticias ===============================
def noticia_es(titulo, horas=2):
    n = noticia(titulo, horas)
    n.idioma = "es"
    return n


def test_puntuar_es_lexico_y_titulares_en_espanol():
    from src.sentimiento import puntuar_es
    assert puntuar_es("Ecopetrol anuncia utilidades récord y dividendo", CFG) > 0.3
    assert puntuar_es("Ecopetrol enfrenta demanda y sanción tras derrame", CFG) < -0.3
    assert puntuar_es("Ecopetrol abre sede en Bogotá", CFG) == 0.0


def test_noticias_en_espanol_usan_su_lexico_y_sus_palabras_clave():
    r = analiza([noticia_es("Ecopetrol enfrenta demanda y sanción tras derrame de crudo", 2)])
    assert r["negativa"] and r["sentimiento"] < -0.3
    assert analiza([noticia_es("Ecopetrol anuncia utilidades récord y reparte dividendo", 2)])["negativa"] is False
    # palabra clave en español anulada por una positiva ("la caída de costos supera expectativas")
    assert analiza([noticia_es("Caída de costos: Ecopetrol supera expectativas y gana más", 2)])["negativa"] is False


def test_respaldo_proxy_sin_noticias_presume_noticia_negativa_con_volumen_y_hueco():
    def con_apertura(e, gap):
        e.precio_apertura = e.cierre_previo * (1 + gap)
        return e
    e = con_apertura(entrada(z=-2.5, z_res=-3.0, vol_rel=2.5, noticias=None), -0.03)
    r = evaluar(e, CFG, puntuar_vader)
    assert r.color == ROJO and r.proxy_noticias and "SIN NOTICIAS" in r.motivo and r.sin_noticias
    # sin hueco de apertura o con volumen normal NO se presume nada
    assert evaluar(con_apertura(entrada(z=-2.5, z_res=-3.0, vol_rel=2.5, noticias=None), 0.0), CFG, puntuar_vader).color == AMARILLO
    assert evaluar(con_apertura(entrada(z=-2.5, z_res=-3.0, vol_rel=1.6, noticias=None), -0.03), CFG, puntuar_vader).color == AMARILLO
    # con datos de noticias el proxy no se usa
    r2 = evaluar(con_apertura(entrada(z=-2.5, z_res=-3.0, vol_rel=2.5, noticias=POS), -0.03), CFG, puntuar_vader)
    assert r2.color == AMARILLO and not r2.proxy_noticias


def test_proxy_se_puede_apagar_por_config():
    cfg = make_cfg()
    cfg["semaforo"]["proxy_sin_noticias"]["activo"] = False
    e = entrada(z=-2.5, z_res=-3.0, vol_rel=2.5, noticias=None)
    e.precio_apertura = e.cierre_previo * 0.97
    assert evaluar(e, cfg, puntuar_vader).color == AMARILLO


def test_cobertura_parcial_se_advierte():
    e = entrada(z=-0.5, z_res=-0.5, noticias=[])
    e.cobertura_parcial = True
    r = evaluar(e, CFG, puntuar_vader)
    assert r.cobertura_parcial and any("cobertura parcial" in a for a in r.advertencias)


# =============================== episodio ROJO → NEGRO ===============================
def test_negro_al_cierre_del_dia_2_si_no_recupera_el_50pct_y_sentimiento_negativo():
    ep, col, _ = avanzar_episodio(None, ROJO, dt.date(2026, 10, 14), 100.0, 90.0, True, 1, True, CFG)
    assert col == ROJO and ep["cierre_dia1"] == 90.0 and ep["cierre_previo"] == 100.0
    ep2, col2, _ = avanzar_episodio(ep, ROJO, dt.date(2026, 10, 15), 90.0, 92.0, True, 2, True, CFG)    # recuperó 20 % de la caída
    assert col2 == NEGRO and ep2["estado"] == NEGRO


def test_no_es_negro_si_recupera_al_menos_la_mitad():
    ep, _, _ = avanzar_episodio(None, ROJO, dt.date(2026, 10, 14), 100.0, 90.0, True, 1, True, CFG)
    ep2, col, _ = avanzar_episodio(ep, VERDE, dt.date(2026, 10, 15), 90.0, 95.0, True, 2, True, CFG)    # recuperó 50 %
    assert ep2 is None and col == VERDE


def test_no_es_negro_si_el_sentimiento_ya_no_es_negativo():
    ep, _, _ = avanzar_episodio(None, ROJO, dt.date(2026, 10, 14), 100.0, 90.0, True, 1, True, CFG)
    ep2, col, _ = avanzar_episodio(ep, AMARILLO, dt.date(2026, 10, 15), 90.0, 91.0, True, 2, False, CFG)
    assert ep2 is None and col == AMARILLO


def test_dia_2_intradia_sigue_en_rojo_y_la_decision_es_al_cierre():
    ep, _, _ = avanzar_episodio(None, ROJO, dt.date(2026, 10, 14), 100.0, 90.0, True, 1, True, CFG)
    ep2, col, nota = avanzar_episodio(ep, VERDE, dt.date(2026, 10, 15), 90.0, 85.0, False, 2, True, CFG)
    assert col == ROJO and ep2["estado"] == ROJO and "cierre" in nota


def test_negro_persiste_hasta_recuperar_el_50pct():
    ep = {"inicio": "2026-10-14", "cierre_previo": 100.0, "cierre_dia1": 90.0, "estado": NEGRO}
    _, col, _ = avanzar_episodio(ep, VERDE, dt.date(2026, 10, 16), 91.0, 92.0, False, 3, False, CFG)
    assert col == NEGRO
    ep2, col2, _ = avanzar_episodio(ep, VERDE, dt.date(2026, 10, 16), 91.0, 96.0, False, 3, False, CFG)
    assert ep2 is None and col2 == VERDE


def test_episodio_completo_con_evaluar_y_contar_sesiones():
    c, _, _ = serie()
    d1 = c.index[-1]                                              # el día 1 ya es parte de los cierres completos de hoy (día 2)
    ep = {"inicio": d1.date().isoformat(), "cierre_previo": float(c.iloc[-2]), "cierre_dia1": float(c.iloc[-1]), "estado": ROJO}
    assert contar_sesiones(ep["inicio"], dt.date(2026, 10, 14), c) == 2
    e = entrada(z=-1.0, z_res=-1.0, noticias=NEG, es_cierre=True, episodio=ep, precio_fijo=float(c.iloc[-1]) * 0.99)
    r = evaluar(e, CFG, puntuar_vader)
    assert r.color == NEGRO and r.episodio["estado"] == NEGRO


def test_si_el_corte_es_manana_rojo_se_trata_como_negro():
    e = entrada(z=-2.5, z_res=-3.0, vol_rel=2.0, noticias=NEG, corte_manana=True)
    r = evaluar(e, CFG, puntuar_vader)
    assert r.color == NEGRO and r.color_calculado == ROJO and "corte es mañana" in r.motivo
    assert evaluar(entrada(z=-2.5, z_res=-3.0, vol_rel=2.0, noticias=POS, corte_manana=True), CFG, puntuar_vader).color == AMARILLO


# =============================== volumen intradía ===============================
def test_proyeccion_del_volumen_con_el_perfil_tipico():
    idx = pd.DatetimeIndex([f"2026-10-{d:02d} {h:02d}:{m:02d}" for d in (6, 7, 8, 9) for h in range(9, 16) for m in (0, 30)], tz="America/New_York")
    df = pd.DataFrame({"Volume": 100.0}, index=idx)
    perfil = perfil_volumen(df, 20)                               # 14 barras iguales por día
    assert perfil.iloc[-1] == pytest.approx(1.0) and perfil.loc["11:30"] == pytest.approx(6 / 14)
    assert proyectar_volumen(600.0, "11:30", perfil) == pytest.approx(1400.0)
    assert proyectar_volumen(100.0, "09:00", perfil) is None      # fracción < 8 %: demasiado temprano para proyectar


# =============================== MEE ===============================
def test_mee_con_iv_y_respaldo():
    v, m = calcular_mee(0.40, 30, 0.02, 5)
    assert m == "IV" and v == pytest.approx(0.40 * math.sqrt(30 / 365))
    v, m = calcular_mee(None, 30, 0.02, 5)                        # respaldo 'consistente': mismo reloj que la IV
    assert m == "σ20" and v == pytest.approx(0.02 * math.sqrt(252) * math.sqrt(30 / 365))
    v, m = calcular_mee(0.40, 30, 0.02, 5, es_local=True)         # acciones locales: siempre σ, nunca IV
    assert m == "σ20" and v == pytest.approx(0.02 * math.sqrt(252) * math.sqrt(30 / 365))
    assert calcular_mee(None, 30, None, 5) == (None, "sin_datos")


def test_mee_respaldo_literal_es_sigma_por_raiz_de_sesiones():
    v, m = calcular_mee(None, 5, 0.02, 5, respaldo="literal")
    assert m == "σ20" and v == pytest.approx(0.02 * math.sqrt(5))


def test_respaldo_consistente_equivale_a_la_iv_del_mismo_activo():
    sigma = 0.025                                                 # activo cuya IV anualizada = σ × √252
    iv = sigma * math.sqrt(252)
    a, _ = calcular_mee(iv, 5, sigma, 5)
    b, _ = calcular_mee(None, 5, sigma, 5)
    assert a == pytest.approx(b)                                  # sin sesgo a favor de los candidatos sin opciones
    c, _ = calcular_mee(None, 5, sigma, 5, respaldo="literal")
    assert c > a * 1.15                                           # el literal sí sesga ~+17 %


def test_reporte_antes_del_corte():
    rep = [{"fecha": "2026-10-28", "hora": "amc"}, {"fecha": "2026-10-21", "hora": "amc"}]
    assert reporte_antes_del_corte(rep, dt.date(2026, 10, 19), dt.date(2026, 10, 23)) == "2026-10-21"
    assert reporte_antes_del_corte(rep, dt.date(2026, 10, 19), dt.date(2026, 10, 20)) is None


# =============================== banco de relevo ===============================
def cand(t, mee, grupo="mgc", **k):
    return Candidato(t, grupo, mee, "IV", valor_negociado_mm=k.pop("liq", 5000.0), **k)


def tickers(banco):
    return [e.c.ticker for e in banco]


def test_lista_negra_nunca_aparece_en_el_banco():
    negra = CFG["universe"]["blacklist"]
    cs = [cand(t, 0.99, grupo="local", liq=10 ** 6) for t in negra] + [cand("META", 0.09)]
    assert tickers(construir_banco(cs, "TSLA", CFG)) == ["META"]


def test_filtros_rojo_negro_actual_liquidez_y_universo():
    cs = [cand("META", 0.09, color=ROJO), cand("AMZN", 0.08, color=NEGRO), cand("TSLA", 0.10), cand("NVDA", 0.07, liq=10.0),
          cand("UBER", 0.06, color=AMARILLO), cand("AMD", 0.20), cand("MSFT", 0.05, liq=None), cand("GOOGL", 0.05, color=AZUL)]
    assert tickers(construir_banco(cs, "TSLA", CFG)) == ["UBER", "GOOGL"]


def test_exigir_noticias_excluye_activos_sin_datos_de_noticias():
    cfg = make_cfg()
    cs = [cand("ISA", 0.09, grupo="local", liq=5000, sin_noticias=True), cand("META", 0.08), cand("AMZN", 0.07), cand("UBER", 0.06),
          cand("GOOGL", 0.05)]
    assert cfg["banco"]["exigir_noticias"] is True                                        # por defecto las noticias influyen
    assert tickers(construir_banco(cs, "TSLA", cfg)) == ["META", "AMZN", "UBER", "GOOGL"]
    cfg["banco"]["exigir_noticias"] = False
    assert tickers(construir_banco(cs, "TSLA", cfg))[0] == "ISA"


def test_respaldo_banco_se_relaja_si_las_noticias_estan_caidas():
    """Si por una caída de las fuentes casi nadie tiene noticias, el banco no queda vacío: se relaja y se marca 'sin datos de noticias'."""
    cs = [cand("ISA", 0.09, grupo="local", liq=5000, sin_noticias=True), cand("META", 0.08, sin_noticias=True),
          cand("AMZN", 0.07, sin_noticias=True), cand("UBER", 0.06)]
    b = construir_banco(cs, "TSLA", CFG)                                                  # sólo 1 con noticias (< 3): se relaja
    assert tickers(b) == ["ISA", "META", "AMZN", "UBER"]
    assert all(any("sin datos de noticias" in r for r in e.razones) for e in b if e.c.sin_noticias)


def test_liquidez_distinta_para_locales():
    cs = [cand("ISA", 0.05, grupo="local", liq=3500), cand("EXITO", 0.06, grupo="local", liq=2500)]      # el mínimo local subió a $3.000 millones (6-oct-2026)
    assert tickers(construir_banco(cs, "TSLA", CFG)) == ["ISA"]


def test_orden_mee_con_tolerancia_correlacion_catalizador_y_desempate():
    cs = [cand("A_" , 0.100), cand("META", 0.100, corr=0.60), cand("AMZN", 0.095, corr=0.20),
          cand("UBER", 0.093, corr=0.20, reporte_antes_corte="2026-10-28"), cand("GOOGL", 0.093, corr=0.20, vol_rel=2.0, z=0.5),
          cand("MSFT", 0.080, corr=0.0)]
    cs = [c for c in cs if c.ticker != "A_"]
    con_empates = {**CFG, "banco": {**CFG["banco"], "tolerancia_mee": 0.10}}
    assert tickers(construir_banco(cs, "TSLA", CFG)) == ["META", "AMZN", "UBER", "GOOGL", "MSFT"]      # configuración actual (tolerancia 0): orden estricto por movimiento
    t = tickers(construir_banco(cs, "TSLA", con_empates))
    # grupo empatado (MEE a menos de 10 % de 0,100): menor correlación primero; entre iguales, con catalizador; luego desempate; MSFT después
    assert t == ["UBER", "GOOGL", "AMZN", "META", "MSFT"] and t[-1] == "MSFT"


def test_top_7_y_motivos():
    cs = [cand(t, 0.10 - i * 0.012, corr=0.3) for i, t in enumerate(["META", "AMZN", "UBER", "GOOGL", "MSFT", "NVDA", "AAPL", "JPM", "V", "MA"])]
    b = construir_banco(cs, "TSLA", CFG)
    assert len(b) == 7 and all(any("MEE" in r for r in e.razones) for e in b)


def test_excluir_candidatos_en_semana_final():
    cs = [cand("NVDA", 0.12), cand("META", 0.10)]
    assert tickers(construir_banco(cs, "TSLA", CFG, excluir={"TSLA", "NVDA"})) == ["META"]


def test_desempate_exige_volumen_sin_caida_o_sentimiento_positivo():
    cs = [cand("META", 0.10, corr=0.3, vol_rel=2.0, z=-3.0), cand("AMZN", 0.10, corr=0.3, sentimiento=0.4), cand("UBER", 0.10, corr=0.3)]
    assert tickers(construir_banco(cs, "TSLA", CFG))[0] == "AMZN" and tickers(construir_banco(cs, "TSLA", CFG))[-1] in ("META", "UBER")


# =============================== Regla Maestra ===============================
def ctx(**k):
    base = dict(mia=6.0, objetivo=11.0, activo="TSLA", color_actual=VERDE, mee_actual=0.065, cambios_restantes=4, cambio_hoy=False)
    base.update(k)
    return ContextoRegla(**base)


def test_tabla_de_multiplos_del_documento():
    esperado = {1: 2.30, 2: 1.65, 3: 1.43, 4: 1.33, 5: 1.26, 6: 1.22, 8: 1.16, 10: 1.13, 15: 1.09, 20: 1.07}
    for g, m in esperado.items():
        assert multiplo_requerido(g, 1.3) == pytest.approx(m, abs=0.006)


def test_ejemplo_del_documento_cambia_a_meta_y_con_10_mantiene():
    cs = [cand("META", 0.090, corr=0.26)]
    d, _ = regla_maestra(ctx(mia=6.0), cs, CFG)
    assert d.accion == CAMBIAR and d.candidato == "META" and d.g == pytest.approx(6.0) and d.multiplo == pytest.approx(1.2167, abs=1e-3)
    assert d.ratio == pytest.approx(0.090 / 0.065)
    d, _ = regla_maestra(ctx(mia=10.0), cs, CFG)
    assert d.accion == MANTENER and d.g == pytest.approx(2.0) and d.multiplo == pytest.approx(1.65)


def test_g_menor_o_igual_a_cero_mantiene():
    d, _ = regla_maestra(ctx(mia=12.0, objetivo=11.0), [cand("META", 0.5)], CFG)
    assert d.accion == MANTENER and d.g == pytest.approx(0.0) and "g = 0,00" in d.razones[0] and g_de(12, 11, CFG) == 0


def test_faltan_datos_de_ranking():
    d, _ = regla_maestra(ctx(mia=None), [], CFG)
    assert d.accion == SIN_DATOS and "/rank" in d.razones[0]


def test_limites_un_cambio_por_dia_y_cuatro_en_total():
    cs = [cand("META", 0.5)]
    assert regla_maestra(ctx(cambio_hoy=True), cs, CFG)[0].accion == MANTENER
    d, _ = regla_maestra(ctx(cambios_restantes=0), cs, CFG)
    assert d.accion == MANTENER and "4 cambios" in " ".join(d.razones)


def test_sin_mee_actual_o_banco_vacio():
    assert regla_maestra(ctx(mee_actual=None), [cand("META", 0.5)], CFG)[0].accion == SIN_DATOS
    d, b = regla_maestra(ctx(), [], CFG)
    assert d.accion == MANTENER and b == [] and "vacío" in " ".join(d.razones)


def test_si_ninguno_cumple_el_multiplo_mantiene_y_muestra_el_mejor_ratio():
    d, _ = regla_maestra(ctx(mia=6.0), [cand("META", 0.070, corr=0.3), cand("AMZN", 0.072, corr=0.3)], CFG)
    assert d.accion == MANTENER and d.ratio == pytest.approx(0.072 / 0.065) and "MANTENER" in d.razones[-1]


def test_elige_el_primero_del_banco_que_cumple_no_el_de_mayor_mee_si_hay_empate():
    cs = [cand("META", 0.100, corr=0.60), cand("AMZN", 0.096, corr=0.10)]
    d, _ = regla_maestra(ctx(mia=6.0), cs, {**CFG, "banco": {**CFG["banco"], "tolerancia_mee": 0.10}})
    assert d.accion == CAMBIAR and d.candidato == "AMZN"          # con tolerancia: empatados (< 10 %), gana la menor correlación
    d0, _ = regla_maestra(ctx(mia=6.0), cs, CFG)
    assert d0.accion == CAMBIAR and d0.candidato == "META"        # configuración actual (tolerancia 0): la de mayor movimiento, la misma que nombran todas las respuestas


CFG_NEGRO = make_cfg()                                              # el modo NEGRO sigue existiendo: se activa a propósito (en config.yaml viene apagado, Fase 3)
CFG_NEGRO["regla_maestra"]["cambio_por_negro"] = True


def test_la_fase_3_dejo_cambio_por_negro_apagado_por_omision():
    assert CFG["regla_maestra"]["cambio_por_negro"] is False          # NEGRO no mostró deriva a la baja en acciones grandes (reports/validacion_semaforo.html)


def test_modo_negro_atras_usa_g_mas_d():
    d, _ = regla_maestra(ctx(mia=6.0, color_actual=NEGRO), [cand("META", 0.066, corr=0.3)], CFG_NEGRO)
    assert d.modo_negro and d.multiplo == pytest.approx((6 + 1.3) / (6 + 1.5))
    assert d.accion == CAMBIAR                                    # 0,066/0,065 = 1,015 ≥ 0,973


def test_modo_negro_exige_menos_que_el_modo_normal():
    cs = [cand("META", 0.068, corr=0.3)]                          # ratio 1,05: no alcanza 1,22 normal, sí alcanza 0,97 en NEGRO
    assert regla_maestra(ctx(mia=6.0), cs, CFG)[0].accion == MANTENER
    assert regla_maestra(ctx(mia=6.0, color_actual=NEGRO), cs, CFG_NEGRO)[0].accion == CAMBIAR


def test_modo_negro_adelante_por_3_puntos_o_menos_tambien_aplica():
    cs = [cand("META", 0.060, corr=0.3)]                          # ratio 0,92 ≥ 1,3/1,5 = 0,867
    d, _ = regla_maestra(ctx(mia=13.0, objetivo=11.0, color_actual=NEGRO), cs, CFG_NEGRO)
    assert d.accion == CAMBIAR and d.multiplo == pytest.approx(1.3 / 1.5)


def test_modo_negro_adelante_por_mas_de_3_solo_cambia_si_cae_15pct_desde_la_entrada():
    cs = [cand("META", 0.090, corr=0.3)]
    d, _ = regla_maestra(ctx(mia=16.0, objetivo=11.0, color_actual=NEGRO, rend_desde_entrada_pp=-8.0), cs, CFG_NEGRO)
    assert d.accion == MANTENER and "-15" in " ".join(d.razones)
    d, _ = regla_maestra(ctx(mia=16.0, objetivo=11.0, color_actual=NEGRO, rend_desde_entrada_pp=-16.0), cs, CFG_NEGRO)
    assert d.accion == CAMBIAR


def test_cambio_por_negro_apagado_trata_negro_como_informacion():
    cfg = make_cfg()
    cfg["regla_maestra"]["cambio_por_negro"] = False
    cs = [cand("META", 0.068, corr=0.3)]
    d, _ = regla_maestra(ctx(mia=6.0, color_actual=NEGRO), cs, cfg)
    assert d.accion == MANTENER and not d.modo_negro


def test_semana_final_sin_ir_primero_excluye_tsla_y_nvda():
    cs = [cand("NVDA", 0.20, corr=0.1), cand("META", 0.10, corr=0.3)]
    d, b = regla_maestra(ctx(mia=4.0, objetivo=15.0, activo="AMZN", mee_actual=0.05, semana_final=True, soy_primero=False), cs, CFG)
    assert tickers(b) == ["META"] and d.candidato == "META" and "excluidos" in " ".join(d.razones)
    d2, b2 = regla_maestra(ctx(mia=4.0, objetivo=15.0, activo="AMZN", mee_actual=0.05, semana_final=False), cs, CFG)
    assert "NVDA" in tickers(b2)


def test_semana_final_encadena_con_multiplo_bajo_cuando_la_brecha_es_grande():
    d, _ = regla_maestra(ctx(mia=2.0, objetivo=22.0, activo="AMZN", mee_actual=0.050, semana_final=True), [cand("META", 0.055, corr=0.3)], CFG)
    assert d.accion == CAMBIAR and d.multiplo == pytest.approx((21 + 1.3) / 21)


def test_semana_final_voy_primero_mantiene_aunque_g_sea_positivo():
    d, _ = regla_maestra(ctx(mia=20.0, objetivo=20.0, semana_final=True, soy_primero=True), [cand("META", 0.9)], CFG)
    assert d.accion == MANTENER and "imita" in " ".join(d.razones)


def test_ultimo_dia_primero_con_ventaja_8_o_mas_vende_todo():
    d, _ = regla_maestra(ctx(mia=20.0, objetivo=20.0, semana_final=True, soy_primero=True, ultimo_dia=True, ventaja_pp=8.5), [], CFG)
    assert d.accion == VENDER_TODO
    d, _ = regla_maestra(ctx(mia=20.0, objetivo=20.0, semana_final=True, soy_primero=True, ultimo_dia=True, ventaja_pp=7.9), [], CFG)
    assert d.accion == MANTENER
    d, _ = regla_maestra(ctx(mia=20.0, objetivo=20.0, semana_final=True, soy_primero=True, ultimo_dia=True, ventaja_pp=None), [], CFG)
    assert d.accion == MANTENER


def test_ultimo_dia_sin_ir_primero_no_vende():
    d, _ = regla_maestra(ctx(mia=5.0, objetivo=20.0, semana_final=True, soy_primero=False, ultimo_dia=True), [cand("META", 0.5, corr=0.3)], CFG)
    assert d.accion != VENDER_TODO


@pytest.mark.parametrize("t", ["CONCONCRET", "ELCONDOR", "ETB", "FABRICATO", "OCCIDENTE", "POPULAR", "VILLAS"])
def test_la_lista_negra_jamas_es_recomendacion_de_la_regla_maestra(t):
    cs = [cand(t, 5.0, grupo="local", liq=10 ** 7, corr=0.0), cand("META", 0.10, corr=0.3)]
    d, b = regla_maestra(ctx(mia=1.0, objetivo=30.0), cs, CFG)
    assert t not in tickers(b) and d.candidato != t and d.candidato == "META"
