"""Pruebas de los tres análisis que juntan todos los filtros: /ganar, /reemplazo y /revisar (src/analisis.py), y de cómo se piden escribiendo normal."""
import datetime as dt
from types import SimpleNamespace

import pytest

from src import analisis as A
from src import formato as F
from src import liquidez as LQ
from src import macro as M
from src import servicio as S
from src.entender import entender
from src.semaforo import ROJO, VERDE
from tests.helpers import make_cfg
from tests.test_fase4 import bog, mk_res, sin_jerga
from tests.test_macro import ctx

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)
CONT = dict(mediana=0.86, flojo=0.79, pausa=15.0, dias=21)
RATOS = dict(mediana=0.63, flojo=0.59, pausa=25.0, dias=21)


def fila(t, mee, vs, nivel=LQ.BUENA, grupo="local", base=False, cont=CONT, a_ratos=False, liq=5000.0):
    return dict(ticker=t, grupo=grupo, mee=mee, vs_base=vs, nivel=nivel, liquida=nivel == LQ.BUENA, a_ratos=a_ratos, continuidad=cont, es_base=base, liquidez_mm=liq, supera=False)


FILAS = [fila("NUCO", 0.153, 1.0, LQ.JUSTA, "mgc", base=True, cont=None),
         fila("GRUPOARGOS", 0.138, 0.90, LQ.JUSTA, cont=RATOS, a_ratos=True),
         fila("TSLA", 0.20, 1.3, LQ.MALA, "mgc", cont=None),                                # se mueve más, pero no es de la BVC ni líquida: nunca se nombra
         fila("PFGRUPOARG", 0.099, 0.64),
         fila("ECOPETROL", 0.093, 0.61),
         fila("GEB", 0.08, 0.5, LQ.JUSTA, cont=dict(mediana=0.51, flojo=0.46, pausa=30.0, dias=21), a_ratos=True)]    # a ratos y lejos del corte: tampoco


def liq(nivel, **k):
    return lambda f, t, cfg: dict(ticker=t, simbolo=t + ".CL", nivel=nivel, mediana_mm=5000.0, hoy_mm=4000.0, acciones_hoy=1000.0, dias_sin_negociar=0, dias=20,
                                  motivos=["poco"] if nivel != LQ.BUENA else [], continuidad=CONT if nivel == LQ.BUENA else None, **k)


@pytest.fixture
def c(tmp_path, monkeypatch):
    x = ctx(tmp_path, SimpleNamespace(avisos=[]))                                        # todos los datos se inyectan: aquí se prueba el armado del análisis
    with S.transaccion(x) as e:
        e.d["posicion"]["activo"] = "NUCO"
    monkeypatch.setattr(A, "_filas", lambda ctx_, base: [dict(f, es_base=f["ticker"] == base) for f in FILAS])
    monkeypatch.setattr(A, "_valores", lambda ctx_, est: {"NUCO": 58e6})
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(VERDE, ticker=t, r_hoy=0.019, z=0.6), None, None))
    monkeypatch.setattr(S, "panorama_macro", lambda ctx_: ([], {}))
    monkeypatch.setattr(S, "senales_recientes", lambda horas, ahora=None: [])
    monkeypatch.setattr(LQ, "medir", liq(LQ.JUSTA))
    return x


def test_ganar_junta_todos_los_filtros_y_concluye_con_la_que_mas_se_mueve(c):
    t = F.plano(A.resp_ganar(c))
    for parte in ("Movimiento hasta el final", "Semáforo", "Macro hoy", "Noticias", "Costo de cambiar", "Liquidez en trii"):
        assert parte in t
    assert "NUCO ±15,3%" in t and "PFGRUPOARG, ±9,9% (0,64 veces)" in t and "ninguna noticia fuerte" in t
    assert "La que más opción te da de quedar arriba es la que ya tienes: NUCO" in t and "Su punto débil es la liquidez: no compres más" in t
    assert "TSLA" not in t and "No sé cómo vas en el ranking" in t and "no garantiza ganar" in t and "Ningún filtro predice" in t
    sin_jerga(t)


def test_ganar_avisa_si_una_liquida_se_mueve_bastante_mas_y_usa_el_ranking(c, monkeypatch):
    monkeypatch.setattr(A, "_filas", lambda ctx_, base: [fila("NUCO", 0.10, 1.0, LQ.JUSTA, "mgc", base=True, cont=None), fila("ECOPETROL", 0.13, 1.3)])
    with S.transaccion(c) as e:
        e.actualizar_rank(12.0, 8.0, AHORA)
    t = F.plano(A.resp_ganar(c))
    assert "ECOPETROL puede moverse bastante más que NUCO (1,30 veces)" in t and "/reemplazo nuco" in t
    assert "vas por encima. Cuida lo ganado" in t
    with S.transaccion(c) as e:
        e.actualizar_rank(-0.3, 10.0, AHORA)
    assert "vas por debajo. Para alcanzarlo te conviene mantener lo que más se mueve" in F.plano(A.resp_ganar(c))


def test_reemplazo_ordena_por_movimiento_y_separa_la_mas_parecida_de_la_que_pasa_todo_el_filtro(c):
    t = F.plano(A.resp_reemplazo(c, ["nuco"]))
    assert "¿Cuál puede reemplazar a NUCO?" in t and "NUCO se espera que se mueva ±15,3%" in t
    assert "1. GRUPOARGOS: ±13,8% (0,90 veces NUCO) · 🟡 liquidez justa: se negocia a ratos (63% del día) · tu posición sería 1,2% de lo que negocia en un día" in t
    assert "2. PFGRUPOARG: ±9,9% (0,64 veces NUCO) · ✅ liquidez buena (operaciones en el 86% del día)" in t
    assert "La que más se le acerca es GRUPOARGOS: conserva cerca del 90% de su movimiento" in t and "se queda fuera del filtro de liquidez por poco" in t
    assert "Si quieres una que pase todo el filtro: PFGRUPOARG" in t and "Cuesta cerca de 1,3%" in t and "no dice hacia dónde" in t
    assert "TSLA" not in t and "GEB" not in t                                              # ni las de EE. UU. ni las que se negocian a ratos lejos del corte
    assert F.plano(A.resp_reemplazo(c, [])).startswith("🔁 ¿Cuál puede reemplazar a NUCO?")  # sin acción: tu principal
    assert "No reconocí" in F.plano(A.resp_reemplazo(c, ["zzzz"]))
    sin_jerga(t)


def noticias(*filas):
    return lambda ctx_, horas=12, lector=None: ([dict(ts=AHORA, emisor="Ecopetrol", ticker=t, titulo=tit, fuente="Valora Analitik", oficial=of, cat="otra", sentido=s,
                                                      impacto=None, tengo=False) for t, tit, s, of in filas], {})


def test_revisar_dice_lo_que_la_accion_tiene_hoy_a_favor_y_en_contra(c, monkeypatch):
    """Caso real (6-oct-2026): Ecopetrol con el petróleo cayendo y una noticia negativa → más en contra que a favor, aunque sea la más líquida."""
    monkeypatch.setattr(LQ, "medir", liq(LQ.BUENA))
    monkeypatch.setattr(S, "mee_de", lambda f, t, s, ahora, cfg: (0.029, "historia", None))
    monkeypatch.setattr(M, "tablero", lambda f, cfg, ahora=None: [dict(clave="petroleo", nombre="el petróleo (Brent)", cambio=-0.02, cuando="hoy")])
    monkeypatch.setattr(M, "sensibilidades", lambda f, cfg, tickers: {"petroleo": {"ECOPETROL": {"beta": 0.3}}})
    monkeypatch.setattr(S, "buscar_noticias", noticias(("ECOPETROL", "Polémica del presidente de Ecopetrol con medios", -1, False), ("ECOPETROL", "Terminó la campaña exploratoria", 0, True),
                                                       ("PFBCOLOM", "Otra empresa", 1, False)))
    t = F.plano(A.resp_revisar(c, ["ecopetrol"]))
    assert "¿Compro ECOPETROL hoy?" in t and "sube 1,9% hoy" in t and "cerca de ±2,9% hasta el corte" in t
    assert "el petróleo (Brent) cae 2,0% → a ECOPETROL suele restarle 0,6%" in t
    assert "🔴 Polémica del presidente" in t and "⚪ Terminó la campaña exploratoria (oficial, Valora Analitik)" in t and "Otra empresa" not in t
    assert "A favor: se compra y se vende fácil en trii." in t and "En contra: la macro de hoy la empuja hacia abajo" in t and "más negativas que positivas" in t
    assert "Hoy tiene más en contra que a favor" in t and "No sé si va a subir o bajar" in t
    sin_jerga(t)


def test_revisar_sin_nada_en_contra_no_lo_vende_como_senal_de_subida_y_sin_liquidez_dice_que_no(c, monkeypatch):
    monkeypatch.setattr(S, "mee_de", lambda f, t, s, ahora, cfg: (0.022, "historia", None))
    monkeypatch.setattr(M, "tablero", lambda f, cfg, ahora=None: [])
    monkeypatch.setattr(S, "buscar_noticias", noticias())
    monkeypatch.setattr(LQ, "medir", liq(LQ.BUENA))
    t = F.plano(A.resp_revisar(c, ["pfbcolom"]))
    assert "Hoy no tiene nada serio en contra" in t and "no es una señal de que vaya a subir" in t and "no encontré noticias de PFBCOLOM" in t
    monkeypatch.setattr(LQ, "medir", liq(LQ.JUSTA, a_ratos=True))
    assert "No la compraría: no pasa el filtro de liquidez en trii." in F.plano(A.resp_revisar(c, ["pfsura"]))
    monkeypatch.setattr(LQ, "medir", liq(LQ.BUENA))
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(ROJO, ticker=t, r_hoy=0.06, z=3.1), None, None))
    t = F.plano(A.resp_revisar(c, ["bcolombia"]))
    assert "el semáforo está en ROJO" in t and "viene de una subida fuerte" in t and "más en contra que a favor" in t
    assert "Cómo usar /revisar" in F.plano(A.resp_revisar(c, []))


def test_un_filtro_sin_datos_no_tumba_el_analisis(c, monkeypatch):
    def falla(*a, **k):
        raise RuntimeError("sin datos")
    monkeypatch.setattr(S, "evaluar_activo", falla)
    monkeypatch.setattr(S, "panorama_macro", falla)
    t = F.plano(A.resp_ganar(c))
    assert "Semáforo: no pude revisarlo ahora." in t and "Macro hoy: no pude revisarla ahora." in t and "NUCO" in t
    monkeypatch.setattr(A, "_filas", lambda ctx_, base: [])
    assert "No pude hacer el análisis ahora" in F.plano(A.resp_ganar(c)) and "No pude calcular" in F.plano(A.resp_reemplazo(c, ["nuco"]))


@pytest.mark.parametrize("texto, consulta, ticker", [
    ("cual podria reemplazar a nuco", "reemplazo", "NUCO"),
    ("por cual cambio nuco", "reemplazo", "NUCO"),
    ("que accion me va a hacer ganar el concurso", "ganar", None),
    ("estas seguro que ecopetrol con todas las noticias de hoy", "revisar", "ECOPETROL"),
    ("que opinas de grupo argos", "revisar", "GRUPOARGOS"),
    ("analiza pfbcolom", "revisar", "PFBCOLOM"),
    ("resumen del dia", "variaciones", None),
    ("variaciones de hoy", "variaciones", None),
    ("como voy", "estado", None),
    ("revisa mi cartera", "cartera", None),                                                 # sin acción no es /revisar
    ("que acciones comprar", "comprar", None),
    ("noticias de ecopetrol", "noticias", "ECOPETROL"),
    ("puedo comprar tesla?", "liquidez", "TSLA"),
])
def test_los_analisis_se_piden_escribiendo_normal(texto, consulta, ticker):
    i = entender(texto, CFG, {"NUCO", "META", "TSLA", "GRUPOARGOS"})
    assert (i.tipo, i.consulta, i.ticker) == ("consulta", consulta, ticker)


def test_el_bot_tiene_los_tres_comandos_y_la_ayuda_los_explica():
    import bot
    fuente = open(bot.__file__, encoding="utf-8").read()
    for nombre in ("ganar", "reemplazo", "revisar", "variaciones"):
        assert f'comando("{nombre}"' in fuente and f"/{nombre}" in F.AYUDA


# =============================== /variaciones: el resumen del día ===============================
def test_las_variaciones_separan_lo_tuyo_lo_que_pasa_el_filtro_y_lo_descartado(c, monkeypatch):
    from src import construir as CO
    monkeypatch.setattr(CO, "universo_candidatos", lambda cfg: ["ECOPETROL", "PFBCOLOM", "PFSURA", "TERPEL", "BVC"])
    ayer = dt.date(2026, 10, 5)
    datos = {"NUCO": (0.019, True), "ECOPETROL": (-0.013, True), "PFBCOLOM": (0.006, True), "PFSURA": (0.007, True), "TERPEL": (0.021, False)}
    monkeypatch.setattr(A, "_variacion", lambda ctx_, t: dict(cambio=datos[t][0], es_hoy=datos[t][1], fecha=AHORA.date() if datos[t][1] else ayer) if t in datos else None)
    niveles = {"NUCO": dict(nivel=LQ.JUSTA), "ECOPETROL": dict(nivel=LQ.BUENA), "PFBCOLOM": dict(nivel=LQ.BUENA), "PFSURA": dict(nivel=LQ.JUSTA, a_ratos=True), "TERPEL": dict(nivel=LQ.MALA)}
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: dict(ticker=t, **niveles[t]))
    monkeypatch.setattr(type(S.leer_estado(c)), "tenidos", lambda self: {"NUCO"})
    t = F.plano(A.resp_variaciones(c))
    mias, resto = t.split("✅ Pasan el filtro de liquidez (2)")
    aptas, fuera = resto.split("⛔ Descartadas por liquidez (2)")
    assert "💼 Las que tienes" in mias and "🟢 NUCO +1,9% · se negocia poco" in mias
    assert aptas.index("🟢 PFBCOLOM +0,6%") < aptas.index("🔴 ECOPETROL -1,3%")                # de la que más sube a la que más baja
    assert "🟢 TERPEL +2,1% 🕘 lun 05/10 · casi no se negocia" in fuera and "🟢 PFSURA +0,7% · se negocia a ratos" in fuera
    assert "Resumen: 3 suben, 1 bajan y 0 no se mueven. La que más sube: NUCO (+1,9%); la que más baja: ECOPETROL (-1,3%)" in t   # lo de ayer (TERPEL) no cuenta
    assert "🕘 = todavía no ha negociado hoy" in t and "Sin dato ahora: BVC." in t and "no es una señal de compra" in t
    sin_jerga(t)


def test_antes_de_abrir_las_variaciones_dicen_que_son_de_la_ultima_sesion(c, monkeypatch):
    from src import construir as CO
    monkeypatch.setattr(CO, "universo_candidatos", lambda cfg: ["ECOPETROL"])
    monkeypatch.setattr(type(S.leer_estado(c)), "tenidos", lambda self: set())
    monkeypatch.setattr(A, "_variacion", lambda ctx_, t: dict(cambio=0.02, es_hoy=False, fecha=dt.date(2026, 10, 5)))
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: dict(ticker=t, nivel=LQ.BUENA))
    t = F.plano(A.resp_variaciones(c))
    assert "Hoy todavía no ha abierto la bolsa: lo que ves es la última sesión." in t and "🟢 ECOPETROL +2,0% 🕘 lun 05/10" in t
    monkeypatch.setattr(A, "_variacion", lambda ctx_, t: None)
    assert "No pude leer las variaciones ahora" in F.plano(A.resp_variaciones(c))
