"""Radar de toda la Bolsa de Colombia, locales y globales. Caso real: el 5 de octubre de 2026 la preferencial de Davivienda subió 9 % en la sesión con seis
veces su volumen normal y el bot no avisó porque no estaba en su lista."""
import datetime as dt
from types import SimpleNamespace

from src import formato as F
from src import noticias_bvc as N
from src import radar_bvc as RB
from src import servicio as S
from src.state import universo_permitido
from tests.helpers import make_cfg
from tests.test_fase4 import bog, sin_jerga

CFG = make_cfg()
EN_SESION = bog(2026, 10, 5, 11, 30)
ANTES = bog(2026, 10, 5, 7, 0)
DESPUES = bog(2026, 10, 5, 17, 0)
TABLA = {
    "PFDAVVNDA": dict(precio=30000.0, cambio=0.091, valor_ult_mm=2964.0, prom10_mm=480.0, nombre="Banco Davivienda SA Pfd", isin="COB51PA00076", global_=False),
    "ECOPETROL": dict(precio=2400.0, cambio=-0.012, valor_ult_mm=21000.0, prom10_mm=35000.0, nombre="Ecopetrol SA", isin="COC04PA00016", global_=False),
    "INVENSA": dict(precio=506.0, cambio=0.30, valor_ult_mm=0.4, prom10_mm=0.0, nombre="Inversiones Venecia SA", isin="COA06PA00015", global_=False),   # 0,4 millones
    "ETB": dict(precio=150.0, cambio=-0.05, valor_ult_mm=12.0, prom10_mm=60.0, nombre="Empresa de Telecomunicaciones de Bogota", isin="COX01PA00010", global_=False),
    "NU": dict(precio=62000.0, cambio=0.0, valor_ult_mm=6000.0, prom10_mm=20000.0, nombre="Nu Holdings Ltd. Class A", isin="KYG6683N1034", global_=True),
    "MSFTCO": dict(precio=2e6, cambio=0.0, valor_ult_mm=0.0, prom10_mm=276.0, nombre="Microsoft Corporation", isin="US5949181045", global_=True),
    "SQMBCO": dict(precio=2e5, cambio=0.0, valor_ult_mm=0.0, prom10_mm=0.0, nombre="Sociedad Quimica y Minera", isin="CLP8716X1082", global_=True),
    "PFDAVIGRP": dict(precio=40620.0, cambio=0.045, valor_ult_mm=900.0, prom10_mm=2000.0, nombre="Davivienda Group SA Pfd", isin="PAP3200A1007", global_=True),   # sólo en la BVC
    "ARKWCO": dict(precio=5e5, cambio=0.0, valor_ult_mm=0.0, prom10_mm=0.0, nombre="ARK Next Generation Technology ETF", isin="US00214Q4010", global_=True),
}
ORIGEN = {
    "KYG6683N1034": dict(simbolo="NU", bolsa="NYSE", cambio=0.013, pre=0.081, post=-0.045, moneda="USD", valor=1e9),
    "US5949181045": dict(simbolo="MSFT", bolsa="NASDAQ", cambio=-0.051, pre=-0.01, post=None, moneda="USD", valor=1e10),
    "CLP8716X1082": dict(simbolo="SQM-B", bolsa="BCS", cambio=0.072, pre=None, post=None, moneda="CLP", valor=1e9),
    "US00214Q4010": dict(simbolo="ARKW", bolsa="CBOE", cambio=0.01, pre=0.02, post=None, moneda="USD", valor=1e7),
}


def _movs(cuando, vistos=None, lista=None):
    f = RB.fase(cuando)
    return RB.movimientos(RB.candidatos(TABLA, ORIGEN, f, CFG), {} if vistos is None else vistos, f"{cuando.astimezone(RB.NY):%Y-%m-%d}", CFG, lista)


def test_las_fases_del_dia_siguen_a_nueva_york():
    assert (RB.fase(ANTES), RB.fase(EN_SESION), RB.fase(DESPUES)) == ("pre", "sesion", "post")
    assert RB.fase(bog(2026, 10, 5, 2, 0)) is None and RB.fase(bog(2026, 10, 10, 11, 0)) is None          # madrugada y sábado


def test_en_la_sesion_salen_locales_y_globales_de_cualquier_bolsa():
    movs = _movs(EN_SESION, lista={"ECOPETROL", "NU"})
    assert [m["simbolo"] for m in movs] == ["PFDAVVNDA", "SQMBCO", "MSFTCO", "ETB", "PFDAVIGRP"]       # Santiago y Nueva York también; no la de 0,4 millones
    assert not movs[-1]["global_"]                                                         # ISIN de Panamá, pero sólo se negocia en la BVC: se mide con la BVC
    dav, sqm, msft = movs[0], movs[1], movs[2]
    assert dav["escalon"] == 7 and not dav["en_lista"] and round(dav["veces"], 1) == 6.2 and not dav["global_"]
    assert sqm["global_"] and sqm["bolsa"] == "Santiago de Chile" and msft["cambio"] < 0 and msft["bolsa"].startswith("Nueva York")


def test_antes_de_abrir_y_despues_del_cierre_salen_las_de_eeuu():
    antes = _movs(ANTES)
    assert [m["simbolo"] for m in antes] == ["NU"] and antes[0]["momento"] == "pre" and antes[0]["escalon"] == 7     # el salto de NU antes de abrir
    despues = _movs(DESPUES)
    assert [m["simbolo"] for m in despues] == ["NU"] and despues[0]["momento"] == "post" and despues[0]["cambio"] < 0


def test_cada_escalon_se_avisa_una_sola_vez_y_el_mismo_movimiento_no_se_repite_al_abrir():
    vistos = {}
    assert _movs(ANTES, vistos)
    assert all(m["simbolo"] != "NU" for m in _movs(bog(2026, 10, 5, 9, 0), vistos))       # sigue en +8 % antes de abrir: nada nuevo
    sube = dict(ORIGEN, **{"KYG6683N1034": dict(ORIGEN["KYG6683N1034"], cambio=0.105)})
    otra = RB.movimientos(RB.candidatos(TABLA, sube, "sesion", CFG), vistos, "2026-10-05", CFG)
    nu = next(m for m in otra if m["simbolo"] == "NU")
    assert nu["escalon"] == 10 and nu["repetido"]
    assert RB.movimientos(RB.candidatos(TABLA, sube, "sesion", CFG), vistos, "2026-10-05", CFG) == []
    assert any(m["simbolo"] == "NU" and m["momento"] == "post" for m in _movs(DESPUES, vistos))   # después del cierre es otro movimiento
    assert {m["simbolo"] for m in _movs(bog(2026, 10, 6, 11, 0), vistos)} >= {"PFDAVVNDA"}      # al otro día empieza de cero


def test_el_aviso_dice_cuanto_donde_si_se_negocia_en_trii_y_la_noticia_que_lo_explica():
    x = _movs(EN_SESION)[0]
    nota = N.Item(EN_SESION.astimezone(dt.timezone.utc) - dt.timedelta(hours=20), "Superfinanciera autoriza integración de DAVIbank con Davivienda",
                  "", "Portafolio", "https://www.portafolio.co/x", "rss")
    otra = N.Item(EN_SESION.astimezone(dt.timezone.utc) - dt.timedelta(hours=2), "Ecopetrol anuncia nuevo pozo", "", "Valora Analitik", "", "rss")
    tit = RB.titulares_de(x["nombre"], [nota, otra], EN_SESION.astimezone(dt.timezone.utc))
    assert tit == [nota]
    t = F.plano(F.msg_radar_bvc(x, tit, EN_SESION))
    for esperado in ("MOVIMIENTO MUY FUERTE", "PFDAVVNDA (Banco Davivienda SA Pfd) sube 9,1% hoy", "6,2 veces lo de un día normal", "volumen fuera de lo normal",
                     "No está en mi lista", "Lo que probablemente lo explica", "integración de DAVIbank", "15 minutos de retraso"):
        assert esperado in t, esperado
    sin_jerga(t)
    nu = F.plano(F.msg_radar_bvc(_movs(ANTES)[0], [], ANTES))
    for esperado in ("NU (Nu Holdings Ltd. Class A) sube 8,1% antes de abrir en Nueva York", "acción extranjera", "llega a trii cuando abra la bolsa de Colombia",
                     "En trii hoy se han negociado", "Todavía no encontré una noticia"):
        assert esperado in nu, esperado
    sqm = F.plano(F.msg_radar_bvc(_movs(EN_SESION)[1], [], EN_SESION))
    assert "en Santiago de Chile (SQM-B)" in sqm and "En trii casi no se negocia" in sqm
    assert RB.nombre_clave("Petroleo Brasileiro SA Sponsored ADR") == "petrobras" and "nubank" in RB.nombre_clave("Nu Holdings Ltd. Class A")


def test_la_ronda_es_automatica_y_cubre_toda_la_bolsa(tmp_path):
    mem = N.Memoria(tmp_path / "m.json")
    enviados = []
    pedidos = []
    f = SimpleNamespace(libro_bvc=lambda: TABLA, origen_globales=lambda isins: pedidos.append(sorted(isins)) or ORIGEN)
    orig = S.enviar_o_encolar
    S.enviar_o_encolar = lambda c, m, bot=None: enviados.append(m) or True
    try:
        assert len(S.tick_radar_bvc(SimpleNamespace(cfg=CFG, ahora=EN_SESION, f=f), mem, [])) == 5 and len(enviados) == 5
        assert pedidos[0] == sorted(v["isin"] for v in TABLA.values() if v["global_"])      # pide TODAS las globales de la BVC
        assert S.tick_radar_bvc(SimpleNamespace(cfg=CFG, ahora=EN_SESION, f=f), mem, []) == []
        assert S.tick_radar_bvc(SimpleNamespace(cfg=CFG, ahora=bog(2026, 10, 5, 2, 0), f=f), N.Memoria(tmp_path / "n.json"), []) == []
    finally:
        S.enviar_o_encolar = orig


def test_davivienda_ya_esta_en_la_lista_con_sus_noticias():
    u = universo_permitido(CFG)
    assert {"PFDAVVNDA", "PFDAVIGRP", "ICOLCAP", "IBIT"} <= u
    em = next(e for e in CFG["noticias_bvc"]["emisores"] if e["nombre"] == "Davivienda")
    assert "davibank" in em["alias"] and em["tickers"] == ["PFDAVVNDA", "PFDAVIGRP"]
    assert RB.nombre_clave("Banco Davivienda SA Pfd") == "davivienda" and RB.nombre_clave("Grupo Cibest S.A. Non-Cum Perp Pfd") == "cibest"


def test_lima_no_cuenta_hasta_que_abre():
    o = {"PEP736001004": dict(simbolo="FERREYC1", bolsa="BVL", cambio=0.06, pre=None, post=None, moneda="PEN", valor=1e6)}
    tabla = {"FERREYC1CO": dict(precio=3000.0, cambio=0.0, valor_ult_mm=0.0, prom10_mm=0.0, nombre="Ferreycorp", isin="PEP736001004", global_=True)}
    assert RB.candidatos(tabla, o, "sesion", CFG, 9 * 60 + 45) == []                       # 9:45 de Nueva York: Lima todavía no abre (es el cambio de ayer)
    assert RB.candidatos(tabla, o, "sesion", CFG, 10 * 60 + 5)[0]["bolsa"] == "Lima"
