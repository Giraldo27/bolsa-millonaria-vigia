"""Radar de toda la Bolsa de Colombia. Caso real: el 5 de octubre de 2026 la preferencial de Davivienda subió 9 % en la sesión con seis veces su volumen
normal y el bot no avisó porque no estaba en su lista."""
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
TABLA = {
    "PFDAVVNDA": dict(precio=30000.0, cambio=0.091, valor_ult_mm=2964.0, prom10_mm=480.0, nombre="Banco Davivienda SA Pfd"),
    "ECOPETROL": dict(precio=2400.0, cambio=-0.012, valor_ult_mm=21000.0, prom10_mm=35000.0, nombre="Ecopetrol SA"),
    "IDSE": dict(precio=99.0, cambio=0.21, valor_ult_mm=0.0, prom10_mm=178.0, nombre="iShares MSCI Europe SRI UCITS ETF"),       # precio sin operaciones hoy
    "RARA": dict(precio=10.0, cambio=0.30, valor_ult_mm=120.0, prom10_mm=20.0, nombre="Acción casi sin negociación"),             # casi no se negocia
    "NU": dict(precio=62000.0, cambio=-0.045, valor_ult_mm=6000.0, prom10_mm=20000.0, nombre="Nu Holdings Ltd. Class A"),
}


def test_avisa_los_movimientos_fuertes_de_lo_que_se_negocia_de_verdad():
    vistos = {}
    movs = RB.movimientos(TABLA, vistos, "2026-10-05", CFG, {"ECOPETROL", "NU"})
    assert [m["simbolo"] for m in movs] == ["PFDAVVNDA", "NU"]                             # ni el precio viejo (IDSE) ni la que casi no se negocia
    dav = movs[0]
    assert dav["escalon"] == 7 and not dav["en_lista"] and round(dav["veces"], 1) == 6.2
    assert movs[1]["en_lista"] and movs[1]["cambio"] < 0                                    # también las caídas


def test_cada_escalon_se_avisa_una_sola_vez_y_cada_dia_empieza_de_cero():
    vistos = {}
    RB.movimientos(TABLA, vistos, "2026-10-05", CFG)
    assert RB.movimientos(TABLA, vistos, "2026-10-05", CFG) == []                           # misma ronda otra vez: nada repetido
    sube = dict(TABLA, PFDAVVNDA=dict(TABLA["PFDAVVNDA"], cambio=0.105))
    otra = RB.movimientos(sube, vistos, "2026-10-05", CFG)
    assert [m["simbolo"] for m in otra] == ["PFDAVVNDA"] and otra[0]["escalon"] == 10 and otra[0]["repetido"]
    assert {m["simbolo"] for m in RB.movimientos(TABLA, vistos, "2026-10-06", CFG)} == {"PFDAVVNDA", "NU"}


def test_el_aviso_dice_cuanto_cuanta_plata_si_la_vigilo_y_la_noticia_que_lo_explica():
    x = RB.movimientos(TABLA, {}, "2026-10-05", CFG)[0]
    nota = N.Item(EN_SESION.astimezone(dt.timezone.utc) - dt.timedelta(hours=20), "Superfinanciera autoriza integración de DAVIbank con Davivienda",
                  "", "Portafolio", "https://www.portafolio.co/x", "rss")
    otra = N.Item(EN_SESION.astimezone(dt.timezone.utc) - dt.timedelta(hours=2), "Ecopetrol anuncia nuevo pozo", "", "Valora Analitik", "", "rss")
    tit = RB.titulares_de(x["nombre"], [nota, otra], EN_SESION.astimezone(dt.timezone.utc))
    assert tit == [nota]
    t = F.plano(F.msg_radar_bvc(x, tit, EN_SESION))
    for esperado in ("MOVIMIENTO MUY FUERTE EN LA BOLSA DE COLOMBIA", "PFDAVVNDA (Banco Davivienda SA Pfd) sube 9,1% hoy", "6,2 veces lo de un día normal",
                     "volumen fuera de lo normal", "No está en mi lista", "Lo que probablemente lo explica", "integración de DAVIbank", "15 minutos de retraso"):
        assert esperado in t, esperado
    sin_jerga(t)
    sin = F.plano(F.msg_radar_bvc(x, [], EN_SESION))
    assert "Todavía no encontré una noticia" in sin


def test_la_ronda_solo_corre_con_la_bolsa_abierta_y_envia_los_avisos(tmp_path):
    mem = N.Memoria(tmp_path / "m.json")
    enviados = []
    ctx = SimpleNamespace(cfg=CFG, ahora=EN_SESION, f=SimpleNamespace(libro_bvc=lambda: TABLA))
    orig = S.enviar_o_encolar
    S.enviar_o_encolar = lambda c, m, bot=None: enviados.append(m) or True
    try:
        assert len(S.tick_radar_bvc(ctx, mem, [])) == 2 and len(enviados) == 2
        assert S.tick_radar_bvc(ctx, mem, []) == []                                         # ya avisado
        cerrada = SimpleNamespace(cfg=CFG, ahora=bog(2026, 10, 5, 6, 0), f=ctx.f)
        assert S.tick_radar_bvc(cerrada, N.Memoria(tmp_path / "n.json"), []) == []         # antes de abrir, el "cambio" es el de ayer: no se avisa
    finally:
        S.enviar_o_encolar = orig


def test_davivienda_ya_esta_en_la_lista_con_sus_noticias():
    u = universo_permitido(CFG)
    assert {"PFDAVVNDA", "PFDAVIGRP", "ICOLCAP", "IBIT"} <= u
    em = next(e for e in CFG["noticias_bvc"]["emisores"] if e["nombre"] == "Davivienda")
    assert "davibank" in em["alias"] and em["tickers"] == ["PFDAVVNDA", "PFDAVIGRP"]
    assert RB.nombre_clave("Banco Davivienda SA Pfd") == "davivienda" and RB.nombre_clave("Grupo Cibest S.A. Non-Cum Perp Pfd") == "cibest"
