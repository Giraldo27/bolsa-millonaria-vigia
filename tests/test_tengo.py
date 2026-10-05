"""Pruebas de "tengo …" (decir lo que tienes sin que cuente como compra) y de la guarda contra compras repetidas.

Caso real (5-oct-2026): el usuario le describió su cartera al bot y quedó con 2.400 NUCO en vez de 1.200, porque se tomó como otra compra."""
import pytest

from src import formato as F
from src import servicio as S
from src.entender import entender
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg
from tests.test_cartera import FuentesCartera
from tests.test_fase4 import bog

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)
TRM = 3200.0
PRECIOS = {"TSLA": (329.0, 325.0), "META": (747.0, 740.0), "GRUPOARGOS": (21500.0, 21560.0), "NUCO": (15.16, 13.4), "ECOPETROL": (2775.0, 2760.0)}
MENSAJE = ("tengo 8 acciones de meta que ahora valen 19120000 cop.   tengo 300 acciones de grupo argos que ahora tienen un valor actual de 6450000 cop.    "
           "tengo 8 acciones de tesla con un valor actual de mercado de 8435000cop. tengo 1200 acciones de nuco con un valor actual de mercado de 58224000 cop.")


def ctx(tmp_path):
    return S.Contexto(CFG, FuentesCartera(precios=PRECIOS, trm=TRM), puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)


def cartera(c):
    return {p["ticker"]: p for p in Estado(CFG, c.estado_path).d["cartera"]}


def test_el_mensaje_real_del_usuario_se_entiende_como_cuatro_tenencias():
    i = entender(MENSAJE, CFG)
    assert i.tipo == "tengo" and [(p.ticker, p.cantidad, p.monto) for p in i.partes] == [("META", 8, 19120000), ("GRUPOARGOS", 300, 6450000), ("TSLA", 8, 8435000), ("NUCO", 1200, 58224000)]


def test_otras_formas_de_decir_lo_que_tienes_y_lo_que_no_lo_es():
    i = entender("tengo 1200 nuco y 300 argos", CFG)
    assert [(p.ticker, p.cantidad, p.monto) for p in i.partes] == [("NUCO", 1200, None), ("GRUPOARGOS", 300, None)]
    assert entender("me quedan 100 ecopetrol", CFG).partes[0].cantidad == 100
    assert entender("tengo una duda", CFG).tipo != "tengo" and entender("¿cuántas tengo de nuco?", CFG).tipo != "tengo"
    assert entender("compré 300 argos, ahora tengo 600", CFG).tipo == "compra"                       # si dice que compró, es una compra


def test_decir_lo_que_tienes_fija_la_cantidad_y_no_cuenta_como_operacion(tmp_path):
    c = ctx(tmp_path)
    S.resp_texto(c, "compré 1200 nuco a 15,16")
    S.resp_texto(c, "compré 1200 nuco a 15,16 confirmo")                                             # el error: quedó duplicada
    assert cartera(c)["NUCO"]["cantidad"] == 2400
    t = F.plano(S.resp_texto(c, MENSAJE)["texto"])
    k = cartera(c)
    assert {x: k[x]["cantidad"] for x in k} == {"NUCO": 1200, "META": 8, "GRUPOARGOS": 300, "TSLA": 8}
    assert k["NUCO"]["monto_cop"] == 58224000 and k["NUCO"]["precio"] == pytest.approx(58224000 / 1200 / TRM) and k["GRUPOARGOS"]["precio"] == pytest.approx(21500)
    assert "Cartera ajustada" in t and "no cuenta como operación" in t and "NUCO: 1200 acciones (antes tenía 2400 registradas)" in t and "TSLA: 8 acciones (no la tenía registrada)" in t
    assert Estado(CFG, c.estado_path).ops_total() == 2 and Estado(CFG, c.estado_path).activo == "NUCO"
    assert "deshice el ajuste de tu cartera" in F.plano(S.resp_texto(c, "me equivoqué")["texto"]) and cartera(c)["NUCO"]["cantidad"] == 2400 and "TSLA" not in cartera(c)


def test_sin_valor_se_conserva_el_precio_que_ya_tenia_o_se_usa_el_de_hoy(tmp_path):
    c = ctx(tmp_path)
    S.resp_texto(c, "compré 300 argos a 21366")
    S.resp_texto(c, "tengo 250 argos y 5000 ecopetrol")
    k = cartera(c)
    assert k["GRUPOARGOS"]["cantidad"] == 250 and k["GRUPOARGOS"]["precio"] == 21366 and k["ECOPETROL"]["precio"] == 2775.0 and k["ECOPETROL"]["monto_cop"] == 5000 * 2775.0
    t = F.plano(S.resp_texto(c, "tengo 250 argos")["texto"])
    assert "ya estaba así" in t and "No toqué: ECOPETROL" in t
    assert "LISTA NEGRA" in S.resp_texto(c, "tengo 100 etb")["texto"] or "No reconocí" in S.resp_texto(c, "tengo 100 etb")["texto"] or S.resp_texto(c, "tengo 100 etb")["texto"] == ""


def test_repetir_la_misma_compra_pregunta_antes_de_duplicar(tmp_path):
    c = ctx(tmp_path)
    S.resp_texto(c, "compré 1200 nuco a 15,16")
    t = F.plano(S.resp_texto(c, "compré 1200 nuco a 15,16")["texto"])
    assert "Ya tienes 1200 NUCO registradas" in t and "quedarías con 2400" in t and "tengo 1200 nuco" in t and cartera(c)["NUCO"]["cantidad"] == 1200
    assert "Compra de NUCO registrada" in S.resp_texto(c, "compré 300 nuco a 15,16")["texto"] and cartera(c)["NUCO"]["cantidad"] == 1500   # otra cantidad: es otra compra
    assert "Compra de NUCO registrada" in S.resp_texto(c, "compré 1500 nuco a 15,16 confirmo")["texto"] and cartera(c)["NUCO"]["cantidad"] == 3000


def test_un_chat_ajeno_no_puede_ajustar_la_cartera(tmp_path):
    c = ctx(tmp_path)
    assert "Sólo el dueño" in S.resp_texto(c, "tengo 1200 nuco", solo_lectura=True)["texto"] and cartera(c) == {}
