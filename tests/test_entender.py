"""Pruebas de "escríbeme normal": entender compras, ventas y ranking en texto libre, corregir precio en pesos/dólares o total en vez de precio,
deshacer el último registro, y el bot con texto libre y botones."""
import asyncio
import copy
from types import SimpleNamespace

import pytest

import bot
from src import entender as T
from src import formato as F
from src import servicio as S
from src.entender import Intencion, aclarar_compra, buscar_ticker, entender
from src.sentimiento import puntuar_vader
from src.state import ErrorEstado, Estado
from tests.helpers import make_cfg
from tests.test_cartera import FuentesCartera
from tests.test_fase4 import bog

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)
TRM = 3196.0


# =============================== números y nombres ===============================
@pytest.mark.parametrize("crudo, valor", [("21.500", 21500), ("1.200.000", 1200000), ("21,5", 21.5), ("384.32", 384.32), ("1,200", 1200), ("$ 48.500", 48500),
                                          ("48500", 48500), ("-2,5", -2.5), ("1.234,56", 1234.56), ("1,234.56", 1234.56), ("3", 3)])
def test_los_numeros_se_leen_como_los_escribe_un_colombiano(crudo, valor):
    assert T.a_numero(crudo) == pytest.approx(valor)


@pytest.mark.parametrize("texto, ticker", [
    ("compre argos", "GRUPOARGOS"), ("grupo argos", "GRUPOARGOS"), ("cementos argos", "CEMARGOS"), ("ecopetrol", "ECOPETROL"), ("ECOPETROL", "ECOPETROL"),
    ("bancolombia", "PFBCOLOM"), ("nuco", "NUCO"), ("nubank", "NUCO"), ("uberco", "UBER"), ("tesla", "TSLA"), ("tslaco", "TSLA"), ("meta", "META"),
    ("la preferencial de argos", "PFGRUPOARG"), ("pf sura", "PFSURA"), ("ecopetro", "ECOPETROL"), ("bancolonbia", "PFBCOLOM"), ("pfgrupoarg", "PFGRUPOARG"),
    ("interconexión eléctrica", "ISA"), ("éxito", "EXITO"),
])
def test_reconoce_la_accion_por_su_nombre_su_ticker_o_como_la_llama_trii(texto, ticker):
    assert buscar_ticker(texto, CFG)[0] == ticker


def test_lo_que_no_reconoce_no_lo_inventa_y_la_lista_negra_no_existe():
    assert buscar_ticker("compre unas acciones", CFG) == (None, [])
    assert buscar_ticker("fabricato", CFG)[0] is None and buscar_ticker("etb", CFG)[0] is None
    assert buscar_ticker("vendi argos", CFG, tenidos={"PFGRUPOARG"})[0] == "PFGRUPOARG"              # si tienes la preferencial, "argos" es esa


# =============================== qué quiso decir ===============================
def test_compras_escritas_de_muchas_formas():
    i = entender("compré 300 argos a 21500", CFG)
    assert (i.tipo, i.ticker, i.cantidad, i.precio, i.monto) == ("compra", "GRUPOARGOS", 300, 21500, None)
    i = entender("Compré 1200 acciones de nuco a 48.500 pesos", CFG)
    assert (i.ticker, i.cantidad, i.precio) == ("NUCO", 1200, 48500)
    i = entender("compre 20 millones de ecopetrol", CFG)
    assert (i.ticker, i.cantidad, i.precio, i.monto) == ("ECOPETROL", None, None, 20e6)
    i = entender("metí 5M en bancolombia", CFG)
    assert (i.tipo, i.ticker, i.monto) == ("compra", "PFBCOLOM", 5e6)
    i = entender("compré 8 meta", CFG)
    assert (i.ticker, i.cantidad, i.precio, i.falta) == ("META", 8, None, [])
    i = entender("compre 55 tesla a 384,32 dolares", CFG)
    assert (i.ticker, i.cantidad, i.precio) == ("TSLA", 55, 384.32)
    i = entender("argos 300 21500", CFG, forzar="compra")                                           # respuesta a un botón: sin verbo
    assert (i.tipo, i.ticker, i.cantidad, i.precio) == ("compra", "GRUPOARGOS", 300, 21500)


def test_si_falta_algo_lo_dice_en_vez_de_adivinar():
    assert entender("compré acciones", CFG).falta == ["ticker", "cantidad"]
    assert entender("compré ecopetrol", CFG).falta == ["cantidad"]
    i = entender("compre 100 ecopetrolito", CFG)
    assert i.ticker is None and "ECOPETROL" in i.sugerencias


def test_ventas():
    i = entender("vendí tesla", CFG, tenidos={"TSLA", "META"})
    assert (i.tipo, i.ticker, i.cantidad, i.fraccion) == ("venta", "TSLA", None, 1.0)
    i = entender("vendi 100 argos", CFG, tenidos={"GRUPOARGOS"})
    assert (i.ticker, i.cantidad) == ("GRUPOARGOS", 100)
    assert entender("vendí la mitad de meta", CFG, tenidos={"META"}).fraccion == 0.5
    assert entender("vendi el 25% de meta", CFG, tenidos={"META"}).fraccion == 0.25
    assert entender("vendí todo", CFG, tenidos={"NUCO"}).ticker == "NUCO"                            # sólo tienes una: es esa
    assert entender("vendí todo", CFG, tenidos={"NUCO", "META"}).falta == ["ticker"]


def test_ranking_en_palabras():
    assert (entender("voy 3,5", CFG).mia, entender("voy 3,5", CFG).objetivo) == (3.5, None)
    i = entender("voy 3,5 y el corte está en 8", CFG)
    assert (i.tipo, i.mia, i.objetivo) == ("rank", 3.5, 8)
    assert entender("llevo -2%", CFG).mia == -2 and entender("voy perdiendo 2", CFG).mia == -2
    assert entender("el corte esta en 6,2", CFG).objetivo == 6.2 and entender("el corte esta en 6,2", CFG).mia is None
    i = entender("rank 3,5 8", CFG)
    assert (i.mia, i.objetivo) == (3.5, 8)


def test_preguntas_y_planes_no_registran_nada():
    for texto in ("¿qué compro?", "que acción me recomiendas", "me conviene vender tesla", "¿compré bien?", "si vendo meta que pasa", "en que invierto"):
        i = entender(texto, CFG, tenidos={"TSLA", "META"})
        assert i.tipo == "consulta" and i.consulta == "comprar", texto
    for texto, tk in (("voy a comprar 300 argos", "GRUPOARGOS"), ("¿puedo comprar tesla?", "TSLA"), ("quiero comprar nuco", "NUCO")):
        i = entender(texto, CFG)                                                                     # antes de comprar: el filtro de liquidez de ESA acción
        assert (i.tipo, i.consulta, i.ticker) == ("consulta", "liquidez", tk), texto
    assert entender("me equivoqué", CFG).tipo == "deshacer" and entender("deshacer", CFG).tipo == "deshacer"
    assert entender("semáforo", CFG).consulta == "semaforo" and entender("mi cartera", CFG).consulta == "cartera" and entender("cómo voy", CFG).consulta == "estado"
    n = entender("noticias de ecopetrol", CFG)
    assert (n.consulta, n.ticker) == ("noticias", "ECOPETROL") and entender("noticias", CFG).ticker is None
    assert entender("hola", CFG).consulta == "ayuda" and entender("jajaja ok", CFG).tipo == "nada"


# =============================== aclarar la compra contra el precio de hoy ===============================
def test_el_caso_real_precio_en_pesos_de_una_accion_de_ee_uu():
    """Lo que le pasó al usuario: compró 1.200 NUCO viéndolas a $48.500 en trii; el sistema las guarda en dólares."""
    a = aclarar_compra(Intencion("compra", "NUCO", 1200, 48500), "USD", 15.2, TRM)
    assert a["ok"] and not a["dudoso"] and a["cantidad"] == 1200 and a["precio"] == pytest.approx(48500 / TRM) and a["monto_cop"] == pytest.approx(1200 * 48500)
    assert "PESOS" in a["notas"][0]


def test_el_otro_caso_real_el_total_en_vez_del_precio():
    a = aclarar_compra(Intencion("compra", "TSLA", 55, 21137.83), "USD", 384.0, TRM)                 # escribió el TOTAL en dólares
    assert a["precio"] == pytest.approx(384.324, abs=0.01) and a["monto_cop"] == pytest.approx(21137.83 * TRM) and "TOTAL" in a["notas"][0]
    b = aclarar_compra(Intencion("compra", "META", 8, 20_260_000), "USD", 792.0, TRM)                # el total en pesos
    assert b["precio"] == pytest.approx(20_260_000 / 8 / TRM) and b["monto_cop"] == 20_260_000
    c = aclarar_compra(Intencion("compra", "GRUPOARGOS", 300, 6_450_000), "COP", 21500.0, TRM)       # BVC: total en vez de precio
    assert c["precio"] == pytest.approx(21500) and c["monto_cop"] == 6_450_000


def test_precio_normal_plata_en_vez_de_cantidad_y_sin_precio():
    a = aclarar_compra(Intencion("compra", "GRUPOARGOS", 300, 21366), "COP", 21500.0, TRM)
    assert a["precio"] == 21366 and a["monto_cop"] == 300 * 21366 and a["notas"] == [] and not a["aproximado"]
    b = aclarar_compra(Intencion("compra", "ECOPETROL", None, None, 20e6), "COP", 2775.0, TRM)
    assert b["cantidad"] == round(20e6 / 2775) and b["precio"] == 2775 and b["aproximado"] and b["monto_cop"] == 20e6
    c = aclarar_compra(Intencion("compra", "NUCO", None, None, 58.2e6), "USD", 15.2, TRM)
    assert c["cantidad"] == round(58.2e6 / (15.2 * TRM)) and c["monto_cop"] == 58.2e6
    d = aclarar_compra(Intencion("compra", "META", 8, None), "USD", 792.0, TRM)
    assert d["precio"] == 792.0 and d["aproximado"] and d["monto_cop"] == pytest.approx(8 * 792 * TRM)


def test_si_ninguna_lectura_cuadra_no_se_adivina():
    a = aclarar_compra(Intencion("compra", "TSLA", 55, 3.8), "USD", 384.0, TRM)
    assert a["ok"] and a["dudoso"] and a["precio"] == 3.8                                            # se devuelve literal y el bot pregunta antes de guardar
    assert aclarar_compra(Intencion("compra", "META", 8, None), "USD", None, TRM) == dict(ok=False, motivo="sin_precio")
    assert aclarar_compra(Intencion("compra", "META", None, 792.0, 20e6), "USD", 792.0, None)["motivo"] == "sin_trm"
    sin_cotizacion = aclarar_compra(Intencion("compra", "META", 8, 792.0), "USD", None, None)       # sin datos: se acepta lo que dijo, sin monto en pesos
    assert sin_cotizacion["ok"] and sin_cotizacion["precio"] == 792.0 and sin_cotizacion["monto_cop"] is None


# =============================== de punta a punta: texto → cartera ===============================
PRECIOS = {"TSLA": (384.0, 380.5), "META": (792.0, 790.0), "GRUPOARGOS": (21500.0, 21560.0), "NUCO": (15.2, 13.43), "ECOPETROL": (2775.0, 2760.0)}


def ctx(tmp_path, precios=PRECIOS):
    return S.Contexto(CFG, FuentesCartera(precios=precios, trm=TRM), puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)


def cartera(c):
    return {p["ticker"]: p for p in Estado(CFG, c.estado_path).d["cartera"]}


def test_escribir_normal_registra_bien_los_casos_que_antes_salian_mal(tmp_path):
    c = ctx(tmp_path)
    t = F.plano(S.resp_texto(c, "compré 1200 nuco a 48500")["texto"])
    assert "Compra de NUCO registrada" in t and "1200 acciones a US$ 15,18" in t and "58,2 millones" in t and "Lo que entendí: tomé 48.500 como precio en PESOS" in t
    assert "No es de la BVC" in t and "me equivoqué" in t
    S.resp_texto(c, "compre 55 tesla a 21137,83")
    S.resp_texto(c, "compré 6,4 millones de argos")
    k = cartera(c)
    assert k["NUCO"]["precio"] == pytest.approx(48500 / TRM) and k["NUCO"]["monto_cop"] == pytest.approx(58.2e6)
    assert k["TSLA"]["precio"] == pytest.approx(384.324, abs=0.01) and k["TSLA"]["cantidad"] == 55
    assert k["GRUPOARGOS"]["cantidad"] == round(6.4e6 / 21500) and k["GRUPOARGOS"]["precio"] == 21500.0


def test_un_precio_que_no_cuadra_se_pregunta_antes_de_guardar_y_se_puede_confirmar(tmp_path):
    c = ctx(tmp_path)
    t = F.plano(S.resp_texto(c, "compré 55 tesla a 3,8")["texto"])
    assert "confirma el precio" in t and "hoy cotiza cerca de US$ 384,00" in t and cartera(c) == {}
    assert "Compra de TSLA registrada" in S.resp_texto(c, "compré 55 tesla a 3,8 confirmo")["texto"] and cartera(c)["TSLA"]["precio"] == 3.8


def test_vender_deshacer_y_ranking_por_texto(tmp_path):
    c = ctx(tmp_path)
    S.resp_texto(c, "compré 300 argos a 21366")
    S.resp_texto(c, "compré 8 meta a 792")
    v = F.plano(S.resp_texto(c, "vendí la mitad de meta")["texto"])
    assert "Venta de 4 acciones de META registrada" in v and cartera(c)["META"]["cantidad"] == 4
    assert "Venta total de GRUPOARGOS registrada" in S.resp_texto(c, "vendi argos")["texto"] and "GRUPOARGOS" not in cartera(c)
    d = F.plano(S.resp_texto(c, "me equivoqué")["texto"])
    assert "deshice la venta de GRUPOARGOS" in d and cartera(c)["GRUPOARGOS"]["cantidad"] == 300 and Estado(CFG, c.estado_path).ops_total() == 3
    assert "No hay nada que deshacer" in S.resp_texto(c, "deshacer")["texto"]                        # sólo se recuerda el último paso
    assert "no está en tu cartera" in S.resp_texto(c, "vendí ecopetrol")["texto"]
    assert "Me falta la rentabilidad del corte" in S.resp_texto(c, "voy 3,5")["texto"]
    assert "vas +3,5% y el objetivo es +8,0%" in F.plano(S.resp_texto(c, "voy 3,5 y el corte está en 8")["texto"])
    assert "vas +4,1% y el objetivo es +8,0%" in F.plano(S.resp_texto(c, "voy 4,1")["texto"])        # el corte se conserva
    assert "vas +4,1% y el objetivo es +9,0%" in F.plano(S.resp_texto(c, "el corte está en 9")["texto"])


def test_lo_que_falta_se_pregunta_y_las_consultas_se_delegan(tmp_path):
    c = ctx(tmp_path)
    r = S.resp_texto(c, "compré ecopetrol")
    assert r["pendiente"] == "compra" and r["ticker"] == "ECOPETROL" and "¿Cuántas acciones de ECOPETROL" in F.plano(r["texto"])
    assert "Compra de ECOPETROL registrada" in S.resp_texto(c, "ECOPETROL 20 millones", forzar="compra")["texto"]      # la respuesta al botón
    r = S.resp_texto(c, "compré 100 ecopetrolito")
    assert r["pendiente"] == "compra" and "¿Quisiste decir ECOPETROL?" in F.plano(r["texto"])
    assert S.resp_texto(c, "¿qué compro?") == {"consulta": "comprar", "ticker": None}
    assert S.resp_texto(c, "noticias de bancolombia") == {"consulta": "noticias", "ticker": "PFBCOLOM"}
    assert S.resp_texto(c, "jajaja") == {"texto": ""}


def test_un_chat_ajeno_no_puede_registrar_si_la_escritura_esta_reservada(tmp_path):
    c = ctx(tmp_path)
    assert "Sólo el dueño" in S.resp_texto(c, "compré 300 argos a 21500", solo_lectura=True)["texto"] and cartera(c) == {}
    assert S.resp_texto(c, "semáforo", solo_lectura=True) == {"consulta": "semaforo", "ticker": None}  # consultar sí puede


def test_el_comando_compra_tambien_corrige_pesos_y_totales_y_acepta_nombres(tmp_path):
    c = ctx(tmp_path)
    assert "Compra de NUCO registrada" in S.resp_compra(c, ["nuco", "1200", "48500"]) and cartera(c)["NUCO"]["precio"] == pytest.approx(48500 / TRM)
    assert "Compra de GRUPOARGOS registrada" in S.resp_compra(c, ["argos", "300", "21.366"]) and cartera(c)["GRUPOARGOS"]["precio"] == 21366
    assert "Compra de ECOPETROL registrada" in S.resp_compra(c, ["ecopetrol", "20M"]) and cartera(c)["ECOPETROL"]["cantidad"] == round(20e6 / 2775)
    assert "Venta total de NUCO" in S.resp_venta(c, ["nubank"])


def test_deshacer_restaura_cartera_principal_y_operaciones(tmp_path):
    e = Estado(CFG, tmp_path / "s.json")
    e.registrar_compra("TSLA", 55, 380.0, AHORA, 80e6)
    e.registrar_compra("GRUPOARGOS", 5000, 21500, AHORA, 107.5e6)
    assert e.activo == "GRUPOARGOS" and e.ops_total() == 2
    assert e.deshacer() == "la compra de 5000 GRUPOARGOS" and e.activo == "TSLA" and e.ops_total() == 1 and e.tenidos() == {"TSLA"}
    with pytest.raises(ErrorEstado, match="nada que deshacer"):
        e.deshacer()
    assert Estado(CFG, tmp_path / "s.json").tenidos() == {"TSLA"}                                    # quedó guardado


# =============================== bot: texto libre y botones ===============================
DUENO = 123456789


class Msg:
    def __init__(self, texto=""):
        self.text, self.textos, self.teclados = texto, [], []

    async def reply_text(self, texto, parse_mode=None, disable_web_page_preview=None, reply_markup=None):
        self.textos.append(texto)
        self.teclados.append(reply_markup)

    async def reply_document(self, **k):
        pass


def app(**bot_cfg):
    cfg = copy.deepcopy(CFG)
    cfg["bot"].update(bot_cfg)
    return bot.construir_app("123456:ABCDEF-token-falso-para-pruebas", DUENO, cfg)


def manejador(a, clase):
    return next(h for g in a.handlers.values() for h in g if type(h).__name__ == clase).callback


def escribir(a, texto, chat=DUENO, tipo="private"):
    m = Msg(texto)
    asyncio.run(manejador(a, "MessageHandler")(SimpleNamespace(effective_chat=SimpleNamespace(id=chat, type=tipo), message=m, callback_query=None), SimpleNamespace(args=[])))
    return m


def tocar(a, dato, chat=DUENO):
    m = Msg()

    async def answer():
        return None
    q = SimpleNamespace(data=dato, message=m, answer=answer)
    asyncio.run(manejador(a, "CallbackQueryHandler")(SimpleNamespace(effective_chat=SimpleNamespace(id=chat, type="private"), message=None, callback_query=q), SimpleNamespace(args=[])))
    return m


def botones_de(markup):
    return [(b.text, b.callback_data) for fila in markup.inline_keyboard for b in fila]


@pytest.fixture
def servicio(tmp_path, monkeypatch):
    c = ctx(tmp_path)
    monkeypatch.setattr(S, "crear_contexto", lambda *a, **k: c)
    return c


def test_el_bot_registra_una_compra_escrita_sin_comando(servicio):
    m = escribir(app(), "compré 300 argos a 21366")
    assert "Compra de GRUPOARGOS registrada" in m.textos[0] and cartera(servicio)["GRUPOARGOS"]["cantidad"] == 300


def test_lo_que_no_entiende_lo_dice_con_ejemplos_y_botones_pero_no_en_grupos(servicio):
    m = escribir(app(), "jajaja ok")
    assert "No te entendí" in m.textos[0] and ("🛒 Qué comprar", "c:comprar") in botones_de(m.teclados[0])
    assert escribir(app(), "jajaja ok", chat=-100777, tipo="group").textos == []                     # en un grupo no interrumpe la conversación


def test_una_consulta_escrita_ejecuta_el_comando(servicio, monkeypatch):
    monkeypatch.setattr(S, "resp_cartera", lambda c: "cartera ok")
    m = escribir(app(), "mi cartera")
    assert m.textos[-1] == "cartera ok"


def test_boton_compre_pregunta_y_la_respuesta_siguiente_se_toma_como_compra(servicio):
    a = app()
    m = tocar(a, "p:compra")
    assert "¿Qué compraste?" in m.textos[0]
    assert "Compra de GRUPOARGOS registrada" in escribir(a, "300 argos a 21366").textos[0]            # sin el verbo: venía del botón
    assert "No te entendí" in escribir(a, "300 argos a 21366").textos[0]                              # la espera se consume una sola vez


def test_boton_la_compre_de_una_noticia_ya_sabe_la_accion(servicio):
    a = app()
    assert "¿Cuánto compraste de ECOPETROL?" in tocar(a, "k:ECOPETROL").textos[0]
    assert "Compra de ECOPETROL registrada" in escribir(a, "20 millones").textos[0] and cartera(servicio)["ECOPETROL"]["cantidad"] == round(20e6 / 2775)


def test_boton_vendi_lista_lo_que_tienes_y_vende_al_tocar(servicio):
    a = app()
    assert "No tienes compras registradas" in tocar(a, "p:venta").textos[0]
    escribir(a, "compré 300 argos a 21366")
    escribir(a, "compré 8 meta a 792")
    m = tocar(a, "p:venta")
    assert botones_de(m.teclados[0]) == [("Vendí todo GRUPOARGOS", "v:GRUPOARGOS"), ("Vendí todo META", "v:META")]
    assert "Venta total de META registrada" in tocar(a, "v:META").textos[0] and set(cartera(servicio)) == {"GRUPOARGOS"}


def test_boton_ranking_acepta_solo_el_numero(servicio):
    a = app()
    assert "¿Cómo vas?" in tocar(a, "p:rank").textos[0]
    assert "Me falta la rentabilidad del corte" in escribir(a, "3,5").textos[0]
    tocar(a, "p:rank")
    assert "vas +3,5% y el objetivo es +8,0%" in F.plano(escribir(a, "voy 3,5 y el corte está en 8").textos[0])


def test_los_botones_de_consulta_ejecutan_el_comando_y_el_menu_trae_todos(servicio, monkeypatch):
    monkeypatch.setattr(S, "resp_cartera", lambda c: "cartera ok")
    assert tocar(app(), "c:cartera").textos[-1] == "cartera ok"
    assert tocar(app(), "c:no_existe").textos == [] and tocar(app(), "basura").textos == []
    comandos = {next(iter(h.commands)): h for g in app().handlers.values() for h in g if hasattr(h, "commands")}
    assert {"comprar", "deshacer", "menu", "noticias", "compra", "venta"} <= set(comandos)
    m = Msg()
    asyncio.run(comandos["menu"].callback(SimpleNamespace(effective_chat=SimpleNamespace(id=DUENO), message=m), SimpleNamespace(args=[])))
    datos = {d for _, d in botones_de(m.teclados[0])}
    assert {"c:semaforo", "c:cartera", "c:nuevas", "c:macro", "c:comprar", "p:compra", "p:venta", "p:rank", "c:estado"} == datos
    todos = set(comandos)
    assert all(d.split(":")[1] in todos for d in datos if d.startswith("c:"))                         # ningún botón apunta a un comando que no existe


def test_con_la_escritura_reservada_otro_chat_no_registra_ni_con_texto_ni_con_botones(servicio):
    a = app(escritura_solo_mi_chat=True)
    assert "Sólo el dueño" in escribir(a, "compré 300 argos a 21366", chat=555).textos[0] and cartera(servicio) == {}
    escribir(a, "compré 300 argos a 21366")
    assert "Sólo el dueño" in tocar(a, "v:GRUPOARGOS", chat=555).textos[0] and "GRUPOARGOS" in cartera(servicio)
    cerrado = app(responder_a="solo_mi_chat")
    assert tocar(cerrado, "c:cartera", chat=555).textos == []                                         # modo cerrado: los botones tampoco contestan a extraños
