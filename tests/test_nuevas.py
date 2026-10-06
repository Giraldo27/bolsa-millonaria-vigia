"""Pruebas de /nuevas: actualizar las noticias en el momento (Superfinanciera primero) y mostrarlas analizadas."""
import datetime as dt
import json

from src import formato as F
from src import liquidez as LQ
from src import noticias_bvc as N
from src import servicio as S
from src.entender import entender
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg
from tests.test_fase4 import bog
from tests.test_noticias_bvc import ESTUDIO, Fuentes, HttpFalso, item

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)
UTC = dt.timezone.utc


class LectorFalso:
    def __init__(self, items, salud=None):
        self.items, self.salud, self.completo = items, salud or {"Superfinanciera": True, "Valora Analitik": True, "Semana": False}, None

    def todo(self, ahora, completo=False):
        self.completo = completo
        return list(self.items), dict(self.salud)


def ctx(tmp_path):
    return S.Contexto(CFG, Fuentes(vol=5e7), puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)


def preparar(monkeypatch, liquidas=("ECOPETROL", "PFBCOLOM", "CELSIA", "GRUPOARGOS")):
    monkeypatch.setattr(N, "cargar_estudio", lambda ruta=None: ESTUDIO)
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: dict(ticker=t, nivel=LQ.BUENA if t in liquidas else LQ.MALA, mediana_mm=9000.0))


def test_el_comando_consulta_todas_las_fuentes_y_muestra_lo_oficial_primero(tmp_path, monkeypatch):
    preparar(monkeypatch)
    oficial = item("Ecopetrol informa que su junta aprobó un dividendo extraordinario", minutos=90, origen="sfc", fuente="Superfinanciera", entidad="ECOPETROL S.A.")
    its = [item("Utilidad neta de Bancolombia creció 20 % en el trimestre", minutos=10), oficial,
           item("Ecopetrol refuerza su apuesta por el gas en el Caribe", minutos=30, fuente="Portafolio"),
           item("Celsia presenta su nuevo logo", minutos=50), item("Enka anuncia recompra de acciones", minutos=5),          # Enka: sin liquidez → no aparece
           item("Bancolombia: utilidad neta creció 20 % en el tercer trimestre", minutos=8, fuente="Semana"),             # repetida
           item("Ecopetrol anunció dividendo hace dos días", minutos=60 * 30), item("El dólar cae tras decisión de la FED", minutos=3)]
    lec = LectorFalso(its)
    filas, salud = S.buscar_noticias(ctx(tmp_path), lector=lec)
    assert lec.completo is True                                                                       # todas las fuentes y búsquedas, sin saltarse ninguna
    assert [(x["ticker"], x["sentido"], x["oficial"]) for x in filas] == [("PFBCOLOM", 1, False), ("ECOPETROL", 1, False), ("CELSIA", 0, False), ("ECOPETROL", 1, True)]
    assert filas[1]["cat"] == "otra" and filas[1]["impacto"] > 0 and filas[2]["impacto"] > 0          # "refuerza su apuesta": positiva por análisis; la neutra lleva magnitud
    t = F.plano(F.msg_noticias_actualizadas(filas, salud, AHORA, 12))
    assert "Noticias actualizadas" in t and "2 de 3 respondieron · Superfinanciera (oficial) ✅ · sin respuesta: Semana" in t
    assert t.index("Información oficial (Superfinanciera)") < t.index("dividendo extraordinario") < t.index("Prensa") < t.index("utilidad neta creció 20 %")
    for esperado in ("🟢 PFBCOLOM · +1,8% · resultados financieros", "🟢 ECOPETROL · +0,3% · noticia de la empresa", "⚪ CELSIA · ±0,3% (neutra)", "hace 8 min", "/comprar"):
        assert esperado in t, esperado
    assert "ENKA" not in t and "dólar" not in t.split("Fuentes")[1].split("El %")[0]


def test_sin_noticias_o_sin_la_fuente_oficial_lo_dice_claro(tmp_path, monkeypatch):
    preparar(monkeypatch)
    filas, salud = S.buscar_noticias(ctx(tmp_path), lector=LectorFalso([], {"Superfinanciera": False, "Valora Analitik": True}))
    t = F.plano(F.msg_noticias_actualizadas(filas, salud, AHORA, 12))
    assert "Superfinanciera (oficial) ❌ no respondió" in t and "No hay noticias de empresas de la BVC" in t
    solo_prensa = [dict(ts=AHORA.astimezone(UTC), emisor="Celsia", ticker="CELSIA", titulo="Celsia gana contrato", fuente="Semana", oficial=False, cat="hallazgo", sentido=1, impacto=0.02, tengo=True)]
    t = F.plano(F.msg_noticias_actualizadas(solo_prensa, {"Superfinanciera": True}, AHORA, 12))
    assert "La Superfinanciera no tiene anuncios de estas empresas en las últimas 12 horas" in t and "CELSIA (la tienes) · +2,0%" in t


def test_lo_que_tienes_se_muestra_aunque_no_sea_liquido(tmp_path, monkeypatch):
    preparar(monkeypatch, liquidas=())
    c = ctx(tmp_path)
    Estado(CFG, c.estado_path).registrar_compra("CELSIA", 100, 4800, AHORA)
    filas, _ = S.buscar_noticias(c, lector=LectorFalso([item("Celsia gana contrato de energía solar"), item("Ecopetrol gana contrato")]))
    assert [(x["ticker"], x["tengo"]) for x in filas] == [("CELSIA", True)]


def test_el_lector_completo_consulta_todos_los_medios_y_todas_las_busquedas():
    http = HttpFalso({"google": "<rss></rss>", "elcolombiano": "<rss></rss>", "semana": "<rss></rss>", "superfinanciera": json.dumps({"content": []})})
    lec = N.Lector(CFG, http=http, env={})
    lec.ronda = 3                                                                                    # una ronda en la que normalmente NO tocarían Google ni El Colombiano
    lec.todo(AHORA, completo=True)
    veces = lambda d: len([u for u, _ in http.pedidas if d in u])                                    # noqa: E731
    assert veces("google") == len(CFG["noticias_bvc"]["fuentes"]["google"]["consultas"]) and veces("elcolombiano") == 1 and veces("semana") == 1 and veces("superfinanciera") == 1


def test_se_puede_pedir_por_comando_por_boton_o_escribiendo():
    import bot
    app = bot.construir_app("123456:ABCDEF-token-falso-para-pruebas", 123456789, CFG)
    nombres = {next(iter(h.commands)) for g in app.handlers.values() for h in g if hasattr(h, "commands")}
    assert {"nuevas", "actualizarnoticias", "noticias"} <= nombres and ("🔄 Buscar noticias", "c:nuevas") in [b for fila in bot.MENU for b in fila]
    for frase in ("actualiza las noticias", "hay noticias nuevas?", "busca noticias", "últimas noticias"):
        assert entender(frase, CFG).consulta == "nuevas", frase
    assert entender("noticias de ecopetrol", CFG).consulta == "noticias" and "/nuevas" in F.AYUDA
