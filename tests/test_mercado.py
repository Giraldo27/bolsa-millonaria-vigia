"""Pruebas del vigía de MERCADO (src/mercado.py): noticias macro y financieras con las acciones y sectores que se benefician o se afectan y por qué."""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import formato as F
from src import liquidez as LQ
from src import mercado as MK
from src import noticias_bvc as N
from src import servicio as S
from src.sentimiento import puntuar_vader
from tests.helpers import make_cfg
from tests.test_fase4 import bog, sin_jerga

CFG = make_cfg()
AHORA = bog(2026, 10, 8, 10, 0)
UTC = dt.timezone.utc
TODAS = set(CFG["universe"]["local"]) | set(CFG["universe"]["mgc"]) | set(CFG["universe"]["etf"])


def item(titulo, minutos=5, fuente="Bloomberg Línea", resumen="", url="https://ejemplo.co/nota"):
    return N.Item(AHORA.astimezone(UTC) - dt.timedelta(minutes=minutos), titulo, resumen, fuente, url, "rss")


def sectores(a):
    return {s["nombre"]: (s["signo"], s["tickers"]) for s in a["sectores"]}


# =============================== qué significa cada noticia ===============================
def test_el_petroleo_al_alza_beneficia_a_las_petroleras_y_afecta_a_quien_paga_el_combustible():
    a = MK.analizar(item("Petróleo se dispara más de 5% y supera los US$105 por temor a una mayor escalada en Medio Oriente"), TODAS)
    assert a["tipo"] == "tema" and a["temas"][0] in ("petroleo", "geopolitica") and a["direccion"] == 1
    assert {"ECOPETROL", "PBR"} <= set(a["beneficiadas"]) and "TERPEL" in a["afectadas"] and "UBER" in a["afectadas"]
    s = sectores(a)
    assert s["petróleo y gas"][0] == 1 and s["consumo y transporte"][0] == -1
    assert all(x["porque"] for x in a["sectores"])                                        # cada sector lleva su porqué
    baja = MK.analizar(item("El crudo Brent cae 4% tras señales de tregua"), TODAS)
    assert "ECOPETROL" in baja["afectadas"] and "TERPEL" in baja["beneficiadas"]             # la misma regla, al revés


@pytest.mark.parametrize("titulo, tema, direccion, gana, pierde", [
    ("La Fed sube las tasas y los rendimientos del Tesoro tocan máximos", "tasas_eeuu", 1, "JPM", "NVDA"),
    ("Fed signals rate cut as Treasury yields fall", "tasas_eeuu", -1, "NVDA", "JPM"),
    ("Inflación en Colombia no encuentra techo: subió a 6,29% en septiembre", "tasas_colombia", 1, None, "CELSIA"),
    ("Dólar en Colombia se dispara y la TRM supera los $3.400", "dolar", 1, "ECOPETROL", "EXITO"),
    ("Gold hits one-week high as dollar softens", "oro", 1, "MINEROS", None),
    ("Las acciones brasileñas se disparan tras el triunfo de Bolsonaro en la primera vuelta de Brasil", "brasil", 1, "NUCO", None),
    ("Trump impone nuevos aranceles a los autos importados", "aranceles", 1, None, "F"),
    ("Israel e Irán acuerdan un cese al fuego tras semanas de ataques", "geopolitica", -1, "NUCO", "ECOPETROL"),
])
def test_cada_tema_dice_quien_gana_y_quien_pierde_segun_la_direccion(titulo, tema, direccion, gana, pierde):
    a = MK.analizar(item(titulo), TODAS)
    assert tema in a["temas"] and a["direccion"] == direccion
    assert gana is None or gana in a["beneficiadas"]
    assert pierde is None or pierde in a["afectadas"]


def test_si_el_titular_no_dice_hacia_donde_se_listan_las_expuestas_sin_signo():
    a = MK.analizar(item("La OPEP se reúne el jueves para definir su producción de petróleo"), TODAS)
    assert a["direccion"] == 0 and a["beneficiadas"] == [] and a["afectadas"] == [] and "ECOPETROL" in a["expuestas"]
    t = F.plano(F.msg_noticia_mercado(dict(a, item=item("La OPEP se reúne el jueves para definir su producción de petróleo")), dict(resumen=""), AHORA))
    assert "Expuestas (el titular no dice hacia dónde): ECOPETROL" in t and "Gana si el precio del petróleo sube." in t and "Gana si el precio del petróleo baja." in t


def test_las_empresas_globales_de_la_bolsa_se_reconocen_y_las_que_solo_opinan_no():
    assert MK.globales_de("Nubank niega acuerdo con Monzo y reafirma su expansión global") == ["NUCO"]
    assert MK.globales_de("Tesla recalls 100,000 vehicles amid probe") == ["TSLA"]
    assert MK.globales_de("JPMorgan sube el precio objetivo de Nu Holdings a US$23") == ["NUCO"]           # JPMorgan opina: la noticia es de Nubank
    assert MK.globales_de("Citi cuestiona al presidente de Ecopetrol, pero ve potencial en Venezuela") == []   # ni Citi es la noticia, ni Ecopetrol es global
    assert MK.globales_de("JPMorgan reports record earnings as trading surges") == ["JPM"]                 # sus resultados sí son noticia propia
    assert MK.globales_de("Cómo sacar la visa a Estados Unidos en 2026") == [] and MK.globales_de("Acciones de Visa caen tras demanda") == ["V"]
    assert MK.globales_de("Marvell fija ambiciosas metas para inversores") == []                           # "metas" no es Meta
    a = MK.analizar(item("Tesla recalls 100,000 vehicles amid probe"), TODAS)
    assert a["tipo"] == "empresa" and a["afectadas"][0] == "TSLA" and "F" in a["afectadas"]                # la empresa y, de rebote, su sector
    assert a["sectores"][0]["porque"] == "es la empresa de la noticia" and "más débil" in a["sectores"][1]["porque"]


def test_lo_que_no_es_de_mercado_se_descarta():
    for titulo in ("Quién es quién en la serie de Timbiriche: los actores", "Precio del dólar hoy en casas de cambio", "Mike Bahía y Greeicy debutan como empresarios de alimentos",
                   "El papa León XIV será operado", "Nu Holdings (NYSE:NU) Stock Forecast & Analyst Predictions"):
        assert MK.analizar(item(titulo), TODAS) is None, titulo
    solo_locales = MK.analizar(item("Nvidia supera expectativas con sus chips de inteligencia artificial"), {"ECOPETROL"})
    assert solo_locales is None                                                            # nunca se nombra una acción que no está en el concurso


# =============================== el aviso ===============================
def test_el_aviso_trae_titular_enlace_resumen_beneficiadas_afectadas_sectores_y_una_semana():
    it = item("Petróleo se dispara más de 5% y supera los US$105 por temor a una mayor escalada en Medio Oriente", url="https://www.bloomberglinea.com/mercados/petroleo-se-dispara/")
    a = dict(MK.analizar(it, TODAS), item=it)
    extras = dict(resumen="El Brent subió 5,2% después de un nuevo cruce de fuego entre EE. UU. e Irán.", tenidos={"NUCO"}, sesiones=5, factor_ahora="el petróleo (Brent) sube 4,1%",
                  sigma={"ECOPETROL": 0.044, "PBR": 0.069}, liquidez={"ECOPETROL": "buena", "PBR": "mala"})
    t = F.plano(F.msg_noticia_mercado(a, extras, AHORA))
    for esperado in ("Noticia de mercado:", "⬆️", "«Petróleo se dispara más de 5%", "Bloomberg Línea · hace 5 min", "🔗 https://www.bloomberglinea.com/mercados/petroleo-se-dispara/",
                     "Resumen: El Brent subió 5,2%", "Ahora en el mercado: el petróleo (Brent) sube 4,1%.", "Se benefician: ECOPETROL, PBR", "Se afectan:", "NUCO (la tienes)",
                     "Sectores y por qué", "🟢 petróleo y gas (ECOPETROL, PBR, CNEC):", "🔴 consumo y transporte", "A una semana", "ECOPETROL ±4,4% · PBR ±6,9%",
                     "En trii: se negocian bien ECOPETROL; se negocian poco PBR", "no cuánto va a subir"):
        assert esperado in t, esperado
    sin_jerga(t)
    sin = F.plano(F.msg_noticia_mercado(a, dict(resumen=""), AHORA))
    assert "La fuente no trae resumen: abre el enlace" in sin and "A una semana" not in sin


def test_el_resumen_se_corta_en_frases_completas():
    largo = "El Brent subió 5,2% tras un nuevo cruce de fuego. " * 12
    r = MK.recortar(largo, 120)
    assert len(r) <= 120 and r.endswith(".") and MK.recortar("corto") == "corto" and MK.recortar("") == ""


# =============================== una ronda: nada viejo, nada repetido ===============================
class LectorFalso:
    def __init__(self, items=None):
        self.items, self.salud = items or [], {"Bloomberg Línea": True, "Investing.com": True}

    def todo(self, ahora, completo=False):
        return list(self.items), dict(self.salud)

    def resumen_de(self, url):
        return "Resumen tomado de la página de la nota."


@pytest.fixture
def mem(tmp_path):
    return N.Memoria(tmp_path / "noticias.json")


def test_la_primera_ronda_solo_toma_la_linea_base_y_despues_avisa_lo_nuevo_sin_repetir(mem):
    lec = LectorFalso([item("Petróleo cae 3% por dudas sobre la demanda")])
    assert MK.ronda(lec, mem, AHORA, CFG)[0] == [] and mem.d["mercado"]["base"]             # lo que ya estaba publicado nunca se avisa
    lec.items.append(item("La Fed sube las tasas medio punto", fuente="Bloomberg (mercados)"))
    (a,), _ = MK.ronda(lec, mem, AHORA, CFG)
    assert a["temas"] == ["tasas_eeuu"] and a["item"].fuente == "Bloomberg (mercados)"
    assert MK.ronda(lec, mem, AHORA, CFG)[0] == []                                          # ya vista
    lec.items.append(item("Fed sube las tasas en medio punto, dice Reuters", fuente="Investing.com"))
    assert MK.ronda(lec, mem, AHORA, CFG)[0] == []                                          # la misma noticia en otro medio
    lec.items.append(item("Rendimientos del Tesoro suben a máximos de 24 años", fuente="Yahoo Finanzas"))
    assert MK.ronda(lec, mem, AHORA, CFG)[0] == []                                          # mismo tema y dirección hace un momento: enfriamiento
    despues = AHORA + dt.timedelta(minutes=CFG["noticias_mercado"]["enfriamiento_tema_min"] + 1)
    lec.items.append(N.Item(despues.astimezone(UTC), "Bonos del Tesoro: los rendimientos suben otra vez tras el dato de empleo", "", "Bloomberg (economía)", "https://e.co/2", "rss"))
    assert len(MK.ronda(lec, mem, despues, CFG)[0]) == 1                                    # pasado el enfriamiento, sí
    assert len(mem.d["mercado"]["ultimas"]) == 2 and mem.d["mercado"]["ultimas"][0]["url"]


def test_lo_viejo_no_se_avisa_y_si_salen_muchas_a_la_vez_el_resto_sale_despues(mem):
    lec = LectorFalso()
    MK.ronda(lec, mem, AHORA, CFG)                                                          # línea base vacía
    mem.d["mercado"]["base"] = "x"
    lec.items = [item("Petróleo sube 4% por tensión en Irán", minutos=400)]                 # publicada hace casi 7 horas
    assert MK.ronda(lec, mem, AHORA, CFG)[0] == []
    titulos = ["Oro sube a máximo de una semana", "Dólar en Colombia cae y la TRM rompe los $3.200", "Tesla supera expectativas de entregas", "Nubank anuncia alianza en México",
               "Inflación en EE. UU.: el CPI se acelera más de lo esperado", "Petrobras sube tras resultados récord"]
    lec.items = [item(t, minutos=i + 1) for i, t in enumerate(titulos)]
    primera, _ = MK.ronda(lec, mem, AHORA, CFG)
    segunda, _ = MK.ronda(lec, mem, AHORA, CFG)
    assert len(primera) == CFG["noticias_mercado"]["max_por_ronda"] == 4 and len(segunda) == 2    # ninguna se pierde


def test_la_noticia_de_una_empresa_colombiana_no_se_repite_aqui_salvo_que_toque_un_tema(mem):
    lec = LectorFalso()
    MK.ronda(lec, mem, AHORA, CFG)
    mem.d["mercado"]["base"] = "x"
    emisores = N.emisores_cfg(CFG)
    local = lambda it: bool(N.emisores_de(it, emisores))                                    # noqa: E731
    lec.items = [item("Ecopetrol anuncia nuevo presidente"), item("Ecopetrol sube mientras el petróleo Brent se dispara 5%", minutos=2)]
    (a,), _ = MK.ronda(lec, mem, AHORA, CFG, locales=local)
    assert "petroleo" in a["temas"]                                                         # la primera la avisa el vigía de empresas; ésta trae un tema de mercado


# =============================== el servicio: con los indicadores del bot ===============================
class Fuentes:
    avisos: list = []

    def diario(self, t, dias=None):
        if t == "CNEC":
            raise RuntimeError("sin historia")
        r = 0.02 * np.array([(-1) ** i for i in range(60)])
        return pd.DataFrame({"Close": 100 * np.cumprod(1 + r)}, index=pd.bdate_range(end="2026-10-07", periods=60))


def ctx(tmp_path):
    return S.Contexto(CFG, Fuentes(), puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)


def test_el_vigia_de_mercado_envia_el_aviso_con_una_semana_y_liquidez_y_guarda_la_memoria(tmp_path, monkeypatch):
    from src import macro as M
    enviados = []
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, botones=None, **k: enviados.append((texto, botones)) or True)
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: dict(ticker=t, nivel=LQ.BUENA if t == "ECOPETROL" else LQ.MALA))
    monkeypatch.setattr(M, "tablero", lambda f, cfg, ahora=None: [dict(clave="petroleo", nombre="el petróleo (Brent)", cambio=0.041, z=2.1, en_puntos=False, cuando="hoy", fuera=None)])
    mem = N.Memoria(tmp_path / "n.json")
    lec = LectorFalso()
    c = ctx(tmp_path)
    assert S.tick_mercado(c, lec, mem) == []
    lec.items = [item("Petróleo Brent se dispara 5% y supera los US$105", resumen="")]
    (m,) = S.tick_mercado(c, lec, mem)
    t = F.plano(m)
    assert "Resumen: Resumen tomado de la página de la nota." in t                          # la fuente no traía resumen: se toma el de la nota
    assert "Ahora en el mercado: el petróleo (Brent) sube 4,1%" in t and "A una semana" in t and "ECOPETROL ±4,6%" in t
    assert "se negocian bien ECOPETROL" in t and "se negocian poco" in t and "CNEC ±" not in t  # una acción sin historia no tumba el aviso
    assert enviados[0][1] == S.BOTONES_MERCADO and "mercado" in N.Memoria(tmp_path / "n.json").d
    apagado = S.Contexto({**CFG, "noticias_mercado": {**CFG["noticias_mercado"], "activo": False}}, Fuentes(), puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)
    lec.items.append(item("Oro sube a máximos"))
    assert S.tick_mercado(apagado, lec, mem) == []


def test_el_comando_mercado_muestra_lo_ultimo_con_enlace(tmp_path, monkeypatch):
    ult = [dict(ts="2026-10-08T14:50:00+00:00", titulo="Petróleo se dispara 5%", fuente="Bloomberg Línea", url="https://e.co/1", nombres=["Petróleo"], empresas=[], direccion=1,
                beneficiadas=["ECOPETROL", "PBR"], afectadas=["TERPEL"], expuestas=[])]
    t = F.plano(F.msg_mercado_recientes(ult, {"Bloomberg Línea": True, "Investing.com": False}, AHORA))
    assert "Fuentes respondiendo: 1 de 2 · sin respuesta: Investing.com" in t and "⬆️ Petróleo" in t and "🔗 https://e.co/1" in t and "🟢 ECOPETROL, PBR · 🔴 TERPEL" in t
    assert "Todavía no he avisado ninguna" in F.plano(F.msg_mercado_recientes([], {}, AHORA))


# =============================== las fuentes ===============================
def test_la_portada_de_bloomberg_linea_se_lee_por_sus_enlaces():
    html = ('<a href="/mercados/petroleo-se-dispara-y-supera-los-us105-por-temor-a-escalada/"><h2>Petróleo se dispara y supera los <b>US$105</b> por temor a una escalada</h2></a>'
            '<a href="/opinion/columna-larga-sobre-cualquier-cosa-que-no-es-noticia/">Una columna de opinión bastante larga sobre cualquier cosa</a>'
            '<a href="/mercados/">Mercados</a><a href="/video/2026/08/16/expira-el-acuerdo/">Un video bastante largo sobre el acuerdo con Irán y sus efectos</a>')
    (it,) = MK.items_de_pagina(html, "Bloomberg Línea (noticias del día)", "https://www.bloomberglinea.com", AHORA)
    assert it.titulo == "Petróleo se dispara y supera los US$105 por temor a una escalada" and it.url == "https://www.bloomberglinea.com/mercados/petroleo-se-dispara-y-supera-los-us105-por-temor-a-escalada/"


def test_las_fuentes_pedidas_estan_configuradas_y_una_caida_no_tumba_a_las_demas():
    m = CFG["noticias_mercado"]
    nombres = " ".join(f["nombre"] for f in m["rss"] + m["paginas"])
    for pedida in ("Bloomberg Línea", "Bloomberg (mercados)", "Valora Analitik", "Investing.com", "Yahoo Finanzas", "La República"):
        assert pedida in nombres, pedida
    assert m["superfinanciera"] and m["yahoo_titulares"]["activo"] and "las-noticias-del-dia" in m["paginas"][0]["url"] and m["horizonte_sesiones"] == 5

    class Http:
        def get(self, url, params=None, headers=None, timeout=None):
            if "investing" in url:
                raise RuntimeError("caída")
            cuerpo = (b"<rss><item><title>Petroleo sube 3%</title><link>https://e.co/a</link><pubDate>Thu, 08 Oct 2026 14:50:00 GMT</pubDate>"
                      b"<content:encoded><![CDATA[<p>El Brent avanza por tension en Medio Oriente.</p>]]></content:encoded></item></rss>")
            return SimpleNamespace(status_code=200, headers={}, content=cuerpo, raise_for_status=lambda: None, json=lambda: {})
    lec = MK.Lector(CFG, Http(), {})
    items, salud = lec.todo(AHORA, completo=True)
    assert salud["Bloomberg Línea"] and not salud["Investing.com"] and salud["Yahoo Finance (por acción)"]
    assert any(i.resumen == "El Brent avanza por tension en Medio Oriente." for i in items)  # el resumen sale de content:encoded cuando no hay description
    assert len(lec.simbolos) == len(CFG["universe"]["mgc"]) + len(CFG["universe"]["local"])  # todas las acciones del concurso: locales y globales


@pytest.mark.parametrize("titulo", [
    "Wall Street ve margen para más ganancias en el S&P 500: las 5 acciones con mayor potencial al alza",          # "gananc-ia-s" no es IA
    "Colombia e Israel acordaron cooperación para fortalecimiento del agro",                                       # un país sin conflicto no es geopolítica
    "La siembra de soja avanza en Brasil y presiona a la baja los precios en el mercado interno de granos",        # Brasil sin bolsa ni elecciones… (el mercado es de granos)
    "China propone prohibir manijas ocultas y exigir controles físicos de seguridad en los automóviles",
    "Musk vows affordable Starlink in India days before Ambani's record Jio IPO",                                  # Musk sin Tesla no es Tesla
    "US Luxury Spending Falls 6% in September as Third-Quarter Demand Declines, Citi Reports",                     # Citi informa, no es la noticia
    "5 Historically Cheap High-Yield Dividend Stocks Ripe for Income Seekers Looking Beyond Tech",
    "Alphabet Stock Is Up 46% Since Last October. Is It Still a Buy?",
    "Las mejores aplicaciones con IA para aprender idiomas lo antes posible",
    "Mauritania Returns Former GTA Gas Deal Minister to Oil Post",
])
def test_los_falsos_positivos_vistos_en_la_prueba_real_ya_no_se_avisan(titulo):
    a = MK.analizar(item(titulo), TODAS)
    assert a is None or (a["tipo"] == "tema" and a["temas"] == ["brasil"] and "soja" in titulo), (titulo, a and a["temas"], a and a["empresas"])


def test_la_caida_de_los_bonos_es_tasas_al_alza_y_un_ataque_real_si_es_geopolitica():
    bonos = MK.analizar(item("Wall Street ve una señal preocupante en la caída del mercado de bonos de EE.UU."), TODAS)
    assert bonos["temas"] == ["tasas_eeuu"] and bonos["direccion"] == 1 and "NVDA" in bonos["afectadas"]         # bonos que caen = rendimientos que suben
    riad = MK.analizar(item("Houthi Attacks on Saudi Arabia Raise Risk of Wider War"), TODAS)
    assert "geopolitica" in riad["temas"] and riad["direccion"] == 1 and "ECOPETROL" in riad["beneficiadas"] and "NUCO" in riad["afectadas"]
    comercio = MK.analizar(item("China Moves to Stave Off Wider EU Trade War With Hybrid Deal"), TODAS)
    assert comercio is None or "geopolitica" not in comercio["temas"]                                            # una guerra comercial no es una guerra
