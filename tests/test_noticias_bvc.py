"""Pruebas del vigía de noticias de la BVC: lectura tolerante de las fuentes, a qué empresa se refiere cada titular, tipo y sentido de la noticia, puntaje
con confirmación del precio, recomendación según la evidencia, mensajes y estudio de eventos."""
import copy
import datetime as dt
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import estudio_noticias as E
from src import formato as F
from src import noticias_bvc as N
from src import servicio as S
from src.construir import universo_candidatos
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg
from tests.test_fase4 import bog, sin_jerga

CFG = make_cfg()
UTC = dt.timezone.utc
AHORA = bog(2026, 10, 6, 10, 0)                      # martes, mercado abierto
EMISORES = N.emisores_cfg(CFG)


def item(titulo, minutos=5, fuente="Valora Analitik", origen="rss", entidad="", tema="", ahora=AHORA, resumen=""):
    return N.Item(ahora.astimezone(UTC) - dt.timedelta(minutes=minutos), titulo, resumen, fuente, "https://ejemplo.co/n", origen, entidad, tema)


def tickers_de(it):
    return [e.tickers[0] for e in N.emisores_de(it, EMISORES)]


# =============================== lectura de las fuentes ===============================
def test_las_fechas_de_todos_los_formatos_de_los_medios_quedan_en_utc():
    esperado = dt.datetime(2026, 10, 5, 19, 40, 49, tzinfo=UTC)
    assert N.parsear_fecha("Mon, 05 Oct 2026 19:40:49 GMT") == esperado
    assert N.parsear_fecha("Mon, 05 Oct 2026 14:40:49 -0500") == esperado
    assert N.parsear_fecha("2026-10-05T14:40:49-05:00") == esperado
    assert N.parsear_fecha("2026-10-05 19:40:49") == esperado                                    # sin zona: los feeds que lo usan publican en UTC
    assert N.parsear_fecha("2026-10-05T14:40:49.000-0500") == esperado                           # formato de la Superfinanciera
    assert N.parsear_fecha("") is None and N.parsear_fecha("ayer") is None


def test_el_lector_de_rss_tolera_xml_mal_formado_y_separa_el_medio_en_google():
    malo = """<rss><channel><title>Feed & roto</title>
      <item><title><![CDATA[Ecopetrol anuncia dividendo extraordinario & recompra]]></title><link>https://x.co/1</link>
            <pubDate>Mon, 05 Oct 2026 19:40:49 GMT</pubDate><description><![CDATA[<p>La junta <b>aprobó</b> el pago.</p>]]></description></item>
      <item><title>Sin fecha</title></item>
      <item><title>Grupo Argos vende activos &amp; reduce deuda</title><pubDate>2026-10-05 18:00:00</pubDate></item>
    </channel></rss>"""
    its = N.items_de_rss(malo, "Valora Analitik")
    assert [i.titulo for i in its] == ["Ecopetrol anuncia dividendo extraordinario & recompra", "Grupo Argos vende activos & reduce deuda"]
    assert its[0].resumen == "La junta aprobó el pago." and its[0].url == "https://x.co/1" and its[0].fuente == "Valora Analitik"
    g = N.items_de_rss("<item><title>Bancolombia compra parque logístico - Portafolio</title><pubDate>Mon, 05 Oct 2026 19:40:49 GMT</pubDate>"
                       "<source url='x'>Portafolio</source></item>", "Google News", "google")
    assert g[0].titulo == "Bancolombia compra parque logístico" and g[0].fuente == "Portafolio" and g[0].origen == "google"


def test_los_registros_de_la_superfinanciera_se_convierten_en_noticias():
    datos = {"content": [{"fechaRegistro": "2026-10-05T15:06:49.000-0500", "resumen": "Ecopetrol informa  el pago de un dividendo\nextraordinario",
                          "tipoTemaInfoRelevante": {"nombre": "Proyecto utilidad o pérdida a presentar a Asamblea. "},
                          "entidad": {"razonSocial": "ECOPETROL S.A."}, "archivoInfoRelevante": {"idArchivoInfoRelevante": 7}},
                         {"fechaRegistro": None, "resumen": "sin fecha"}]}
    (it,) = N.items_de_sfc(datos)
    assert it.origen == "sfc" and it.fuente == "Superfinanciera" and it.entidad == "ECOPETROL S.A." and it.titulo == "Ecopetrol informa el pago de un dividendo extraordinario"
    assert it.ts == dt.datetime(2026, 10, 5, 20, 6, 49, tzinfo=UTC) and "Proyecto utilidad" in N.texto_para_clasificar(it)


class HttpFalso:
    """Responde por URL; una fuente puede fallar sin tumbar a las demás."""
    def __init__(self, respuestas):
        self.respuestas, self.pedidas = respuestas, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.pedidas.append((url, headers or {}))
        r = next((v for k, v in self.respuestas.items() if k in url), None)
        if r is None or isinstance(r, Exception):
            raise r or RuntimeError("sin respuesta")
        cuerpo, cabeceras, estado = r if isinstance(r, tuple) else (r, {}, 200)
        return SimpleNamespace(status_code=estado, headers=cabeceras, content=cuerpo.encode("utf-8") if isinstance(cuerpo, str) else b"",
                               json=lambda: json.loads(cuerpo), raise_for_status=lambda: None)


def test_una_fuente_caida_no_tumba_a_las_demas_y_se_usan_peticiones_condicionales():
    rss = "<item><title>Celsia gana contrato</title><pubDate>Mon, 05 Oct 2026 19:40:49 GMT</pubDate></item>"
    http = HttpFalso({"valoraanalitik": (rss, {"ETag": "abc"}, 200), "superfinanciera": json.dumps({"content": []}), "google": rss})
    lec = N.Lector(CFG, http=http, env={})
    items, salud = lec.todo(AHORA)
    assert salud["Valora Analitik"] and salud["Superfinanciera"] and salud["Google News"] and not salud["La República"]
    assert [i.titulo for i in items].count("Celsia gana contrato") == 2
    http.respuestas["valoraanalitik"] = ("", {}, 304)                                            # la segunda vez el medio dice "no cambió": no se baja de nuevo
    items2, salud2 = lec.todo(AHORA)
    cab = next(h for u, h in http.pedidas[::-1] if "valoraanalitik" in u)
    assert cab.get("If-None-Match") == "abc" and salud2["Valora Analitik"] and not any(i.fuente == "Valora Analitik" for i in items2)
    assert any("api-key" in h for u, h in http.pedidas if "superfinanciera" in u)


def test_google_rota_las_busquedas_una_por_ronda():
    http = HttpFalso({"google": "<rss></rss>"})
    lec = N.Lector(CFG, http=http, env={})
    for _ in range(3):
        lec.todo(AHORA)
    assert len([u for u, _ in http.pedidas if "google" in u]) == 3 and lec.ronda == 3


# =============================== ¿de qué empresa habla? ===============================
def test_los_titulares_se_asignan_a_la_empresa_correcta():
    assert tickers_de(item("Ecopetrol anuncia dividendo extraordinario")) == ["ECOPETROL"]
    assert tickers_de(item("Bancolombia compra parque logístico en Bogotá")) == ["PFBCOLOM"]
    assert tickers_de(item("Grupo Cibest reporta utilidad récord")) == ["PFBCOLOM"]
    assert tickers_de(item("Cementos Argos vende su negocio en Estados Unidos")) == ["CEMARGOS"]
    assert set(tickers_de(item("Grupo Argos y Grupo Sura avanzan en su escisión"))) == {"GRUPOARGOS", "GRUPOSURA"}
    assert tickers_de(item("El dólar cae por decisión de la FED")) == []


def test_los_nombres_que_son_palabras_corrientes_exigen_contexto_de_bolsa():
    assert tickers_de(item("El éxito de la selección Colombia en las eliminatorias")) == []
    assert tickers_de(item("Grupo Éxito reporta utilidad neta de $50.000 millones")) == ["EXITO"]
    assert tickers_de(item("Rescatan a mineros atrapados en Antioquia")) == []
    assert tickers_de(item("Mineros aumenta el dividendo para sus accionistas")) == ["MINEROS"]
    assert tickers_de(item("ISA gana contrato de transmisión en Perú por US$200 millones", resumen="La acción de ISA en la bolsa reaccionó")) == ["ISA"]


def test_los_resumenes_de_la_jornada_no_son_noticias_de_la_empresa():
    for ruido in ("Bolsa de Valores de Colombia hoy: Éxito lidera las ganancias y Ecopetrol cae", "Acción de Ecopetrol sube 3 % en la jornada",
                  "Ecopetrol lidera las pérdidas del Colcap este lunes", "Las acciones que recomiendan los analistas: Bancolombia y Ecopetrol",
                  "Video | Acciones de Ecopetrol lideran el volumen de la semana"):
        assert tickers_de(item(ruido)) == [], ruido


def test_en_la_superfinanciera_solo_cuenta_la_razon_social_exacta_no_las_filiales():
    def sfc(entidad):
        return tickers_de(item("Informa decisión de la junta", origen="sfc", fuente="Superfinanciera", entidad=entidad))
    assert sfc("ECOPETROL S.A.") == ["ECOPETROL"]
    assert sfc("BANCOLOMBIA S.A. podrá girar también con la denominación social Banco de Colombia") == ["PFBCOLOM"]
    assert sfc("GRUPO CIBEST S.A.") == ["PFBCOLOM"]
    assert sfc("FIDUCIARIA BANCOLOMBIA S.A.") == [] and sfc("CELSIA COLOMBIA S.A. E.S.P.") == [] and sfc("CELSIA S.A.") == ["CELSIA"]
    assert sfc("CORPORACIÓN FINANCIERA COLOMBIANA S.A., pudiendo utilizar las siglas CORFICOLOMBIANA") == ["CORFICOLCF"]
    assert sfc("FIDUCIARIA DAVIBANK S.A.") == []


def test_la_lista_negra_nunca_aparece_como_emisor():
    negra = set(CFG["universe"]["blacklist"])
    assert all(not (set(e.tickers) & negra) for e in EMISORES)
    cfg = copy.deepcopy(CFG)
    cfg["universe"]["blacklist"] = list(negra) + ["ECOPETROL"]
    assert "ECOPETROL" not in [t for e in N.emisores_cfg(cfg) for t in e.tickers]


# =============================== tipo de noticia y sentido ===============================
@pytest.mark.parametrize("titulo, cat, sentido", [
    ("Gilinski lanza OPA por el 20 % de Grupo Éxito", "opa", 1),
    ("Ecopetrol aprueba recompra de acciones por $1 billón", "recompra", 1),
    ("Grupo Argos propone dividendo extraordinario", "dividendo", 1),
    ("Ecopetrol recorta el dividendo tras caída de utilidades", "dividendo", -1),
    ("Utilidad neta de Bancolombia creció 20 % en el tercer trimestre", "resultados", 1),
    ("Ganancias de Cementos Argos cayeron 35 %", "resultados", -1),
    ("Ecopetrol presenta resultados del tercer trimestre", "resultados", 0),
    ("SIC abre investigación contra Grupo Éxito por precios", "legal", -1),
    ("Archivan investigación contra ISA", "legal", 1),
    ("Atentado obliga a Ecopetrol a suspender operaciones en Caño Limón", "operativo", -1),
    ("Bancolombia compra parque logístico en Bogotá", "fusion", 0),
    ("Ecopetrol confirma hallazgo de gas en el Caribe", "hallazgo", 1),
    ("Fitch rebaja la calificación de Ecopetrol", "calificacion", -1),
])
def test_cada_titular_se_clasifica_por_tipo_y_sentido(titulo, cat, sentido):
    c = N.clasificar(titulo, CFG)
    assert c is not None and c.cat == cat and c.sentido == sentido, (c, titulo)


def test_lo_que_no_mueve_la_accion_no_tiene_tipo_y_los_tramites_pesan_menos():
    assert N.clasificar("Ecopetrol y Petrobras culminan campaña exploratoria en el Caribe", CFG) is None
    assert N.clasificar("Grupo Argos y Nicky Jam donarán 100 viviendas", CFG) is None
    normal = N.clasificar("Ecopetrol propone dividendo extraordinario", CFG)
    tramite = N.clasificar("Ecopetrol pagará la segunda cuota del dividendo el 15 de octubre", CFG)
    assert tramite.peso == pytest.approx(normal.peso * CFG["noticias_bvc"]["castigo_rutina"])


def test_los_pesos_cubren_todos_los_tipos_y_solo_los_raros_pasan_con_el_titular():
    n = CFG["noticias_bvc"]
    assert set(n["pesos"]) == set(N.ETIQUETA)
    def solo_texto(cat, origen="rss"):
        return N.puntuar(N.Clasif(cat, 1, n["pesos"][cat]), {origen}, 1, None, CFG)[0]
    assert solo_texto("opa") >= n["umbral_alto"] and solo_texto("legal") >= n["umbral_alto"]
    for frecuente in ("resultados", "dividendo", "calificacion", "fusion"):                       # el estudio dice que casi nunca mueven fuerte: necesitan al precio
        assert solo_texto(frecuente) < n["umbral_alto"], frecuente
        assert N.puntuar(N.Clasif(frecuente, 1, n["pesos"][frecuente]), {"rss"}, 1, 1.4, CFG)[0] >= n["umbral_alto"], frecuente


def test_el_precio_confirma_contradice_o_pone_el_sentido():
    n = CFG["noticias_bvc"]
    c = N.Clasif("resultados", 1, n["pesos"]["resultados"])
    base, _ = N.puntuar(c, {"rss"}, 1, 0.2, CFG)                                                  # el precio no se ha movido
    assert N.puntuar(c, {"rss"}, 1, 1.5, CFG)[0] == pytest.approx(base + n["bono_precio"])
    assert N.puntuar(c, {"rss"}, 1, -1.5, CFG)[0] == pytest.approx(base - n["castigo_precio_contra"])
    assert N.puntuar(c, {"rss", "sfc"}, 2, None, CFG)[0] > N.puntuar(c, {"rss"}, 1, None, CFG)[0]  # fuente oficial y dos fuentes: más fiable
    neutro = N.Clasif("resultados", 0, n["pesos"]["resultados"])
    assert N.puntuar(neutro, {"rss"}, 1, None, CFG) == (pytest.approx(base * n["castigo_sin_sentido"]), 0)      # sin sentido no se inventa
    assert N.puntuar(neutro, {"rss"}, 1, -2.0, CFG)[1] == -1 and N.puntuar(neutro, {"rss"}, 1, 2.0, CFG)[1] == 1  # lo pone el precio


# =============================== precio y recomendación ===============================
class Fuentes:
    """Precios falsos: variación diaria de `sigma` y el cambio de hoy que se pida por ticker."""
    def __init__(self, hoy=None, sigma=0.02, vol=1e6, precio=1000.0):
        self.hoy, self.sigma, self.vol, self.precio, self.consultas = hoy or {}, sigma, vol, precio, []
        self.avisos = []

    def diario(self, t, dias=None):
        n = 80
        r = self.sigma * np.array([(-1) ** i for i in range(n)])
        return pd.DataFrame({"Close": self.precio * np.cumprod(1 + r), "Volume": np.full(n, self.vol if not callable(self.vol) else self.vol(t))},
                            index=pd.bdate_range(end="2026-10-05", periods=n))

    def cotizacion(self, t):
        self.consultas.append(t)
        if t in self.hoy and self.hoy[t] is None:
            return None
        return {"precio": self.precio * (1 + self.hoy.get(t, 0.0)), "cierre_previo": self.precio, "fuente": "falsa"}

    def traducir(self, t):
        return None


ESTUDIO = {"n_minimo": 15, "horizontes": [1, 3, 5], "n_total": 1883,
           "global": {"todos": {"n": 831, "abs_media": 0.0183},
                      "+1": {"n": 138, "pct_fuerte": 0.41, "h3": dict(n=138, media=-0.0062, mediana=-0.004, pct_pos=0.46, ic_lo=-0.0134, ic_hi=0.0001)},
                      "-1": {"n": 118, "pct_fuerte": 0.27, "h3": dict(n=118, media=0.0087, mediana=0.004, pct_pos=0.56, ic_lo=0.0005, ic_hi=0.0184)}},
           "apertura": {"texto+": {"n": 299, "pct_fuerte": 0.09, "h3": dict(n=299, media=0.0019, mediana=0.0, pct_pos=0.52, ic_lo=-0.0017, ic_hi=0.0058)},
                        "sin_hueco": {"n": 584, "pct_fuerte": 0.09, "h3": dict(n=584, media=0.0012, mediana=0.0, pct_pos=0.49, ic_lo=-0.0017, ic_hi=0.0042)}},
           "categorias": {"opa|todos": {"n": 22, "abs_media": 0.0369}, "legal|todos": {"n": 4, "abs_media": 0.09},
                          "resultados|todos": {"n": 264, "pct_fuerte": 0.11, "abs_media": 0.0178, "h3": dict(n=264, media=0.0015, mediana=0.0, pct_pos=0.48, ic_lo=-0.0035, ic_hi=0.0066)}}}


def test_el_pulso_mide_el_cambio_de_hoy_contra_lo_normal_sin_contar_la_barra_de_hoy():
    p = N.pulso(Fuentes({"ECOPETROL": 0.03}, sigma=0.02, vol=5e6), "ECOPETROL", CFG, AHORA)
    assert p["r_hoy"] == pytest.approx(0.03) and p["sigma"] == pytest.approx(0.02, rel=0.1) and p["z"] == pytest.approx(1.5, rel=0.1)
    assert p["liquidez_mm"] == pytest.approx(5e6 * 1000 / 1e6, rel=0.05)
    assert N.pulso(Fuentes({"ECOPETROL": None}), "ECOPETROL", CFG, AHORA) is None                  # sin cotización no se inventa


def test_se_recomienda_la_serie_mas_liquida_del_emisor():
    argos = next(e for e in EMISORES if e.nombre == "Grupo Argos")
    t, p = N.elegir_ticker(Fuentes(vol=lambda t: 9e6 if t == "PFGRUPOARG" else 1e6), argos, CFG, AHORA)
    assert t == "PFGRUPOARG" and p["ticker"] == "PFGRUPOARG"


def rec(sentido=1, hoy=0.0, tengo=False, ahora=AHORA, estudio=ESTUDIO, vol=5e6, texto=1, cat="resultados"):
    f = Fuentes({"ECOPETROL": hoy}, vol=vol)
    return N.recomendar(cat, sentido, "ECOPETROL", N.pulso(f, "ECOPETROL", CFG, ahora), ahora, CFG, tengo, estudio, texto)


def test_una_buena_noticia_no_basta_para_comprar_si_la_evidencia_no_paga_el_costo():
    r = rec(hoy=0.002)
    assert r["accion"] == "NO_COMPRAR" and r["evidencia"]["clave"] == "texto+" and r["evidencia"]["casos"] == 299 and r["reaccion"] == 0
    assert r["sesiones"] == 3 and r["salida"] == dt.date(2026, 10, 8) and r["rango"] == pytest.approx(0.02 * 3 ** 0.5, rel=0.1)
    assert r["stop"] == pytest.approx(max(1.5 * r["pulso"]["sigma"], 0.03)) and r["stop"] >= 0.03                # 1,5 veces lo normal, nunca menos de 3 %


def test_si_ya_subio_fuerte_no_se_persigue():
    r = rec(hoy=0.05)
    assert r["accion"] == "NO_PERSEGUIR" and r["reaccion"] == 1 and r["evidencia"]["clave"] == "+1" and r["evidencia"]["media"] < 0


def test_solo_se_recomienda_comprar_cuando_el_extremo_bajo_del_intervalo_supera_el_costo():
    bueno = copy.deepcopy(ESTUDIO)
    bueno["apertura"]["texto+"]["h3"].update(media=0.035, ic_lo=0.02, ic_hi=0.05)
    assert rec(hoy=0.002, estudio=bueno)["accion"] == "COMPRAR"
    casi = copy.deepcopy(ESTUDIO)
    casi["apertura"]["texto+"]["h3"].update(media=0.02, ic_lo=0.005, ic_hi=0.035)                   # promedio mayor que el costo, pero no con seguridad
    assert rec(hoy=0.002, estudio=casi)["accion"] == "NO_COMPRAR"
    de_noche = rec(hoy=0.0, estudio=bueno, ahora=bog(2026, 10, 6, 20, 0))
    assert de_noche["accion"] == "COMPRAR_MANANA" and de_noche["cuando"] == dt.date(2026, 10, 7) and de_noche["hora"] == "08:45"
    antes_de_abrir = rec(hoy=0.0, estudio=bueno, ahora=bog(2026, 10, 6, 7, 0))
    assert antes_de_abrir["accion"] == "COMPRAR_MANANA" and antes_de_abrir["cuando"] == dt.date(2026, 10, 6)


def test_sin_estudio_o_con_pocos_casos_nunca_se_recomienda_comprar():
    assert rec(hoy=0.002, estudio={})["accion"] == "NO_COMPRAR" and rec(hoy=0.002, estudio={})["evidencia"] is None
    pocos = copy.deepcopy(ESTUDIO)
    pocos["apertura"] = {"texto+": {"n": 8, "h3": dict(n=8, media=0.05, mediana=0.05, pct_pos=0.9, ic_lo=0.03, ic_hi=0.07)}}
    assert rec(hoy=0.002, estudio=pocos)["accion"] == "NO_COMPRAR"


def test_lo_que_ya_tienes_y_las_malas_noticias():
    assert rec(tengo=True, hoy=0.05)["accion"] == "MANTENER"
    assert rec(sentido=-1, tengo=True, hoy=-0.05)["accion"] == "NO_VENDER_PANICO"                 # ya cayó: en casos así rebotó en promedio
    assert rec(sentido=-1, tengo=True, hoy=-0.002)["accion"] == "VIGILAR"
    assert rec(sentido=-1, tengo=False, hoy=-0.05)["accion"] == "EVITAR"
    assert rec(hoy=0.002, vol=100)["accion"] == "POCO_LIQUIDA"


def test_el_plazo_nunca_pasa_del_final_del_concurso():
    assert N.sesiones_desde(bog(2026, 10, 6, 10), 3, CFG) == dt.date(2026, 10, 8)                   # hoy cuenta si el mercado sigue abierto
    assert N.sesiones_desde(bog(2026, 10, 6, 16), 3, CFG) == dt.date(2026, 10, 9)                   # ya cerró: desde mañana
    assert N.sesiones_desde(bog(2026, 10, 9, 16), 1, CFG) == dt.date(2026, 10, 13)                  # salta el festivo del lunes 12
    assert N.sesiones_desde(bog(2026, 11, 5, 10), 5, CFG) == dt.date(2026, 11, 6)
    assert N.sesiones_desde(bog(2026, 11, 6, 17), 3, CFG) is None


# =============================== la ronda de cada 2 minutos ===============================
class LectorFalso:
    def __init__(self):
        self.items, self.salud = [], {"Superfinanciera": True, "Valora Analitik": True}

    def todo(self, ahora):
        return list(self.items), dict(self.salud)


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setattr(N, "cargar_estudio", lambda ruta=None: ESTUDIO)
    return N.Memoria(tmp_path / "noticias.json")


def test_la_primera_ronda_solo_toma_la_linea_base(mem):
    lec = LectorFalso()
    lec.items = [item("Gilinski lanza OPA por Grupo Éxito")]
    s, salud = N.ronda(lec, mem, Fuentes(), AHORA, CFG)
    assert s == [] and mem.d["base"] and len(mem.d["vistos"]) == 1 and salud["Superfinanciera"]
    assert N.ronda(lec, mem, Fuentes(), AHORA, CFG)[0] == []                                       # lo que ya existía nunca se avisa


def test_sin_ninguna_fuente_no_hay_linea_base(mem):
    lec = LectorFalso()
    lec.salud = {"Superfinanciera": False, "Valora Analitik": False}
    N.ronda(lec, mem, Fuentes(), AHORA, CFG)
    assert "base" not in mem.d


def con_base(mem):
    lec = LectorFalso()
    N.ronda(lec, mem, Fuentes(), AHORA, CFG)
    return lec


def test_una_opa_nueva_se_avisa_una_sola_vez_con_su_recomendacion(mem):
    lec = con_base(mem)
    lec.items = [item("Gilinski lanza OPA por el 20 % de Grupo Éxito", fuente="Valora Analitik"), item("Lanzan OPA por Grupo Éxito a $5.000 por acción", fuente="La República")]
    (s,), _ = N.ronda(lec, mem, Fuentes({"EXITO": 0.001}, vol=5e6), AHORA, CFG)
    assert s.emisor == "Grupo Éxito" and s.ticker == "EXITO" and s.cat == "opa" and s.sentido == 1 and s.fuentes == ["Valora Analitik", "La República"]
    assert s.puntaje >= CFG["noticias_bvc"]["umbral_alto"] and s.rec["accion"] == "NO_COMPRAR"
    assert N.ronda(lec, mem, Fuentes({"EXITO": 0.001}, vol=5e6), AHORA, CFG)[0] == []              # ya vista
    lec.items.append(item("OPA por Grupo Éxito: lo que deben saber los accionistas", fuente="Portafolio"))
    assert N.ronda(lec, mem, Fuentes({"EXITO": 0.001}, vol=5e6), bog(2026, 10, 6, 10, 20), CFG)[0] == []      # mismo emisor y sentido: enfriamiento
    assert mem.d["senales"][-1]["ticker"] == "EXITO" and mem.d["senales"][-1]["accion"] == "NO_COMPRAR" and len(mem.d["senales"]) == 1


def test_una_noticia_frecuente_solo_pasa_si_el_precio_la_confirma(mem):
    lec = con_base(mem)
    lec.items = [item("Utilidad neta de Ecopetrol creció 20 % en el trimestre")]
    f = Fuentes({"ECOPETROL": 0.001}, vol=5e7)
    (m,), _ = N.ronda(lec, mem, f, AHORA, CFG)
    assert m.nivel == "medio" and m.puntaje < CFG["noticias_bvc"]["umbral_alto"] and m.impacto == pytest.approx(0.0178)    # el precio ni se movió: relevante, no de alto impacto
    lec.items = [item("Ganancias de Bancolombia subieron 30 % y superan lo esperado")]
    (s,), _ = N.ronda(lec, mem, Fuentes({"PFBCOLOM": 0.035, "BCOLOMBIA": 0.03}, vol=5e7), AHORA, CFG)
    assert s.nivel == "alto" and s.cat == "resultados" and s.sentido == 1 and s.rec["accion"] == "NO_PERSEGUIR" and s.rec["reaccion"] == 1
    assert [x["nivel"] for x in mem.d["senales"]] == ["medio", "alto"] and mem.d["senales"][0]["impacto"] == pytest.approx(0.0178)


def test_el_impacto_estimado_lleva_signo_y_usa_el_promedio_general_si_hay_pocos_casos():
    assert N.impacto_estimado("resultados", 1, ESTUDIO) == pytest.approx(0.0178) and N.impacto_estimado("resultados", -1, ESTUDIO) == pytest.approx(-0.0178)
    assert N.impacto_estimado("opa", 1, ESTUDIO) == pytest.approx(0.0369)
    assert N.impacto_estimado("legal", -1, ESTUDIO) == pytest.approx(-0.0183)                        # sólo 4 casos medidos: se usa el promedio de todos los tipos
    assert N.impacto_estimado("resultados", 1, {}) is None


def test_las_noticias_relevantes_tienen_tope_por_hora(mem):
    lec = con_base(mem)
    tope = CFG["noticias_bvc"]["max_medias_hora"]
    titulares = ["Utilidad neta de Ecopetrol creció 20 %", "Ganancias de Celsia subieron 15 %", "Utilidad de Promigas creció 9 %", "Ganancias de Terpel subieron 12 %",
                 "Utilidad neta de Nutresa creció 7 %", "Ganancias de Corficolombiana subieron 30 %"]
    total = 0
    for k, t in enumerate(titulares):
        lec.items = [item(t)]
        total += len(N.ronda(lec, mem, Fuentes(vol=5e7), bog(2026, 10, 6, 10, k), CFG)[0])
    assert total == tope and len(titulares) > tope


def test_no_se_consultan_precios_de_lo_que_no_puede_llegar_al_umbral(mem):
    lec = con_base(mem)
    lec.items = [item("Ecopetrol y Petrobras culminan campaña exploratoria"), item("El dólar cae tras decisión de la FED")]
    f = Fuentes()
    assert N.ronda(lec, mem, f, AHORA, CFG)[0] == [] and f.consultas == []


def test_noticia_sin_sentido_claro_no_se_avisa_aunque_sea_oficial(mem):
    lec = con_base(mem)
    lec.items = [item("Ecopetrol informa decisiones de la asamblea sobre fusión", origen="sfc", fuente="Superfinanciera", entidad="ECOPETROL S.A.")]
    assert N.ronda(lec, mem, Fuentes({"ECOPETROL": 0.002}, vol=5e7), AHORA, CFG)[0] == []


def test_lo_viejo_no_se_avisa_y_la_mala_noticia_de_lo_que_no_tienes_solo_si_es_muy_fuerte(mem):
    lec = con_base(mem)
    lec.items = [item("Gilinski lanza OPA por Grupo Éxito", minutos=60 * 5)]                        # de hace 5 horas: ya está en el precio
    assert N.ronda(lec, mem, Fuentes(vol=5e6), AHORA, CFG)[0] == []
    lec.items = [item("SIC sanciona a Celsia con multa millonaria")]
    (m,), _ = N.ronda(lec, mem, Fuentes({"CELSIA": -0.002}, vol=5e6), AHORA, CFG)
    assert m.nivel == "medio" and m.sentido == -1 and m.impacto < 0                                # no la tienes y el precio no cae: informativa, no alarma
    lec.items = [item("Superservicios impone sanción a Celsia y abre investigación", ahora=bog(2026, 10, 6, 11, 30))]
    (s,), _ = N.ronda(lec, mem, Fuentes({"CELSIA": -0.002}, vol=5e6), bog(2026, 10, 6, 11, 30), CFG, tenidos={"CELSIA"})
    assert s.nivel == "alto" and s.sentido == -1 and s.rec["accion"] == "VIGILAR" and s.rec["tengo"]


def test_la_memoria_se_guarda_y_se_poda(mem, tmp_path):
    lec = con_base(mem)
    lec.items = [item("Gilinski lanza OPA por Grupo Éxito")]
    N.ronda(lec, mem, Fuentes(vol=5e6), AHORA, CFG)
    mem.guardar()
    otra = N.Memoria(tmp_path / "noticias.json")
    assert otra.d["senales"][0]["ticker"] == "EXITO" and otra.d["base"] and not otra.sucia
    otra.d["vistos"]["viejo"] = "2026-09-01T00:00:00+00:00"
    otra.podar(AHORA.astimezone(UTC))
    assert "viejo" not in otra.d["vistos"] and len(otra.d["vistos"]) == 1
    (tmp_path / "roto.json").write_text("{no es json", encoding="utf-8")
    assert N.Memoria(tmp_path / "roto.json").d["senales"] == []                                   # una memoria dañada no tumba al vigía


# =============================== mensajes ===============================
def senal(accion_hoy=0.05, sentido=1, tengo=False, cat="resultados", titulo="Ganancias de Ecopetrol subieron 30 % y superan lo esperado"):
    f = Fuentes({"ECOPETROL": accion_hoy}, vol=5e7)
    p = N.pulso(f, "ECOPETROL", CFG, AHORA)
    r = N.recomendar(cat, sentido, "ECOPETROL", p, AHORA, CFG, tengo, ESTUDIO, sentido)
    return N.Senal("Ecopetrol", "ECOPETROL", cat, sentido, 0.77, [item(titulo), item(titulo + " hoy", fuente="La República")], r, AHORA.astimezone(UTC),
                   "alto", N.impacto_estimado(cat, sentido, ESTUDIO))


def test_el_aviso_dice_que_paso_que_hacer_que_esperar_y_cuanto_tiempo():
    t = F.plano(F.msg_noticia_bvc(senal(), AHORA, CFG, {"ticker": "PFGRUPOARG", "mee": 0.062, "corte": dt.date(2026, 10, 9)}))
    for esperado in ("Noticia de alto impacto: Ecopetrol", "Ganancias de Ecopetrol subieron 30 %", "Valora Analitik, La República", "resultados financieros",
                     "Impacto estimado: +1,8%", "hoy la acción va +5,0%", "ECOPETROL sube 5,0% hoy", "el precio ya reaccionó", "Qué hacer: No compres ECOPETROL ahora: ya subió",
                     "Qué esperar: en 138 anuncios oficiales", "bajó en promedio 0,6% frente al mercado", "subió 46% de las veces",
                     "movieron fuerte la acción solo 11% de las veces", "Cuánto tiempo: Si aun así decides entrar", "máximo 3 sesiones (vende a más tardar el jue 08/10)",
                     "sal antes si cae 3%", "PFGRUPOARG", "±6,2% hasta el corte del vie 09/10", "/comprar", "no una garantía"):
        assert esperado in t, esperado
    sin_jerga(F.msg_noticia_bvc(senal(), AHORA, CFG))


def test_el_aviso_cambia_segun_el_caso():
    quieto = F.plano(F.msg_noticia_bvc(senal(0.002), AHORA, CFG))
    assert "todavía no reacciona" in quieto and "No compres ECOPETROL solo por esta noticia" in quieto and "subió en promedio 0,2%" in quieto and "no alcanza a pagarlo" in quieto
    tengo = F.plano(F.msg_noticia_bvc(senal(0.05, tengo=True), AHORA, CFG))
    assert "Ya tienes ECOPETROL: mantenla" in tengo and "Cuánto tiempo" not in tengo
    mala = F.plano(F.msg_noticia_bvc(senal(-0.05, sentido=-1, tengo=True, cat="legal", titulo="SIC sanciona a Ecopetrol"), AHORA, CFG))
    assert "Impacto estimado: -1,8%" in mala and "hoy la acción va -5,0%" in mala and "no vendas por pánico" in mala and "subió en promedio 0,9%" in mala and "/semaforo ECOPETROL" in mala
    cerrado = F.plano(F.msg_noticia_bvc(senal(0.0), bog(2026, 10, 6, 20), CFG))
    assert "el mercado está cerrado" in cerrado


def test_el_aviso_corto_de_noticia_relevante_trae_el_impacto_en_porcentaje_con_signo():
    s = senal(0.004)
    s.nivel = "medio"
    m = F.msg_noticia_relevante(s, AHORA, CFG)
    t = F.plano(m)
    for esperado in ("Ecopetrol (ECOPETROL) · resultados financieros", "Ganancias de Ecopetrol subieron 30 %", "Valora Analitik, La República", "Impacto estimado: +1,8%",
                     "hoy la acción va +0,4%", "por sí sola no es motivo para comprar ni vender"):
        assert esperado in t, esperado
    neg = senal(-0.01, sentido=-1, tengo=True, cat="legal", titulo="SIC sanciona a Ecopetrol")
    assert "Impacto estimado: -1,8%" in F.plano(F.msg_noticia_relevante(neg, AHORA, CFG)) and "la tienes" in F.plano(F.msg_noticia_relevante(neg, AHORA, CFG))
    sin = senal(0.004)
    sin.impacto = None
    assert "positivo (+), sin una cifra medida" in F.plano(F.msg_noticia_relevante(sin, AHORA, CFG))
    assert m.count("<b>") == m.count("</b>") and len(t) < 600
    sin_jerga(m)


def test_el_aviso_no_rompe_el_formato_con_titulares_raros():
    m = F.msg_noticia_bvc(senal(titulo="Ecopetrol <b>gana</b> & sube > 5 %"), AHORA, CFG)
    assert m.count("<b>") == m.count("</b>") and m.count("<i>") == m.count("</i>") and "&lt;b&gt;gana&lt;/b&gt; &amp; sube &gt; 5 %" in m


def test_el_resumen_de_noticias_recientes_y_el_estado_de_las_fuentes():
    vacio = F.plano(F.msg_noticias_bvc_recientes([], {"Superfinanciera": True, "Semana": False}, AHORA))
    assert "No ha salido ninguna de alto impacto" in vacio and "sin respuesta de Semana" in vacio
    s = [dict(ts="2026-10-06T14:30+00:00", emisor="Ecopetrol", ticker="ECOPETROL", cat="opa", sentido=1, puntaje=0.8, titulo="Lanzan OPA por Ecopetrol", fuente="Valora Analitik", url="", accion="NO_COMPRAR", salida=None)]
    t = F.plano(F.msg_noticias_bvc_recientes(s, {"Superfinanciera": True}, AHORA))
    assert "ECOPETROL" in t and "hace 30 min" in t and "Lanzan OPA por Ecopetrol" in t and "las 1 responden" in t


# =============================== servicio: la ronda envía, y el modo sólo BVC ===============================
def ctx(tmp_path, f=None, dry=True, ahora=AHORA):
    return S.Contexto(CFG, f or Fuentes(vol=5e6), puntuar_vader, "vader", ahora, dry, tmp_path / "s.json", salida=lambda s: None)


def test_la_ronda_del_servicio_envia_el_aviso_con_botones_y_guarda_la_memoria(tmp_path, mem, monkeypatch):
    enviados = []
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, botones=None, **k: enviados.append((texto, botones)) or True)
    lec = con_base(mem)
    lec.items = [item("Gilinski lanza OPA por el 20 % de Grupo Éxito")]
    c = ctx(tmp_path, Fuentes({"EXITO": 0.001}, vol=5e6), dry=False)
    out = S.tick_noticias(c, lec, mem, {"ticker": "PFGRUPOARG", "mee": 0.06, "corte": dt.date(2026, 10, 9)})
    assert len(out) == 1 and len(enviados) == 1 and "Noticia de alto impacto: Grupo Éxito" in F.plano(enviados[0][0]) and "PFGRUPOARG" in enviados[0][0]
    assert enviados[0][1][0] == [("✅ La compré", "k:EXITO")] and ("🛒 Qué comprar", "c:comprar") in enviados[0][1][1]
    assert json.loads((tmp_path / "noticias.json").read_text(encoding="utf-8"))["senales"][0]["ticker"] == "EXITO"
    assert S.tick_noticias(c, lec, mem) == [] and len(enviados) == 1
    lec.items.append(item("Utilidad neta de Ecopetrol creció 20 % en el trimestre"))
    (corto,) = S.tick_noticias(ctx(tmp_path, Fuentes({"ECOPETROL": 0.001}, vol=5e7), dry=False), lec, mem)
    assert "Impacto estimado: +1,8%" in F.plano(corto) and "Noticia de alto impacto" not in corto and enviados[-1][1] is None    # relevante: aviso corto, sin botones


def test_con_las_fuentes_bien_la_ronda_no_reescribe_el_estado(tmp_path, mem, monkeypatch):
    """La ronda corre cada 2 minutos: si tocara state.json cada vez, la nube lo respaldaría sin parar."""
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, **k: True)
    lec = con_base(mem)
    c = ctx(tmp_path, dry=False)
    for _ in range(3):
        S.tick_noticias(c, lec, mem)
    assert not (tmp_path / "s.json").exists()


def test_si_todas_las_fuentes_caen_varias_rondas_se_avisa_una_vez_y_luego_la_recuperacion(tmp_path, mem, monkeypatch):
    enviados = []
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, **k: enviados.append(texto) or True)
    lec = con_base(mem)
    lec.salud = {"Superfinanciera": False, "Valora Analitik": False}
    for k in range(6):
        S.tick_noticias(ctx(tmp_path, dry=False, ahora=bog(2026, 10, 6, 10, 2 * k)), lec, mem)
    assert len(enviados) == 1 and "Falló una fuente de datos: las noticias de la BVC" in F.plano(enviados[0]) and "cada 2 minutos" in enviados[0]
    lec.salud = {"Superfinanciera": True, "Valora Analitik": False}
    S.tick_noticias(ctx(tmp_path, dry=False, ahora=bog(2026, 10, 6, 10, 14)), lec, mem)
    assert len(enviados) == 2 and "Ya funciona de nuevo" in F.plano(enviados[1])


def test_el_bot_solo_recomienda_acciones_de_la_bvc():
    assert CFG["modo"]["solo_bvc"] is True and CFG["banco"]["grupos_candidatos"] == ["local"]
    cands = universo_candidatos(CFG)
    assert cands and set(cands) <= set(CFG["universe"]["local"]) and not (set(cands) & set(CFG["universe"]["blacklist"]))
    assert "TSLA" not in cands and "PFBCOLOM" in cands and "BCOLOMBIA" in cands
    assert CFG["yahoo_symbol"]["PFBCOLOM"] == "PFCIBEST.CL" and CFG["yahoo_symbol"]["BCOLOMBIA"] == "CIBEST.CL"     # en pesos, no el ADR en dólares


def test_noticias_de_una_accion_de_ee_uu_que_no_tienes_remite_a_la_bvc(tmp_path):
    c = ctx(tmp_path)
    assert "sólo con acciones de la BVC" in S.resp_noticias(c, "TSLA")
    assert "❌" in S.resp_noticias(c, "popular")                                                    # lista negra
    Estado(CFG, c.estado_path).registrar_compra("TSLA", 5, 380.0, AHORA, 5 * 380 * 3900)
    c.f.noticias = lambda t, d: []
    assert "no hay noticias recientes" in S.resp_noticias(c, "tesla")                               # la tienes: sí se consulta (y entiende el nombre)


def test_que_comprar_lista_la_bvc_por_movimiento_esperado_con_plazo_y_costo():
    from src.regla_maestra import Decision, SIN_DATOS
    from src.relevo import Candidato, EntradaBanco
    banco = [EntradaBanco(Candidato("PFGRUPOARG", "local", 0.062, "σ20", valor_negociado_mm=19390)), EntradaBanco(Candidato("ISA", "local", 0.05, "σ20", valor_negociado_mm=9043))]
    corte = dict(fecha=dt.date(2026, 10, 9), top_pct=50)
    s = [dict(ts="2026-10-06T14:30+00:00", ticker="ECOPETROL", sentido=1, titulo="Lanzan OPA por Ecopetrol", fuente="Valora Analitik")]
    t = F.plano(F.msg_que_comprar(banco, "NUCO", 0.04, Decision(SIN_DATOS, codigo="sin_ranking"), corte, 4, s, "08:45", CFG, n_estudio=1883))
    for esperado in ("¿Qué acción de la BVC comprar?", "hasta el corte del vie 09/10 (quedan 4 sesiones)", "1. 🟢 PFGRUPOARG: se espera ±6,2%", "1,6 veces lo que se mueve tu cartera",
                     "negocia $19,4 mil millones al día", "Si vas a comprar una: PFGRUPOARG", "Cuánto tiempo: hasta ese corte", "cuesta cerca de 1,3%",
                     "Noticias fuertes de las últimas 24 horas", "Lanzan OPA por Ecopetrol", "medido en 1.883 anuncios oficiales", "Solo acciones de la BVC"):
        assert esperado in t, esperado
    assert "no tengo ninguna candidata" in F.plano(F.msg_que_comprar([], "NUCO", None, Decision(SIN_DATOS, codigo="sin_ranking"), corte, 4, [], "08:45", CFG))


# =============================== estudio de eventos ===============================
def precios(n=60, saltos=None):
    idx = pd.bdate_range("2026-01-05", periods=n)
    r = np.zeros(n)
    for k, v in (saltos or {}).items():
        r[k] = v
    r = r + 0.01 * np.array([(-1) ** i for i in range(n)])
    c = 100 * np.cumprod(1 + r)
    return pd.DataFrame({"Open": np.r_[100.0, c[:-1]], "Close": c, "Volume": 1e6}, index=idx), pd.Series(100.0, index=idx)


def test_el_dia_de_reaccion_es_el_mismo_si_salio_en_horario_y_el_siguiente_si_salio_de_noche():
    px, _ = precios()
    zona = dt.timezone(dt.timedelta(hours=-5))
    martes = px.index[30].date()
    en_horario = dt.datetime.combine(martes, dt.time(10, 30), zona)
    de_noche = dt.datetime.combine(martes, dt.time(19, 0), zona)
    madrugada = dt.datetime.combine(martes, dt.time(6, 0), zona)
    assert E.dia_de_reaccion(en_horario, px.index) == (30, True) and E.dia_de_reaccion(de_noche, px.index) == (31, False)
    assert E.dia_de_reaccion(madrugada, px.index) == (30, False)
    sabado = dt.datetime.combine(px.index[34].date() + dt.timedelta(days=1), dt.time(11, 0), zona)
    assert E.dia_de_reaccion(sabado, px.index)[0] == 35                                              # fin de semana: reacciona el lunes
    assert E.dia_de_reaccion(dt.datetime(2030, 1, 1, 10, tzinfo=zona), px.index) is None


def test_lo_que_vino_despues_se_mide_sin_trampa_desde_el_cierre_del_dia_de_reaccion():
    zona = dt.timezone(dt.timedelta(hours=-5))
    px, ref = precios(saltos={31: 0.06, 32: -0.02})                                                # sube 6 % el día de reacción y devuelve 2 % al siguiente
    ev = dict(emisor="X", cat="resultados", sentido_texto=1, ts=dt.datetime.combine(px.index[30].date(), dt.time(19, 0), zona))
    m = E.medir(ev, px, ref, [1, 3])
    assert m["fecha"] == px.index[31].date() and not m["en_horario"] and m["z"] > 3 and m["reaccion"] == pytest.approx(px["Close"].iloc[31] / px["Close"].iloc[30] - 1)
    assert m["despues_1"] == pytest.approx(px["Close"].iloc[32] / px["Close"].iloc[31] - 1) and m["despues_1"] < 0     # NO incluye la subida del día 31
    assert m["apertura_1"] == pytest.approx(px["Close"].iloc[31] / px["Open"].iloc[31] - 1)                           # comprando en la apertura sí la captura
    assert m["hueco_z"] is not None
    assert E.medir(dict(ev, ts=dt.datetime.combine(px.index[3].date(), dt.time(19, 0), zona)), px, ref, [1]) is None    # sin historia para medir lo normal


def test_la_tabla_resume_por_tipo_y_por_reaccion_y_un_anuncio_por_emisor_y_dia():
    zona = dt.timezone(dt.timedelta(hours=-5))
    px, ref = precios(saltos={31: 0.06, 41: -0.06})
    def ev(i, cat="resultados"):
        return dict(emisor="X", cat=cat, sentido_texto=1, ts=dt.datetime.combine(px.index[i].date(), dt.time(19, 0), zona))
    medidos = [m for m in (E.medir(e, px, ref, CFG["noticias_bvc"]["estudio"]["horizontes"]) for e in (ev(30), ev(30, "dividendo"), ev(40), ev(45, "otros"))) if m]
    dep = E.depurar(medidos, CFG["noticias_bvc"]["pesos"])
    assert len(medidos) == 4 and len(dep) == 3 and [m["cat"] for m in dep] == ["resultados", "resultados", "otros"]      # del mismo día queda el de más peso
    t = E.tabla(dep, CFG)
    assert t["n_total"] == 3 and t["categorias"]["resultados|+1"]["n"] == 1 and t["categorias"]["resultados|-1"]["n"] == 1 and t["categorias"]["resultados|todos"]["n"] == 2
    assert t["global"]["sin_tipo"]["n"] == 1 and t["apertura"]["todos"]["n"] == 2 and "h1" in t["categorias"]["resultados|todos"]
    assert E.resumir([], [1]) == {"n": 0}


def test_la_evidencia_elige_el_horizonte_mas_cercano_y_exige_casos_suficientes():
    ev = N.evidencia(ESTUDIO, "global", "+1", 5)
    assert ev["h"] == 3 and ev["casos"] == 138 and ev["media"] == pytest.approx(-0.0062)
    assert N.evidencia(ESTUDIO, "global", "no existe", 3) is None and N.evidencia({}, "global", "+1", 3) is None
    assert N.evidencia({**ESTUDIO, "n_minimo": 500}, "global", "+1", 3) is None


def test_el_estudio_real_guardado_tiene_la_forma_que_usa_el_bot():
    t = N.cargar_estudio()
    if not t:
        pytest.skip("aún no se ha corrido run_estudio_noticias.py")
    assert t["n_total"] > 500 and {"+1", "-1"} <= set(t["global"]) and {"texto+", "sin_hueco"} <= set(t["apertura"])
    ev = N.evidencia(t, "global", "+1", 3)
    assert ev and {"media", "pct_pos", "ic_lo", "ic_hi", "h", "casos"} <= set(ev)
