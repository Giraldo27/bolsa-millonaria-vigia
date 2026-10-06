"""Vigía de noticias de la BVC: cada 2 minutos lee las fuentes, se queda SÓLO con lo que es de un emisor de la BVC y puede mover su acción, y arma
la recomendación (qué acción, qué esperar y por cuánto tiempo). NUNCA envía órdenes.

Fuentes, de la más rápida y fiable a la menos:
  1. Superfinanciera · "Información relevante" (SIMEV): lo que cada emisor está OBLIGADO a publicar (dividendos, OPA, resultados, sanciones…). Oficial.
  2. RSS de medios económicos colombianos (Valora Analitik, La República, Portafolio, Semana, El Tiempo…).
  3. Google News (una búsqueda agrupada por ronda, rotando): respaldo y corroboración.

Reglas para no molestar: la primera ronda sólo toma la "línea base" (no avisa de lo que ya existía), cada titular se avisa una vez, no se repite el mismo
emisor y sentido antes de `enfriamiento_emisor_min` y sólo pasa lo que supera `umbral_alto`. Una noticia sin sentido claro (ni en el texto ni en el precio)
NO se avisa: no se inventa una dirección.

La recomendación NO sale de la intuición ("buena noticia = comprar") sino de lo que pasó después de anuncios parecidos (estudio de eventos sobre la
información relevante histórica: run_estudio_noticias.py → data/estudio_noticias_bvc.json). Sólo se recomienda COMPRAR cuando esa evidencia supera el costo
de entrar y salir; si no, el aviso lo dice con los números. Ver ASSUMPTIONS.md."""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import html
import json
import math
import os
import re
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import concurso as C
from .config import ROOT

UTC = dt.timezone.utc
ARCHIVO = "data/noticias_bvc.json"
ESTUDIO = "data/estudio_noticias_bvc.json"


# ------------------------------------------------------------------ texto
def norm(s: str) -> str:
    """Minúsculas, sin tildes y con espacios simples: 'Éxito  S.A.' → 'exito s.a.'."""
    t = unicodedata.normalize("NFKD", s or "")
    return " ".join("".join(c for c in t if not unicodedata.combining(c)).lower().split())


def huella(titulo: str) -> str:
    return hashlib.sha1("".join(c for c in norm(titulo) if c.isalnum() or c == " ")[:160].encode("utf-8")).hexdigest()[:14]


_ETIQ = re.compile(r"<[^>]+>")


def limpiar(s: str | None) -> str:
    """Quita CDATA, etiquetas y entidades HTML de un campo de RSS."""
    t = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", s or "", flags=re.S)
    return " ".join(html.unescape(_ETIQ.sub(" ", html.unescape(t))).split())


def parsear_fecha(s: str | None) -> dt.datetime | None:
    """Acepta los formatos que usan los medios: RFC 822 ('Mon, 05 Oct 2026 19:40:49 GMT'), ISO ('2026-10-05T14:50:41-05:00') y
    'AAAA-MM-DD HH:MM:SS' (sin zona: los feeds que lo usan publican en UTC). Devuelve UTC o None."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        d = email.utils.parsedate_to_datetime(s)
        return (d if d.tzinfo else d.replace(tzinfo=UTC)).astimezone(UTC)
    except (TypeError, ValueError, IndexError):
        pass
    try:
        t = re.sub(r"([+-]\d{2})(\d{2})$", r"\1:\2", s.replace("Z", "+00:00"))        # -0500 → -05:00
        t = re.sub(r"\.\d+(?=[+-]|$)", "", t)                                          # sin milisegundos
        d = dt.datetime.fromisoformat(t.replace(" ", "T", 1))
        return (d if d.tzinfo else d.replace(tzinfo=UTC)).astimezone(UTC)
    except ValueError:
        return None


# ------------------------------------------------------------------ noticias y emisores
@dataclass
class Item:
    ts: dt.datetime                     # UTC
    titulo: str
    resumen: str = ""
    fuente: str = ""                    # medio ("Valora Analitik") o "Superfinanciera"
    url: str = ""
    origen: str = "rss"                 # sfc | rss | google
    entidad: str = ""                   # razón social (sólo SFC)
    tema: str = ""                      # tema oficial (sólo SFC)

    @property
    def texto(self) -> str:
        return f"{self.titulo}. {self.resumen}" if self.resumen else self.titulo


@dataclass
class Emisor:
    nombre: str
    tickers: list[str]
    alias: list[str]                    # cómo lo nombran los titulares (ya normalizados)
    sfc: list[str] = field(default_factory=list)       # comienzo de la razón social en la Superfinanciera
    ambiguo: bool = False               # el nombre es una palabra corriente ("éxito", "mineros"): exige contexto de bolsa en el titular


def emisores_cfg(cfg: dict[str, Any]) -> list[Emisor]:
    negra = set(cfg["universe"]["blacklist"])
    out = []
    for e in cfg["noticias_bvc"]["emisores"]:
        tk = [t for t in e["tickers"] if t not in negra]
        if tk:
            out.append(Emisor(e["nombre"], tk, [norm(a) for a in e.get("alias", [])], [norm(a) for a in e.get("sfc", [])], bool(e.get("ambiguo"))))
    return out


CONTEXTO_BOLSA = re.compile(r"\b(accion|acciones|accionistas?|bolsa|bvc|colcap|dividendos?|utilidad(es)?|ganancias?|resultados|ebitda|opa|inversionistas?|"
                            r"emisor|titulo|ingresos|perdidas?|calificacion|trimestre|recompra|asamblea|junta directiva|s\.a\.|capitalizacion)\b")
# Titulares que NO son una noticia de la empresa: resúmenes de la jornada ("X lidera las ganancias", "la acción de X sube 3 %": eso ya es el precio), guías, opinión…
RUIDO = re.compile(r"(precio del dolar|dolar hoy|asi cerro|asi abrio|asi amanec|cierre de (la )?bolsa|resumen de (la )?jornada|horoscopo|loteria|"
                   r"en vivo|minuto a minuto|como invertir|que es y como|guia para|pico y placa|podcast|video \||opinion:|editorial|"
                   r"las acciones que (mas|recomiendan)|acciones mas (valorizadas|negociadas|desvalorizadas)|bolsa de valores de colombia hoy|"
                   r"lidera(n|ron)? (las )?(ganancias|perdidas|alzas|bajas|valorizaciones|caidas)|colcap (sube|cae|cierra|abre|avanza|retrocede|gana|pierde)|"
                   r"accion(es)? de [a-z .]{2,40} (sube|suben|cae|caen|baja|bajan|repunta|se desploma|se dispara|se valoriza)|"
                   r"(sube|cae|baja|repunta|se dispara|se desploma) la accion)")
# Anuncios de trámite: existen, pero casi nunca mueven la acción (se multiplica el peso por noticias_bvc.castigo_rutina)
RUTINA = re.compile(r"((fecha|calendario|cronograma) de pago|pagara (la |el )?(primera|segunda|tercera|cuarta|ultima) cuota|\bcuota\b|ex ?- ?dividendo|periodo ex|"
                    r"citacion a|convoca(toria)? a (la )?(asamblea|reunion)|convoco a reunion|aviso de convocatoria|publica el documento tecnico|"
                    r"codigo de buen gobierno|encuesta codigo pais|representacion de accionistas|se permite informar que en el diario)")


def _hay(alias: str, texto: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", texto) is not None


def emisores_de(item: Item, emisores: list[Emisor]) -> list[Emisor]:
    """Emisores de la BVC a los que se refiere la noticia. Superfinanciera: por la razón social (exacta, desde el comienzo: así no entran las filiales,
    'Fiduciaria Bancolombia' no es 'Bancolombia'). Medios: por el TITULAR (el cuerpo menciona empresas de pasada)."""
    if item.origen == "sfc":
        ent = norm(item.entidad)
        return [e for e in emisores if any(ent == s or ent.startswith(s + " ") or ent.startswith(s + ",") for s in e.sfc)]
    t = norm(item.titulo)
    if RUIDO.search(t):
        return []
    out = []
    for e in emisores:
        if any(_hay(a, t) for a in e.alias) and (not e.ambiguo or CONTEXTO_BOLSA.search(t + " " + norm(item.resumen))):
            out.append(e)
    return out


def items_de_rss(texto: str, fuente: str, origen: str = "rss") -> list[Item]:
    """Lector de RSS TOLERANTE (varios medios publican XML mal formado que un lector estricto rechaza): extrae cada <item> con expresiones regulares."""
    out = []
    for bloque in re.findall(r"<item[\s>](.*?)</item>", texto or "", flags=re.S | re.I):
        def campo(nombre: str) -> str:
            m = re.search(rf"<{nombre}(?:\s[^>]*)?>(.*?)</{nombre}>", bloque, flags=re.S | re.I)
            return limpiar(m.group(1)) if m else ""
        titulo, ts = campo("title"), parsear_fecha(campo("pubDate") or campo("dc:date") or campo("published"))
        if not titulo or ts is None:
            continue
        medio = fuente
        if origen == "google":
            medio = campo("source") or fuente
            if " - " in titulo:
                titulo, _, resto = titulo.rpartition(" - ")
                medio = medio or resto
        out.append(Item(ts, titulo, "" if origen == "google" else campo("description")[:300], medio, campo("link"), origen))
    return out


def items_de_sfc(datos: Any) -> list[Item]:
    """Registros de 'Información relevante' de la Superfinanciera → Item (titular = el resumen oficial)."""
    filas = datos.get("content", []) if isinstance(datos, dict) else (datos or [])
    out = []
    for x in filas:
        ts = parsear_fecha(x.get("fechaRegistro"))
        resumen = " ".join((x.get("resumen") or "").split())
        ent = x.get("entidad") or {}
        if ts is None or not resumen:
            continue
        idarch = (x.get("archivoInfoRelevante") or {}).get("idArchivoInfoRelevante")
        out.append(Item(ts, resumen[:400], "", "Superfinanciera", "https://www.superfinanciera.gov.co/SIMEV2/informacionrelevantegeneral", "sfc",
                        (ent.get("razonSocial") or "").split(" , pero ")[0], ((x.get("tipoTemaInfoRelevante") or {}).get("nombre") or "").strip()
                        + (f" #{idarch}" if idarch else "")))
    return out


# ------------------------------------------------------------------ ¿qué tipo de noticia es y hacia dónde empuja?
# (clave, etiqueta en palabras sencillas, sentido de partida, sesiones sugeridas, patrones sobre texto normalizado)
CATEGORIAS: list[tuple[str, str, int, int, list[str]]] = [
    ("opa", "oferta para comprar la empresa (OPA)", +1, 5, [r"\bopa\b", r"oferta publica de adquisicion", r"oferta de adquisicion", r"oferta para (comprar|adquirir)",
                                                              r"ofrece comprar"]),
    ("deslistado", "salida de la bolsa", -1, 3, [r"cancelacion de la inscripcion", r"deslist", r"retir(o|ar|ara) (sus acciones )?de la bolsa"]),
    ("recompra", "recompra de acciones", +1, 5, [r"recompra", r"readquisicion de acciones", r"readquirir (sus )?acciones"]),
    ("dividendo", "dividendo", +1, 5, [r"dividendo", r"distribucion de utilidades"]),
    ("resultados", "resultados financieros", 0, 3, [r"utilidad(es)? (neta|del|de |operacional|consolidad)", r"\bganancias?\b", r"\bgano\b", r"\bperdio\b",
                                                     r"resultados (financieros|del|de su|trimestral|consolidados|al cierre)", r"\bebitda\b", r"perdidas? (neta|de |por )",
                                                     r"ingresos (operacionales|crecieron|cayeron|subieron|aumentaron|bajaron|de )", r"estados financieros",
                                                     r"(primer|segundo|tercer|cuarto) trimestre"]),
    ("legal", "sanción, investigación o demanda", -1, 3, [r"sancion", r"\bmultas?\b", r"pliego de cargos", r"investiga", r"demanda (contra|a |de |por )", r"embargo",
                                                           r"toma de posesion", r"intervencion", r"imputa", r"fraude", r"corrupcion", r"liquidacion (judicial|forzosa)"]),
    ("operativo", "problema en la operación", -1, 3, [r"atentado", r"derrame", r"\bparo\b", r"huelga", r"suspend(e|io|era) (operaciones|la operacion|la produccion|produccion)",
                                                       r"accidente", r"bloqueo", r"cierre de (planta|mina|campo|tiendas)", r"incendio", r"explosion", r"ciberataque"]),
    ("fusion", "compra, venta o fusión de un negocio", 0, 3, [r"\bfusion(a|ara|o)?\b", r"\bescision\b", r"\badquier[ea]\b", r"\badquirio\b", r"adquisicion de", r"\bcompr(a|o|ara)\b",
                                                               r"\bvend(e|io|era)\b", r"\bventa de (su|sus|la|el|una|un) ", r"desinversion", r"enajenacion", r"se queda con"]),
    ("hallazgo", "contrato, hallazgo o adjudicación", +1, 3, [r"hallazgo", r"descubri", r"adjudic", r"gan(a|o|aron) (el |la |un |una )?(contrato|licitacion|concesion)",
                                                               r"contrato (por|de) ", r"firm(a|o|aron) (un |el )?(contrato|acuerdo)", r"licencia ambiental"]),
    ("indice", "entrada o salida de un índice", 0, 5, [r"\bmsci\b", r"\bftse\b", r"rebalanceo", r"(canasta|composicion) del (indice|colcap)"]),
    ("capital", "emisión de acciones nuevas", -1, 3, [r"emision de acciones", r"aumento de capital", r"capitalizacion de"]),
    ("guia", "metas y planes de la empresa", 0, 3, [r"proyecci", r"\bmetas? (de|para) ", r"plan de inversi", r"\brecort(a|o|ara) (la |su |sus |el )?(meta|inversion|proyeccion|prevision|produccion|plan)",
                                                    r"\beleva(n)? (la |su |sus )?(meta|prevision|proyeccion)"]),
    ("calificacion", "calificación de riesgo", 0, 2, [r"calificacion", r"perspectiva (negativa|positiva|estable)", r"\bfitch\b", r"\bmoody", r"standard & poor", r"\bs&p\b", r"grado de inversion"]),
    ("analistas", "recomendación de analistas", 0, 3, [r"precio objetivo", r"sobreponderar", r"infraponderar", r"recomienda(n)? (comprar|vender|mantener)", r"recomendacion de (compra|venta)"]),
    ("direccion", "cambio en la dirección", 0, 2, [r"\brenunci", r"destitu", r"nuev[oa] presidente", r"sale (el|la) presidente", r"cambio de presidente", r"\bnombr(a|o|an|aron) "]),
]
ETIQUETA = {c[0]: c[1] for c in CATEGORIAS} | {"otra": "noticia de la empresa"}
SESIONES = {c[0]: c[3] for c in CATEGORIAS} | {"otra": 3}

POSITIVAS = re.compile(r"\b(sube|subio|subieron|suben|crece|crecio|crecieron|crecen|aument(a|o|an|aron)|record|super(a|o|an|aron)|mejor(a|o|an|aron)|gana|gano|ganaron|positiv[oa]s?|"
                       r"alza|repunt(a|o)|dispar(a|o)|avanz(a|o)|duplic(a|o)|triplic(a|o)|elev(a|o)|fortalece|aprueb(a|an)|aprobo|aprobaron|mayor(es)?|incluid[ao]|entra|ingresa|"
                       r"sobreponderar|comprar|extraordinario|archiv(a|an|o)|absuel(ve|to)|levant(a|an|o)|reanud(a|o))\b")
NEGATIVAS = re.compile(r"\b(cae|cayo|cayeron|caen|baj(a|o|an|aron)|disminuy(e|o|eron)|reduj(o|eron)|reduce|pierde|perdio|perdieron|perdidas?|retroced(e|io)|desplom(a|o)|"
                       r"negativ[oa]s?|rebaj(a|o|an)|recort(a|o|an)|deterior(a|o)|menor(es)?|excluid[ao]|sale|infraponderar|vender|suspend(e|io|en)|cancel(a|o|an)|"
                       r"no pagara|sin dividendo|incumpl(e|io))\b")


def sentido_del_texto(texto: str) -> int:
    """+1 si el titular trae más palabras de subida/mejora que de caída/problema, −1 al revés, 0 si no se sabe."""
    t = norm(texto)
    pos, neg = len(POSITIVAS.findall(t)), len(NEGATIVAS.findall(t))
    return (pos > neg) - (neg > pos)


def palabras_clave(titulo: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]{4,}", norm(titulo))}


def es_repetida(titulo: str, previos: list[list[str]], umbral: float) -> bool:
    """¿Es la misma noticia que otra ya avisada (otro medio, otro titular)? Parecido = palabras en común ÷ palabras del titular más corto."""
    a = palabras_clave(titulo)
    for p in previos:
        b = set(p)
        if a and b and len(a & b) / min(len(a), len(b)) >= umbral:
            return True
    return False


def lleno(lista: list[Any], tope: int | None) -> bool:
    """¿Se alcanzó el tope? Un tope de 0 (o vacío) significa SIN tope."""
    return bool(tope) and len(lista) >= tope


def texto_para_clasificar(it: Item) -> str:
    """Medios: sólo el TITULAR (el cuerpo trae palabras sueltas que confunden). Superfinanciera: el resumen oficial más el tema con que lo radicaron."""
    return f"{it.titulo}. {it.tema.split(' #')[0]}" if it.origen == "sfc" else it.titulo


@dataclass
class Clasif:
    cat: str
    sentido: int                        # +1 empuja al alza, -1 a la baja, 0 no se sabe por el texto
    peso: float


def clasificar(texto: str, cfg: dict[str, Any]) -> Clasif | None:
    """Tipo de noticia de mayor peso que aparece en el texto y hacia dónde empuja. None si no es de ningún tipo que mueva la acción."""
    t = norm(texto)
    pesos = cfg["noticias_bvc"]["pesos"]
    mejor: tuple[float, str, int] | None = None
    for clave, _, base, _, patrones in CATEGORIAS:
        if any(re.search(p, t) for p in patrones) and (mejor is None or pesos[clave] > mejor[0]):
            mejor = (pesos[clave], clave, base)
    if mejor is None:
        return None
    peso, clave, base = mejor
    pos, neg = len(POSITIVAS.findall(t)), len(NEGATIVAS.findall(t))
    if base == 0:
        sentido = (pos > neg) - (neg > pos)
    elif base > 0:
        sentido = -1 if (neg > pos and neg >= 1) else +1                               # "recorta el dividendo", "no pagará dividendo"
    else:
        sentido = +1 if (pos > neg and re.search(r"\b(archiv|absuel|levant|reanud|gan(a|o) (la |el )?(demanda|pleito|caso))", t)) else -1
    if RUTINA.search(t):
        peso *= cfg["noticias_bvc"]["castigo_rutina"]
    return Clasif(clave, sentido, float(peso))


def puntuar(c: Clasif, origenes: set[str], n_fuentes: int, z_hoy: float | None, cfg: dict[str, Any]) -> tuple[float, int]:
    """(puntaje 0–1, sentido final). El puntaje parte del peso del tipo de noticia, se ajusta por la fiabilidad de la fuente, sube si varias fuentes la
    repiten o si el precio ya la confirma, y baja si el precio va en contra. Si el texto no dice hacia dónde empuja, el sentido lo pone el precio
    (cuando se mueve ≥ 1 vez lo normal); si tampoco, se penaliza (no se inventa una dirección)."""
    n = cfg["noticias_bvc"]
    fiab = max(n["fiabilidad"].get(o, 0.8) for o in origenes)
    p = c.peso * fiab + (n["bono_varias_fuentes"] if n_fuentes >= 2 else 0.0)
    sentido = c.sentido
    confirma = z_hoy is not None and z_hoy == z_hoy and abs(z_hoy) >= n["z_confirma"]
    if sentido == 0:
        if confirma:
            sentido = 1 if z_hoy > 0 else -1
            p += n["bono_precio"]
        else:
            p *= n["castigo_sin_sentido"]
    elif confirma:
        p += n["bono_precio"] if (z_hoy > 0) == (sentido > 0) else -n["castigo_precio_contra"]
    return max(0.0, min(1.0, p)), sentido


# ------------------------------------------------------------------ lectura de las fuentes
class Lector:
    """Descarga las fuentes en paralelo. Cada una falla por separado; usa peticiones condicionales (ETag / If-Modified-Since) para no bajar lo que no cambió."""

    def __init__(self, cfg: dict[str, Any], http: Any = None, env: dict[str, str] | None = None):
        self.cfg = cfg
        self.n = cfg["noticias_bvc"]
        if http is None:
            import requests
            http = requests.Session()
        self.http = http
        self.env = env if env is not None else dict(os.environ)
        self.cond: dict[str, dict[str, str]] = {}
        self.fallos: dict[str, int] = {}
        self.ronda = 0

    def _get(self, url: str, params: dict[str, Any] | None = None, cabeceras: dict[str, str] | None = None, condicional: bool = True) -> Any:
        h = {"User-Agent": self.n["agente"], **(cabeceras or {})}
        if condicional:
            h.update(self.cond.get(url, {}))
        r = self.http.get(url, params=params, headers=h, timeout=self.n["timeout_s"])
        if r.status_code == 304:
            return None
        r.raise_for_status()
        if condicional:
            c = {k2: r.headers[k1] for k1, k2 in (("ETag", "If-None-Match"), ("Last-Modified", "If-Modified-Since")) if r.headers.get(k1)}
            if c:
                self.cond[url] = c
        return r

    def sfc(self, ahora: dt.datetime) -> list[Item]:
        s = self.n["fuentes"]["sfc"]
        hoy = ahora.astimezone(C.bogota(self.cfg)).date()
        r = self._get(s["url"], {"page": 0, "size": s["size"], "fechaDesde": f"{hoy - dt.timedelta(days=1):%Y/%m/%d}", "fechaHasta": f"{hoy:%Y/%m/%d}"},
                      {"api-key": self.env.get("SFC_API_KEY") or s["api_key"], "Accept": "application/json"}, condicional=False)
        return items_de_sfc(r.json())

    def rss(self, fuente: dict[str, Any]) -> list[Item]:
        r = self._get(fuente["url"])
        return [] if r is None else items_de_rss(r.content.decode("utf-8", "replace"), fuente["nombre"])

    def google(self, consulta: str) -> list[Item]:
        g = self.n["fuentes"]["google"]
        r = self._get("https://news.google.com/rss/search", {"q": f"{consulta} when:{g['cuando']}", "hl": "es-419", "gl": "CO", "ceid": "CO:es-419"}, condicional=False)
        return items_de_rss(r.content.decode("utf-8", "replace"), "Google News", "google")

    def todo(self, ahora: dt.datetime) -> tuple[list[Item], dict[str, bool]]:
        """Todas las fuentes activas de esta ronda. Devuelve (noticias, salud por fuente). Una fuente que falla 3 veces seguidas descansa unas rondas."""
        f = self.n["fuentes"]
        tareas: list[tuple[str, Callable[[], list[Item]]]] = []
        if f["sfc"]["activo"]:
            tareas.append(("Superfinanciera", lambda: self.sfc(ahora)))
        for x in f["rss"]:
            if self.ronda % max(int(x.get("cada_rondas", 1)), 1) == 0:
                tareas.append((x["nombre"], lambda x=x: self.rss(x)))
        g = f["google"]
        cada = max(int(g.get("cada_rondas", 1)), 1)
        if g["activo"] and g["consultas"] and self.ronda % cada == 0:
            q = g["consultas"][(self.ronda // cada) % len(g["consultas"])]
            tareas.append(("Google News", lambda: self.google(q)))
        self.ronda += 1
        activas = [(n, fn) for n, fn in tareas if self.fallos.get(n, 0) < 3 or self.ronda % 10 == 0]

        def una(par: tuple[str, Callable[[], list[Item]]]) -> tuple[str, list[Item] | None]:
            try:
                return par[0], par[1]()
            except Exception:                                                          # noqa: BLE001 — una fuente caída no tumba a las demás
                return par[0], None
        items: list[Item] = []
        salud: dict[str, bool] = {}
        with ThreadPoolExecutor(max_workers=max(len(activas), 1)) as ex:
            for nombre, res in ex.map(una, activas):
                salud[nombre] = res is not None
                self.fallos[nombre] = 0 if res is not None else self.fallos.get(nombre, 0) + 1
                items += res or []
        self.ultimos = items                                                           # los reutiliza la revisión macro (src/macro.py) sin volver a descargar
        return items, salud


# ------------------------------------------------------------------ memoria (qué se vio y qué se avisó)
class Memoria:
    def __init__(self, ruta: Path | None = None):
        self.ruta = ruta or ROOT / ARCHIVO
        try:
            self.d = json.loads(self.ruta.read_text(encoding="utf-8")) if self.ruta.exists() else {}
        except (json.JSONDecodeError, OSError):
            self.d = {}
        self.d.setdefault("vistos", {})            # huella → ISO
        self.d.setdefault("avisos", {})            # "EMISOR|sentido" → ISO del último aviso
        self.d.setdefault("senales", [])           # últimas señales enviadas (para /noticias y "qué comprar")
        self.sucia = False

    def guardar(self) -> None:
        if not self.sucia:
            return
        self.ruta.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.ruta.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(self.d, fh, ensure_ascii=False)
        os.replace(tmp, self.ruta)
        self.sucia = False

    def podar(self, ahora: dt.datetime, dias: int = 4) -> None:
        lim = (ahora - dt.timedelta(days=dias)).isoformat()
        self.d["vistos"] = {k: v for k, v in self.d["vistos"].items() if v >= lim}
        self.d["senales"] = self.d["senales"][-40:]


# ------------------------------------------------------------------ pulso del precio y recomendación
def pulso(f: Any, ticker: str, cfg: dict[str, Any], ahora: dt.datetime | None = None) -> dict[str, Any] | None:
    """Precio de ahora, cambio de hoy, variación diaria típica (20 días), liquidez media y cuántas veces lo normal se movió hoy. None sin datos."""
    try:
        d = f.diario(ticker, 120)
        q = f.cotizacion(ticker)
        if d is None or d.empty or len(d) < 25 or not q or not q.get("cierre_previo"):
            return None
        hoy = (ahora or C.ahora_bogota(cfg)).date()
        hist = d[[x.date() < hoy for x in d.index]]                                    # la barra de hoy (incompleta) no entra en "lo normal"
        if len(hist) < 22:
            return None
        sigma = float(hist["Close"].pct_change().tail(20).std(ddof=1))
        from .liquidez import clasificar
        lq = clasificar(hist["Close"] * hist["Volume"], hist["Volume"], cfg)             # liquidez real en trii: mediana y días sin negociar
        liq = (lq["mediana_mm"] or 0.0) if (lq["dias_sin_negociar"] or 0) <= cfg["liquidez"]["max_dias_sin_negociar"] else 0.0
        r = q["precio"] / q["cierre_previo"] - 1
        return dict(ticker=ticker, precio=float(q["precio"]), r_hoy=float(r), sigma=sigma, z=(r / sigma if sigma > 0 else None), liquidez_mm=liq)
    except Exception:                                                                  # noqa: BLE001 — sin precio no hay recomendación, pero no se cae
        return None


def elegir_ticker(f: Any, e: Emisor, cfg: dict[str, Any], ahora: dt.datetime | None = None) -> tuple[str, dict[str, Any] | None]:
    """De las acciones del emisor (ordinaria y preferencial), la más líquida: es la que se puede comprar y vender sin problema."""
    mejor: tuple[str, dict[str, Any] | None] = (e.tickers[0], None)
    for t in e.tickers:
        p = pulso(f, t, cfg, ahora)
        if p and (mejor[1] is None or p["liquidez_mm"] > mejor[1]["liquidez_mm"]):
            mejor = (t, p)
    return mejor


def sesiones_desde(ahora: dt.datetime, n: int, cfg: dict[str, Any]) -> dt.date | None:
    """Fecha de la n-ésima sesión contando la de hoy si el mercado sigue abierto (si no, desde la próxima). Nunca después del final del concurso."""
    ses = C.sesiones(cfg)
    h = C.horario(ahora.date(), cfg)
    desde = ahora.date() if (h and ahora.time() < h[1]) else ahora.date() + dt.timedelta(days=1)
    futuras = [d for d in ses if d >= desde]
    if not futuras:
        return None
    return futuras[min(n, len(futuras)) - 1]


def cargar_estudio(ruta: Path | None = None) -> dict[str, Any]:
    """Tabla del estudio de eventos (run_estudio_noticias.py): qué pasó con la acción después de anuncios parecidos. {} si aún no se ha corrido."""
    try:
        return json.loads((ruta or ROOT / ESTUDIO).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def evidencia(estudio: dict[str, Any], grupo: str, clave: str, sesiones: int) -> dict[str, Any] | None:
    """Fila del estudio para ese caso, con el horizonte más cercano al pedido. None si no hay estudio o los casos son menos que el mínimo."""
    fila = (estudio.get(grupo) or {}).get(clave)
    if not fila or fila.get("n", 0) < estudio.get("n_minimo", 15):
        return None
    hs = [h for h in estudio.get("horizontes", []) if fila.get(f"h{h}")]
    if not hs:
        return None
    h = min(hs, key=lambda x: abs(x - sesiones))
    return dict(fila[f"h{h}"], h=h, casos=fila["n"], grupo=grupo, clave=clave, pct_fuerte=fila.get("pct_fuerte"))


def impacto_estimado(cat: str, sentido: int, estudio: dict[str, Any]) -> float | None:
    """Impacto estimado sobre el precio, con signo: lo que se movió en promedio la acción (frente al mercado) el día de un anuncio de ESE tipo, en el
    sentido de la noticia. Si ese tipo tiene pocos casos medidos, se usa el promedio de todos los tipos. None sin estudio."""
    fila = (estudio.get("categorias") or {}).get(f"{cat}|todos") or {}
    if fila.get("n", 0) < estudio.get("n_minimo", 15) or not fila.get("abs_media"):
        fila = (estudio.get("global") or {}).get("todos") or {}
    return sentido * float(fila["abs_media"]) if fila.get("abs_media") else None


def recomendar(cat: str, sentido: int, ticker: str, p: dict[str, Any] | None, ahora: dt.datetime, cfg: dict[str, Any], tengo: bool = False,
               estudio: dict[str, Any] | None = None, sentido_texto: int = 0) -> dict[str, Any]:
    """Qué hacer con la noticia, SEGÚN LA EVIDENCIA (no según la intuición de "buena noticia = comprar").

    Se busca en el estudio el caso que corresponde (el precio ya reaccionó hoy / aún no) y sólo se recomienda COMPRAR cuando, en los anuncios parecidos,
    lo que vino después superó el costo de entrar y salir incluso en el extremo bajo del intervalo de confianza. Si no, se dice claramente que no paga.
    Acciones: COMPRAR · COMPRAR_MANANA · NO_PERSEGUIR (ya subió) · NO_COMPRAR (la evidencia no paga el costo) · POCO_LIQUIDA · MANTENER (la tienes, buena noticia)
    · NO_VENDER_PANICO (la tienes, ya cayó) · VIGILAR (la tienes, mala noticia sin reacción) · EVITAR (mala noticia y no la tienes)."""
    nb = cfg["noticias_bvc"]
    r, zc = nb["recomendacion"], nb["z_confirma"]
    estudio = estudio if estudio is not None else cargar_estudio()
    n = SESIONES.get(cat, 3)
    z = p["z"] if (p and p.get("z") is not None) else None
    reaccion = 0 if z is None else (1 if z >= zc else (-1 if z <= -zc else 0))
    if reaccion != 0:                                                                  # el precio ya se movió: ¿qué pasó DESPUÉS de días así?
        ev = evidencia(estudio, "global", f"{reaccion:+d}", n)
    else:                                                                              # aún no reacciona: ¿qué pasó comprando en la siguiente apertura?
        ev = evidencia(estudio, "apertura", {1: "texto+", -1: "texto-", 0: "texto0"}[sentido_texto], n) or evidencia(estudio, "apertura", "sin_hueco", n)
    costo = r["costo_ida_vuelta"]
    base = dict(ticker=ticker, sesiones=n, salida=sesiones_desde(ahora, n, cfg), tengo=tengo, pulso=p, evidencia=ev, costo=costo, reaccion=reaccion,
                tipico=evidencia(estudio, "categorias", f"{cat}|todos", n))
    if p is not None:
        base.update(rango=p["sigma"] * math.sqrt(n), stop=max(r["stop_sigmas"] * p["sigma"], r["stop_min"]))
    if sentido < 0:
        if tengo:
            return {**base, "accion": "NO_VENDER_PANICO" if reaccion < 0 else "VIGILAR"}
        return {**base, "accion": "EVITAR"}
    if tengo:
        return {**base, "accion": "MANTENER"}
    if p is not None and p["liquidez_mm"] < r["liquidez_min_cop_mm"]:
        return {**base, "accion": "POCO_LIQUIDA"}
    paga = bool(ev and ev.get("ic_lo") is not None and ev["ic_lo"] > costo)            # exigente a propósito: el extremo BAJO del intervalo debe superar el costo
    if not paga:
        return {**base, "accion": "NO_PERSEGUIR" if reaccion > 0 else "NO_COMPRAR"}
    if not C.mercado_abierto(ahora, cfg):
        h = C.horario(ahora.date(), cfg)
        prox = ahora.date() if (h and ahora.time() < h[0]) else C.proxima_sesion(ahora, cfg)
        return {**base, "accion": "COMPRAR_MANANA", "cuando": prox, "hora": C.hora_orden_manana(prox, cfg) if prox else "-"}
    return {**base, "accion": "COMPRAR"}


# ------------------------------------------------------------------ la ronda de cada 2 minutos
@dataclass
class Senal:
    emisor: str
    ticker: str
    cat: str
    sentido: int
    puntaje: float
    items: list[Item]
    rec: dict[str, Any]
    ts: dt.datetime
    nivel: str = "alto"                     # alto (aviso completo con recomendación) | medio (relevante, aviso corto) | bajo (otra noticia de la empresa, aviso corto)
    impacto: float | None = None            # impacto estimado sobre el precio, con signo (+0,012 = +1,2 %). Si sentido == 0 es la magnitud (se muestra como ±)

    @property
    def fuentes(self) -> list[str]:
        return list(dict.fromkeys(i.fuente for i in self.items))


def ronda(lector: Lector, mem: Memoria, f: Any, ahora: dt.datetime, cfg: dict[str, Any], tenidos: set[str] | None = None) -> tuple[list[Senal], dict[str, bool]]:
    """Una pasada completa: leer → quedarse con lo nuevo de emisores de la BVC → clasificar → puntuar (con el precio) → recomendar.
    Devuelve (señales de alto impacto para avisar, salud de las fuentes). La primera vez sólo toma la línea base."""
    n = cfg["noticias_bvc"]
    tenidos = tenidos or set()
    items, salud = lector.todo(ahora)
    ahora_utc = ahora.astimezone(UTC)
    primera = not mem.d.get("base")
    nuevos = []
    for it in items:
        h = huella(it.titulo)
        if h not in mem.d["vistos"]:
            mem.d["vistos"][h] = ahora_utc.isoformat(timespec="seconds")
            mem.sucia = True
            nuevos.append(it)
    if primera:
        if any(salud.values()):                                                        # sólo hay línea base si al menos una fuente respondió
            mem.d["base"] = ahora_utc.isoformat(timespec="seconds")
            mem.sucia = True
        return [], salud
    limite = ahora_utc - dt.timedelta(minutes=n["ventana_min"])
    emisores = emisores_cfg(cfg)
    grupos: dict[tuple[str, str], dict[str, Any]] = {}
    for it in nuevos:
        if it.ts < limite or it.ts > ahora_utc + dt.timedelta(minutes=10):
            continue
        c = clasificar(texto_para_clasificar(it), cfg)
        if c is None:
            if not n.get("todas"):
                continue
            c = Clasif("otra", sentido_del_texto(it.titulo), float(n["pesos"]["otra"]))   # noticia de la empresa sin un tipo reconocido: también se avisa
        for e in emisores_de(it, emisores):
            g = grupos.setdefault((e.nombre, c.cat), dict(emisor=e, items=[], clasifs=[]))
            g["items"].append(it)
            g["clasifs"].append(c)
    senales: list[Senal] = []
    estudio = cargar_estudio()
    hace_1h = (ahora_utc - dt.timedelta(hours=1)).isoformat(timespec="seconds")
    medias = mem.d["medias"] = [x for x in mem.d.get("medias", []) if x >= hace_1h]       # avisos de noticias relevantes en la última hora (tope)
    otras = mem.d["otras"] = [x for x in mem.d.get("otras", []) if x >= hace_1h]          # ídem para "otras noticias"
    hace_12h = (ahora_utc - dt.timedelta(hours=12)).isoformat(timespec="seconds")
    titulos = mem.d["titulos"] = {k: [x for x in v if x[0] >= hace_12h] for k, v in mem.d.get("titulos", {}).items()}     # titulares ya avisados por empresa
    liq_min = n["recomendacion"]["liquidez_min_cop_mm"]
    for (nombre, cat), g in grupos.items():
        e: Emisor = g["emisor"]
        suma = sum(c.sentido for c in g["clasifs"])
        c = Clasif(cat, (suma > 0) - (suma < 0), max(x.peso for x in g["clasifs"]))
        origenes, n_f = {i.origen for i in g["items"]}, len({i.fuente for i in g["items"]})
        if not n.get("todas") and puntuar(Clasif(cat, c.sentido or 1, c.peso), origenes, n_f, None, cfg)[0] + n["bono_precio"] < n["umbral_medio"]:
            continue                                                                   # ni con el precio a favor llegaría: no se gasta una consulta de precios
        ticker, p = elegir_ticker(f, e, cfg, ahora)
        tengo = bool(set(e.tickers) & tenidos)
        if not tengo and (p is None or p["liquidez_mm"] < liq_min):
            continue                                                                   # sin liquidez buena en trii (o sin precio para comprobarla): no se avisa, salvo que la tengas
        if cat == "otra":                                                              # sin tipo reconocido: el precio de hoy NO le pone sentido ni la vuelve "fuerte"
            puntaje, sentido = min(puntuar(c, origenes, n_f, None, cfg)[0], n["umbral_medio"] - 0.01), c.sentido      # (que la acción suba hoy no prueba que sea por este titular)
        else:
            puntaje, sentido = puntuar(c, origenes, n_f, p["z"] if p else None, cfg)
        if puntaje >= n["umbral_alto"] and sentido != 0:
            nivel = "alto"
            if sentido < 0 and not tengo and puntaje < n["umbral_negativa_sin_tener"]:
                nivel = "medio"                                                        # mala noticia de una acción que no tienes: informativa, salvo que sea muy fuerte
        elif puntaje >= n["umbral_medio"] and sentido != 0:
            nivel = "medio"
        elif n.get("todas"):
            nivel = "bajo"
        else:
            continue
        items = sorted(g["items"], key=lambda i: (i.origen != "sfc", -i.ts.timestamp()))
        previos = [x[1] for x in titulos.get(nombre, [])]
        if nivel == "bajo":
            if lleno(otras, n["max_otras_hora"]) or es_repetida(items[0].titulo, previos, n["parecido_repetida"]):
                continue                                                               # misma noticia en otro medio (o tope por hora, si se configuró)
        else:
            if nivel == "medio" and lleno(medias, n["max_medias_hora"]):
                if not n.get("todas") or lleno(otras, n["max_otras_hora"]):
                    continue
                nivel = "bajo"                                                         # ya van muchas relevantes esta hora: sale como "otra noticia" (no se pierde)
            clave = f"{nombre}|{sentido:+d}"
            ult = mem.d["avisos"].get(clave)
            if ult and (ahora_utc - dt.datetime.fromisoformat(ult)).total_seconds() / 60 < n["enfriamiento_emisor_min"]:
                if not n.get("todas") or es_repetida(items[0].titulo, previos, n["parecido_repetida"]) or lleno(otras, n["max_otras_hora"]):
                    continue
                nivel = "bajo"                                                         # ya hubo un aviso fuerte de esta empresa hace poco: esta sale como "otra noticia"
        rec = recomendar(cat, sentido or 1, ticker, p, ahora, cfg, tengo, estudio, c.sentido)
        base_imp = impacto_estimado(cat, 1, estudio)
        senales.append(Senal(nombre, ticker, cat, sentido, puntaje, items, rec, ahora_utc, nivel,
                             (sentido * base_imp if sentido else base_imp) if base_imp is not None else None))
    senales.sort(key=lambda s: -s.puntaje)                                              # las más importantes salen primero
    if n["max_por_ronda"]:
        senales = senales[: n["max_por_ronda"]]
    for s in senales:
        marca = ahora_utc.isoformat(timespec="seconds")
        if s.rec.get("pulso"):                                                         # se vigila el precio: si la noticia lo mueve, sale un segundo aviso
            mem.d.setdefault("seguimiento", []).append(dict(emisor=s.emisor, ticker=s.ticker, ts=marca, ref=s.rec["pulso"]["precio"], sigma=s.rec["pulso"]["sigma"],
                                                            titulo=s.items[0].titulo[:200], sentido=s.sentido, tengo=bool(s.rec.get("tengo"))))
        if s.nivel == "medio":
            medias.append(marca)
        if s.nivel == "bajo":
            otras.append(marca)
        else:
            mem.d["avisos"][f"{s.emisor}|{s.sentido:+d}"] = marca
        titulos.setdefault(s.emisor, []).append([marca, sorted(palabras_clave(s.items[0].titulo))])
        mem.d["senales"].append(dict(ts=ahora_utc.isoformat(timespec="minutes"), emisor=s.emisor, ticker=s.ticker, cat=s.cat, sentido=s.sentido,
                                     puntaje=round(s.puntaje, 2), titulo=s.items[0].titulo[:200], fuente=s.items[0].fuente, url=s.items[0].url,
                                     accion=s.rec["accion"], salida=s.rec["salida"].isoformat() if s.rec.get("salida") else None, nivel=s.nivel, impacto=s.impacto))
        mem.sucia = True
    mem.podar(ahora_utc)
    return senales, salud


def movimientos_tras_noticia(mem: Memoria, f: Any, ahora: dt.datetime, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Noticias ya avisadas cuyo precio se movió fuerte DESPUÉS del aviso (≥ 1 vez la variación diaria normal y al menos 1 %): es la confirmación de que la
    noticia sí está afectando el precio. Cada una se avisa una sola vez; se vigilan `seguimiento_min` minutos y sólo con el mercado abierto (el precio
    no se mueve de noche). Los precios llegan con ≈ 15 minutos de retraso."""
    n = cfg["noticias_bvc"]
    pend = mem.d.get("seguimiento", [])
    if not pend:
        return []
    ahora_utc = ahora.astimezone(UTC)
    vivos, out = [], []
    abierto = C.mercado_abierto(ahora, cfg)
    cambio = False
    for x in pend:
        edad = (ahora_utc - dt.datetime.fromisoformat(x["ts"])).total_seconds() / 60
        if edad > 4 * 24 * 60:                                                         # nunca llegó a verse con el mercado abierto (fin de semana largo): se olvida
            cambio = True
            continue
        if not abierto:
            vivos.append(x)
            continue
        if not x.get("inicio"):                                                        # el reloj del seguimiento corre desde que el mercado está abierto
            x["inicio"] = ahora_utc.isoformat(timespec="seconds")
            cambio = True
        if (ahora_utc - dt.datetime.fromisoformat(x["inicio"])).total_seconds() / 60 > n["seguimiento_min"]:
            cambio = True
            continue
        p = pulso(f, x["ticker"], cfg, ahora)
        if p is None or not x.get("ref"):
            vivos.append(x)
            continue
        mov = p["precio"] / x["ref"] - 1
        if abs(mov) >= max(n["z_confirma"] * (x.get("sigma") or 0.0), 0.01):
            out.append(dict(x, movimiento=mov, minutos=edad, pulso=p))
            cambio = True
        else:
            vivos.append(x)
    if cambio:
        mem.d["seguimiento"] = vivos
        mem.sucia = True
    return out
