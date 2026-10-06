"""Entiende lo que escribes en el chat sin obligarte a usar comandos ni a acordarte del formato.

"compré 300 argos a 21500" · "compré 20 millones de ecopetrol" · "vendí tesla" · "vendí la mitad de meta" · "voy 3,5 y el corte está en 8" · "me equivoqué".

Funciones puras: devuelven una `Intencion` (qué quisiste decir y con qué datos). Lo que falte o sea dudoso (precio en pesos o en dólares, total en vez de
precio por acción) lo resuelve `aclarar_compra` comparando con el precio de hoy, y SIEMPRE se le muestra al usuario lo que se entendió (y se puede deshacer)."""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .noticias_bvc import norm

# Cómo llama la gente (y trii) a cada acción. Las de la BVC salen además de noticias_bvc.emisores (config.yaml).
ALIAS_FIJOS = {
    "tesla": "TSLA", "nvidia": "NVDA", "nu": "NUCO", "nubank": "NUCO", "nu holdings": "NUCO", "meta": "META", "facebook": "META", "apple": "AAPL", "amazon": "AMZN",
    "google": "GOOGL", "alphabet": "GOOGL", "microsoft": "MSFT", "uber": "UBER", "disney": "DIS", "ford": "F", "coca cola": "KO", "cocacola": "KO", "visa": "V",
    "mastercard": "MA", "walmart": "WMT", "pfizer": "PFE", "petrobras": "PBR", "jpmorgan": "JPM", "jp morgan": "JPM", "bank of america": "BAC", "citigroup": "C",
    "citi": "C", "johnson": "JNJ", "general electric": "GE",
    "argos": "GRUPOARGOS", "grupo argos": "GRUPOARGOS", "cementos argos": "CEMARGOS", "cemargos": "CEMARGOS", "ecopetrol": "ECOPETROL", "eco": "ECOPETROL",
    "bancolombia": "PFBCOLOM", "cibest": "PFBCOLOM", "grupo cibest": "PFBCOLOM", "pfcibest": "PFBCOLOM", "sura": "GRUPOSURA", "grupo sura": "GRUPOSURA",
    "exito": "EXITO", "grupo exito": "EXITO", "interconexion electrica": "ISA", "grupo energia bogota": "GEB", "energia de bogota": "GEB", "nutresa": "NUTRESA",
    "corficolombiana": "CORFICOLCF", "corfi": "CORFICOLCF", "aval": "PFAVAL", "grupo aval": "PFAVAL", "banco de bogota": "BOGOTA", "canacol": "CNEC",
    "bac holding": "BHI", "bolsa de valores": "BVC",
}
PREFERENCIAL = {"GRUPOARGOS": "PFGRUPOARG", "CEMARGOS": "PFCEMARGOS", "GRUPOSURA": "PFSURA", "CORFICOLCF": "PFCORFICOL", "AVAL": "PFAVAL", "BCOLOMBIA": "PFBCOLOM"}
ORDINARIA = {"PFBCOLOM": "BCOLOMBIA", "PFAVAL": "AVAL"}

_NUM = r"-?\$?\s?\d[\d.,]*"
_MILLONES = r"(millones|millon|mill|palos|mm|m)\b"
VERBO_COMPRA = re.compile(r"\b(compre|compro|compra|comprar|comprado|comprando|meti|inverti|adquiri)\b")
TENGO = re.compile(r"\b(tengo|poseo|me quedan|quedo con|mi posicion (es|son)|mis acciones son)\b")
VERBO_VENTA = re.compile(r"\b(vendi|vendo|venta|vender|vendido|liquide|sali de|me sali)\b")
PREGUNTA = re.compile(r"(\b(que|cual|cuales|cuando|como) (accion(es)? )?(compro|comprar|vendo|vender|registro)\b|\b(conviene|deberia|recomiend\w*|vale la pena|pienso|planeo|"
                      r"podria|(voy a|quiero|puedo) (comprar|vender)|si (compro|vendo))\b|\?)")
DESHACER = re.compile(r"\b(deshacer|deshaz|deshaga|me equivoque|equivocacion|borra (lo|la) ultim|anula|anular|corrige lo ultimo|cancela (lo )?ultim)")
CON_TICKER = ("noticias", "semaforo", "reemplazo", "revisar")                          # consultas que pueden venir con una acción ("noticias de ecopetrol")
CONSULTAS = [
    ("variaciones", re.compile(r"\b(variacion(es)?|resumen del dia|resumen de hoy|rentabilidad(es)? de (las|cada|todas las) accion(es)?|como (cerraron|cerro la bolsa|va la bolsa|van las acciones|amanecio la bolsa)|cuanto (subieron|bajaron)|que acciones (suben|bajan|subieron|bajaron))\b")),
    ("reemplazo", re.compile(r"\b(reemplaz\w*|sustitu\w*|en (vez|lugar) de|por (cual|que) (la |lo |las )?cambio|cambi\w+ \w+ por (cual|que|otra))\b")),
    ("ganar", re.compile(r"\b(ganar|gane|ganador\w*|todos los filtros|analisis completo|analiza\w* (todo|mi cartera))\b")),
    ("revisar", re.compile(r"\b(segur[oa]|analiza\w*|analisis|revisa\w*|evalua\w*|que (opinas|piensas|me dices) de|que tal|como (ves|esta)|vale la pena)\b")),   # sólo con acción
    ("comprar", re.compile(r"\b(que (accion )?(compro|comprar|comprarias)|en que (invierto|meto)|recomiend\w*|recomendacion(es)?|que hago con la plata|(cuales?|que) (son |es )?(las? )?mejor(es)? (accion(es)?|opcion(es)?))\b")),
    ("semaforo", re.compile(r"\b(semaforo|como (estan|esta|van) mis acciones)\b")),
    ("cartera", re.compile(r"\b(cartera|portafolio|mis acciones|que tengo)\b")),
    ("nuevas", re.compile(r"\b(actualiza\w* (las )?noticias|noticias nuevas|nuevas noticias|busca\w* noticias|hay noticias|ultimas noticias|refresca\w*)\b")),
    ("liquidez", re.compile(r"\b(liquidez|se negocia|es liquida|tiene liquidez)\b")),
    ("macro", re.compile(r"\b(macro|macroeconomia|petroleo|dolar|brent|tasas? de interes|inflacion|wall street|fed)\b")),
    ("noticias", re.compile(r"\b(noticias?|que paso con|novedades)\b")),
    ("estado", re.compile(r"\b(como voy|estado|resumen)\b")),
    ("ayuda", re.compile(r"\b(ayuda|help|menu|que puedes hacer|como funciona|comandos|hola|buenas|buenos dias)\b")),
]


@dataclass
class Intencion:
    tipo: str                                   # compra | venta | rank | deshacer | consulta | nada
    ticker: str | None = None
    cantidad: float | None = None
    precio: float | None = None
    monto: float | None = None                  # pesos
    fraccion: float | None = None               # venta: 1.0 = todo, 0.5 = la mitad
    mia: float | None = None
    objetivo: float | None = None
    consulta: str | None = None
    falta: list[str] = field(default_factory=list)
    sugerencias: list[str] = field(default_factory=list)
    partes: list["Intencion"] = field(default_factory=list)      # "tengo 8 meta… tengo 300 argos…": una por acción


def a_numero(crudo: str) -> float:
    """'21.500' → 21500 · '1.200.000' → 1200000 · '21,5' → 21.5 · '384.32' → 384.32 · '1,200' → 1200 · '$ 48.500' → 48500."""
    t = crudo.replace("$", "").replace(" ", "")
    neg = t.startswith("-")
    t = t.lstrip("-").strip(".,")
    if re.fullmatch(r"\d{1,3}([.,]\d{3})+", t):                                         # miles con punto o coma
        v = float(re.sub(r"[.,]", "", t))
    elif "," in t and "." in t:
        v = float(t.replace(".", "").replace(",", ".")) if t.rfind(",") > t.rfind(".") else float(t.replace(",", ""))
    else:
        v = float(t.replace(",", "."))
    return -v if neg else v


def numeros(texto: str) -> list[dict[str, Any]]:
    """Números del texto en orden, con lo que los rodea: ¿son millones?, ¿van después de 'a' / 'en' / 'precio' (precio)?, ¿llevan %?"""
    out = []
    for m in re.finditer(_NUM, texto):
        crudo = m.group(0)
        if not re.search(r"\d", crudo):
            continue
        try:
            v = a_numero(crudo)
        except ValueError:
            continue
        despues, antes = texto[m.end(): m.end() + 12], texto[max(0, m.start() - 12): m.start()]
        mill = re.match(r"\s?" + _MILLONES, despues)
        mil = re.match(r"\s?(mil|k)\b", despues)
        out.append(dict(valor=v * (1e6 if mill else (1e3 if mil else 1)), millones=bool(mill), es_precio=bool(re.search(r"\b(a|en|precio|por|cada una a)\s*\$?\s*$", antes)),
                        pct=despues.lstrip().startswith("%") or "por ciento" in despues, pos=m.start(), fin=m.end()))
    return out


def catalogo(cfg: dict[str, Any]) -> dict[str, str]:
    """alias normalizado → ticker (todos los tickers permitidos, sus nombres de trii con el sufijo 'co', los alias fijos y los de noticias_bvc.emisores)."""
    u = cfg["universe"]
    permitidos = [t for g in ("mgc", "etf", "local") for t in u[g] if t not in u["blacklist"]]
    c = {t.lower(): t for t in permitidos}
    for t in u["mgc"]:
        c.setdefault(t.lower() + "co", t)                                               # trii le pone "CO" a las de EE. UU.: UBERCO, TSLACO…
    for a, t in ALIAS_FIJOS.items():
        if t in permitidos:
            c.setdefault(a, t)
    for e in cfg.get("noticias_bvc", {}).get("emisores", []):
        tk = [t for t in e["tickers"] if t in permitidos]
        for a in e.get("alias", []):
            if tk:
                c.setdefault(norm(a), tk[0])
    return c


def buscar_ticker(texto: str, cfg: dict[str, Any], tenidos: set[str] | None = None) -> tuple[str | None, list[str]]:
    """(ticker, sugerencias). Busca primero el nombre más largo que aparezca tal cual; si no hay, el más parecido (errores de tecleo: 'ecopetro', 'bancolonbia').
    'preferencial' / 'pf' elige la preferencial del emisor; si ya tienes una de las dos series de esa empresa, se prefiere la que tienes."""
    t = norm(texto)
    cat = catalogo(cfg)
    hallado = None
    for a in sorted(cat, key=len, reverse=True):
        if len(a) >= 2 and re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", t):
            hallado = cat[a]
            break
    sugerencias: list[str] = []
    if hallado is None:
        palabras = [w for w in re.findall(r"[a-z]{4,}", t) if not (VERBO_COMPRA.search(w) or VERBO_VENTA.search(w))
                    and w not in ("acciones", "accion", "millones", "pesos", "dolares", "todo", "toda", "mitad", "precio", "cada")]
        for w in palabras:
            cerca = difflib.get_close_matches(w, list(cat), n=3, cutoff=0.8)
            if cerca:
                if difflib.SequenceMatcher(None, w, cerca[0]).ratio() >= 0.88:
                    hallado = cat[cerca[0]]
                    break
                sugerencias += [cat[x] for x in cerca]
    if hallado is None:
        return None, list(dict.fromkeys(sugerencias))[:3]
    if re.search(r"\b(preferencial|preferenciales|pf)\b", t) and hallado in PREFERENCIAL:
        hallado = PREFERENCIAL[hallado]
    elif re.search(r"\b(ordinaria|ordinarias)\b", t) and hallado in ORDINARIA:
        hallado = ORDINARIA[hallado]
    elif tenidos and hallado not in tenidos:                                            # "vendí argos" teniendo PFGRUPOARG
        hermana = PREFERENCIAL.get(hallado) or ORDINARIA.get(hallado) or next((k for k, v in PREFERENCIAL.items() if v == hallado), None)
        if hermana in tenidos:
            hallado = hermana
    return hallado, []


def entender(texto: str, cfg: dict[str, Any], tenidos: set[str] | None = None, forzar: str | None = None) -> Intencion:
    """Qué quiso decir el usuario. `forzar` = 'compra' | 'venta' cuando viene de un botón ("➕ Compré") y el texto ya no trae el verbo."""
    t = norm(texto)
    if DESHACER.search(t):
        return Intencion("deshacer")
    nums = numeros(t)
    mc, mv = VERBO_COMPRA.search(t), VERBO_VENTA.search(t)
    # "¿qué compro?", "voy a comprar…", "¿me conviene vender?" son preguntas o planes: NO se registra nada (sólo se anota lo que ya hiciste)
    pregunta = PREGUNTA.search(t) is not None
    if TENGO.search(t) and not mc and not mv and forzar is None and nums and not pregunta:
        partes = _tengo(t, cfg)                                                         # "tengo 1200 nuco": es lo que TIENES, no una compra nueva
        if partes:
            return Intencion("tengo", partes=partes)
    if forzar == "compra" or (mc and not pregunta and (not mv or mc.start() < mv.start())):
        return _compra(t, nums, cfg, tenidos)
    if forzar == "venta" or (mv and not pregunta):
        return _venta(t, nums, cfg, tenidos)
    if pregunta and mc and not mv:
        tk = buscar_ticker(t, cfg, tenidos)[0]
        if tk:                                                                          # "¿puedo comprar tesla?", "voy a comprar nuco": primero el filtro de liquidez de ESA acción
            return Intencion("consulta", consulta="liquidez", ticker=tk)
    if pregunta and (mc or mv):
        return Intencion("consulta", consulta="comprar")
    r = _rank(t, nums)
    if r is not None:
        return r
    for nombre, patron in CONSULTAS:
        if patron.search(t):
            tk = buscar_ticker(t, cfg, tenidos)[0] if nombre in CON_TICKER else None
            if nombre == "revisar" and not tk:                                          # "revisa mi cartera", "¿estás seguro?": sin acción no es esta consulta
                continue
            return Intencion("consulta", consulta=nombre, ticker=tk)
    return Intencion("nada")


def _compra(t: str, nums: list[dict[str, Any]], cfg: dict[str, Any], tenidos: set[str] | None) -> Intencion:
    tk, sug = buscar_ticker(t, cfg, None)
    i = Intencion("compra", ticker=tk, sugerencias=sug)
    plata = [x for x in nums if x["millones"] or (x["valor"] >= 1e6 and not x["es_precio"])]
    resto = [x for x in nums if x not in plata]
    if plata:
        i.monto = plata[0]["valor"]
    precio = next((x for x in resto if x["es_precio"]), None)
    cant = next((x for x in resto if x is not precio), None)
    if precio is None and len(resto) >= 2:                                             # "argos 300 21500": el primero es la cantidad y el segundo el precio
        precio = [x for x in resto if x is not cant][0]
    if cant is not None and precio is not None and "accion" in t[precio["fin"]: precio["fin"] + 12] and cant["pos"] > precio["pos"]:
        cant, precio = precio, cant                                                    # "a 21500 300 acciones"
    i.cantidad = cant["valor"] if cant else None
    i.precio = precio["valor"] if precio else None
    if tk is None:
        i.falta.append("ticker")
    if i.cantidad is None and i.monto is None:
        i.falta.append("cantidad")
    return i


def _tengo(t: str, cfg: dict[str, Any]) -> list[Intencion]:
    """Cada "tengo N acciones de X [que valen V]" del mensaje → (acción, cantidad, valor total en pesos si lo dijo). Los trozos sin acción o sin cantidad se ignoran."""
    trozos = [x for x in re.split(r"\b(?:tengo|poseo|me quedan|quedo con)\b|[.;\n]| y (?=\d)", t) if x and x.strip()]
    out: list[Intencion] = []
    for trozo in trozos:
        tk, _ = buscar_ticker(trozo, cfg)
        ns = numeros(trozo)
        if tk is None or not ns:
            continue
        valor = next((x for x in ns if x["millones"] or x["valor"] >= 1e5 or re.search(r"\b(val\w+|valor\w*|por|equivale\w*)\b[^0-9]*$", trozo[max(0, x["pos"] - 30): x["pos"]])), None)
        cant = next((x for x in ns if x is not valor and not x["pct"]), None)
        if cant is None or cant["valor"] <= 0:
            continue
        if any(p.ticker == tk for p in out):
            continue
        out.append(Intencion("tengo", ticker=tk, cantidad=cant["valor"], monto=valor["valor"] if valor else None))
    return out


def _venta(t: str, nums: list[dict[str, Any]], cfg: dict[str, Any], tenidos: set[str] | None) -> Intencion:
    tk, sug = buscar_ticker(t, cfg, tenidos)
    if tk is None and tenidos and len(tenidos) == 1:
        tk = next(iter(tenidos))
    i = Intencion("venta", ticker=tk, sugerencias=sug)
    if re.search(r"\b(la mitad|mitad)\b", t):
        i.fraccion = 0.5
    elif nums and not nums[0]["millones"] and not nums[0]["es_precio"]:
        if nums[0]["pct"]:
            i.fraccion = max(0.0, min(1.0, nums[0]["valor"] / 100))
        else:
            i.cantidad = nums[0]["valor"]
    else:
        i.fraccion = 1.0
    if tk is None:
        i.falta.append("ticker")
    return i


def _rank(t: str, nums: list[dict[str, Any]]) -> Intencion | None:
    """'voy 3,5' · 'llevo -2' · 'voy perdiendo 2' · 'el corte está en 8' · 'voy 3,5 y el corte 8' · 'rank 3,5 8'."""
    if not nums or not re.search(r"\b(voy|llevo|rentabilidad|ranking|rank|corte|objetivo|umbral|primero|lider|puntero)\b", t):
        return None
    i = Intencion("rank")
    for x in nums:
        antes = t[max(0, x["pos"] - 34): x["pos"]]
        v = x["valor"]
        if re.search(r"\b(corte|objetivo|umbral|primero|lider|puntero|pasar)\b[^0-9]*$", antes):
            i.objetivo = v if i.objetivo is None else i.objetivo
        elif re.search(r"\b(voy|llevo|rentabilidad|ranking|rank|estoy)\b[^0-9]*$", antes) and i.mia is None:
            i.mia = -abs(v) if re.search(r"\b(perdiendo|abajo|negativo|menos|en rojo)\b[^0-9]*$", antes) else v
        elif i.mia is not None and i.objetivo is None:
            i.objetivo = v                                                             # "rank 3,5 8": el segundo es el corte
    return i if (i.mia is not None or i.objetivo is not None) else None


# ------------------------------------------------------------------ aclarar una compra con el precio de hoy
def aclarar_compra(i: Intencion, moneda: str, precio_hoy: float | None, trm: float | None, tolerancia: float = 0.35) -> dict[str, Any]:
    """Convierte lo que escribió el usuario en (cantidad, precio por acción en la moneda de la acción, monto en pesos) y explica cómo lo interpretó.

    El problema real: en trii las acciones de EE. UU. se ven en PESOS, pero el sistema las guarda en dólares; y es fácil escribir el TOTAL donde iba el precio.
    Se prueba cada lectura posible del número contra el precio de hoy y gana la que queda cerca (± `tolerancia`). Si ninguna cuadra, se devuelve dudoso=True
    con la lectura literal, para que el bot pregunte en vez de guardar un disparate."""
    usd = moneda == "USD"
    a_cop = (trm if usd else 1.0) or None
    notas: list[str] = []
    cant, precio, monto = i.cantidad, i.precio, i.monto
    dudoso = aproximado = False

    def cerca(x: float, ref: float) -> bool:
        return ref > 0 and abs(x / ref - 1) <= tolerancia
    if precio is None:
        if precio_hoy is None:
            return dict(ok=False, motivo="sin_precio")
        precio, aproximado = precio_hoy, True
        notas.append("usé el precio de hoy porque no me dijiste el tuyo")
    elif precio_hoy:
        if cerca(precio, precio_hoy):
            pass
        elif usd and trm and cerca(precio, precio_hoy * trm):
            notas.append(f"tomé {precio:,.0f} como precio en PESOS por acción (≈ US$ {precio / trm:,.2f})".replace(",", "."))
            monto = monto or (cant * precio if cant else None)
            precio = precio / trm
        elif cant and cerca(precio, precio_hoy * cant):
            notas.append("tomé ese número como el TOTAL que pagaste, no como el precio de una acción")
            monto = monto or precio * (trm if usd and trm else 1.0)
            precio = precio / cant
        elif cant and usd and trm and cerca(precio, precio_hoy * cant * trm):
            notas.append("tomé ese número como el TOTAL en pesos")
            monto = monto or precio
            precio = precio / cant / trm
        else:
            dudoso = True
    if cant is None:
        if monto is None or not a_cop:
            return dict(ok=False, motivo="sin_cantidad" if monto is None else "sin_trm")
        cant = monto / (precio * a_cop)
        cant = round(cant) if abs(cant - round(cant)) / max(cant, 1) < 0.02 or cant > 50 else round(cant, 4)
        notas.append(f"con esa plata salen unas {cant:g} acciones al precio que tengo")
    if monto is None:
        monto = cant * precio * a_cop if a_cop else None
    return dict(ok=True, cantidad=float(cant), precio=float(precio), monto_cop=monto, notas=notas, dudoso=dudoso, aproximado=aproximado)
