"""Macroeconomía que mueve tus acciones: qué está pasando AHORA con el petróleo, el dólar, Wall Street, el oro, Brasil y el mercado colombiano, y qué
acciones suelen ganar o perder con cada uno.

"Quién se beneficia y quién se afecta" NO sale de una lista hecha a mano: se mide. Para cada acción y cada factor se calcula, con los últimos 12 meses,
cuánto se movió la acción el mismo día por cada 1 % del factor (sensibilidad) y qué tan firme es esa relación; sólo se muestran las relaciones firmes
(|t| ≥ `macro.t_minimo`). Es la relación del MISMO día: sirve para entender un movimiento y saber a qué estás expuesto, no para adivinar el día siguiente.

Dos disparadores automáticos (src/servicio.tick_macro): (1) un factor se mueve hoy mucho más de lo normal; (2) sale un titular macro importante (Banco de
la República, inflación, FED, calificación del país, reforma tributaria, petróleo, elecciones en Brasil…): se muestra junto con cómo están reaccionando
los mercados en ese momento. Funciones sin estado propio: lo ya avisado vive en la memoria de noticias."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

import numpy as np
import pandas as pd

from .config import group_of
from .noticias_bvc import Item, norm

UTC = dt.timezone.utc


# ------------------------------------------------------------------ factores
def _historia(f: Any, simbolo: str, dias: int = 420) -> pd.Series:
    def bajar() -> pd.DataFrame:
        d = f.yf.download(simbolo, period=f"{dias}d", interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if d is None or d.empty:
            raise RuntimeError("Yahoo devolvió vacío")
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        return d[["Close"]].dropna()
    d = f._con_cache(f"macro_hist|{simbolo}|{dias}", 360, bajar, f"macro {simbolo}")
    return d["Close"] if d is not None and len(d) else pd.Series(dtype=float)


def tablero(f: Any, cfg: dict[str, Any], ahora: Any = None) -> list[dict[str, Any]]:
    """Cada factor macro: su último cambio, DE QUÉ DÍA es (`cuando`: 'hoy' o 'ayer' = la última sesión, porque hoy aún no ha negociado), cuántas veces lo
    normal es (z) y si es fuerte. Para lo que cotiza en Nueva York se agrega `fuera` (movimiento antes de abrir o después de cerrar).
    Un factor sin datos se omite (no se inventa). `fuerte` y `notable` sólo valen para movimientos de HOY: lo de ayer ya pasó y ya se avisó."""
    from . import sesion as SE
    m = cfg["macro_vivo"]
    out = []
    for clave, fc in m["factores"].items():
        q = SE.movimiento(f, fc["simbolo"], ahora)
        try:
            h = _historia(f, fc["simbolo"])
        except Exception:                                                              # noqa: BLE001
            h = pd.Series(dtype=float)
        if q is None or len(h) < 60:
            continue
        en_puntos = bool(fc.get("en_puntos"))                                          # tasas: el cambio se mide en puntos, no en %
        previas = h[[pd.Timestamp(x).date() < q["fecha"] for x in h.index]]            # lo "normal" se mide sin contar la sesión que se está juzgando
        cambios = (previas.diff() if en_puntos else previas.pct_change()).dropna().tail(60)
        sigma = float(cambios.std(ddof=1)) if len(cambios) >= 30 else float("nan")
        r = (q["precio"] - q["previo"]) if en_puntos else (q["precio"] / q["previo"] - 1)
        if not sigma or sigma != sigma or abs(r) > 0.5 and not en_puntos:              # un salto de más del 50 % es un tick erróneo de Yahoo
            continue
        z = r / sigma
        hoy = bool(q["es_hoy"])
        fuera = SE.fuera_de_horario(f, fc["simbolo"], ahora) if fc.get("nueva_york") else None
        out.append(dict(clave=clave, nombre=fc["nombre"], cambio=float(r), z=float(z), sigma=sigma, en_puntos=en_puntos, precio=q["precio"],
                        cuando="hoy" if hoy else "ayer", fecha=q["fecha"], fuera=fuera,
                        fuerte=hoy and abs(z) >= m["z_alerta"], notable=hoy and abs(z) >= m["z_notable"]))
    return out


# ------------------------------------------------------------------ sensibilidades medidas
def _beta(y: pd.Series, x: pd.Series, control: pd.Series | None = None) -> dict[str, float] | None:
    """Cuánto se mueve y por cada unidad de x (mismo día) y qué tan firme es (t). Con `control`, descontando lo que explica esa otra serie."""
    cols = {"y": y, "x": x} | ({"c": control} if control is not None else {})
    j = pd.concat(cols, axis=1, sort=True).dropna()
    if len(j) < 120 or j["x"].std() == 0:
        return None
    X = np.column_stack([np.ones(len(j)), j["x"].to_numpy()] + ([j["c"].to_numpy()] if control is not None else []))
    coef, *_ = np.linalg.lstsq(X, j["y"].to_numpy(), rcond=None)
    res = j["y"].to_numpy() - X @ coef
    gl = len(j) - X.shape[1]
    try:
        se = float(np.sqrt((res @ res) / gl * np.linalg.inv(X.T @ X)[1, 1]))
    except np.linalg.LinAlgError:
        return None
    b = float(coef[1])
    return {"beta": b, "t": b / se if se > 0 else 0.0, "n": len(j), "corr": float(j["y"].corr(j["x"]))}


def sensibilidades(f: Any, cfg: dict[str, Any], tickers: list[str]) -> dict[str, dict[str, dict[str, float]]]:
    """{factor: {ticker: {beta, t, n, corr}}} con los últimos `macro.dias` días. Sólo quedan las relaciones firmes (|t| ≥ t_minimo). Caché de 12 horas."""
    m = cfg["macro_vivo"]
    clave = "macro_sens|" + ",".join(sorted(tickers))

    def calcular() -> dict[str, Any]:
        acciones = {}
        for t in tickers:
            try:
                d = f.diario(t, m["dias"] + 30)
                if d is not None and len(d) >= 130:
                    acciones[t] = d["Close"].pct_change().tail(m["dias"])
            except Exception:                                                          # noqa: BLE001 — una acción sin datos no tumba el cálculo
                continue
        out: dict[str, Any] = {}
        for fk, fc in m["factores"].items():
            try:
                h = _historia(f, fc["simbolo"])
            except Exception:                                                          # noqa: BLE001
                continue
            if len(h) < 130:
                continue
            x = (h.diff() if fc.get("en_puntos") else h.pct_change()).tail(m["dias"] + 10)
            ctrl = None
            if fc.get("controlar") in m["factores"]:
                try:
                    ctrl = _historia(f, m["factores"][fc["controlar"]]["simbolo"]).pct_change().tail(m["dias"] + 10)
                except Exception:                                                      # noqa: BLE001 — sin el control se mide sin descontar
                    ctrl = None
            fila = {}
            for t, y in acciones.items():
                b = _beta(y, x, ctrl)
                if b and abs(b["t"]) >= m["t_minimo"]:
                    fila[t] = b
            out[fk] = fila
        return out
    v = f._con_cache(clave, 12 * 60, calcular, "sensibilidades macro")
    return v or {}


def sector_de(cfg: dict[str, Any]) -> dict[str, str]:
    """{acción: nombre de su sector} según noticias_bvc.indirectos.sectores."""
    sec = (cfg["noticias_bvc"].get("indirectos") or {}).get("sectores") or {}
    return {t: s["nombre"] for s in sec.values() for t in s["tickers"]}


def relacionadas(cfg: dict[str, Any]) -> dict[str, dict[str, str]]:
    """{acción: {otra: 'misma' | 'grupo' | 'sector'}}: 'misma' = otra serie del mismo emisor (ordinaria/preferencial); 'grupo' = vínculo de propiedad
    (noticias_bvc.grupos); 'sector' = competidoras del mismo sector (noticias_bvc.indirectos.sectores). Si hay dos vínculos, queda el más fuerte."""
    out: dict[str, dict[str, str]] = {}
    for s in ((cfg["noticias_bvc"].get("indirectos") or {}).get("sectores") or {}).values():
        for a in s["tickers"]:
            for b in s["tickers"]:
                if a != b:
                    out.setdefault(a, {})[b] = "sector"
    for g in cfg["noticias_bvc"].get("grupos", []):
        for a in g:
            for b in g:
                if a != b:
                    out.setdefault(a, {})[b] = "grupo"
    for e in cfg["noticias_bvc"]["emisores"]:
        for a in e["tickers"]:
            for b in e["tickers"]:
                if a != b:
                    out.setdefault(a, {})[b] = "misma"
    return out


def contagio(f: Any, cfg: dict[str, Any], tickers: list[str]) -> dict[str, dict[str, float]]:
    """{acción con la noticia: {otra acción de la BVC: cuánto suele moverse por cada 1 % PROPIO de la primera}}.

    Sólo entre empresas con vínculo real (`relacionadas`): la otra serie del mismo emisor cuenta 1 a 1 (es la misma empresa) y las del mismo grupo cuentan
    lo que se midió (descontando el mercado colombiano, y sólo si la relación es firme, |t| ≥ t_minimo). Sin vínculo no se muestra nada: una regresión entre
    25 acciones siempre encuentra parejas "relacionadas" por casualidad. Caché de 12 horas."""
    m = cfg["macro_vivo"]
    rel = relacionadas(cfg)
    tope_sector = float((cfg["noticias_bvc"].get("indirectos") or {}).get("tope_sector", 0.5))

    def calcular() -> dict[str, Any]:
        r = {}
        for t in tickers:
            try:
                d = f.diario(t, m["dias"] + 30)
                if d is not None and len(d) >= 130:
                    r[t] = d["Close"].pct_change().tail(m["dias"])
            except Exception:                                                          # noqa: BLE001
                continue
        try:
            mercado = _historia(f, m["factores"]["colombia"]["simbolo"]).pct_change().tail(m["dias"] + 10)
        except Exception:                                                              # noqa: BLE001
            mercado = None
        out: dict[str, Any] = {}
        for origen, x in r.items():
            fila = {}
            for destino, tipo in rel.get(origen, {}).items():
                if tipo == "misma":
                    fila[destino] = 1.0
                elif destino in r:
                    b = _beta(r[destino], x, mercado)
                    if b and abs(b["t"]) >= m["t_minimo"] and b["beta"] > 0:
                        fila[destino] = round(min(b["beta"], tope_sector if tipo == "sector" else 1.0), 3)
            out[origen] = fila
        for origen, vecinos in rel.items():                                             # acción sin historia: al menos su otra serie
            out.setdefault(origen, {t: 1.0 for t, tipo in vecinos.items() if tipo == "misma"})
        return out
    return f._con_cache("macro_contagio3|" + ",".join(sorted(tickers)), 12 * 60, calcular, "contagio entre acciones") or {}


def afectadas_por_noticia(ticker: str, impacto_pct: float | None, sentido: int, vecinos: dict[str, float], cfg: dict[str, Any], tenidos: set[str],
                          cat: str | None = None) -> list[dict[str, Any]]:
    """Acciones de la BVC a las que les pega una noticia de `ticker`, con su porcentaje estimado y signo: primero la propia (efecto directo) y luego las
    indirectas, cada una con su `via` y su `motivo`:
      · 'misma'  = la otra serie de la misma empresa (se mueve igual);
      · 'grupo'  = empresa del mismo grupo (matriz, filial o accionista), según cuánto suele moverse con ella;
      · 'sector' = competidora del mismo sector, sólo si la noticia es de un tipo que habla del negocio del sector (`cat` en indirectos.cats_sector).
    Efecto = su sensibilidad medida × el impacto estimado de la noticia. Sin impacto medido sólo se indica la propia (efecto None)."""
    out = [dict(ticker=ticker, efecto=impacto_pct, sentido=sentido, tengo=ticker in tenidos, propia=True, via="directo", motivo="")]
    if impacto_pct is None:
        return out
    limite = cfg["macro_vivo"]["efecto_minimo"]
    rel, sector = relacionadas(cfg).get(ticker, {}), sector_de(cfg)
    cats = set((cfg["noticias_bvc"].get("indirectos") or {}).get("cats_sector") or [])
    otras = []
    for t, b in vecinos.items():
        via = rel.get(t, "grupo")
        if via == "sector" and cat not in cats:                                        # una compra o un nombramiento de un banco no dice nada de los demás bancos
            continue
        motivo = {"misma": "la misma empresa", "grupo": "mismo grupo empresarial"}.get(via) or f"mismo sector: {sector.get(t, 'el suyo')}"
        otras.append(dict(ticker=t, efecto=b * impacto_pct, sentido=0 if sentido == 0 else (1 if b * impacto_pct > 0 else -1), tengo=t in tenidos, propia=False,
                          via=via, motivo=motivo))
    orden = {"misma": 0, "grupo": 1, "sector": 2}
    return out + sorted([x for x in otras if abs(x["efecto"]) >= limite], key=lambda x: (orden[x["via"]], -abs(x["efecto"])))[:7]


def impacto(factor: dict[str, Any], sens: dict[str, dict[str, float]], cfg: dict[str, Any], tenidos: set[str], minimo: float | None = None) -> dict[str, Any]:
    """Para el movimiento de HOY de un factor: qué acciones suelen subir con él (beneficiadas) y cuáles bajar (afectadas), con el efecto estimado
    (sensibilidad × movimiento). Las de EE. UU. que tienes suman el efecto directo del dólar: en trii se ven en pesos."""
    filas = []
    for t, b in sens.items():
        filas.append(dict(ticker=t, beta=b["beta"], efecto=b["beta"] * factor["cambio"], tengo=t in tenidos, directo=False))
    if factor["clave"] == "dolar":
        for t in tenidos:
            try:
                global_ = group_of(t, cfg) == "mgc"
            except KeyError:
                global_ = False
            if global_ and not any(x["ticker"] == t for x in filas):
                filas.append(dict(ticker=t, beta=1.0, efecto=factor["cambio"], tengo=True, directo=True))    # aritmética: precio en pesos = precio en dólares × dólar
    limite = cfg["macro_vivo"]["efecto_minimo"] if minimo is None else minimo
    benef = sorted([x for x in filas if x["efecto"] >= limite], key=lambda x: -x["efecto"])
    afect = sorted([x for x in filas if x["efecto"] <= -limite], key=lambda x: x["efecto"])
    return dict(beneficiadas=benef, afectadas=afect, mias=[x for x in benef + afect if x["tengo"]])


# ------------------------------------------------------------------ titulares macro
# (clave, nombre, factores que suelen reaccionar, patrones sobre el titular sin tildes)
TEMAS: list[tuple[str, str, list[str], list[str]]] = [
    ("banrep", "Banco de la República (tasas de interés)", ["dolar", "colombia"],
     [r"banco de la republica", r"\bbanrep\b", r"junta del emisor", r"tasa de (intervencion|interes) (del|de la|en colombia)", r"tasas de interes en colombia"]),
    ("inflacion_co", "inflación en Colombia", ["dolar", "colombia"], [r"\bipc\b.*\b(colombia|dane)\b", r"inflacion (en|de) colombia", r"\bdane\b.*inflacion", r"inflacion.*\bdane\b"]),
    ("fed", "Reserva Federal de EE. UU.", ["nasdaq", "dolar", "tasas_eeuu"], [r"\bfed\b", r"reserva federal", r"\bpowell\b", r"\bfomc\b"]),
    ("inflacion_eeuu", "inflación o empleo en EE. UU.", ["nasdaq", "dolar", "tasas_eeuu"],
     [r"inflacion (en|de) (ee\.? ?uu|estados unidos)", r"\bcpi\b", r"nominas no agricolas", r"empleo (en|de) (ee\.? ?uu|estados unidos)"]),
    ("riesgo_pais", "calificación y finanzas del país", ["dolar", "colombia"],
     [r"calificacion (de|crediticia de|soberana de) colombia", r"(fitch|moody'?s|s&p|standard).*colombia", r"regla fiscal", r"deficit fiscal", r"reforma tributaria", r"ley de financiamiento"]),
    ("petroleo", "petróleo", ["petroleo"], [r"\bopep\b", r"precio del (petroleo|crudo|brent)", r"\bbrent\b", r"petroleo (cae|sube|se desploma|se dispara|baja)"]),
    ("brasil", "elecciones y economía de Brasil", ["brasil"], [r"(elecciones|segunda vuelta|balotaje).*brasil", r"brasil.*(elecciones|segunda vuelta|balotaje)", r"\blula\b", r"banco central de brasil"]),
    ("dolar", "dólar en Colombia", ["dolar"], [r"dolar (se dispara|se desploma|rompe|supera|cae por debajo|toca)", r"\btrm\b.*(record|maximo|minimo)"]),
    ("crecimiento", "crecimiento de Colombia", ["colombia"], [r"\bpib\b.*colombia", r"economia colombiana (crecio|cayo|se contrajo)", r"\bise\b.*dane"]),
]
NOMBRE_TEMA = {t[0]: t[1] for t in TEMAS}
FACTORES_TEMA = {t[0]: t[2] for t in TEMAS}
RUIDO_MACRO = re.compile(r"(precio del dolar hoy|dolar hoy|a como esta|casas de cambio|horoscopo|en vivo|minuto a minuto|opinion|editorial|que es y como|pico y placa|loteria)")


def tema_de(titulo: str) -> str | None:
    t = norm(titulo)
    if RUIDO_MACRO.search(t):
        return None
    for clave, _, _, patrones in TEMAS:
        if any(re.search(p, t) for p in patrones):
            return clave
    return None


def titulares_macro(items: list[Item], mem: dict[str, Any], ahora: dt.datetime, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Titulares macro NUEVOS y recientes, agrupados por tema. Devuelve los temas que merecen aviso: los que traen ≥ `min_fuentes` medios distintos (o una
    sola fuente si `min_fuentes` es 1) y no se avisaron hace poco. Actualiza `mem` (vistos y último aviso por tema)."""
    m = cfg["macro_vivo"]["titulares"]
    ahora_utc = ahora.astimezone(UTC)
    vistos = mem.setdefault("macro_vistos", {})
    limite = ahora_utc - dt.timedelta(minutes=m["ventana_min"])
    primera = "macro_base" not in mem
    por_tema: dict[str, list[Item]] = {}
    for it in items:
        if it.origen == "sfc":
            continue
        tema = tema_de(it.titulo)
        if tema is None:
            continue
        h = norm(it.titulo)[:120]
        if h in vistos:
            continue
        vistos[h] = ahora_utc.isoformat(timespec="minutes")
        if it.ts >= limite:
            por_tema.setdefault(tema, []).append(it)
    viejo = (ahora_utc - dt.timedelta(days=3)).isoformat()
    for k in [k for k, v in vistos.items() if v < viejo]:
        del vistos[k]
    if primera:
        mem["macro_base"] = ahora_utc.isoformat(timespec="minutes")                      # la primera vez no se avisa de lo que ya existía
        return []
    out = []
    avisos = mem.setdefault("macro_avisos", {})
    for tema, its in por_tema.items():
        fuentes = list(dict.fromkeys(i.fuente for i in its))
        ult = avisos.get(tema)
        if len(fuentes) < m["min_fuentes"] or (ult and (ahora_utc - dt.datetime.fromisoformat(ult)).total_seconds() / 60 < m["enfriamiento_min"]):
            continue
        avisos[tema] = ahora_utc.isoformat(timespec="minutes")
        out.append(dict(tema=tema, nombre=NOMBRE_TEMA[tema], items=sorted(its, key=lambda i: -i.ts.timestamp())[:3], fuentes=fuentes, factores=FACTORES_TEMA[tema]))
    return out


def movimientos_nuevos(tab: list[dict[str, Any]], mem: dict[str, Any], ahora: dt.datetime) -> list[dict[str, Any]]:
    """Factores con un movimiento fuerte que aún no se avisó hoy en ese sentido (o que ya se avisó pero ahora es mucho mayor: +1,5 en z)."""
    hoy = ahora.date().isoformat()
    hechos = mem.setdefault("macro_mov", {})
    out = []
    for x in tab:
        if not x["fuerte"]:
            continue
        k = f"{hoy}|{x['clave']}|{'+' if x['z'] > 0 else '-'}"
        previo = hechos.get(k)
        if previo is None or abs(x["z"]) >= abs(previo) + 1.5:
            hechos[k] = x["z"]
            out.append(x)
    for k in [k for k in hechos if not k.startswith(hoy)]:
        del hechos[k]
    return out
