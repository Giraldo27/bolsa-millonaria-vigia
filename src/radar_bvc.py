"""Radar de TODA la Bolsa de Colombia: avisa solo, sin que lo pidas, cuando cualquier acción o fondo de la BVC —locales y globales— se mueve fuerte, esté o no
en la lista del bot.

Por qué existe: el 5 de octubre de 2026 la preferencial de Davivienda subió 9 % en la sesión (y 14 % en la semana) con seis veces su volumen normal, tras la
autorización de la Superfinanciera para integrar a DAVIbank (Scotiabank Colpatria). El bot no la vigilaba y no avisó: los primeros del concurso sí la vieron.

De dónde sale cada movimiento:
  · LOCALES (ISIN colombiano): el libro de la BVC (`FuentesDatos.libro_bvc`, 15 minutos de retraso), con la bolsa abierta. Se exige algo negociado HOY
    (`min_hoy_mm`) para no confundir una operación de unas pocas acciones con un movimiento.
  · GLOBALES (ISIN extranjero, 151 en la BVC): su bolsa de ORIGEN (`FuentesDatos.origen_globales`). En trii casi no se negocian, pero su precio en pesos sigue al
    de allá: el movimiento real se ve allá. Las de EE. UU. también antes de abrir (desde las 4:00 de Nueva York) y después de cerrar (hasta las 20:00).

Avisa al cruzar cada escalón del día (4 %, 7 %, 10 %, 15 %) una sola vez por acción y escalón; lo que pasa después del cierre cuenta aparte (es un movimiento
nuevo frente al cierre)."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any
from zoneinfo import ZoneInfo

from . import noticias_bvc as N

NY = ZoneInfo("America/New_York")
APERTURA_NY = {"BVL": 600}                                                             # Lima abre a las 10:00 de Nueva York: antes, su "cambio" es el de ayer
BOLSAS_EEUU = {"NYSE", "NASDAQ", "AMEX", "CBOE", "BATS", "NYSE ARCA", "ARCA"}
NOMBRE_BOLSA = {"NYSE": "Nueva York", "NASDAQ": "Nueva York (Nasdaq)", "AMEX": "Nueva York", "CBOE": "Nueva York", "BATS": "Nueva York", "LSE": "Londres",
                "XETR": "Fráncfort", "BCS": "Santiago de Chile", "BVL": "Lima", "TSX": "Toronto", "SIX": "Zúrich", "EURONEXT": "Europa (Euronext)",
                "BIVA": "México", "BMV": "México", "BMFBOVESPA": "São Paulo"}
GENERICAS = {"banco", "grupo", "sa", "s.a.", "inc", "inc.", "corp", "corp.", "corporation", "company", "co", "the", "de", "del", "la", "el", "y", "and", "class", "a",
             "pfd", "registered", "shs", "ltd", "ltd.", "plc", "fondo", "bursatil", "ishares", "etf", "ucits", "trust", "holdings", "holding", "organizacion",
             "empresa", "almacenes", "compania", "corporacion", "sponsored", "adr", "acciones", "valores", "non-cum", "perp", "esp", "e.s.p", "s.a.s", "vanguard",
             "invesco", "spdr", "first", "global", "funds", "icav", "x-trackers", "xtrackers"}
ALIAS = {"petroleo brasileiro": "petrobras", "taiwan semiconductor": "tsmc", "alphabet": "alphabet|google", "meta platforms": "meta|facebook",
         "nu holdings": "nu holdings|nubank|nu"}


def fase(ahora: dt.datetime) -> str | None:
    """'pre' (4:00–9:30 de Nueva York), 'sesion' (9:30–16:00), 'post' (16:00–20:00) o None (noche y fines de semana). La BVC abre y cierra con Nueva York."""
    a = ahora.astimezone(NY)
    if a.weekday() >= 5:
        return None
    m = a.hour * 60 + a.minute
    return "pre" if 240 <= m < 570 else "sesion" if 570 <= m < 960 else "post" if 960 <= m < 1200 else None


def nombre_clave(descripcion: str) -> str:
    """Cómo nombran los titulares a la empresa (patrón): 'Banco Davivienda SA Pfd' → 'davivienda'; 'Petroleo Brasileiro SA' → 'petrobras'."""
    t = N.norm(descripcion).replace(",", " ")
    for k, v in ALIAS.items():
        if t.startswith(k):
            return v
    for p in t.split():
        if p not in GENERICAS and len(p) >= 2 and not p.isdigit():
            return re.escape(p)
    return ""


def candidatos(tabla: dict[str, dict[str, Any]], origen: dict[str, dict[str, Any]] | None, fase_: str, cfg: dict[str, Any],
               minuto_ny: int | None = None) -> list[dict[str, Any]]:
    """Cada acción de la BVC con el movimiento que cuenta en esta fase del día (local: el de la BVC con la bolsa abierta; global: el de su bolsa de origen)."""
    r = cfg["radar_bvc"]
    out = []
    for sym, f in tabla.items():
        o = (origen or {}).get(f.get("isin") or "") if f.get("global_") else None
        es_global = bool(o)                                                            # ISIN extranjero pero sólo se negocia en la BVC (Davivienda Group, BHI): como local
        base = dict(simbolo=sym, nombre=f.get("nombre") or sym, hoy_mm=f.get("valor_ult_mm") or 0.0, prom_mm=f.get("prom10_mm") or 0.0, global_=es_global)
        if not es_global:
            if fase_ != "sesion" or f.get("cambio") is None or base["hoy_mm"] < r["min_hoy_mm"]:
                continue
            out.append(dict(base, cambio=f["cambio"], donde="bvc", bolsa="BVC", momento="sesion"))
            continue
        eeuu = o["bolsa"] in BOLSAS_EEUU
        if fase_ == "pre" and eeuu and o.get("pre") is not None:
            cambio, momento = o["pre"], "pre"
        elif fase_ == "post" and eeuu and o.get("post") is not None:
            cambio, momento = o["post"], "post"
        elif fase_ == "sesion" and (minuto_ny is None or minuto_ny >= APERTURA_NY.get(o["bolsa"], 570)):
            cambio, momento = o["cambio"], "sesion"                                     # las de Londres o Fráncfort, con su cambio del día ya conocido
        else:
            continue
        if cambio is None:
            continue
        out.append(dict(base, cambio=cambio, donde=o["bolsa"], bolsa=NOMBRE_BOLSA.get(o["bolsa"], o["bolsa"]), momento=momento, simbolo_origen=o["simbolo"]))
    return out


def movimientos(filas: list[dict[str, Any]], vistos: dict[str, Any], hoy: str, cfg: dict[str, Any], en_lista: set[str] | None = None) -> list[dict[str, Any]]:
    """De los candidatos, los que cruzaron un escalón que aún no se avisó hoy. Marca en `vistos` lo que devuelve. Orden: el más grande primero.
    Antes de abrir y en la sesión es el MISMO movimiento (un 5 % de las 7:00 que sigue en 5 % a las 10:00 no se repite); después del cierre cuenta aparte."""
    escalones = sorted(float(x) for x in cfg["radar_bvc"]["umbrales"])
    for k in [k for k, v in vistos.items() if v.get("fecha") != hoy]:
        del vistos[k]                                                                  # cada día empieza de cero
    out = []
    for x in filas:
        pct = abs(x["cambio"]) * 100
        alcanzado = max((e for e in escalones if pct >= e), default=None)
        if alcanzado is None:
            continue
        clave = x["simbolo"] + ("|post" if x["momento"] == "post" else "")
        previo = vistos.get(clave)
        if previo and previo["escalon"] >= alcanzado and (previo.get("signo", 1) > 0) == (x["cambio"] > 0):
            continue
        vistos[clave] = dict(fecha=hoy, escalon=alcanzado, signo=1 if x["cambio"] > 0 else -1)
        out.append(dict(x, escalon=alcanzado, repetido=bool(previo), en_lista=bool(en_lista and x["simbolo"] in en_lista),
                        veces=(x["hoy_mm"] / x["prom_mm"] if x["prom_mm"] else None)))
    return sorted(out, key=lambda x: -abs(x["cambio"]))


def titulares_de(nombre: str, titulares: list[Any], ahora_utc: dt.datetime, horas: float = 72) -> list[Any]:
    """Titulares recientes (de los que ya leyó el bot) que nombran a la empresa: la explicación probable del movimiento."""
    clave = nombre_clave(nombre)
    if not clave:
        return []
    desde = ahora_utc - dt.timedelta(hours=horas)
    hall = [it for it in titulares if getattr(it, "ts", None) and it.ts >= desde and re.search(r"\b(" + clave + r")\b", N.norm(it.titulo))]
    return sorted(hall, key=lambda it: -it.ts.timestamp())[:2]
