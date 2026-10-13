"""Radar de TODA la Bolsa de Colombia: avisa cuando cualquier acción o fondo que se negocie de verdad se mueve fuerte en el día, esté o no en la lista del bot.

Por qué existe: el 5 de octubre de 2026 la preferencial de Davivienda subió 9 % en la sesión (y 14 % en la semana) con seis veces su volumen normal, tras la
autorización de la Superfinanciera para integrar a DAVIbank (Scotiabank Colpatria). El bot no la vigilaba y no avisó: los primeros del concurso sí la vieron.

Fuente: el libro de la bolsa (`FuentesDatos.libro_bvc`, tablero público con 15 minutos de retraso). Sólo cuenta lo que se negocia: un promedio diario mínimo
y algo negociado HOY, para no confundir un precio viejo o una sola operación pequeña con un movimiento real. Avisa al cruzar cada escalón del día (4 %, 7 %,
10 %…) una sola vez por acción y escalón."""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from . import noticias_bvc as N

GENERICAS = {"banco", "grupo", "sa", "s.a.", "inc", "inc.", "corp", "corporation", "company", "co", "the", "de", "del", "la", "el", "y", "and", "class", "a", "pfd",
             "registered", "shs", "ltd", "ltd.", "plc", "fondo", "bursatil", "ishares", "etf", "ucits", "trust", "holdings", "holding", "organizacion", "empresa",
             "almacenes", "compania", "corporacion", "sponsored", "adr", "acciones", "valores", "non-cum", "perp", "esp", "e.s.p", "s.a.s"}


def nombre_clave(descripcion: str) -> str:
    """La palabra que nombra a la empresa en los titulares: 'Banco Davivienda SA Pfd' → 'davivienda'."""
    for p in N.norm(descripcion).replace(",", " ").split():
        if p not in GENERICAS and len(p) >= 3 and not p.isdigit():
            return p
    return ""


def movimientos(tabla: dict[str, dict[str, Any]], vistos: dict[str, Any], hoy: str, cfg: dict[str, Any], en_lista: set[str] | None = None) -> list[dict[str, Any]]:
    """Movimientos fuertes de hoy que aún no se avisaron en ese escalón. Marca en `vistos` lo que devuelve. Orden: el más grande primero."""
    r = cfg["radar_bvc"]
    escalones = sorted(float(x) for x in r["umbrales"])
    for k in [k for k, v in vistos.items() if v.get("fecha") != hoy]:
        del vistos[k]                                                                  # cada día empieza de cero
    out = []
    for sym, f in tabla.items():
        cambio, prom, hoy_mm = f.get("cambio"), f.get("prom10_mm") or 0.0, f.get("valor_ult_mm") or 0.0
        if cambio is None or prom < r["min_prom_mm"] or hoy_mm < r["min_hoy_mm"]:
            continue
        pct = abs(cambio) * 100
        alcanzado = max((e for e in escalones if pct >= e), default=None)
        if alcanzado is None:
            continue
        previo = vistos.get(sym)
        if previo and previo["escalon"] >= alcanzado:
            continue
        vistos[sym] = dict(fecha=hoy, escalon=alcanzado)
        out.append(dict(simbolo=sym, nombre=f.get("nombre") or sym, cambio=cambio, precio=f.get("precio"), hoy_mm=hoy_mm, prom_mm=prom,
                        veces=(hoy_mm / prom if prom else None), escalon=alcanzado, repetido=bool(previo), en_lista=bool(en_lista and sym in en_lista)))
    return sorted(out, key=lambda x: -abs(x["cambio"]))


def titulares_de(nombre: str, titulares: list[Any], ahora_utc: dt.datetime, horas: float = 72) -> list[Any]:
    """Titulares recientes (de los que ya leyó el bot) que nombran a la empresa: la explicación probable del movimiento."""
    clave = nombre_clave(nombre)
    if not clave:
        return []
    desde = ahora_utc - dt.timedelta(hours=horas)
    hall = [it for it in titulares if getattr(it, "ts", None) and it.ts >= desde and re.search(r"\b" + re.escape(clave) + r"\b", N.norm(it.titulo))]
    return sorted(hall, key=lambda it: -it.ts.timestamp())[:2]
