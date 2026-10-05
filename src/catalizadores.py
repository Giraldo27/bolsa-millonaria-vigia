"""Calendario de catalizadores de los próximos días hábiles: reportes del universo, FED, BanRep, IPC, empleo y cortes del concurso."""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from . import concurso as C
from .data_sources import NEWS_SYMBOL, FuentesDatos


def ventana(ahora: dt.datetime, dias_habiles: int) -> tuple[dt.date, dt.date]:
    """Desde hoy hasta N días hábiles adelante (lunes a viernes; los festivos colombianos no cuentan porque EE. UU. sí opera)."""
    ini = ahora.date()
    fin = pd.bdate_range(ini, periods=dias_habiles + 1)[-1].date()
    return ini, fin


def agrupar_reportes(filas: list[dict[str, Any]], tolerancia_dias: int = 4) -> list[dict[str, Any]]:
    """Fusiona fechas de la misma empresa que difieren pocos días (Finnhub y yfinance no siempre coinciden): queda una sola fila
    marcada como incierta con todas las fechas y sus fuentes."""
    out: list[dict[str, Any]] = []
    for f in sorted(filas, key=lambda x: x["fecha"]):
        d = pd.Timestamp(f["fecha"]).date()
        if out and (d - out[-1]["fechas"][-1]).days <= tolerancia_dias:
            out[-1]["fechas"].append(d)
            out[-1]["fuentes"].append(f"{f['fuente']} {d:%d/%m}")
            out[-1]["hora"] = out[-1]["hora"] or f.get("hora", "")
        else:
            out.append({"fechas": [d], "fuentes": [f"{f['fuente']} {d:%d/%m}"], "hora": f.get("hora", "")})
    return out


def catalizadores(f: FuentesDatos, cfg: dict[str, Any], ahora: dt.datetime, tickers: list[str], dias_habiles: int | None = None) -> list[dict[str, Any]]:
    dias_habiles = dias_habiles or cfg["radar"]["dias_catalizadores"]
    ini, fin = ventana(ahora, dias_habiles)
    items: list[dict[str, Any]] = []

    def rep(t: str) -> list[dict[str, Any]]:
        try:
            return f.reportes(t, dias_adelante=(fin - ini).days + 3)
        except Exception:                                                          # noqa: BLE001 — una acción sin datos no tumba el resto
            return []
    with ThreadPoolExecutor(max_workers=cfg["banco"]["max_hilos"]) as ex:
        resultados = list(ex.map(rep, tickers))
    for t, filas in zip(tickers, resultados):
        for ev in agrupar_reportes(filas):
            en_ventana = [d for d in ev["fechas"] if ini <= d <= fin]
            if not en_ventana:
                continue
            incierta = len(ev["fechas"]) > 1
            hora = {"amc": "después del cierre", "bmo": "antes de abrir", "dmh": "durante el mercado"}.get(ev["hora"], "")
            fecha = min(en_ventana)
            det = f"{t} reporta resultados" + (f" ({hora})" if hora else "")
            if incierta:
                det += f" — las fuentes no coinciden en la fecha ({' / '.join(ev['fuentes'])})"
            items.append({"fecha": fecha, "hora": "", "detalle": det, "tipo": "reporte", "ticker": t, "incierta": incierta})
    for e in cfg["eventos_macro"]:
        d = pd.Timestamp(e["fecha"]).date()
        if ini <= d <= fin:
            items.append({"fecha": d, "hora": e.get("hora", ""), "detalle": e["evento"], "tipo": "macro", "verificada": e.get("verificada", False)})
    for c in C.cortes(cfg):
        if ini <= c["fecha"] <= fin:
            etiqueta = (f"Corte del concurso: pasa el top {c['top_pct']}% ({c['liga']})" if c.get("top_pct") else "Final del concurso: gana el #1")
            items.append({"fecha": c["fecha"], "hora": "", "detalle": etiqueta, "tipo": "corte"})
    return sorted(items, key=lambda x: (x["fecha"], x["tipo"]))


def tickers_con_reporte(cfg: dict[str, Any]) -> list[str]:
    """Universo del que se esperan reportes: acciones MGC y ECOPETROL (vía ADR EC). Las demás locales no tienen fechas gratuitas."""
    return list(cfg["universe"]["mgc"]) + [t for t in cfg["universe"]["local"] if t in NEWS_SYMBOL]
