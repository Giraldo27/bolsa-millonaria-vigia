"""¿Cuál es la mejor BASE para todo el concurso? Compara TODAS las acciones permitidas (EE. UU. y BVC) por el movimiento esperado hasta el final.

Criterio (config `seleccion_base`): movimiento esperado = volatilidad anual × √(días hasta el final / 365), con volatilidad anual =
peso_iv × IV de opciones + (1 − peso_iv) × volatilidad realizada de 60 días (sólo realizada si no hay opciones fiables, p. ej. acciones locales).
Sólo cuentan acciones líquidas. Otra acción reemplaza a la base únicamente si supera su movimiento esperado por `ventaja_minima` (20 %): con menos que eso
no compensa el costo de cambiar ni el riesgo de equivocarse con una racha corta. Es movimiento ESPERADO (riesgo y oportunidad por igual), no una promesa de ganancia."""
from __future__ import annotations

import datetime as dt
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from . import concurso as C
from .config import group_of
from .construir import universo_candidatos
from .data_sources import FuentesDatos
from .relevo import Candidato, liquidez_ok


def _una(f: FuentesDatos, t: str, base: str, fin: dt.date, hoy: dt.date, cfg: dict[str, Any]) -> dict[str, Any] | None:
    s = cfg["seleccion_base"]
    try:
        d = f.diario(t, 300)
        if d.empty or len(d) < s["vol_realizada_dias"] + 5:
            return None
        r = d["Close"].pct_change().dropna().tail(s["vol_realizada_dias"])
        real = float(r.std(ddof=1)) * math.sqrt(252)
        liq = float((d["Close"] * d["Volume"]).tail(20).mean() / 1e6)
    except Exception:                                                                    # noqa: BLE001 — una acción sin datos se omite, no se inventa
        return None
    grupo = group_of(t, cfg)
    iv, venc = None, None
    if grupo != "local":
        try:
            o = f.opciones_iv(t, fin)
            if o.get("iv") and o.get("calidad", "sin_datos") != "sin_datos":
                iv, venc = float(o["iv"]), o.get("vencimiento")
        except Exception:                                                                # noqa: BLE001
            pass
    anual = s["peso_iv"] * iv + (1 - s["peso_iv"]) * real if iv else real
    dias = max((fin - hoy).days, 1)
    return dict(ticker=t, grupo=grupo, anual=anual, iv=iv, real=real, vencimiento=venc, mee=anual * math.sqrt(dias / 365), liquidez_mm=liq,
                liquida=liquidez_ok(Candidato(t, grupo, anual, valor_negociado_mm=liq), cfg) and (liq >= s["valor_negociado_min_usd_mm"] if grupo != "local" else True),
                es_base=t == base)


def ranking_base(f: FuentesDatos, cfg: dict[str, Any], ahora: dt.datetime, base: str | None = None) -> list[dict[str, Any]]:
    """Filas ordenadas de mayor a menor movimiento esperado hasta el final del concurso (sólo acciones líquidas), con la ventaja de cada una sobre la base
    y si supera el umbral (`supera`). La base aparece siempre, aunque no cumpla la liquidez."""
    base = base or cfg["posicion_base"]["ticker"]
    fin = pd.Timestamp(cfg["concurso"]["final"]["fecha"]).date()
    hoy = ahora.date()
    tickers = list(dict.fromkeys([base] + universo_candidatos(cfg)))
    with ThreadPoolExecutor(max_workers=cfg["banco"]["max_hilos"]) as ex:
        filas = [x for x in ex.map(lambda t: _una(f, t, base, fin, hoy, cfg), tickers) if x]
    mb = next((x["mee"] for x in filas if x["ticker"] == base), None)
    ventaja = cfg["seleccion_base"]["ventaja_minima"]
    for x in filas:
        x["vs_base"] = (x["mee"] / mb) if mb else None
        x["supera"] = bool(mb and x["ticker"] != base and x["liquida"] and x["mee"] >= mb * (1 + ventaja))
    return sorted([x for x in filas if x["liquida"] or x["es_base"]], key=lambda x: -x["mee"])


def veredicto(filas: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """('mantener' | 'revisar' | 'sin_datos', las que superan a la base por el umbral)."""
    if not any(x["es_base"] for x in filas):
        return "sin_datos", []
    mejores = [x for x in filas if x["supera"]]
    return ("revisar" if mejores else "mantener"), mejores
