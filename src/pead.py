"""Señal de reacción a resultados (PEAD / gap-and-go): matriz (T x A) con la reacción r0 de la sesión posterior al reporte."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_loader import load_earnings, load_prices
from .panel import Panel


def reaction_dates(px: pd.Series, events: pd.DataFrame) -> pd.DataFrame:
    """Sesión de reacción de cada reporte (BMO: el mismo día; AMC: el día siguiente) y r0 = cierre/cierre previo - 1.
    La reacción se mide en la moneda de cotización (sin ruido cambiario)."""
    rows = []
    idx = px.index
    for _, ev in events.iterrows():
        i = int(idx.searchsorted(ev.report_date))
        if i >= len(idx):
            continue
        react = i + 1 if (ev.timing == "AMC" and idx[i] == ev.report_date) else i
        if 1 <= react < len(idx):
            rows.append((idx[react], px.iloc[react] / px.iloc[react - 1] - 1))
    return pd.DataFrame(rows, columns=["fecha", "r0"])


def make_pead_matrix(p: Panel, cfg: dict[str, Any]) -> np.ndarray:
    """Matriz (T x A) con r0 en la fila de la sesión de reacción (NaN en el resto). Sólo acciones con fechas de resultados."""
    earn = load_earnings(cfg, [t for t in p.tickers if t in cfg["universe"]["mgc"]])
    m = np.full(p.C.shape, np.nan)
    for a, t in enumerate(p.tickers):
        e = earn[earn.ticker == t]
        if e.empty:
            continue
        r = reaction_dates(load_prices(t, cfg)["Close"], e)
        for _, row in r.iterrows():
            i = int(p.dates.searchsorted(row.fecha))
            if i < len(p.dates):
                m[i, a] = row.r0 if np.isnan(m[i, a]) else max(m[i, a], row.r0)
    return m
