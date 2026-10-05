"""Indicadores técnicos y filtros de entrada (usan sólo datos hasta la fecha evaluada)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

TRENDS = ("none", "close_gt_sma20", "close_gt_sma50", "sma20_gt_sma50", "close_lt_sma50")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return 100 - 100 / (1 + up / dn)          # dn=0 y up>0 => inf => RSI 100; 0/0 => NaN (serie plana)


def indicator_frame(close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"close": close, "sma20": close.rolling(20).mean(), "sma50": close.rolling(50).mean(),
                         "rsi": rsi(close)})


def passes_filter(row: dict[str, float], flt: dict[str, Any] | None) -> bool:
    """row: close, sma20, sma50, rsi del día evaluado. flt: {trend, rsi_max, rsi_min}. Sin filtro => True.
    Si el indicador no existe todavía (inicio de la serie) el filtro NO deja pasar (conservador)."""
    if not flt:
        return True
    t = flt.get("trend", "none")
    if t != "none":
        c, s20, s50 = row["close"], row["sma20"], row["sma50"]
        if any(np.isnan(v) for v in (c, s20, s50)):
            return False
        ok = {"close_gt_sma20": c > s20, "close_gt_sma50": c > s50, "sma20_gt_sma50": s20 > s50,
              "close_lt_sma50": c < s50}[t]
        if not ok:
            return False
    r = row["rsi"]
    if flt.get("rsi_max") is not None and not (r <= flt["rsi_max"]):
        return False
    if flt.get("rsi_min") is not None and not (r >= flt["rsi_min"]):
        return False
    return True
