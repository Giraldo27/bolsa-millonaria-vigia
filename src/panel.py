"""Panel de precios en COP (Open/High/Low/Close) del universo líquido, alineado al calendario de sesiones del concurso."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import group_of
from .data_loader import currency_of, load_macro, load_prices
from .universe import fx_to_cop


@dataclass
class Panel:
    dates: pd.DatetimeIndex
    tickers: list[str]
    O: np.ndarray            # (T x A) en COP
    H: np.ndarray
    L: np.ndarray
    C: np.ndarray
    spread: np.ndarray       # (A,) spread de entrada
    slip: np.ndarray         # (A,) slippage de salida a mercado
    group: list[str]
    ndx: np.ndarray          # (T,) cierre del Nasdaq-100 (régimen de mercado)
    brent: np.ndarray        # (T,) cierre del Brent

    def idx(self, t: str) -> int:
        return self.tickers.index(t)


def liquid_universe(cfg: dict[str, Any], ranking_csv, min_value_mm: float, max_zero_pct: float) -> list[str]:
    """Activos permitidos con liquidez suficiente (valor negociado diario y % de días sin volumen); sin duplicados."""
    m = pd.read_csv(ranking_csv, index_col=0)
    ok = m[(m.valor_cop_mm >= min_value_mm) & (m.pct_vol0 <= max_zero_pct)].index.tolist()
    ok = [t for t in ok if t != "BCOLOMBIA"]          # mismo ADR (CIB) que PFBCOLOM
    assert not set(ok) & set(cfg["universe"]["blacklist"])
    order = list(cfg["universe"]["mgc"]) + list(cfg["universe"]["etf"]) + list(cfg["universe"]["local"])
    return [t for t in order if t in ok]


def build_panel(cfg: dict[str, Any], tickers: list[str], cal: pd.DatetimeIndex) -> Panel:
    usdcop, gbpusd = load_macro("usdcop", cfg)["Close"], load_macro("gbpusd", cfg)["Close"]
    g = cfg["costs"]["generic"]
    cols = {k: {} for k in "OHLC"}
    spread, slip, groups = [], [], []
    for t in tickers:
        df = load_prices(t, cfg)
        fx = fx_to_cop(t, df, currency_of(t, cfg), usdcop, gbpusd)
        for k, name in zip("OHLC", ("Open", "High", "Low", "Close")):
            s = (df[name] * fx)
            cols[k][t] = s.reindex(s.index.union(cal)).ffill(limit=5).reindex(cal)
        grp = group_of(t, cfg)
        groups.append(grp)
        spread.append(cfg["costs"]["entry_spread"].get(t, g["spread"][grp]))
        slip.append(cfg["costs"]["exit_slippage"].get(t, g["slippage"][grp]))
    ndx = load_macro("ndx", cfg)["Close"].reindex(cal.union(load_macro("ndx", cfg).index)).ffill().reindex(cal)
    brent = load_macro("brent", cfg)["Close"].reindex(cal.union(load_macro("brent", cfg).index)).ffill().reindex(cal)
    arr = {k: pd.DataFrame(v)[tickers].values for k, v in cols.items()}
    return Panel(cal, tickers, arr["O"], arr["H"], arr["L"], arr["C"], np.array(spread), np.array(slip), groups,
                 ndx.values, brent.values)
