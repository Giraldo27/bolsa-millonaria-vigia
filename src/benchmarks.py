"""Benchmarks: comprar y mantener, portafolio 1/3, efectivo y estrategia aleatoria (Monte Carlo)."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .backtest import Inputs
from .costs import commission, gross_from_budget


def _fx(inp: Inputs, a: str, idx: pd.DatetimeIndex) -> np.ndarray:
    fx = inp.assets[a].fx
    if fx is None:
        return np.ones(len(idx))
    return fx.reindex(fx.index.union(idx)).ffill().reindex(idx).bfill().values


def bh_window_return(cfg: dict[str, Any], inp: Inputs, weights: dict[str, float], start: pd.Timestamp,
                     end: pd.Timestamp, n_sessions: int, capital: float | None = None) -> float:
    """Compra a la apertura del primer día (con spread y comisión) y vende al cierre del último (slippage y comisión).
    Resta además la micro-compra de actividad de las sesiones sin compra real (n_sessions-1)."""
    cap = capital or cfg["capital"]
    c, total = cfg["costs"], 0.0
    for a, w in weights.items():
        px = inp.assets[a].ohlc
        sub = px.loc[start:end]
        if sub.empty:
            return np.nan
        idx = sub.index
        fx = _fx(inp, a, idx)
        v = gross_from_budget(w * cap, c)
        buy_px = sub["Open"].iloc[0] * (1 + c["entry_spread"][a])
        sell_px = sub["Close"].iloc[-1] * (1 - c["exit_slippage"][a])
        proceeds = v * (sell_px * fx[-1]) / (buy_px * fx[0])
        total += proceeds - commission(proceeds, c)
    total -= (n_sessions - 1) * c["activity_micro_buy_cop"]
    return total / cap - 1


def cash_window_return(cfg: dict[str, Any], n_sessions: int) -> float:
    return -n_sessions * cfg["costs"]["activity_micro_buy_cop"] / cfg["capital"]


def bh_equity(cfg: dict[str, Any], inp: Inputs, weights: dict[str, float], dates: pd.DatetimeIndex) -> pd.Series:
    """Curva continua comprar-y-mantener marcada a mercado (costos de entrada incluidos, sin rebalanceo)."""
    cap, c = cfg["capital"], cfg["costs"]
    val = pd.Series(0.0, index=dates)
    for a, w in weights.items():
        px = inp.assets[a].ohlc["Close"].reindex(dates).ffill()
        fx = pd.Series(_fx(inp, a, dates), index=dates)
        o0 = inp.assets[a].ohlc["Open"].reindex(dates).bfill().iloc[0]
        v = gross_from_budget(w * cap, c)
        shares = v / (o0 * (1 + c["entry_spread"][a]) * fx.iloc[0])
        val += shares * px * fx
    return val


def random_strategy_mc(cfg: dict[str, Any], inp: Inputs, trades: pd.DataFrame, start: pd.Timestamp,
                       n_runs: int = 1000, seed: int = 2026) -> dict[str, Any]:
    """Estrategia aleatoria con el mismo nº de trades, mismos activos, misma duración (sesiones) y mismo monto que la
    real, pero con fechas de entrada al azar, salida al cierre tras k sesiones y los mismos costos (sin TP/SL).
    Devuelve la distribución de (a) retorno neto medio por trade y (b) P&L neto total en COP."""
    rng = np.random.default_rng(seed)
    c = cfg["costs"]
    means = np.zeros(n_runs)
    totals = np.zeros(n_runs)
    for a, g in trades.groupby("asset"):
        px = inp.assets[a].ohlc.loc[start:]
        close = px["Close"].values
        fx = _fx(inp, a, px.index)
        k = g.sessions.values.astype(int)
        invested = g.invested_cop.values
        n = len(g)
        ent = (rng.random((n_runs, n)) * (len(close) - k - 1)).astype(int)       # índice de entrada válido
        ex = ent + k
        v = np.vectorize(lambda b: gross_from_budget(b, c))(invested)
        ratio = (close[ex] * (1 - c["exit_slippage"][a]) * fx[ex]) / (close[ent] * (1 + c["entry_spread"][a]) * fx[ent])
        proceeds = v * ratio
        fee_out = np.where(proceeds <= c["fixed_threshold_cop"], c["fixed_fee_cop"], proceeds * c["pct_fee"])
        net = proceeds - fee_out - invested
        means += (net / invested).sum(axis=1)
        totals += net.sum(axis=1)
    means /= len(trades)
    return dict(mean_ret=means, total_pnl=totals)
