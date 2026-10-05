"""Métricas de desempeño: curva de patrimonio, operaciones, ventanas y robustez estadística."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EXPECTANCY_BASE_COP = 40_000_000


def equity_stats(equity: pd.Series) -> dict[str, float]:
    eq = equity.dropna()
    total = eq.iloc[-1] / eq.iloc[0] - 1
    years = max((eq.index[-1] - eq.index[0]).days / 365.25, 1e-9)
    cagr = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    r = eq.pct_change().dropna()
    sharpe = r.mean() / r.std() * np.sqrt(252) if r.std() > 0 else np.nan
    down = r[r < 0]
    dd_dev = np.sqrt((down ** 2).sum() / len(r)) if len(r) else np.nan
    sortino = r.mean() / dd_dev * np.sqrt(252) if dd_dev and dd_dev > 0 else np.nan
    mdd = (eq / eq.cummax() - 1).min()
    return dict(rent_neta=total, cagr=cagr, max_dd=mdd, sharpe=sharpe, sortino=sortino, anios=years)


def trade_stats(trades: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, float]:
    if trades.empty:
        return dict(n_trades=0)
    t = trades
    win = t.net_pnl > 0
    wins, losses = t.ret_net[win], t.ret_net[~win]
    costs = (t.spread_cost + t.slippage_cost + t.fees).sum()
    gross = t.gross_pnl.sum()
    mins = cfg["assets"]
    min_wr = np.mean([mins[a]["min_win_rate"] for a in t.asset])        # acierto mínimo teórico ponderado por mezcla
    return dict(
        n_trades=len(t), win_rate=win.mean(), win_rate_min=min_wr,
        payoff=(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else np.nan,
        profit_factor=(t.net_pnl[win].sum() / abs(t.net_pnl[~win].sum())) if (~win).any() and t.net_pnl[~win].sum() != 0 else np.nan,
        expectancy_pct=t.ret_net.mean(), expectancy_cop_40m=t.ret_net.mean() * EXPECTANCY_BASE_COP,
        dias_prom=t.sessions.mean(), costos_cop=costs, pnl_bruto_cop=gross, pnl_neto_cop=t.net_pnl.sum(),
        costos_pct_bruto=(costs / gross) if gross > 0 else np.nan)


def summarize(res, cfg: dict[str, Any]) -> dict[str, float]:
    out = equity_stats(res.equity)
    out.update(trade_stats(res.trades, cfg))
    out["exposicion_media"] = float(res.exposure.mean()) if res.exposure is not None else np.nan
    return out


def window_stats(rets: pd.Series) -> dict[str, float]:
    return dict(n=len(rets), media=rets.mean(), mediana=rets.median(), p10=rets.quantile(0.10), p90=rets.quantile(0.90),
                pct_gt0=(rets > 0).mean(), pct_gt3=(rets > 0.03).mean(), pct_gt10=(rets > 0.10).mean(),
                peor=rets.min(), mejor=rets.max())


def bootstrap_expectancy(ret: pd.Series, n_boot: int = 10_000, seed: int = 11) -> dict[str, float]:
    """IC 95 % bootstrap de la media del retorno neto por trade y p-valor (t de una muestra vs 0 y bootstrap)."""
    x = ret.dropna().values
    if len(x) < 3:
        return dict(media=np.nan, ic_lo=np.nan, ic_hi=np.nan, p_t=np.nan, p_boot=np.nan)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    centered = means - x.mean()                                  # distribución bajo H0: media = 0
    p_boot = float((np.abs(centered) >= abs(x.mean())).mean())
    return dict(media=x.mean(), ic_lo=np.percentile(means, 2.5), ic_hi=np.percentile(means, 97.5),
                p_t=float(stats.ttest_1samp(x, 0.0).pvalue), p_boot=p_boot)
