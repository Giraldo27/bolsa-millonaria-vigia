"""Ventanas tipo concurso: cada ventana arranca con el capital completo, sin posiciones, y descuenta el costo de
actividad. Al final de la ventana las posiciones abiertas se liquidan al cierre (con slippage y comisión)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .backtest import Backtester, Inputs
from .benchmarks import bh_window_return, cash_window_return


def rolling_windows(cal: pd.DatetimeIndex, n: int, start: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    cal = cal[cal >= pd.Timestamp(start)]
    return [(cal[i], cal[i + n - 1]) for i in range(len(cal) - n + 1)]


def contest_windows(cal: pd.DatetimeIndex, years: range, cfg: dict[str, Any]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Ventana 5-oct a 6-nov de cada año (primera sesión >= 5-oct, última sesión <= 6-nov)."""
    w = cfg["ranking"]["contest_window"]
    out = []
    for y in years:
        s = pd.Timestamp(y, w["start_month"], w["start_day"])
        e = pd.Timestamp(y, w["end_month"], w["end_day"])
        sub = cal[(cal >= s) & (cal <= e)]
        if len(sub) >= 20:
            out.append((sub[0], sub[-1]))
    return out


def run_windows(cfg: dict[str, Any], inp: Inputs, mode: str, wins: list, with_benchmarks: bool = False) -> pd.DataFrame:
    rows = []
    sys_assets = cfg["universe"]["system_assets"]
    for s, e in wins:
        r = Backtester(cfg, inp, mode, start=s, end=e, deduct_activity=True).run()
        n_sess = len(inp.activity_cal[(inp.activity_cal >= s) & (inp.activity_cal <= e)])
        row = dict(inicio=s, fin=e, sesiones=n_sess, ret=r.equity.iloc[-1] / cfg["capital"] - 1, n_trades=len(r.trades),
                   pnl_neto=r.trades.net_pnl.sum() if len(r.trades) else 0.0, dias_actividad=r.activity_days)
        if with_benchmarks:
            for a in sys_assets:
                row[f"bh_{a}"] = bh_window_return(cfg, inp, {a: 1.0}, s, e, n_sess)
            row["bh_tercios"] = bh_window_return(cfg, inp, {a: 1 / 3 for a in sys_assets}, s, e, n_sess)
            row["efectivo"] = cash_window_return(cfg, n_sess)
        rows.append(row)
    return pd.DataFrame(rows)
