"""Fase 4b: estrategia con las mejoras validadas vs. la original (antes vs. después).
Uso: python run_fase4b.py [--skip-windows]"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.benchmarks import random_strategy_mc
from src.config import load_config
from src.data_prep import build_inputs
from src.metrics import bootstrap_expectancy, equity_stats, trade_stats, window_stats
from src.sweep import override
from src.windows import contest_windows, rolling_windows, run_windows

pd.set_option("display.width", 230, "display.max_columns", 40, "display.float_format", "{:,.4f}".format)


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4"
    imp = override(cfg, **cfg["mejoras_validadas"])
    inp = build_inputs(cfg)
    cal = inp.activity_cal
    start, split = pd.Timestamp(cfg["backtest"]["start"]), pd.Timestamp(cfg["backtest"]["oos_split"])
    periods = (("completo", start, None), ("dentro (<2022)", start, cal[cal < split][-1]), ("fuera (>=2022)", split, None))
    rows, boots, mcs = [], [], []
    for version, c in (("original", cfg), ("mejorada", imp)):
        for m in cfg["modes"]:
            for nombre, s0, e0 in periods:
                r = Backtester(c, inp, m, start=s0, end=e0, deduct_activity=False).run()
                fx = Backtester(c, inp, m, start=s0, end=e0, deduct_activity=False, compound=False).run()
                row = dict(version=version, modo=m, periodo=nombre)
                row.update(trade_stats(r.trades, c)); row.update(equity_stats(r.equity))
                row["pnl_fijo_cop"] = fx.trades.net_pnl.sum() if len(fx.trades) else 0.0
                rows.append(row)
                if nombre == "completo":
                    r.trades.to_csv(out / f"trades_{version}_{m}.csv", index=False)
                    bo = bootstrap_expectancy(r.trades.ret_net); bo.update(version=version, modo=m); boots.append(bo)
                    mc = random_strategy_mc(c, inp, r.trades, start)
                    mcs.append(dict(version=version, modo=m, strat=r.trades.ret_net.mean(),
                                    percentil=float((mc["mean_ret"] < r.trades.ret_net.mean()).mean())))
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(out / "antes_vs_despues.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "bootstrap_antes_despues.csv", index=False)
    pd.DataFrame(mcs).to_csv(out / "montecarlo_antes_despues.csv", index=False)
    cols = ["version", "modo", "periodo", "n_trades", "win_rate", "expectancy_pct", "profit_factor", "rent_neta", "max_dd",
            "pnl_fijo_cop", "costos_cop"]
    print(cmp_[cols].to_string(index=False))
    print("\nBOOTSTRAP\n", pd.DataFrame(boots).to_string(index=False))
    print("\nMONTE CARLO (percentil de la estrategia vs. aleatoria)\n", pd.DataFrame(mcs).to_string(index=False))
    if "--skip-windows" in sys.argv:
        return

    wins = rolling_windows(cal, cfg["contest"]["sessions"], cfg["backtest"]["start"])
    yrs = [w for w in contest_windows(cal, range(start.year, pd.Timestamp.today().year + 1), cfg) if w[1] < cal[-1]]
    ws, ys = [], []
    for m in cfg["modes"]:
        w = run_windows(imp, inp, m, wins)
        w.to_parquet(out / f"ventanas_mejorada_{m}.parquet")
        s = window_stats(w["ret"]); s["modo"] = m; ws.append(s)
        y = run_windows(imp, inp, m, yrs); y["modo"] = m; ys.append(y)
        print("ventanas", m, flush=True)
    pd.DataFrame(ws).to_csv(out / "ventanas_estadisticas_mejorada.csv", index=False)
    pd.concat(ys).to_csv(out / "ventanas_concurso_por_anio_mejorada.csv", index=False)
    orig = pd.read_csv(cfg["paths"]["outputs_dir"] / "fase3" / "ventanas_estadisticas.csv", index_col="modo")
    print("\nVENTANAS 23 sesiones — original\n", orig.loc[list(cfg["modes"])].to_string())
    print("\nVENTANAS 23 sesiones — mejorada\n", pd.DataFrame(ws).set_index("modo").to_string())


if __name__ == "__main__":
    main()
