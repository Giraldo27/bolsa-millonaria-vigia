"""Fase 3: resultados.
Uso: python run_fase3.py                 (completo, ~8 min por las ~3.000 ventanas x 4 modos)
     python run_fase3.py --skip-windows  (sólo continuo, benchmarks, dentro/fuera de muestra y Monte Carlo)"""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.benchmarks import bh_equity, random_strategy_mc
from src.config import load_config
from src.data_prep import build_inputs
from src.metrics import bootstrap_expectancy, equity_stats, summarize, trade_stats, window_stats
from src.windows import contest_windows, rolling_windows, run_windows

pd.set_option("display.width", 230, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)


def breakdown(t: pd.DataFrame) -> pd.DataFrame:
    """P&L, acierto, retorno bruto y costo por trade, por setup y activo."""
    keys = [t.setup, t.asset]
    cost = t.spread_cost + t.slippage_cost + t.fees
    out = pd.DataFrame({
        "n": t.groupby(keys).size(),
        "pnl_neto_mm": t.net_pnl.groupby(keys).sum() / 1e6,
        "acierto": (t.net_pnl > 0).groupby(keys).mean(),
        "bruto_pct": (t.gross_pnl / t.invested_cop).groupby(keys).mean(),
        "costo_pct": (cost / t.invested_cop).groupby(keys).mean(),
        "neto_pct": t.ret_net.groupby(keys).mean(),
        "costos_mm": cost.groupby(keys).sum() / 1e6,
        "dias": t.sessions.groupby(keys).mean()})
    out.index.names = ["setup", "asset"]
    return out.reset_index()


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase3"
    out.mkdir(exist_ok=True)
    inp = build_inputs(cfg)
    cal = inp.activity_cal
    start = pd.Timestamp(cfg["backtest"]["start"])
    split = pd.Timestamp(cfg["backtest"]["oos_split"])
    modes, sysa = list(cfg["modes"]), cfg["universe"]["system_assets"]
    summary, fixed, isoos, boots, mcs, brk = [], [], [], [], [], []
    res = {}

    # ---------- continuo (con reinversión y con tamaño fijo) ----------
    for m in modes:
        r = Backtester(cfg, inp, m, deduct_activity=False).run()
        res[m] = r
        t = r.trades
        t.to_csv(out / f"trades_{m}.csv", index=False)
        r.equity.to_frame().join(r.exposure).to_parquet(out / f"equity_{m}.parquet")
        s = summarize(r, cfg); s["modo"] = m; summary.append(s)
        rf = Backtester(cfg, inp, m, deduct_activity=False, compound=False).run()
        sf = summarize(rf, cfg); sf["modo"] = m; fixed.append(sf)
        rf.equity.to_frame().to_parquet(out / f"equity_fijo_{m}.parquet")
        b = breakdown(t); b["modo"] = m; brk.append(b)
        bo = bootstrap_expectancy(t.ret_net); bo["modo"] = m; boots.append(bo)
        mc = random_strategy_mc(cfg, inp, t, start)
        mcs.append(dict(modo=m, strat_mean_ret=t.ret_net.mean(), mc_mean_p5=np.percentile(mc["mean_ret"], 5),
                        mc_mean_p50=np.median(mc["mean_ret"]), mc_mean_p95=np.percentile(mc["mean_ret"], 95),
                        percentil_media=float((mc["mean_ret"] < t.ret_net.mean()).mean()),
                        strat_total_mm=t.net_pnl.sum() / 1e6, mc_total_p50_mm=np.median(mc["total_pnl"]) / 1e6,
                        percentil_total=float((mc["total_pnl"] < t.net_pnl.sum()).mean())))
        np.save(out / f"mc_{m}_mean.npy", mc["mean_ret"])
        np.save(out / f"mc_{m}_total.npy", mc["total_pnl"])
        # dentro / fuera de muestra: cada periodo arranca con capital nuevo
        for nombre, s0, e0 in (("dentro (<2022)", start, cal[cal < split][-1]), ("fuera (>=2022)", split, None)):
            rp = Backtester(cfg, inp, m, start=s0, end=e0, deduct_activity=False).run()
            rx = Backtester(cfg, inp, m, start=s0, end=e0, deduct_activity=False, compound=False).run()
            ts = trade_stats(rp.trades, cfg)
            ts.update(equity_stats(rp.equity))
            ts.update(modo=m, periodo=nombre, rent_tamano_fijo=rx.equity.iloc[-1] / cfg["capital"] - 1)
            isoos.append(ts)
        yr = r.equity.groupby(r.equity.index.year).last()
        yr.pct_change().fillna(yr / cfg["capital"] - 1).rename(m).to_csv(out / f"anual_{m}.csv")

    pd.DataFrame(summary).set_index("modo").to_csv(out / "metricas_continuo.csv")
    pd.DataFrame(fixed).set_index("modo").to_csv(out / "metricas_tamano_fijo.csv")
    pd.DataFrame(isoos).to_csv(out / "dentro_fuera_muestra.csv", index=False)
    pd.DataFrame(boots).to_csv(out / "bootstrap.csv", index=False)
    pd.DataFrame(mcs).to_csv(out / "montecarlo.csv", index=False)
    pd.concat(brk).to_csv(out / "desglose_setup_activo.csv", index=False)

    # ---------- benchmarks continuos ----------
    dates = res["blindaje"].equity.index
    bench = pd.DataFrame({f"bh_{a}": bh_equity(cfg, inp, {a: 1.0}, dates) for a in sysa})
    bench["bh_tercios"] = bh_equity(cfg, inp, {a: 1 / 3 for a in sysa}, dates)
    bench["efectivo"] = float(cfg["capital"])
    bench.to_parquet(out / "benchmarks_continuo.parquet")
    bst = pd.DataFrame({c: equity_stats(bench[c]) for c in bench.columns}).T
    bst.to_csv(out / "benchmarks_metricas.csv")

    def show(titulo: str, df: pd.DataFrame) -> None:
        print(f"\n=== {titulo} ===\n{df.to_string()}")

    sm, sfx = pd.DataFrame(summary).set_index("modo"), pd.DataFrame(fixed).set_index("modo")
    show("CONTINUO CON REINVERSIÓN", sm[["rent_neta", "cagr", "max_dd", "sharpe", "sortino", "n_trades", "win_rate", "win_rate_min", "payoff",
                                         "profit_factor", "expectancy_pct", "expectancy_cop_40m", "dias_prom", "exposicion_media",
                                         "costos_cop", "pnl_bruto_cop", "pnl_neto_cop"]].T)
    show("CONTINUO TAMAÑO FIJO (sin reinversión)", sfx[["rent_neta", "max_dd", "sharpe", "pnl_bruto_cop", "pnl_neto_cop", "costos_cop"]])
    show("BENCHMARKS CONTINUOS", bst[["rent_neta", "cagr", "max_dd", "sharpe"]])
    show("BOOTSTRAP", pd.DataFrame(boots).set_index("modo"))
    show("MONTE CARLO", pd.DataFrame(mcs).set_index("modo").T)
    show("DENTRO/FUERA DE MUESTRA", pd.DataFrame(isoos)[["modo", "periodo", "n_trades", "win_rate", "expectancy_pct",
                                                         "pnl_neto_cop", "rent_neta", "rent_tamano_fijo"]])
    show("DESGLOSE (blindaje)", pd.concat(brk).query("modo=='blindaje'").set_index(["setup", "asset"]))
    if "--skip-windows" in sys.argv:
        return

    # ---------- ventanas ----------
    wins = rolling_windows(cal, cfg["contest"]["sessions"], cfg["backtest"]["start"])
    yrs = [w for w in contest_windows(cal, range(start.year, pd.Timestamp.today().year + 1), cfg) if w[1] < cal[-1]]
    wstats, ystats = [], []
    for k, m in enumerate(modes):
        t0 = time.time()
        w = run_windows(cfg, inp, m, wins, with_benchmarks=(k == 0))
        bcols = [c for c in w.columns if c.startswith("bh_") or c == "efectivo"]
        if k == 0:
            bench_w = w[["inicio"] + bcols]
            bench_w.to_parquet(out / "ventanas_benchmarks.parquet")
        w.drop(columns=bcols).to_parquet(out / f"ventanas_{m}.parquet")
        s = window_stats(w["ret"]); s["modo"] = m; wstats.append(s)
        y = run_windows(cfg, inp, m, yrs, with_benchmarks=True); y["modo"] = m; ystats.append(y)
        print(f"ventanas {m}: {time.time() - t0:.0f}s", flush=True)
    for c in bench_w.columns[1:]:
        s = window_stats(bench_w[c]); s["modo"] = c; wstats.append(s)
    ws = pd.DataFrame(wstats).set_index("modo")
    ws.to_csv(out / "ventanas_estadisticas.csv")
    yy = pd.concat(ystats)
    yy.to_csv(out / "ventanas_concurso_por_anio.csv", index=False)
    show("VENTANAS DE 23 SESIONES", ws)
    piv = yy.pivot_table(index=yy.inicio.dt.year, columns="modo", values="ret")
    ref = yy[yy.modo == "blindaje"].set_index(yy[yy.modo == "blindaje"].inicio.dt.year)
    show("VENTANAS 5-oct a 6-nov POR AÑO", piv.join(ref[["bh_TSLA", "bh_NVDA", "bh_ECOPETROL", "bh_tercios"]]))


if __name__ == "__main__":
    main()
