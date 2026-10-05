"""Fase 4c (opcional): simulador de ligas. Percentil semanal, matriz dinámica, prob. de pasar cada corte y de ser #1.
Uso: python run_fase4c.py   (~10 min)"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.backtest import Backtester
from src.config import load_config
from src.data_prep import build_inputs
from src.league import build_cop_prices, dynamic_mode, participant_paths, percentile_from_top, sample_participants
from src.sweep import override

pd.set_option("display.width", 230, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)


def main() -> None:
    cfg = load_config()
    lg = cfg["league"]
    out = cfg["paths"]["outputs_dir"] / "fase4"
    imp = override(cfg, **cfg["mejoras_validadas"])
    inp = build_inputs(cfg)
    cal, cap, n = inp.activity_cal, float(cfg["capital"]), cfg["contest"]["sessions"]
    cut_T = {int(k): v for k, v in lg["cut_T"].items()}
    cuts = sorted(cut_T)
    opens, closes = build_cop_prices(cfg, cal)
    tickers = list(opens.columns)
    w = sample_participants(len(tickers), lg["n_participants"], lg["seed"])
    entry_cost = lg["participant_entry_cost"] + cfg["costs"]["pct_fee"]
    daily_act = cfg["costs"]["activity_micro_buy_cop"] / cap
    O, C = opens.values, closes.values
    starts = [i for i in range(len(cal) - n + 1) if cal[i] >= pd.Timestamp(cfg["backtest"]["start"])][::lg["step"]]
    print(f"{len(tickers)} activos, {len(starts)} ventanas, {len(w)} participantes", flush=True)

    sys_idx = {a: tickers.index(a) for a in cfg["universe"]["system_assets"]}
    bh_w = {f"bh_{a}": np.eye(len(tickers))[i][None, :] for a, i in sys_idx.items()}
    bh_w["bh_tercios"] = (sum(np.eye(len(tickers))[i] for i in sys_idx.values()) / 3)[None, :]
    static_modes = list(cfg["modes"])
    rows = []
    t0 = time.time()
    for k, s in enumerate(starts):
        cw = cal[s:s + n]
        part = participant_paths(w, O[s:s + n], C[s:s + n], entry_cost, daily_act)          # (1000 x n)
        pos = {d: j for j, d in enumerate(cw)}
        field_max = np.nanmax(part, axis=0)

        def record(name: str, ret_path: np.ndarray) -> None:
            r = dict(inicio=cw[0], estrategia=name, final=ret_path[-1],
                     p_final=percentile_from_top(ret_path[-1], part[:, -1]), es_primero=bool(ret_path[-1] > field_max[-1]))
            for c in cuts:
                r[f"p{c}"] = percentile_from_top(ret_path[c - 1], part[:, c - 1])
            rows.append(r)

        for name, wv in bh_w.items():
            record(name, participant_paths(wv, O[s:s + n], C[s:s + n], entry_cost, daily_act)[0])
        for version, c_ in (("orig", cfg), ("mejor", imp)):
            for m in static_modes:
                res = Backtester(c_, inp, m, start=cw[0], end=cw[-1], deduct_activity=True, liquidate_end=False).run()
                record(f"{version}_{m}", (res.equity.reindex(cw).ffill() / cap - 1).values)

            def ctrl(i, d, eq, pos=pos, part=part):
                j = pos.get(d)
                if j is None:
                    return None
                sess = j + 1
                nxt = next((c for c in cuts if c > sess), cuts[-1])
                return dynamic_mode(percentile_from_top(eq / cap - 1, part[:, j]), cut_T[nxt], nxt - sess)
            res = Backtester(c_, inp, lg["start_mode"], start=cw[0], end=cw[-1], deduct_activity=True,
                             liquidate_end=False, controller=ctrl).run()
            record(f"{version}_dinamica", (res.equity.reindex(cw).ffill() / cap - 1).values)
        if k % 100 == 0:
            print(f"  ventana {k}/{len(starts)}  {time.time() - t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "liga_ventanas.parquet")
    summ = []
    for name, g in df.groupby("estrategia"):
        passed = np.ones(len(g), dtype=bool)
        row = dict(estrategia=name, n_ventanas=len(g), ret_medio=g.final.mean(), ret_mediana=g.final.median(),
                   percentil_medio_final=g.p_final.mean())
        for c in cuts[:-1]:
            passed &= (g[f"p{c}"].values <= cut_T[c])
            row[f"pasa_corte_{c}"] = passed.mean()
        row["top30_final"] = (g.p_final <= 30).mean()
        row["top10_final"] = (g.p_final <= 10).mean()
        row["es_primero"] = g.es_primero.mean()
        summ.append(row)
    sm = pd.DataFrame(summ).set_index("estrategia").sort_values("percentil_medio_final")
    sm.to_csv(out / "liga_resumen.csv")
    print("\n=== SIMULADOR DE LIGAS (percentil: 0 = primero, 100 = último; 'pasa_corte' = acumulado) ===")
    print(sm.to_string())


if __name__ == "__main__":
    main()
