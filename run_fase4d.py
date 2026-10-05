"""Fase 4d: rediseño de la estrategia. Barrido de reglas de momentum/tendencia evaluadas por ventanas de concurso
contra el campo de 1.000 participantes sintéticos, con entrenamiento (< 2022) y validación (>= 2022).
Uso: python run_fase4d.py"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.data_loader import load_prices
from src.evalwin import Field, precompute_field, summarize_paths, window_starts
from src.panel import build_panel, liquid_universe
from src.rotation import Params, make_features, make_universes, simulate

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)

LS = [20, 60, 120, 250]
NS = [1, 2, 3]
UNIS = ["liq", "mgc"]
KS = [0, 5]
STOPS = [None, 0.10]
REGIMES = [False, True]
TRENDS = ["none", "sma50"]


def run_paths(p, f, uni, prm, starts, n, cfg) -> np.ndarray:
    return np.stack([simulate(p, f, uni, prm, int(s), n, cfg) for s in starts])


def subset(field: Field, sel: np.ndarray) -> Field:
    return Field(field.starts[sel], field.dates[sel], field.sorted[sel], field.n)


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4d"
    out.mkdir(exist_ok=True)
    cal = load_prices("ECOPETROL", cfg).index
    cal = cal[cal >= "2011-01-01"]
    n = cfg["contest"]["sessions"]
    cut_T = {int(k): v for k, v in cfg["league"]["cut_T"].items()}
    tick = liquid_universe(cfg, cfg["paths"]["outputs_dir"] / "fase1_ranking.csv", 1000, 12)
    print(len(tick), "activos líquidos:", tick, flush=True)
    p = build_panel(cfg, tick, cal)
    f = make_features(p, [(L, 0) for L in LS] + [(250, 20)])
    uni = make_universes(p, cfg)
    starts = window_starts(cal, n, cfg["backtest"]["start"], cfg["league"]["step"])
    t0 = time.time()
    field = precompute_field(cfg, cal, starts, n)
    print(f"{len(starts)} ventanas, campo listo en {time.time() - t0:.0f}s", flush=True)
    train = field.dates < pd.Timestamp(cfg["backtest"]["oos_split"])
    ftr, fte = subset(field, train), subset(field, ~train)

    refs = {}
    for t in ("NVDA", "TSLA", "ECOPETROL"):
        u = np.array([x == t for x in p.tickers])
        refs[f"B&H {t}"] = (Params(N=1, K=0, stop=None, trend="none", min_mom=-1.0), u)
    refs["B&H 1/3 (TSLA,NVDA,ECOP)"] = (Params(N=3, K=0, stop=None, trend="none", min_mom=-1.0), uni["sys3"])
    rows = []
    for name, (prm, u) in refs.items():
        paths = run_paths(p, f, u, prm, starts, n, cfg)
        for tag, fl, sel in (("train", ftr, train), ("test", fte, ~train)):
            r = summarize_paths(fl, paths[sel], cut_T); r.update(config=name, tramo=tag); rows.append(r)
    ref_df = pd.DataFrame(rows)
    print("\n=== REFERENCIAS ===\n", ref_df[["config", "tramo", "ret_medio", "ret_mediana", "pct_pos", "p_final", "top30", "top10", "pasa_20"]].to_string(index=False), flush=True)

    grid = list(itertools.product(LS, NS, UNIS, KS, STOPS, REGIMES, TRENDS))
    res, allpaths = [], {}
    t0 = time.time()
    for i, (L, N, U, K, stp, reg, tr) in enumerate(grid):
        prm = Params(L=L, N=N, universe=U, K=K, stop=stp, regime=reg, trend=tr)
        paths = run_paths(p, f, uni[U], prm, starts, n, cfg)
        allpaths[prm] = paths
        for tag, fl, sel in (("train", ftr, train), ("test", fte, ~train)):
            r = summarize_paths(fl, paths[sel], cut_T)
            r.update(L=L, N=N, uni=U, K=K, stop=stp or 0, regime=reg, trend=tr, tramo=tag)
            res.append(r)
        if i % 40 == 0:
            print(f"  {i}/{len(grid)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(res)
    df.to_csv(out / "barrido.csv", index=False)
    wide = df.pivot_table(index=["L", "N", "uni", "K", "stop", "regime", "trend"], columns="tramo",
                          values=["ret_medio", "ret_mediana", "p_final", "top30", "top10", "pasa_20", "pct_pos"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index()
    wide.to_csv(out / "barrido_ancho.csv", index=False)
    # ¿la selección en entrenamiento generaliza? correlación de rangos entre entrenamiento y validación
    for m in ("ret_medio", "p_final", "top30", "pasa_20"):
        rho, pv = stats.spearmanr(wide[f"{m}_train"], wide[f"{m}_test"])
        print(f"Spearman {m}: train vs test = {rho:+.2f} (p={pv:.3f})")
    best = wide.sort_values("p_final_train").head(12)
    print("\n=== TOP 12 por percentil medio en ENTRENAMIENTO (menor es mejor) y su VALIDACIÓN ===")
    print(best[["L", "N", "uni", "K", "stop", "regime", "trend", "ret_medio_train", "p_final_train", "top30_train", "pasa_20_train",
                "ret_medio_test", "ret_mediana_test", "p_final_test", "top30_test", "top10_test", "pasa_20_test"]].to_string(index=False))
    print("\n=== TOP 12 por percentil medio en VALIDACIÓN (sólo informativo: NO se usa para elegir) ===")
    print(wide.sort_values("p_final_test").head(12)[["L", "N", "uni", "K", "stop", "regime", "trend", "ret_medio_train", "p_final_train",
                                                       "ret_medio_test", "p_final_test", "top30_test", "pasa_20_test"]].to_string(index=False))
    print(f"\nconfig con retorno medio test > 0: {(wide.ret_medio_test > 0).mean():.0%}; "
          f"mediana de p_final test: {wide.p_final_test.median():.1f}")


if __name__ == "__main__":
    main()
