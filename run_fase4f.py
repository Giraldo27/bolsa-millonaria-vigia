"""Fase 4f: diseños con señal PEAD (+ núcleo de momentum), evaluados contra el campo de 1.000 participantes.
Uso: python run_fase4f.py"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from src.config import load_config
from src.data_loader import load_prices
from src.evalwin import by_year, precompute_field, summarize_paths, window_starts
from src.panel import build_panel, liquid_universe
from src.pead import make_pead_matrix
from src.rotation import Params, make_features, make_universes, simulate
from run_fase4d import subset

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)
COLS = ["ret_medio", "ret_mediana", "pct_pos", "p10", "p90", "p_final", "top30", "top10", "pasa_20"]


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4f"
    out.mkdir(exist_ok=True)
    cal = load_prices("ECOPETROL", cfg).index
    cal = cal[cal >= "2011-01-01"]
    n = cfg["contest"]["sessions"]
    cut_T = {int(k): v for k, v in cfg["league"]["cut_T"].items()}
    tick = liquid_universe(cfg, cfg["paths"]["outputs_dir"] / "fase1_ranking.csv", 1000, 12)
    p = build_panel(cfg, tick, cal)
    pead = make_pead_matrix(p, cfg)
    print("eventos PEAD en el panel:", int(np.isfinite(pead).sum()), " con reacción >= 5 %:", int((pead >= 0.05).sum()), flush=True)
    f = make_features(p, [(60, 0), (120, 0), (250, 0)], pead=pead)
    uni = make_universes(p, cfg)
    starts = window_starts(cal, n, cfg["backtest"]["start"], cfg["league"]["step"])
    field = precompute_field(cfg, cal, starts, n)
    train = field.dates < pd.Timestamp(cfg["backtest"]["oos_split"])
    ftr, fte = subset(field, train), subset(field, ~train)

    designs: dict[str, tuple[Params, str]] = {}
    nv = np.array([x == "NVDA" for x in p.tickers])
    designs["B&H NVDA (referencia)"] = (Params(N=1, K=0, stop=None, trend="none", min_mom=-1.0, universe="NVDA"), "NVDA")
    for reg in (False, True):
        designs[f"Núcleo L120 N2 hold reg={reg}"] = (Params(L=120, N=2, stop=None, trend="none", regime=reg, universe="mgc"), "mgc")
    for thr, nt, reg in itertools.product((0.03, 0.05, 0.08), (1, 2, 3), (False, True)):
        designs[f"PEAD solo thr={thr} n={nt} reg={reg}"] = (
            Params(core=False, pead=thr, n_total=nt, N=nt, stop=None, trend="none", regime=reg, universe="mgc"), "mgc")
    for thr, reg, nt in itertools.product((0.03, 0.05), (False, True), (2, 3)):
        designs[f"Núcleo+PEAD thr={thr} n={nt} reg={reg}"] = (
            Params(L=120, N=2, n_total=nt, pead=thr, replace=True, stop=None, trend="none", regime=reg, universe="mgc"), "mgc")
    # robustez: sin NVDA ni TSLA
    for thr in (0.05,):
        designs[f"[sin NVDA/TSLA] PEAD solo thr={thr} n=3"] = (
            Params(core=False, pead=thr, n_total=3, N=3, stop=None, trend="none", universe="mgc_sin_nvda_tsla"), "mgc_sin_nvda_tsla")
        designs[f"[sin NVDA/TSLA] Núcleo+PEAD thr={thr} n=3"] = (
            Params(L=120, N=2, n_total=3, pead=thr, stop=None, trend="none", universe="mgc_sin_nvda_tsla"), "mgc_sin_nvda_tsla")

    rows, store = [], {}
    t0 = time.time()
    for k, (name, (prm, u)) in enumerate(designs.items()):
        paths = np.stack([simulate(p, f, uni[prm.universe], prm, int(s), n, cfg, uni_pead=uni[u]) for s in starts])
        store[name] = paths
        for tag, fl, sel in (("train", ftr, train), ("test", fte, ~train)):
            r = summarize_paths(fl, paths[sel], cut_T); r.update(diseno=name, tramo=tag); rows.append(r)
        if k % 10 == 0:
            print(f"  {k}/{len(designs)} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "disenos.csv", index=False)
    wide = df.pivot(index="diseno", columns="tramo", values=COLS)
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    show = wide[[f"{m}_{t}" for m in ("ret_medio", "pct_pos", "p_final", "top30", "top10", "pasa_20") for t in ("train", "test")]]
    print("\n=== DISEÑOS (train = <2022, test = >=2022) ===")
    print(show.sort_values("p_final_test").to_string())
    np.save(out / "paths_keys.npy", np.array(list(store)))
    for kname, v in store.items():
        np.save(out / ("paths_" + str(abs(hash(kname)))[:8] + ".npy"), v)


if __name__ == "__main__":
    main()
