"""Fase 4g: sesgo hacia la volatilidad y campo exigente (rivales sólo con acciones MGC).
Uso: python run_fase4g.py"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd

from run_fase4d import subset
from src.config import load_config
from src.data_loader import load_prices
from src.evalwin import precompute_field, summarize_paths, window_starts
from src.panel import build_panel, liquid_universe
from src.pead import make_pead_matrix
from src.rotation import Params, make_features, make_universes, simulate

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4g"
    out.mkdir(exist_ok=True)
    cal = load_prices("ECOPETROL", cfg).index
    cal = cal[cal >= "2011-01-01"]
    n = cfg["contest"]["sessions"]
    cut_T = {int(k): v for k, v in cfg["league"]["cut_T"].items()}
    tick = liquid_universe(cfg, cfg["paths"]["outputs_dir"] / "fase1_ranking.csv", 1000, 12)
    p = build_panel(cfg, tick, cal)
    f = make_features(p, [(60, 0), (120, 0)], pead=make_pead_matrix(p, cfg))
    uni = make_universes(p, cfg)
    starts = window_starts(cal, n, cfg["backtest"]["start"], cfg["league"]["step"])
    fields = {"campo_amplio": precompute_field(cfg, cal, starts, n),
              "campo_mgc": precompute_field(cfg, cal, starts, n, only_groups=("mgc",))}
    split = pd.Timestamp(cfg["backtest"]["oos_split"])
    train = fields["campo_amplio"].dates < split

    designs: dict[str, Params] = {"B&H NVDA": Params(N=1, K=0, stop=None, trend="none", min_mom=-1.0, universe="NVDA"),
                                  "B&H TSLA": Params(N=1, K=0, stop=None, trend="none", min_mom=-1.0, universe="TSLA")}
    for L, N, vt, reg in itertools.product((60, 120), (1, 2), (None, 6, 10), (False, True)):
        designs[f"L{L} N{N} voltop={vt} reg={reg}"] = Params(L=L, N=N, universe="mgc", stop=None, trend="none", regime=reg, vol_top=vt)
    rows = []
    t0 = time.time()
    for name, prm in designs.items():
        paths = np.stack([simulate(p, f, uni[prm.universe], prm, int(s), n, cfg, uni_pead=uni["mgc"]) for s in starts])
        for fname, fl in fields.items():
            for tag, sel in (("train", train), ("test", ~train)):
                r = summarize_paths(subset(fl, sel), paths[sel], cut_T)
                r.update(diseno=name, campo=fname, tramo=tag)
                rows.append(r)
    print(f"{len(designs)} diseños en {time.time() - t0:.0f}s")
    df = pd.DataFrame(rows)
    df.to_csv(out / "volatilidad.csv", index=False)
    for fname in fields:
        sub = df[df.campo == fname].pivot(index="diseno", columns="tramo", values=["ret_medio", "pct_pos", "p_final", "top10", "primero", "pasa_20"])
        sub.columns = [f"{a}_{b}" for a, b in sub.columns]
        print(f"\n=== {fname} (ordenado por p_final en validación) ===")
        print(sub.sort_values("p_final_test").to_string())


if __name__ == "__main__":
    main()
