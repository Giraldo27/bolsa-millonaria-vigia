"""Fase 4j: regla a priori para elegir la acción única (mayor volatilidad / momentum) y probabilidad de ser 'campeón'.
Uso: python run_fase4j.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_fase4d import subset
from src.config import load_config
from src.data_loader import load_prices
from src.evalwin import precompute_field, summarize_paths, window_starts
from src.panel import build_panel, liquid_universe
from src.pead import make_pead_matrix
from src.rotation import Params, make_features, make_universes, pick, simulate

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4j"
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
    fields = {"amplio": precompute_field(cfg, cal, starts, n), "mgc": precompute_field(cfg, cal, starts, n, only_groups=("mgc",))}
    train = fields["amplio"].dates < pd.Timestamp(cfg["backtest"]["oos_split"])
    base = dict(stop=None, trend="none", universe="mgc")
    sel = {}
    for vt in (1, 2, 3):
        sel[f"vol_top{vt} sin momentum (a priori: la más volátil)"] = Params(N=vt, vol_top=vt, min_mom=-1.0, L=60, **base)
        sel[f"vol_top{vt} + mom60>0"] = Params(N=vt, vol_top=vt, L=60, **base)
        sel[f"vol_top{vt} + mom120>0"] = Params(N=vt, vol_top=vt, L=120, **base)
    sel["vol_top3 N1: mayor momentum120 entre las 3 más volátiles"] = Params(N=1, vol_top=3, L=120, **base)
    sel["vol_top5 N1: mayor momentum120 entre las 5 más volátiles"] = Params(N=1, vol_top=5, L=120, **base)
    sel["TSLA fijo"] = Params(N=1, min_mom=-1.0, **{**base, "universe": "TSLA"})
    sel["NVDA fijo"] = Params(N=1, min_mom=-1.0, **{**base, "universe": "NVDA"})
    rows = []
    which = {}
    for name, prm in sel.items():
        paths = np.stack([simulate(p, f, uni[prm.universe], prm, int(s), n, cfg) for s in starts])
        for fname, fl in fields.items():
            for tag, m in (("train", train), ("test", ~train)):
                r = summarize_paths(subset(fl, m), paths[m], cut_T)
                r.update(seleccion=name, campo=fname, tramo=tag)
                rows.append(r)
        # ¿qué activo eligió la regla? (frecuencia de la compra inicial)
        c = {}
        for s in starts[::10]:
            for a in pick(p, f, prm, uni[prm.universe], int(s) - 1):
                c[p.tickers[a]] = c.get(p.tickers[a], 0) + 1
        which[name] = dict(sorted(c.items(), key=lambda kv: -kv[1])[:4])
    df = pd.DataFrame(rows)
    df.to_csv(out / "seleccion_unica.csv", index=False)
    for fname in fields:
        sub = df[df.campo == fname].pivot(index="seleccion", columns="tramo", values=["ret_medio", "p_final", "top10", "pasa_20", "primero", "campeon"])
        sub.columns = [f"{a}_{b}" for a, b in sub.columns]
        print(f"\n=== campo {fname} ===")
        print(sub[[f"{m}_{t}" for m in ("ret_medio", "p_final", "top10", "pasa_20", "primero", "campeon") for t in ("train", "test")]]
              .sort_values("campeon_train", ascending=False).to_string())
    print("\nActivos que elige cada regla (conteo en ventanas muestreadas):")
    for k, v in which.items():
        print(f"  {k}: {v}")
    t = len(cal) - 1
    vol = pd.Series(f.vol20[t], index=p.tickers)[uni["mgc"]].sort_values(ascending=False)
    print("\nVolatilidad 20d hoy (MGC):", (vol.head(8) * 100).round(2).to_dict())
    print("Momentum 60 / 120 de las 6 más volátiles:")
    for a in vol.index[:6]:
        i = p.tickers.index(a)
        print(f"  {a}: mom60={f.mom[(60, 0)][t, i]:+.1%} mom120={f.mom[(120, 0)][t, i]:+.1%}")


if __name__ == "__main__":
    main()
