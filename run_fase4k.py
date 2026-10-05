"""Fase 4k: ¿qué medida de volatilidad elige mejor la acción única? Detalle por año de la regla elegida.
Uso: python run_fase4k.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_fase4d import subset
from src.config import load_config
from src.data_loader import load_prices
from src.evalwin import by_year, precompute_field, summarize_paths, window_starts
from src.panel import build_panel, liquid_universe
from src.rotation import Params, make_features, make_universes, pick, simulate

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4k"
    out.mkdir(exist_ok=True)
    cal = load_prices("ECOPETROL", cfg).index
    cal = cal[cal >= "2011-01-01"]
    n = cfg["contest"]["sessions"]
    cut_T = {int(k): v for k, v in cfg["league"]["cut_T"].items()}
    tick = liquid_universe(cfg, cfg["paths"]["outputs_dir"] / "fase1_ranking.csv", 1000, 12)
    p = build_panel(cfg, tick, cal)
    f = make_features(p, [(60, 0), (120, 0)])
    uni = make_universes(p, cfg)
    starts = window_starts(cal, n, cfg["backtest"]["start"], cfg["league"]["step"])
    fields = {"amplio": precompute_field(cfg, cal, starts, n), "mgc": precompute_field(cfg, cal, starts, n, only_groups=("mgc",))}
    train = fields["amplio"].dates < pd.Timestamp(cfg["backtest"]["oos_split"])
    base = dict(stop=None, trend="none", min_mom=-1.0, N=1, vol_top=1)
    sel = {f"más volátil {w}d ({u})": Params(vol_window=w, universe=u, **base) for w in (20, 60, 250) for u in ("mgc", "mgc_etf")}
    sel["TSLA fijo"] = Params(N=1, universe="TSLA", min_mom=-1.0, stop=None, trend="none")
    rows, store = [], {}
    for name, prm in sel.items():
        paths = np.stack([simulate(p, f, uni[prm.universe], prm, int(s), n, cfg) for s in starts])
        store[name] = paths
        for fname, fl in fields.items():
            for tag, m in (("train", train), ("test", ~train), ("todo", np.ones(len(train), bool))):
                r = summarize_paths(subset(fl, m), paths[m], cut_T)
                r.update(seleccion=name, campo=fname, tramo=tag)
                rows.append(r)
        c = {}
        for s in starts[::10]:
            for a in pick(p, f, prm, uni[prm.universe], int(s) - 1):
                c[p.tickers[a]] = c.get(p.tickers[a], 0) + 1
        print(name, dict(sorted(c.items(), key=lambda kv: -kv[1])[:5]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "volatilidad_ventana.csv", index=False)
    for fname in fields:
        sub = df[df.campo == fname].pivot(index="seleccion", columns="tramo", values=["ret_medio", "p_final", "top10", "pasa_20", "primero", "campeon"])
        sub.columns = [f"{a}_{b}" for a, b in sub.columns]
        print(f"\n=== campo {fname} ===")
        print(sub[[f"{m}_{t}" for m in ("ret_medio", "p_final", "top10", "pasa_20", "primero", "campeon") for t in ("train", "test", "todo")]]
              .sort_values("campeon_todo", ascending=False).to_string())
    plan_name = "más volátil 250d (mgc)"                       # regla elegida para el plan final
    np.save(out / "paths_plan.npy", store[plan_name])
    np.save(out / "paths_tsla.npy", store["TSLA fijo"])
    np.save(out / "dates_plan.npy", fields["amplio"].dates.values)
    best = "más volátil 60d (mgc)"
    byy = by_year(fields["amplio"], store[best], cut_T)
    print(f"\n=== {best}: por año de inicio (campo amplio) ===\n", byy[["ret_medio", "pct_pos", "p_final", "top10", "pasa_20", "primero", "campeon"]].to_string())
    paths = store[best]
    print("\nDistribución del retorno final por ventana:", {q: round(float(np.percentile(paths[:, -1], q)), 3) for q in (5, 25, 50, 75, 90, 95, 99)})
    print("P(ret > +10 %):", round(float((paths[:, -1] > 0.10).mean()), 3), " P(ret > +20 %):", round(float((paths[:, -1] > 0.20).mean()), 3),
          " P(ret < -10 %):", round(float((paths[:, -1] < -0.10).mean()), 3))


if __name__ == "__main__":
    main()
