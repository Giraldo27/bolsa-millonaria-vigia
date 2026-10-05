"""Fase 4i: estrategia para GANAR el concurso: posición concentrada + gestión de exposición por percentil en los cortes.
Mide la probabilidad de pasar los 4 cortes semanales y terminar #1 ('campeón'), no el retorno medio.
Uso: python run_fase4i.py"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from run_fase4d import subset
from src.config import load_config
from src.data_loader import load_prices
from src.evalwin import precompute_field, summarize_paths, window_starts
from src.league import dynamic_mode
from src.panel import build_panel, liquid_universe
from src.pead import make_pead_matrix
from src.rotation import Params, make_features, make_universes, simulate

pd.set_option("display.width", 260, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)


def policy(cfg, field, k: int, kind: str):
    """Devuelve checkpoint(j, ret) -> exposición objetivo en el corte (sesión j+1) o None."""
    cut_T = {int(a): b for a, b in cfg["league"]["cut_T"].items()}
    cuts = sorted(cut_T)
    expo = {m: v[0] for m, v in cfg["v2"]["size_modes"].items()}
    srt, m = field.sorted[k], field.sorted.shape[2]
    if kind == "ninguna":
        return None

    def cp(j: int, ret: float):
        sess = j + 1
        P = 100.0 * (m - np.searchsorted(srt[j], ret, side="right")) / m
        nxt = next((c for c in cuts if c > sess), cuts[-1])
        T = cut_T[nxt]
        if kind == "matriz":
            return expo[dynamic_mode(P, T, nxt - sess)]
        if kind == "proteger_liderazgo":                       # sólo reduce si vas muy arriba (P <= T/2)
            return 0.5 if P <= T / 2 else 1.0
        if kind == "proteger_final":                           # sólo protege en la última semana si vas casi primero
            return 0.5 if (sess >= 20 and P <= 3) else 1.0
        return None
    return cp


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4i"
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
    selections = {
        "NVDA fijo (ref. hindsight)": Params(N=1, min_mom=-1.0, **{**base, "universe": "NVDA"}),
        "TSLA fijo (ref. hindsight)": Params(N=1, min_mom=-1.0, **{**base, "universe": "TSLA"}),
        "mom120 N1": Params(L=120, N=1, **base),
        "mom60 N1 vol6": Params(L=60, N=1, vol_top=6, **base),
        "mom120 N1 vol6": Params(L=120, N=1, vol_top=6, **base),
        "mom120 N1 vol10": Params(L=120, N=1, vol_top=10, **base),
        "mom120 N2 vol10": Params(L=120, N=2, vol_top=10, **base),
        "mom120 N1 vol6 +régimen": Params(L=120, N=1, vol_top=6, regime=True, **base),
        "mom120 N1 vol6 +PEAD": Params(L=120, N=1, vol_top=6, pead=0.05, n_total=1, **base),
    }
    policies = ["ninguna", "matriz", "proteger_liderazgo", "proteger_final"]
    rows = []
    t0 = time.time()
    for sname, prm in selections.items():
        for pol in policies:
            paths = np.zeros((len(starts), n))
            for k, s in enumerate(starts):
                cp = policy(cfg, fields["amplio"], k, pol)
                paths[k] = simulate(p, f, uni[prm.universe], prm, int(s), n, cfg, uni_pead=uni["mgc"], checkpoint=cp)
            for fname, fl in fields.items():
                for tag, sel in (("train", train), ("test", ~train), ("todo", np.ones(len(train), bool))):
                    r = summarize_paths(subset(fl, sel), paths[sel], cut_T)
                    r.update(seleccion=sname, politica=pol, campo=fname, tramo=tag)
                    rows.append(r)
        print(f"  {sname} {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "concentrada.csv", index=False)
    for fname in fields:
        sub = df[(df.campo == fname) & (df.tramo.isin(["train", "test"]))].pivot(
            index=["seleccion", "politica"], columns="tramo", values=["ret_medio", "p_final", "top10", "primero", "pasa_20", "campeon"])
        sub.columns = [f"{a}_{b}" for a, b in sub.columns]
        print(f"\n=== campo {fname} (orden: campeón en validación) ===")
        print(sub[[f"{m}_{t}" for m in ("ret_medio", "p_final", "top10", "pasa_20", "primero", "campeon") for t in ("train", "test")]]
              .sort_values("campeon_test", ascending=False).to_string())


if __name__ == "__main__":
    main()
