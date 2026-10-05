"""Fase 4h: evaluación final de la estrategia v2 (núcleo de momentum con filtro de régimen + PEAD + matriz dinámica).
Uso: python run_fase4h.py"""
from __future__ import annotations

import numpy as np
import pandas as pd

from run_fase4d import subset
from src.config import load_config
from src.data_loader import load_prices
from src.evalwin import by_year, precompute_field, summarize_paths, window_starts
from src.league import dynamic_mode
from src.panel import build_panel, liquid_universe
from src.pead import make_pead_matrix
from src.rotation import Params, make_features, make_universes, simulate

pd.set_option("display.width", 250, "display.max_columns", 40, "display.float_format", "{:,.3f}".format)


def v2_params(cfg, **over) -> Params:
    v = cfg["v2"]
    base = dict(L=v["core"]["L"], N=v["core"]["N"], universe=v["universe"], K=v["core"]["K"], stop=v["core"]["stop"],
                trend=v["core"]["trend"], regime=v["core"]["regime"], pead=v["pead"]["threshold"],
                pead_hold=v["pead"]["hold"], n_total=v["pead"]["n_total"], replace=v["pead"]["replace"])
    base.update(over)
    return Params(**base)


def dynamic_scale(cfg, field, k: int):
    """Matriz dinámica para la ventana k: devuelve (fracción, nº de posiciones) según el percentil del día anterior."""
    cut_T = {int(a): b for a, b in cfg["league"]["cut_T"].items()}
    cuts = sorted(cut_T)
    modes = cfg["v2"]["size_modes"]
    srt = field.sorted[k]
    m = srt.shape[1]

    def scale(j: int, ret: float):
        if j == 0:
            return tuple(modes["ataque"])                       # sin información: se arranca invertido
        P = 100.0 * (m - np.searchsorted(srt[j - 1], ret, side="right")) / m
        sess = j                                               # sesiones transcurridas
        nxt = next((c for c in cuts if c > sess), cuts[-1])
        return tuple(modes[dynamic_mode(P, cut_T[nxt], nxt - sess)])
    return scale


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["outputs_dir"] / "fase4h"
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
    dates = fields["campo_amplio"].dates
    train = dates < pd.Timestamp(cfg["backtest"]["oos_split"])

    variants = {
        "v2 estática (invertido 100 %)": (v2_params(cfg), False),
        "v2 con matriz dinámica": (v2_params(cfg), True),
        "v2 sin filtro de régimen": (v2_params(cfg, regime=False), False),
        "v2 sin PEAD": (v2_params(cfg, pead=None), False),
        "v2 sin NVDA/TSLA": (v2_params(cfg, universe="mgc_sin_nvda_tsla"), False),
        "B&H NVDA": (Params(N=1, K=0, stop=None, trend="none", min_mom=-1.0, universe="NVDA"), False),
        "B&H TSLA": (Params(N=1, K=0, stop=None, trend="none", min_mom=-1.0, universe="TSLA"), False),
    }
    rows, paths_all = [], {}
    for name, (prm, dyn) in variants.items():
        paths = np.zeros((len(starts), n))
        for k, s in enumerate(starts):
            sc = dynamic_scale(cfg, fields["campo_amplio"], k) if dyn else None
            paths[k] = simulate(p, f, uni[prm.universe], prm, int(s), n, cfg, uni_pead=uni["mgc_sin_nvda_tsla" if "sin NVDA" in name else "mgc"], scale=sc)
        paths_all[name] = paths
        np.save(out / f"paths_{name.replace(' ', '_').replace('/', '-').replace('%', 'pct').replace('(', '').replace(')', '')}.npy", paths)
        for fname, fl in fields.items():
            for tag, sel in (("train", train), ("test", ~train), ("todo", np.ones(len(train), bool))):
                r = summarize_paths(subset(fl, sel), paths[sel], cut_T)
                r.update(variante=name, campo=fname, tramo=tag)
                rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "v2_resumen.csv", index=False)
    cols = ["ret_medio", "ret_mediana", "pct_pos", "p10", "p90", "p_final", "top30", "top10", "primero", "pasa_20"]
    for fname in fields:
        sub = df[(df.campo == fname) & (df.tramo != "todo")].pivot(index="variante", columns="tramo", values=cols)
        sub.columns = [f"{a}_{b}" for a, b in sub.columns]
        keep = [f"{m}_{t}" for m in ("ret_medio", "pct_pos", "p_final", "top30", "top10", "primero", "pasa_20") for t in ("train", "test")]
        print(f"\n=== {fname} ===\n", sub[keep].to_string())

    # ventanas del 5-oct de cada año (la competencia real) y desglose por año
    yrs = pd.DataFrame({"inicio": dates, "v2": paths_all["v2 estática (invertido 100 %)"][:, -1],
                        "v2_dinamica": paths_all["v2 con matriz dinámica"][:, -1], "bh_nvda": paths_all["B&H NVDA"][:, -1],
                        "bh_tsla": paths_all["B&H TSLA"][:, -1]})
    yrs["anio"] = yrs.inicio.dt.year
    print("\n=== RETORNO MEDIO POR AÑO DE INICIO (todas las ventanas de 23 sesiones) ===")
    print(yrs.groupby("anio")[["v2", "v2_dinamica", "bh_nvda", "bh_tsla"]].mean().to_string())
    yrs.to_csv(out / "v2_por_anio.csv", index=False)
    byy = by_year(fields["campo_amplio"], paths_all["v2 estática (invertido 100 %)"], cut_T)
    print("\nv2 estática por año (campo amplio):\n", byy[["ret_medio", "pct_pos", "p_final", "top30", "top10", "pasa_20"]].to_string())
    y = yrs.groupby("anio").v2.mean()
    from scipy import stats
    print(f"\nv2 estática: retorno medio por ventana {yrs.v2.mean():+.2%}; años con media > 0: {(y > 0).sum()}/{len(y)}; "
          f"t-test sobre medias anuales p={stats.ttest_1samp(y, 0).pvalue:.3f}")


if __name__ == "__main__":
    main()
