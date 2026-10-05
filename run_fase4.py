"""Fase 4a: barrido de parámetros con validación temporal (optimiza < 2022, valida >= 2022).
Uso: python run_fase4.py [--force]   Guarda cada experimento en outputs/fase4/*.csv y un resumen de decisiones."""
from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from src.config import load_config
from src.data_prep import build_inputs
from src.sweep import evaluate, override, pick_and_validate, run_sweep

pd.set_option("display.width", 230, "display.max_columns", 40, "display.float_format", "{:,.4f}".format)
ASSETS = ["ECOPETROL", "NVDA", "TSLA"]
TPS = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]
SLS = [0.02, 0.03, 0.035, 0.04, 0.05, 0.06, 0.08]
THRS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
BRENT = [0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
TSTOPS_C = [3, 5, 8]
TSTOPS_A = [2, 3, 4, 5, 6, 8]
B_DAYS = [1, 2, 3, 4, 5]
TRAIL_FRAC = [None, 0.5, 0.67, 0.8]
TRAIL_SL = [0.005, 0.015, 0.025]
TRENDS = ["none", "close_gt_sma20", "close_gt_sma50", "sma20_gt_sma50", "close_lt_sma50"]
RSI_MAX = [None, 30, 40]

OUT = None
FORCE = "--force" in sys.argv
decisions: list[dict] = []


def cached(name: str, fn):
    f = OUT / f"{name}.csv"
    if f.exists() and not FORCE:
        return pd.read_csv(f)
    t0 = time.time()
    df = fn()
    df.to_csv(f, index=False)
    print(f"  {name}: {len(df)} corridas, {time.time() - t0:.0f}s", flush=True)
    return df


def decide(exp: str, asset: str, df: pd.DataFrame, keys: list[str], base_key, axes=None, params: list[str] | None = None) -> None:
    r = pick_and_validate(df, keys, base_key, axes)
    row = dict(experimento=exp, activo=asset, aceptada=r["aceptada"], motivo=r["motivo"])
    if "mejor" in r:
        b, m = r["base"], r["mejor"]
        row.update({p: m[p] for p in (params or keys)})
        row.update(exp_is_mejor=m.exp_is, exp_oos_mejor=m.exp_oos, n_oos_mejor=m.n_oos, exp_oos_base=b.exp_oos,
                   exp_is_base=b.exp_is, mejora_oos=r["mejora_oos"],
                   pct_celdas_oos_positivas=float((df.exp_oos > 0).mean()))
    decisions.append(row)


def main() -> None:
    global OUT
    cfg = load_config()
    OUT = cfg["paths"]["outputs_dir"] / "fase4"
    OUT.mkdir(exist_ok=True)
    split = pd.Timestamp(cfg["backtest"]["oos_split"])
    inp = build_inputs(cfg)
    base = cfg["assets"]
    ev = lambda tr: evaluate(tr, split)                                                    # noqa: E731

    # ---------- E0: ablación por setup y activo (¿qué conviene eliminar?) ----------
    def e0():
        rows = []
        for s, assets in (("A", ASSETS), ("B", ASSETS), ("C", ["ECOPETROL"])):
            for a in assets:
                rows.append(dict(setup=s, activo=a, **ev(run_sweep(cfg, inp, a, [s]))))
        return pd.DataFrame(rows)
    e0df = cached("e0_ablacion_setup_activo", e0)
    print("\nE0 ablación por setup y activo (tamaño fijo):\n", e0df.to_string(index=False))

    # ---------- E1: TP x SL por activo (Setups A+B) ----------
    for a in ASSETS:
        def e1(a=a):
            rows = []
            for tp in TPS:
                for sl in SLS:
                    kw = {f"assets__{a}__tp": tp, f"assets__{a}__sl": sl}
                    if (tp, sl) != (base[a]["tp"], base[a]["sl"]):                          # trailing proporcional (0,67 x TP)
                        kw[f"assets__{a}__trail_trigger"] = round(0.67 * tp, 4)
                    rows.append(dict(tp=tp, sl=sl, **ev(run_sweep(override(cfg, **kw), inp, a, ["A", "B"]))))
            return pd.DataFrame(rows)
        df = cached(f"e1_tp_sl_{a}", e1)
        decide("TP x SL (A+B)", a, df, ["tp", "sl"], (base[a]["tp"], base[a]["sl"]), ("tp", "sl"))

    # ---------- E2: umbral del Setup A ----------
    for a in ASSETS:
        def e2(a=a):
            return pd.DataFrame([dict(umbral=t, **ev(run_sweep(override(cfg, **{f"setups__A__drop_threshold__{a}": t}), inp, a, ["A"])))
                                 for t in THRS])
        df = cached(f"e2_umbral_A_{a}", e2)
        decide("Umbral Setup A", a, df, ["umbral"], cfg["setups"]["A"]["drop_threshold"][a])

    # ---------- E3: Setup C (Brent) ----------
    def e3():
        return pd.DataFrame([dict(brent=b, tstop=ts, **ev(run_sweep(override(cfg, setups__C__brent_up=b, setups__C__time_stop=ts),
                                                                      inp, "ECOPETROL", ["C"]))) for b in BRENT for ts in TSTOPS_C])
    df = cached("e3_setup_C", e3)
    decide("Setup C (Brent x stop tiempo)", "ECOPETROL", df, ["brent", "tstop"], (0.025, 5), ("brent", "tstop"))

    # ---------- E4: stop de tiempo del Setup A ----------
    for a in ASSETS:
        def e4(a=a):
            return pd.DataFrame([dict(tstop=ts, **ev(run_sweep(override(cfg, setups__A__time_stop=ts), inp, a, ["A"])))
                                 for ts in TSTOPS_A])
        df = cached(f"e4_tstop_A_{a}", e4)
        decide("Stop de tiempo Setup A", a, df, ["tstop"], 3)

    # ---------- E5: sesiones de anticipación del Setup B ----------
    def e5():
        rows = []
        for n in B_DAYS:
            c = override(cfg, setups__B__entry_sessions_before_report=n)
            inp_n = build_inputs(c)
            for a in ASSETS:
                rows.append(dict(activo=a, dias=n, **ev(run_sweep(c, inp_n, a, ["B"]))))
        return pd.DataFrame(rows)
    df = cached("e5_dias_antes_B", e5)
    for a in ASSETS:
        decide("Sesiones antes del reporte (B)", a, df[df.activo == a], ["dias"], 2)

    # ---------- E6: trailing ----------
    for a in ASSETS:
        def e6(a=a):
            rows = []
            for fr in TRAIL_FRAC:
                for tsl in TRAIL_SL:
                    tp = base[a]["tp"]
                    trig = 9.99 if fr is None else round(fr * tp, 4)
                    c = override(cfg, **{f"assets__{a}__trail_trigger": trig, f"assets__{a}__trail_sl": tsl})
                    rows.append(dict(frac=-1.0 if fr is None else fr, trail_sl=tsl, **ev(run_sweep(c, inp, a, ["A", "B"]))))
            return pd.DataFrame(rows)
        df = cached(f"e6_trailing_{a}", e6)
        # base: trailing original (gatillo en 0,67 x TP aprox.; se compara contra la fila más cercana: 0,67 / 1,5 %)
        decide("Trailing (gatillo x TP, SL trailing)", a, df, ["frac", "trail_sl"], (0.67, 0.015), ("frac", "trail_sl"))

    # ---------- E7: filtros de tendencia y RSI en el Setup A ----------
    for a in ASSETS:
        def e7(a=a):
            rows = []
            for tr in TRENDS:
                for rm in RSI_MAX:
                    c = override(cfg, filters={"A": {"trend": tr, "rsi_max": rm}})
                    rows.append(dict(tendencia=tr, rsi_max=-1 if rm is None else rm, **ev(run_sweep(c, inp, a, ["A"]))))
            return pd.DataFrame(rows)
        df = cached(f"e7_filtros_A_{a}", e7)
        decide("Filtro tendencia/RSI (A)", a, df, ["tendencia", "rsi_max"], ("none", -1))

    dec = pd.DataFrame(decisions)
    dec.to_csv(OUT / "decisiones.csv", index=False)
    print("\n=== DECISIONES (elige con dentro de muestra, valida fuera de muestra) ===")
    cols = [c for c in ["experimento", "activo", "aceptada", "motivo", "tp", "sl", "umbral", "brent", "tstop", "dias", "frac",
                        "trail_sl", "tendencia", "rsi_max", "exp_is_mejor", "exp_oos_mejor", "exp_oos_base", "mejora_oos",
                        "n_oos_mejor", "pct_celdas_oos_positivas"] if c in dec.columns]
    print(dec[cols].to_string(index=False))
    print(f"\nConfiguraciones probadas en total: {sum(len(pd.read_csv(f)) for f in OUT.glob('e*.csv'))}")


if __name__ == "__main__":
    main()
