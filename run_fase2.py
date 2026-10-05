"""Fase 2: prueba de humo del motor sobre datos reales (4 modos). Los resultados formales son de la Fase 3."""
from __future__ import annotations

import pandas as pd

from src.backtest import Backtester
from src.config import load_config
from src.data_prep import build_inputs

pd.set_option("display.width", 220, "display.max_columns", 30, "display.float_format", "{:,.2f}".format)


def main() -> None:
    cfg = load_config()
    inp = build_inputs(cfg)
    for a, ad in inp.assets.items():
        print(f"{a}: {ad.ohlc.index[0].date()} -> {ad.ohlc.index[-1].date()} | pre_exit={len(ad.own.pre_exit)} "
              f"reaction={len(ad.own.reaction)} B_entradas={len(ad.cat.b_entry)}")
    rows = []
    for mode in cfg["modes"]:
        r = Backtester(cfg, inp, mode, deduct_activity=False).run()
        t, eq = r.trades, r.equity
        rows.append(dict(modo=mode, trades=len(t), neto_pct=(eq.iloc[-1] / eq.iloc[0] - 1) * 100,
                         win_pct=(t.net_pnl > 0).mean() * 100, dias_prom=t.sessions.mean(),
                         costos_mm=(t.spread_cost + t.slippage_cost + t.fees).sum() / 1e6,
                         actividad_mm=r.activity_cost / 1e6, dd_pct=((eq / eq.cummax() - 1).min()) * 100))
        print(f"\n== {mode} ==  saltadas: {r.skipped}")
        print(t.groupby(["setup", "asset"]).agg(n=("net_pnl", "size"), pnl_mm=("net_pnl", lambda s: s.sum() / 1e6),
                                                 win=("net_pnl", lambda s: (s > 0).mean() * 100)).to_string())
        print(t.exit_reason.value_counts().to_dict())
        if mode == "mantener":
            t.to_csv(cfg["paths"]["outputs_dir"] / "fase2_trades_mantener_humo.csv", index=False)
    print("\n", pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
