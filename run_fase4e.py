"""Fase 4e: estudio de eventos de resultados (PEAD / gap-and-go) en las acciones MGC.
Pregunta: tras una reacción fuerte al reporte, ¿el precio sigue en la misma dirección lo bastante como para pagar los costos?"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config
from src.data_loader import load_earnings, load_prices

pd.set_option("display.width", 220, "display.float_format", "{:,.4f}".format)
HORIZONS = [5, 10, 15, 20]
COST = 0.0118          # ida y vuelta aproximada (entrada 0,6 % + salida 0,6 %); a mercado con slippage ~1,2 %


def events_frame(cfg) -> pd.DataFrame:
    mgc = list(cfg["universe"]["mgc"])
    earn = load_earnings(cfg, mgc)
    rows = []
    for t in mgc:
        px = load_prices(t, cfg)["Close"]
        e = earn[(earn.ticker == t) & (earn.report_date <= px.index[-1] - pd.Timedelta(days=40))]
        idx = px.index
        for _, ev in e.iterrows():
            i = int(idx.searchsorted(ev.report_date))
            react = i + 1 if (ev.timing == "AMC" and i < len(idx) and idx[i] == ev.report_date) else i       # sesión de reacción
            if react < 1 or react + max(HORIZONS) >= len(idx) or idx[react] < pd.Timestamp("2012-01-01"):
                continue
            r = dict(ticker=t, fecha=idx[react], r0=px.iloc[react] / px.iloc[react - 1] - 1)
            for h in HORIZONS:
                r[f"f{h}"] = px.iloc[react + h] / px.iloc[react] - 1
            # deriva incondicional del mismo activo: media de retornos de h sesiones en la muestra
            rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    ev = events_frame(cfg)
    split = pd.Timestamp(cfg["backtest"]["oos_split"])
    ev["tramo"] = np.where(ev.fecha < split, "train", "test")
    print(f"{len(ev)} eventos MGC; reacción media {ev.r0.mean():+.2%}; |r0| mediana {ev.r0.abs().median():.2%}")
    base = {h: ev[f"f{h}"].mean() for h in HORIZONS}                           # deriva media de cualquier evento
    rows = []
    for nombre, mask in (("reacción >= +3 %", ev.r0 >= 0.03), ("reacción >= +5 %", ev.r0 >= 0.05), ("reacción >= +8 %", ev.r0 >= 0.08),
                         ("reacción <= -5 %", ev.r0 <= -0.05), ("-2 % < reacción < +2 %", ev.r0.abs() < 0.02)):
        for tr in ("train", "test"):
            s = ev[mask & (ev.tramo == tr)]
            for h in (10, 20):
                x = s[f"f{h}"]
                if len(x) < 5:
                    continue
                rows.append(dict(grupo=nombre, tramo=tr, h=h, n=len(x), media=x.mean(), mediana=x.median(),
                                 exceso_vs_base=x.mean() - base[h], neto_costos=x.mean() - COST,
                                 acierto=(x > COST).mean(), p_vs_base=stats.ttest_1samp(x, base[h]).pvalue))
    out = pd.DataFrame(rows)
    out.to_csv(cfg["paths"]["outputs_dir"] / "fase4e_pead.csv", index=False)
    print("\nderiva base (cualquier evento):", {h: round(v, 4) for h, v in base.items()})
    print(out.to_string(index=False))
    # ¿hay deriva en sentido contrario (reversión) tras caídas fuertes? (para un posible Setup de rebote post-reporte)


if __name__ == "__main__":
    main()
