"""Barrido de parámetros con validación temporal (optimizar < 2022, validar >= 2022).

Todas las corridas usan tamaño fijo (sin reinversión) y un solo activo a la vez, para que cada celda del barrido mida
sólo el efecto del parámetro (sin que el interés compuesto o la competencia por cupos tapen la señal)."""
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

from .backtest import Backtester, Inputs

MIN_TRADES = 30                 # mínimo de operaciones por tramo para fiarse de una celda
MIN_IMPROVEMENT = 0.0025        # mejora mínima en expectancy fuera de muestra: 0,25 puntos porcentuales


def override(cfg: dict[str, Any], **paths: Any) -> dict[str, Any]:
    """Copia profunda de cfg con valores cambiados por ruta con '__' (p. ej. assets__TSLA__tp=0.08)."""
    out = copy.deepcopy(cfg)
    for path, val in paths.items():
        node = out
        keys = path.split("__")
        for k in keys[:-1]:
            node = node[k]
        node[keys[-1]] = val
    return out


def run_sweep(cfg: dict[str, Any], inp: Inputs, asset: str | None, setups: list[str]) -> pd.DataFrame:
    """Trades de una corrida de tamaño fijo, con un activo (o todos si asset=None) y los setups indicados."""
    c = copy.deepcopy(cfg)
    mode = {"positions": 1 if asset else 2, "size": 0.20, "setups": setups}
    if asset:
        mode["only_asset"] = asset
    c["modes"]["sweep"] = mode
    r = Backtester(c, inp, "sweep", deduct_activity=False, compound=False).run()
    return r.trades


def evaluate(trades: pd.DataFrame, split: pd.Timestamp) -> dict[str, float]:
    """Expectancy neta por trade y P&L (COP) dentro (< split) y fuera (>= split) de muestra."""
    out: dict[str, float] = {}
    for tag, part in (("is", trades[trades.entry_date < split] if len(trades) else trades),
                      ("oos", trades[trades.entry_date >= split] if len(trades) else trades)):
        out[f"n_{tag}"] = len(part)
        out[f"exp_{tag}"] = part.ret_net.mean() if len(part) else np.nan
        out[f"pnl_{tag}"] = part.net_pnl.sum() if len(part) else 0.0
        out[f"win_{tag}"] = (part.net_pnl > 0).mean() if len(part) else np.nan
    return out


def smooth(grid: pd.DataFrame) -> pd.DataFrame:
    """Media de la celda y sus vecinas (3x3): premia zonas planas y castiga picos aislados."""
    v = grid.values.astype(float)
    pad = np.pad(v, 1, constant_values=np.nan)
    stack = [pad[i:i + v.shape[0], j:j + v.shape[1]] for i in range(3) for j in range(3)]
    with np.errstate(all="ignore"):
        m = np.nanmean(np.stack(stack), axis=0)
    return pd.DataFrame(m, index=grid.index, columns=grid.columns)


def pick_and_validate(df: pd.DataFrame, key_cols: list[str], baseline_key: tuple, grid_axes: tuple | None = None
                      ) -> dict[str, Any]:
    """Elige la mejor configuración usando SÓLO el tramo dentro de muestra y la valida fuera de muestra.
    Si hay dos ejes (grid_axes), la elección usa la expectancy suavizada (zona plana) en vez del pico."""
    d = df.copy()
    d["_key"] = list(zip(*[d[c] for c in key_cols])) if len(key_cols) > 1 else d[key_cols[0]]
    base = d[d["_key"] == baseline_key].iloc[0]
    ok = d[d.n_is >= MIN_TRADES].copy()
    if ok.empty:
        return dict(aceptada=False, motivo="sin celdas con suficientes operaciones", base=base)
    if grid_axes:
        rows, cols = grid_axes
        g = ok.pivot(index=rows, columns=cols, values="exp_is").reindex(
            index=sorted(d[rows].unique()), columns=sorted(d[cols].unique()))
        sm = smooth(g)
        ok = ok.join(sm.stack().rename("exp_is_suav"), on=[rows, cols])
        best = ok.sort_values("exp_is_suav", ascending=False).iloc[0]
    else:
        best = ok.sort_values("exp_is", ascending=False).iloc[0]
    mejora = best.exp_oos - base.exp_oos
    plateau = True
    if grid_axes:
        goos = d.pivot(index=grid_axes[0], columns=grid_axes[1], values="exp_oos").reindex(
            index=sorted(d[grid_axes[0]].unique()), columns=sorted(d[grid_axes[1]].unique()))
        plateau = bool(smooth(goos).loc[best[grid_axes[0]], best[grid_axes[1]]] > base.exp_oos)
    acepta = bool(best.n_oos >= MIN_TRADES and mejora >= MIN_IMPROVEMENT and plateau)
    motivo = ("mejora fuera de muestra sostenida" if acepta else
              "pocas operaciones fuera de muestra" if best.n_oos < MIN_TRADES else
              "la mejora fuera de muestra es < 0,25 pp" if mejora < MIN_IMPROVEMENT else
              "es un pico aislado (vecinas peores que la base)")
    return dict(aceptada=acepta, motivo=motivo, mejor=best, base=base, mejora_oos=mejora)
