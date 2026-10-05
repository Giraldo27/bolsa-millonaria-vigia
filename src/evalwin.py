"""Evaluación de estrategias por ventanas de concurso frente a un campo de participantes sintéticos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .league import build_cop_prices, participant_paths, sample_participants


@dataclass
class Field:
    starts: np.ndarray            # índices de inicio (sobre el calendario)
    dates: pd.DatetimeIndex       # fecha de inicio de cada ventana
    sorted: np.ndarray            # (W x n x M) float32, rentabilidad ordenada de los M participantes
    n: int


def window_starts(cal: pd.DatetimeIndex, n: int, start: str, step: int) -> np.ndarray:
    first = int(cal.searchsorted(pd.Timestamp(start)))
    return np.arange(max(first, 300), len(cal) - n + 1, step)      # >=300 sesiones previas para los indicadores


def precompute_field(cfg: dict[str, Any], cal: pd.DatetimeIndex, starts: np.ndarray, n: int,
                     only_groups: tuple[str, ...] | None = None) -> Field:
    """Campo de participantes sintéticos. only_groups=('mgc',) restringe sus activos a las acciones MGC
    (campo 'agresivo': rivales que sólo operan acciones globales)."""
    lg = cfg["league"]
    opens, closes = build_cop_prices(cfg, cal)
    if only_groups:
        keep = [t for t in opens.columns if any(t in cfg["universe"][g] for g in only_groups)]
        opens, closes = opens[keep], closes[keep]
    w = sample_participants(opens.shape[1], lg["n_participants"], lg["seed"])
    entry = lg["participant_entry_cost"] + cfg["costs"]["pct_fee"]
    daily = cfg["costs"]["activity_micro_buy_cop"] / cfg["capital"]
    O, C = opens.values, closes.values
    out = np.zeros((len(starts), n, w.shape[0]), dtype=np.float32)
    for k, s in enumerate(starts):
        part = participant_paths(w, O[s:s + n], C[s:s + n], entry, daily)         # (M x n)
        part = np.where(np.isnan(part), -9.0, part)
        out[k] = np.sort(part, axis=0).T
    return Field(starts, cal[starts], out, n)


def percentile_paths(field: Field, paths: np.ndarray, days: list[int]) -> np.ndarray:
    """P (0 = primero, 100 = último) de cada ventana en los días indicados (índices 0-based). Devuelve (W x len(days))."""
    m = field.sorted.shape[2]
    out = np.zeros((paths.shape[0], len(days)))
    for k in range(paths.shape[0]):
        for q, d in enumerate(days):
            out[k, q] = 100.0 * (m - np.searchsorted(field.sorted[k, d], paths[k, d], side="right")) / m
    return out


def summarize_paths(field: Field, paths: np.ndarray, cut_T: dict[int, float]) -> dict[str, float]:
    cuts = sorted(cut_T)
    pc = percentile_paths(field, paths, [c - 1 for c in cuts])
    fin = paths[:, -1]
    passed = np.ones(len(fin), bool)
    out = dict(ret_medio=fin.mean(), ret_mediana=np.median(fin), pct_pos=(fin > 0).mean(), pct_gt3=(fin > 0.03).mean(),
               pct_gt10=(fin > 0.10).mean(), p10=np.percentile(fin, 10), p90=np.percentile(fin, 90),
               p_final=pc[:, -1].mean(), top30=(pc[:, -1] <= 30).mean(), top10=(pc[:, -1] <= 10).mean(),
               primero=(pc[:, -1] <= 100.0 / field.sorted.shape[2] + 1e-9).mean())
    for q, c in enumerate(cuts[:-1]):
        passed &= pc[:, q] <= cut_T[c]
        out[f"pasa_{c}"] = passed.mean()
    out["campeon"] = float((passed & (pc[:, -1] <= 100.0 / field.sorted.shape[2] + 1e-9)).mean())   # pasa todos los cortes y termina #1
    return out


def by_year(field: Field, paths: np.ndarray, cut_T: dict[int, float]) -> pd.DataFrame:
    yrs = field.dates.year
    rows = []
    for y in sorted(set(yrs)):
        sel = yrs == y
        r = summarize_paths(Field(field.starts[sel], field.dates[sel], field.sorted[sel], field.n), paths[sel], cut_T)
        r["anio"] = y
        rows.append(r)
    return pd.DataFrame(rows).set_index("anio")
