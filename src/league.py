"""Simulador de ligas: 1.000 participantes sintéticos con portafolios concentrados de 2 a 4 activos del universo
permitido (comprar y mantener, pesos iguales) y la matriz dinámica de modos de riesgo.

Supuestos (ver ASSUMPTIONS.md): el percentil se calcula sobre la rentabilidad acumulada desde el inicio del concurso,
P = % de participantes por delante de ti (P pequeño = vas liderando), y se observa a diario."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .data_loader import currency_of, load_macro, load_prices
from .universe import fx_to_cop


def build_cop_prices(cfg: dict[str, Any], cal: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Open y Close en COP de todo el universo permitido (sin lista negra), alineados al calendario `cal`.
    Los festivos propios de cada mercado se rellenan con el último precio (máx. 5 sesiones)."""
    u = cfg["universe"]
    usdcop, gbpusd = load_macro("usdcop", cfg)["Close"], load_macro("gbpusd", cfg)["Close"]
    opens, closes = {}, {}
    for t in list(u["mgc"]) + list(u["etf"]) + list(u["local"]):
        df = load_prices(t, cfg)
        if df.empty:
            continue
        fx = fx_to_cop(t, df, currency_of(t, cfg), usdcop, gbpusd)
        if fx.isna().all():
            continue
        px = df[["Open", "Close"]].mul(fx, axis=0)
        px = px.reindex(px.index.union(cal)).ffill(limit=5).reindex(cal)
        opens[t], closes[t] = px["Open"], px["Close"]
    return pd.DataFrame(opens), pd.DataFrame(closes)


def sample_participants(n_assets: int, n: int = 1000, seed: int = 2026) -> np.ndarray:
    """Matriz (n x n_assets) de pesos: cada participante elige 2, 3 o 4 activos al azar, pesos iguales."""
    rng = np.random.default_rng(seed)
    w = np.zeros((n, n_assets))
    for i in range(n):
        k = rng.integers(2, 5)
        w[i, rng.choice(n_assets, k, replace=False)] = 1.0 / k
    return w


def participant_paths(w: np.ndarray, opens: np.ndarray, closes: np.ndarray, entry_cost: float,
                      daily_activity: float) -> np.ndarray:
    """Rentabilidad acumulada (n x sesiones) de los participantes en una ventana.
    opens/closes: (sesiones x activos) en COP. Compran a la apertura de la sesión 1 (costo de entrada) y pagan
    la micro-compra de actividad desde la sesión 2."""
    ok = ~np.isnan(opens[0]) & ~np.isnan(closes).any(axis=0)         # activos con datos completos en la ventana
    wa = w * ok
    tot = wa.sum(axis=1)
    rets = np.nan_to_num(closes[:, ok] / opens[0, ok] - 1)           # (sesiones x activos válidos)
    path = (wa[:, ok] @ rets.T) / np.where(tot > 0, tot, np.nan)[:, None]
    penalty = daily_activity * np.arange(path.shape[1])
    return path - entry_cost - penalty


def dynamic_mode(P: float, T: float, D: int) -> str:
    """Matriz dinámica: Blindaje si P<=T/2; Mantener si P<=T; Ataque si P<=2T; All-In si P>2T, o si P>T y D<=2."""
    if P > 2 * T or (P > T and D <= 2):
        return "allin"
    if P <= T / 2:
        return "blindaje"
    if P <= T:
        return "mantener"
    return "ataque"


def percentile_from_top(mine: float, field: np.ndarray) -> float:
    f = field[~np.isnan(field)]
    return 100.0 * float((f > mine).mean())
