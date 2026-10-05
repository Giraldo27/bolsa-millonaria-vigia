"""Simulador multi-activo para ventanas de concurso: núcleo de momentum/tendencia + señal de reacción a resultados (PEAD).

Sin look-ahead: toda decisión del cierre del día t usa sólo datos hasta t y se ejecuta a la apertura de t+1.
Costos: spread de entrada, slippage en salidas a mercado (stop, rotación, sustitución), comisión de trii (escalonada).
Al final de la ventana las posiciones se valoran a mercado (sin costo de venta); `liquidate=True` descuenta ese costo."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd

from .costs import commission, gross_from_budget
from .panel import Panel


@dataclass(frozen=True)
class Params:
    L: int = 60                      # lookback del momentum (sesiones)
    skip: int = 0                    # sesiones recientes que se omiten del momentum
    N: int = 2                       # nº de posiciones del núcleo
    universe: str = "liq"            # clave de universos (ver make_universes)
    K: int = 0                       # rotación cada K sesiones (0 = comprar y mantener)
    stop: float | None = 0.10        # stop duro bajo el precio de compra
    trail: float | None = None       # stop móvil bajo el máximo cierre desde la compra
    trend: str = "sma50"             # none | sma50 | sma100 : sólo activos con cierre sobre su media
    regime: bool = False             # sólo comprar si el Nasdaq-100 cierra sobre su media de 100 días
    voladj: bool = False             # ordena por momentum / volatilidad
    size: float = 1.0                # fracción del patrimonio invertida entre las posiciones
    min_mom: float = 0.0             # momentum mínimo para comprar
    core: bool = True                # False = sólo señales PEAD (sin núcleo de momentum)
    pead: float | None = None        # umbral de reacción al reporte (p. ej. 0,05) para abrir una posición PEAD
    pead_universe: str | None = None # universo de la señal PEAD (None = el mismo del núcleo)
    n_total: int | None = None       # máximo de posiciones simultáneas (núcleo + PEAD); None = N
    replace: bool = True             # una señal PEAD sin cupo reemplaza a la posición del núcleo con peor rendimiento
    pead_hold: int = 20              # sesiones máximas de una posición PEAD
    vol_top: int | None = None       # núcleo: sólo los k activos de mayor volatilidad del universo
    vol_window: int = 20             # ventana (sesiones) de la volatilidad usada por vol_top: 20, 60 o 250


@dataclass
class Features:
    ret: np.ndarray
    sma50_ok: np.ndarray
    sma100_ok: np.ndarray
    vol20: np.ndarray
    ndx_ok: np.ndarray
    mom: dict = field(default_factory=dict)          # (L, skip) -> (T x A)
    pead: np.ndarray | None = None                   # (T x A) reacción al reporte en la sesión de reacción
    vol: dict = field(default_factory=dict)          # ventana -> (T x A) desviación estándar de retornos diarios


def make_features(p: Panel, lookbacks: list[tuple[int, int]], pead: np.ndarray | None = None) -> Features:
    c = pd.DataFrame(p.C, index=p.dates)
    ret = c.pct_change()
    ndx = pd.Series(p.ndx, index=p.dates)
    f = Features(ret=ret.values, sma50_ok=(c > c.rolling(50).mean()).values, sma100_ok=(c > c.rolling(100).mean()).values,
                 vol20=ret.rolling(20).std().values, ndx_ok=(ndx > ndx.rolling(100).mean()).values, pead=pead)
    for w in (20, 60, 250):
        f.vol[w] = ret.rolling(w).std().values
    for L, sk in lookbacks:
        f.mom[(L, sk)] = (c.shift(sk) / c.shift(sk + L) - 1).values
    return f


def make_universes(p: Panel, cfg: dict[str, Any]) -> dict[str, np.ndarray]:
    t = p.tickers
    mgc = np.array([g == "mgc" for g in p.group])
    ex = ~np.array([x in ("NVDA", "TSLA") for x in t])
    out = {"liq": np.ones(len(t), bool), "mgc": mgc, "mgc_etf": np.array([g in ("mgc", "etf") for g in p.group]),
           "sys3": np.array([x in cfg["universe"]["system_assets"] for x in t]), "sin_nvda_tsla": ex,
           "mgc_sin_nvda_tsla": mgc & ex}
    for x in t:
        out[x] = np.array([y == x for y in t])
    return out


def pick(p: Panel, f: Features, prm: Params, uni: np.ndarray, t: int) -> list[int]:
    """Activos del núcleo tras el cierre de la sesión t. Usa sólo información hasta t."""
    if prm.regime and not f.ndx_ok[t]:
        return []
    mom = f.mom[(prm.L, prm.skip)][t]
    score = mom / np.where(f.vol20[t] > 0, f.vol20[t], np.nan) if prm.voladj else mom
    ok = uni & np.isfinite(score) & (mom > prm.min_mom) & np.isfinite(p.C[t])
    if prm.vol_top:
        vv = f.vol[prm.vol_window][t]
        v = np.where(uni & np.isfinite(vv), vv, -np.inf)
        ok &= v >= np.sort(v)[-prm.vol_top]
    if prm.trend == "sma50":
        ok &= f.sma50_ok[t]
    elif prm.trend == "sma100":
        ok &= f.sma100_ok[t]
    cand = np.where(ok)[0]
    if cand.size == 0:
        return []
    order = cand[np.argsort(-score[cand])]
    return list(order[: prm.N])


def pead_signals(p: Panel, f: Features, prm: Params, uni: np.ndarray, t: int) -> list[int]:
    """Activos con reacción >= umbral en la sesión t (el más fuerte primero)."""
    if prm.pead is None or f.pead is None or (prm.regime and not f.ndx_ok[t]):
        return []
    r0 = f.pead[t]
    ok = uni & np.isfinite(r0) & (r0 >= prm.pead) & np.isfinite(p.C[t])
    cand = np.where(ok)[0]
    return list(cand[np.argsort(-r0[cand])])


def simulate(p: Panel, f: Features, uni: np.ndarray, prm: Params, s: int, n: int, cfg: dict[str, Any],
             capital: float = 1e8, liquidate: bool = False, scale=None, uni_pead: np.ndarray | None = None,
             checkpoint=None, checkpoint_days: tuple[int, ...] = (5, 10, 15, 20)) -> np.ndarray:
    """Retorno acumulado diario (n,) de una ventana que empieza en la sesión s.
    scale(j, rentabilidad_actual) -> factor que reescala `size` al comprar (matriz dinámica de riesgo)."""
    c = cfg["costs"]
    act = c["activity_micro_buy_cop"]
    n_total = prm.n_total or prm.N
    up = uni_pead if uni_pead is not None else uni
    cash = capital
    pos: dict[int, list] = {}                  # a -> [acciones, px_compra, stop, máx_cierre, etiqueta, sesiones]
    path = np.zeros(n)
    pending: list[tuple[int, str]] = []        # (activo, etiqueta) a comprar en la próxima apertura
    adds: dict[int, float] = {}                # compras adicionales (presupuesto) a la próxima apertura

    def equity(d: int) -> float:
        return cash + sum(v[0] * p.C[d, a] for a, v in pos.items())

    def sell(a: int, d: int, ref: float) -> None:
        nonlocal cash
        proceeds = pos[a][0] * ref * (1 - p.slip[a])
        cash += proceeds - commission(proceeds, c)
        del pos[a]

    if prm.core:                               # decisión del núcleo al cierre de s-1, compra a la apertura de s
        pending += [(a, "core") for a in pick(p, f, prm, uni, s - 1)]
    for a in pead_signals(p, f, prm, up, s - 1):
        if len(pending) + len(pos) < n_total and a not in {x for x, _ in pending}:
            pending.append((a, "pead"))

    for j in range(n):
        d = s + j
        bought = False
        if pending:
            eq = equity(d - 1) if j else capital
            sc = scale(j, eq / capital - 1) if scale else 1.0
            frac, n_use = (sc if isinstance(sc, tuple) else (sc, n_total))      # (fracción invertida, nº de posiciones)
            budget_each = prm.size * frac * eq / max(n_use, 1)
            for a, tag in pending[: max(n_use - len(pos), 0)]:
                if a in pos or not np.isfinite(p.O[d, a]):
                    continue
                budget = min(budget_each, cash)
                if budget < c["min_position_cop"]:
                    continue
                v = gross_from_budget(budget, c)
                px = p.O[d, a] * (1 + p.spread[a])
                cash -= v + commission(v, c)
                stop = px * (1 - prm.stop) if (prm.stop and tag == "core") else 0.0
                pos[a] = [v / px, px, stop, px, tag, 0]
                bought = True
            pending = []
        for a, budget in adds.items():                              # compras adicionales decididas en el corte previo
            budget = min(budget, cash)
            if a in pos and np.isfinite(p.O[d, a]) and budget >= c["min_position_cop"]:
                v = gross_from_budget(budget, c)
                pos[a][0] += v / (p.O[d, a] * (1 + p.spread[a]))
                cash -= v + commission(v, c)
                bought = True
        adds = {}
        for a in list(pos):
            sh, ent, stp, hi, tag, ses = pos[a]
            pos[a][5] += 1
            lo, op = p.L[d, a], p.O[d, a]
            if np.isfinite(lo) and stp > 0 and lo <= stp:
                sell(a, d, op if op < stp else stp)
                continue
            cl = p.C[d, a]
            if np.isfinite(cl) and prm.trail and tag == "core":
                pos[a][3] = max(hi, cl)
                pos[a][2] = max(stp, pos[a][3] * (1 - prm.trail))
            if tag == "pead" and pos[a][5] >= prm.pead_hold and j < n - 1:
                sell(a, d, p.C[d, a])
        if not bought:
            cash -= act
        if j < n - 1:
            if prm.K and prm.core and (j + 1) % prm.K == 0:
                want = pick(p, f, prm, uni, d)
                for a in list(pos):
                    if pos[a][4] == "core" and a not in want:      # rotación: se vende al cierre a mercado
                        sell(a, d, p.C[d, a])
                pending = [(a, "core") for a in want if a not in pos]
            for a in pead_signals(p, f, prm, up, d):               # señal de reacción a resultados
                if a in pos or a in {x for x, _ in pending}:
                    continue
                if len(pos) + len(pending) < n_total:
                    pending.append((a, "pead"))
                elif prm.replace:
                    cores = [b for b, v in pos.items() if v[4] == "core"]
                    if cores:
                        worst = min(cores, key=lambda b: p.C[d, b] / pos[b][1])
                        sell(worst, d, p.C[d, worst])
                        pending.append((a, "pead"))
        if checkpoint is not None and (j + 1) in checkpoint_days and j < n - 1 and pos:
            tgt = checkpoint(j, equity(d) / capital - 1)           # exposición objetivo en el corte semanal
            if tgt is not None:
                eq = equity(d)
                cur = sum(v[0] * p.C[d, a] for a, v in pos.items()) / eq if eq > 0 else 0.0
                if tgt < cur - 0.02:                                # proteger: se vende una parte al cierre (a mercado)
                    keep = tgt / cur
                    for a in list(pos):
                        sh_sell = pos[a][0] * (1 - keep)
                        proceeds = sh_sell * p.C[d, a] * (1 - p.slip[a])
                        cash += proceeds - commission(proceeds, c)
                        pos[a][0] -= sh_sell
                elif tgt > cur + 0.02:                              # atacar: se compra más a la próxima apertura
                    adds = {a: (tgt - cur) * eq / len(pos) for a in pos}
        eq_d = equity(d)
        if liquidate and j == n - 1:
            eq_d = cash + sum(v[0] * p.C[d, a] * (1 - p.slip[a]) - commission(v[0] * p.C[d, a], c) for a, v in pos.items())
        path[j] = eq_d / capital - 1
    return path


def with_(prm: Params, **kw) -> Params:
    return replace(prm, **kw)
