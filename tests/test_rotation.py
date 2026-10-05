import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.evalwin import Field, percentile_paths, summarize_paths
from src.panel import Panel
from src.rotation import Params, make_features, make_universes, pick, simulate

CFG = load_config()
T, A = 400, 3
IDX = pd.bdate_range("2022-01-03", periods=T)


def panel(close: np.ndarray, high=None, low=None, open_=None) -> Panel:
    c = close.astype(float)
    o = c.copy() if open_ is None else open_
    h = c.copy() if high is None else high
    lo = c.copy() if low is None else low
    return Panel(IDX, ["A", "B", "C"], o, h, lo, c, np.full(A, 0.003), np.full(A, 0.004), ["mgc"] * A,
                 np.linspace(100, 200, T), np.full(T, 80.0))


def trending(slopes=(0.002, 0.0005, -0.001)) -> np.ndarray:
    return np.stack([100 * np.exp(np.arange(T) * s) for s in slopes], axis=1)


def feats(p, L=60):
    return make_features(p, [(L, 0)])


def test_pick_chooses_highest_momentum_and_respects_trend_and_N():
    p = panel(trending())
    f = feats(p)
    prm = Params(L=60, N=2, universe="liq", trend="sma50")
    assert pick(p, f, prm, make_universes(p, CFG)["liq"], 300) == [0, 1]       # el activo en caída no entra
    assert pick(p, f, Params(L=60, N=1, trend="none"), make_universes(p, CFG)["liq"], 300) == [0]


def test_pick_has_no_look_ahead():
    c = trending()
    p1, c2 = panel(c), c.copy()
    c2[301:] *= 0.1                                                        # el futuro cambia radicalmente
    p2 = panel(c2)
    u = make_universes(p1, CFG)["liq"]
    prm = Params(L=60, N=2)
    assert pick(p1, feats(p1), prm, u, 300) == pick(p2, feats(p2), prm, u, 300)


def test_regime_filter_blocks_when_nasdaq_below_average():
    p = panel(trending())
    p.ndx[:] = np.linspace(200, 100, T)                                     # Nasdaq en caída
    f = feats(p)
    assert pick(p, f, Params(regime=True), make_universes(p, CFG)["liq"], 300) == []


def test_buy_and_hold_flat_prices_pays_entry_costs_and_activity_only():
    p = panel(np.full((T, A), 100.0))
    f = feats(p)
    prm = Params(L=60, N=1, trend="none", stop=None, min_mom=-1.0, universe="liq")
    path = simulate(p, f, make_universes(p, CFG)["liq"], prm, 300, 23, CFG)
    # compra de 100M a la apertura con spread 0,3 % + comisión 0,2975 %; 22 sesiones sin compra pagan 14.875
    v = 100e6 / 1.002975
    expected = (v / 1.003 * 100 / 100 + (100e6 - 100e6)) / 1e8                # valor de mercado de lo comprado
    assert path[0] == pytest.approx(v / 1.003 / 1e8 - 1, abs=1e-6)
    assert path[-1] == pytest.approx(v / 1.003 / 1e8 - 1 - 22 * 14_875 / 1e8, abs=1e-6)


def test_hard_stop_exits_at_stop_with_slippage_and_gap_uses_open():
    c = np.full((T, A), 100.0)
    c[:, 0] = 100 * np.exp(np.arange(T) * 0.003)
    p = panel(c)
    f = feats(p)
    u = make_universes(p, CFG)["liq"]
    s = 300
    base = p.C[s - 1, 0]
    entry = p.O[s, 0] * 1.003
    # día 2 de la ventana: el mínimo cruza el stop (10 %), sin gap
    lo = p.L.copy(); lo[s + 2, 0] = entry * 0.85
    p2 = panel(c, low=lo)
    path = simulate(p2, f, u, Params(L=60, N=1, stop=0.10, trend="none"), s, 5, CFG)
    # tras el stop el activo ya no está en cartera: el patrimonio queda fijo (sólo baja por actividad)
    assert path[3] < path[1] and path[4] == pytest.approx(path[3] - 14_875 / 1e8, abs=1e-9)
    from src.costs import commission, gross_from_budget
    v = gross_from_budget(1e8, CFG["costs"])
    proceeds = v * 0.90 * 0.996                                           # venta al stop con slippage de 0,4 %
    cash = 1e8 - v - commission(v, CFG["costs"]) + proceeds - commission(proceeds, CFG["costs"]) - 2 * 14_875
    assert path[2] == pytest.approx(cash / 1e8 - 1, abs=1e-9)


def test_rotation_sells_losers_and_buys_leaders():
    c = np.full((T, A), 100.0)
    c[:, 0] = 100 * np.exp(np.arange(T) * 0.006)
    c[300:, 0] = c[299, 0] * np.exp(-0.05 * np.arange(1, T - 299))            # el líder se desploma desde la sesión 300
    c[:, 1] = 100 * np.exp(np.arange(T) * 0.003)
    p = panel(c)
    f = feats(p)
    u = make_universes(p, CFG)["liq"]
    prm = Params(L=20, N=1, K=5, trend="none", stop=None)
    f = feats(p, L=20)
    path_rot = simulate(p, f, u, prm, 296, 23, CFG)
    path_hold = simulate(p, f, u, Params(L=20, N=1, K=0, trend="none", stop=None), 296, 23, CFG)
    assert path_rot[-1] > path_hold[-1]


def pead_setup(r0_day=302, r0=0.06):
    c = np.full((T, A), 100.0)
    c[:, 0] = 100 * np.exp(np.arange(T) * 0.004)                         # núcleo: A sube de forma constante
    c[303:, 1] = 100 * np.exp(np.arange(T - 303) * 0.01)                  # B sube con fuerza tras la señal
    p = panel(c)
    m = np.full((T, A), np.nan)
    m[r0_day, 1] = r0
    return p, make_features(p, [(60, 0)], pead=m)


def test_pead_buys_next_open_after_signal_and_not_before():
    p, f = pead_setup()
    u = make_universes(p, CFG)["liq"]
    prm = Params(L=60, N=1, core=False, pead=0.05, n_total=1, stop=None, trend="none")
    path = simulate(p, f, u, prm, 300, 10, CFG)
    assert path[:3].max() <= 0.0 and path[2] == pytest.approx(-3 * 14_875 / 1e8, abs=1e-9)   # sin señal: sólo actividad
    assert path[-1] > 0.02                                                                     # entra el día 3 y captura la subida


def test_pead_ignores_reactions_below_threshold():
    p, f = pead_setup(r0=0.04)
    u = make_universes(p, CFG)["liq"]
    path = simulate(p, f, u, Params(core=False, pead=0.05, n_total=1, N=1), 300, 10, CFG)
    assert path[-1] == pytest.approx(-10 * 14_875 / 1e8, abs=1e-9)


def test_pead_replaces_weakest_core_position_when_full():
    p, f = pead_setup()
    u = make_universes(p, CFG)["liq"]
    prm = Params(L=60, N=1, pead=0.05, n_total=1, stop=None, trend="none", replace=True)
    no_replace = Params(L=60, N=1, pead=0.05, n_total=1, stop=None, trend="none", replace=False)
    assert simulate(p, f, u, prm, 300, 10, CFG)[-1] > simulate(p, f, u, no_replace, 300, 10, CFG)[-1]


def test_pead_has_no_look_ahead():
    p, f = pead_setup()
    u = make_universes(p, CFG)["liq"]
    f2 = make_features(p, [(60, 0)], pead=np.where(np.arange(T)[:, None] > 305, 0.5, f.pead))     # señal futura distinta
    prm = Params(core=False, pead=0.05, n_total=1, N=1)
    assert np.allclose(simulate(p, f, u, prm, 300, 4, CFG), simulate(p, f2, u, prm, 300, 4, CFG))


def test_pead_position_is_closed_after_hold_limit():
    p, f = pead_setup()
    u = make_universes(p, CFG)["liq"]
    prm = Params(core=False, pead=0.05, n_total=1, N=1, pead_hold=3)
    path = simulate(p, f, u, prm, 300, 12, CFG)
    # tras vender (sesión 3 de la posición) el patrimonio ya no sigue la subida de B: crece poco desde ahí
    assert path[-1] < simulate(p, f, u, Params(core=False, pead=0.05, n_total=1, N=1, pead_hold=20), 300, 12, CFG)[-1]


def _flat_then_up():
    c = np.full((T, A), 100.0)
    c[:, 0] = 100 * np.exp(np.arange(T) * 0.004)
    p = panel(c)
    return p, make_features(p, [(60, 0)]), make_universes(p, CFG)["liq"]


def test_checkpoint_reduces_exposure_and_pays_costs():
    p, f, u = _flat_then_up()
    prm = Params(L=60, N=1, stop=None, trend="none")
    hold = simulate(p, f, u, prm, 300, 12, CFG)
    prot = simulate(p, f, u, prm, 300, 12, CFG, checkpoint=lambda j, r: 0.4, checkpoint_days=(5,))
    assert np.allclose(prot[:4], hold[:4])                       # antes del corte (al cierre de la sesión 5) son iguales
    assert prot[4] < hold[4]                                      # la venta cuesta slippage y comisión
    assert prot[-1] < hold[-1]                                    # menos exposición en una tendencia alcista: menos ganancia


def test_checkpoint_none_keeps_position_and_increase_buys_more():
    p, f, u = _flat_then_up()
    prm = Params(L=60, N=1, stop=None, trend="none", size=0.5)
    base = simulate(p, f, u, prm, 300, 12, CFG)
    same = simulate(p, f, u, prm, 300, 12, CFG, checkpoint=lambda j, r: None, checkpoint_days=(5,))
    assert np.allclose(base, same)
    more = simulate(p, f, u, prm, 300, 12, CFG, checkpoint=lambda j, r: 1.0, checkpoint_days=(5,))
    assert more[-1] > base[-1]                                    # sube la exposición de 50 % a 100 % en tendencia alcista


def test_percentiles_and_summary():
    n, m = 23, 100
    srt = np.tile(np.linspace(-0.1, 0.1, m, dtype=np.float32), (4, n, 1))
    f = Field(np.arange(4), pd.date_range("2020-01-01", periods=4), srt, n)
    paths = np.full((4, n), 0.0)
    p = percentile_paths(f, paths, [4, 22])
    assert p[0, 0] == pytest.approx(50.0, abs=1.0)
    best = np.full((4, n), 0.5)
    s = summarize_paths(f, best, {5: 30, 10: 30, 15: 10, 20: 10, 23: 10})
    assert s["top10"] == 1.0 and s["primero"] == 1.0 and s["pasa_20"] == 1.0
    worst = np.full((4, n), -0.5)
    assert summarize_paths(f, worst, {5: 30, 23: 10})["top30"] == 0.0
