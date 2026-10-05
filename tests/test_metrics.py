import numpy as np
import pandas as pd
import pytest

from src.benchmarks import bh_window_return, cash_window_return, random_strategy_mc
from src.metrics import bootstrap_expectancy, equity_stats, trade_stats, window_stats
from src.windows import contest_windows, rolling_windows
from tests.helpers import candles, dates, inputs, make_cfg, run, drop_then

E = "ECOPETROL"


def test_equity_stats_known_values():
    idx = pd.bdate_range("2020-01-01", periods=5)
    eq = pd.Series([100, 110, 99, 120, 120.0], index=idx)
    s = equity_stats(eq)
    assert s["rent_neta"] == pytest.approx(0.2)
    assert s["max_dd"] == pytest.approx(99 / 110 - 1)


def test_trade_stats_payoff_profit_factor_expectancy():
    t = pd.DataFrame({"asset": ["TSLA"] * 4, "net_pnl": [200.0, 100.0, -100.0, -50.0], "ret_net": [0.04, 0.02, -0.02, -0.01],
                      "sessions": [2, 2, 3, 1], "spread_cost": 1.0, "slippage_cost": 1.0, "fees": 1.0,
                      "gross_pnl": [210.0, 110, -90, -40]})
    s = trade_stats(t, make_cfg())
    assert s["win_rate"] == 0.5 and s["win_rate_min"] == 0.43
    assert s["payoff"] == pytest.approx(0.03 / 0.015)
    assert s["profit_factor"] == pytest.approx(300 / 150)
    assert s["expectancy_pct"] == pytest.approx(0.0075)
    assert s["expectancy_cop_40m"] == pytest.approx(0.0075 * 40_000_000)
    assert s["costos_cop"] == 12 and s["costos_pct_bruto"] == pytest.approx(12 / 190)


def test_window_stats_thresholds():
    s = window_stats(pd.Series([-0.05, 0.0, 0.02, 0.04, 0.12]))
    assert s["pct_gt0"] == 0.6 and s["pct_gt3"] == 0.4 and s["pct_gt10"] == 0.2


def test_bootstrap_detects_clear_edge_and_noise():
    rng = np.random.default_rng(0)
    edge = bootstrap_expectancy(pd.Series(rng.normal(0.02, 0.01, 200)))
    assert edge["ic_lo"] > 0 and edge["p_t"] < 0.001 and edge["p_boot"] < 0.01
    noise = bootstrap_expectancy(pd.Series(rng.normal(0.0, 0.02, 200)))
    assert noise["ic_lo"] < 0 < noise["ic_hi"] and noise["p_t"] > 0.05


def test_cash_window_return_is_only_activity_cost():
    assert cash_window_return(make_cfg(), 23) == pytest.approx(-23 * 14_875 / 100_000_000)


def test_buy_and_hold_window_return_includes_all_costs():
    cfg = make_cfg()
    inp = inputs({E: candles([(100, 100, 100, 100)] * 40)})
    cal = dates(40)
    r = bh_window_return(cfg, inp, {E: 1.0}, cal[0], cal[22], 23)
    v = (100_000_000 - 119_000 * 0) / 1.002975                       # compra 0,2975 % de comisión incluida
    buy_px, sell_px = 100 * 1.002, 100 * 0.997
    proceeds = v * sell_px / buy_px
    expected = (proceeds - proceeds * 0.002975 - 22 * 14_875) / 100_000_000 - 1
    assert r == pytest.approx(expected)


def test_windows_helpers():
    cal = pd.bdate_range("2024-01-01", periods=60)
    w = rolling_windows(cal, 23, "2024-01-01")
    assert len(w) == 38 and (w[0][1] - w[0][0]).days > 0
    cfg = make_cfg()
    cw = contest_windows(pd.bdate_range("2023-09-01", "2023-12-31"), range(2023, 2024), cfg)
    assert cw[0][0] == pd.Timestamp("2023-10-05") and cw[0][1] == pd.Timestamp("2023-11-06")


def test_random_strategy_mc_shape_and_zero_edge_on_flat_prices():
    cfg = make_cfg()
    inp = inputs({E: drop_then([(97, 102.5, 96.5, 101)], total=60)})
    tr = run(cfg, inp, "t_a").trades
    mc = random_strategy_mc(cfg, inp, tr, pd.Timestamp("2024-01-02"), n_runs=200)
    assert mc["mean_ret"].shape == (200,)
    flat = inputs({E: candles([(100, 100, 100, 100)] * 60, total=60)})
    mc = random_strategy_mc(cfg, flat, tr, pd.Timestamp("2024-01-02"), n_runs=50)
    assert (mc["mean_ret"] < 0).all()                                # precios planos: sólo se pagan costos
