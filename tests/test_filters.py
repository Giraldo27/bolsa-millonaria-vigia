import numpy as np
import pandas as pd
import pytest

from src.indicators import indicator_frame, passes_filter, rsi
from tests.helpers import FLAT, candles, inputs, make_cfg, run

E = "ECOPETROL"


def long_drop(n_flat: int = 80, close: float = 96.0):
    return candles([FLAT] * n_flat + [(100, 100, close, close)] + [(close, close + 1, close - 0.5, close)] * 3, total=n_flat + 10)


def trades(cfg, df):
    return run(cfg, inputs({E: df}), "t_a").trades


def test_rsi_extremes_and_no_look_ahead():
    up = pd.Series(np.arange(1.0, 60.0))
    assert rsi(up).iloc[-1] == pytest.approx(100.0)
    down = pd.Series(np.arange(60.0, 1.0, -1.0))
    assert rsi(down).iloc[-1] == pytest.approx(0.0, abs=1e-9)
    s = pd.Series(np.random.default_rng(0).normal(0, 1, 200).cumsum() + 100)
    pd.testing.assert_series_equal(rsi(s).iloc[:120], rsi(s.iloc[:120]))      # el futuro no cambia el pasado


def test_trend_filters_logic():
    row = dict(close=96.0, sma20=99.0, sma50=100.0, rsi=20.0)
    assert not passes_filter(row, {"trend": "close_gt_sma50"})
    assert passes_filter(row, {"trend": "close_lt_sma50"})
    assert not passes_filter(row, {"trend": "sma20_gt_sma50"})
    assert passes_filter(row, {"rsi_max": 30}) and not passes_filter(row, {"rsi_min": 30})
    assert passes_filter(row, None) and passes_filter(row, {})
    assert not passes_filter(dict(close=96.0, sma20=np.nan, sma50=np.nan, rsi=20.0), {"trend": "close_gt_sma20"})


def test_default_has_no_filters_and_signal_fires():
    assert len(trades(make_cfg(), long_drop())) == 1


def test_trend_filter_blocks_dip_below_sma20():
    cfg = make_cfg()
    cfg["filters"] = {"A": {"trend": "close_gt_sma20"}}
    assert trades(cfg, long_drop()).empty
    cfg["filters"] = {"A": {"trend": "close_lt_sma50"}}
    assert len(trades(cfg, long_drop())) == 1


def test_rsi_filter_blocks_when_not_oversold_enough():
    cfg = make_cfg()
    cfg["filters"] = {"A": {"rsi_min": 30}}                  # exige RSI >= 30, pero tras la caída el RSI ~ 0
    assert trades(cfg, long_drop()).empty
    cfg["filters"] = {"A": {"rsi_max": 30}}
    assert len(trades(cfg, long_drop())) == 1


def test_filter_without_enough_history_is_conservative():
    cfg = make_cfg()
    cfg["filters"] = {"A": {"trend": "close_lt_sma50"}}
    assert trades(cfg, candles([FLAT] * 10 + [(100, 100, 96, 96)] + [(96, 97, 95, 96)] * 3)).empty


def test_per_asset_filters():
    cfg = make_cfg()
    cfg["filters"] = {"A": {"TSLA": {"rsi_min": 30}}}        # el filtro de TSLA no afecta a ECOPETROL
    assert len(trades(cfg, long_drop())) == 1
    cfg["filters"] = {"A": {E: {"rsi_min": 30}}}
    assert trades(cfg, long_drop()).empty


def test_enabled_assets_per_setup():
    cfg = make_cfg()
    cfg["setups"]["enabled_assets"] = {"A": ["TSLA"]}
    assert trades(cfg, long_drop()).empty
    cfg["setups"]["enabled_assets"] = {"A": [E]}
    assert len(trades(cfg, long_drop())) == 1
