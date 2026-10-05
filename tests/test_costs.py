import pytest

from src.config import load_config
from src.costs import commission, entry_price, gross_from_budget, market_exit_price

C = load_config()["costs"]


def test_commission_fixed_up_to_5m_inclusive():
    assert commission(1_000_000, C) == 14_875
    assert commission(5_000_000, C) == 14_875


def test_commission_percentage_above_5m():
    assert commission(5_000_001, C) == pytest.approx(5_000_001 * 0.002975)
    assert commission(40_000_000, C) == pytest.approx(119_000)


def test_commission_includes_iva():
    assert 12_500 * 1.19 == pytest.approx(14_875)
    assert 0.0025 * 1.19 == pytest.approx(0.002975)


@pytest.mark.parametrize("budget", [6_000_000, 5_010_000, 20_000_000, 40_000_000, 100_000_000])
def test_gross_plus_fee_equals_budget(budget):
    v = gross_from_budget(budget, C)
    assert v + commission(v, C) == pytest.approx(budget, rel=1e-6)


def test_entry_spread_and_slippage_per_asset():
    assert entry_price(100, "ECOPETROL", C) == pytest.approx(100.2)
    assert entry_price(100, "TSLA", C) == pytest.approx(100.3)
    assert entry_price(100, "NVDA", C) == pytest.approx(100.3)
    assert market_exit_price(100, "ECOPETROL", C) == pytest.approx(99.7)
    assert market_exit_price(100, "TSLA", C) == pytest.approx(99.6)
    assert market_exit_price(100, "NVDA", C) == pytest.approx(99.6)
