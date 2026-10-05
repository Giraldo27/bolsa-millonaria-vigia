import numpy as np
import pytest

from src.league import dynamic_mode, participant_paths, percentile_from_top, sample_participants


def test_dynamic_matrix_bronze_silver_T30():
    T = 30
    assert dynamic_mode(10, T, 10) == "blindaje"        # P <= 15
    assert dynamic_mode(15, T, 10) == "blindaje"
    assert dynamic_mode(20, T, 10) == "mantener"        # 15 < P <= 30
    assert dynamic_mode(30, T, 10) == "mantener"
    assert dynamic_mode(45, T, 10) == "ataque"          # 30 < P <= 60
    assert dynamic_mode(61, T, 10) == "allin"           # P > 2T
    assert dynamic_mode(45, T, 2) == "allin"            # P > T y quedan <= 2 sesiones
    assert dynamic_mode(45, T, 3) == "ataque"
    assert dynamic_mode(20, T, 1) == "mantener"         # dentro de T no se arriesga aunque quede poco


def test_dynamic_matrix_gold_T10():
    assert dynamic_mode(4, 10, 5) == "blindaje" and dynamic_mode(8, 10, 5) == "mantener"
    assert dynamic_mode(15, 10, 5) == "ataque" and dynamic_mode(25, 10, 5) == "allin"


def test_participants_have_2_to_4_equal_weight_assets():
    w = sample_participants(30, 500)
    k = (w > 0).sum(axis=1)
    assert k.min() == 2 and k.max() == 4 and np.allclose(w.sum(axis=1), 1.0)
    assert np.array_equal(w, sample_participants(30, 500))          # reproducible


def test_participant_paths_equal_weights_and_costs():
    w = np.array([[0.5, 0.5, 0.0], [0.0, 0.0, 1.0]])
    opens = np.array([[100.0, 100.0, 50.0], [0, 0, 0], [0, 0, 0]])
    closes = np.array([[110.0, 100.0, 50.0], [120.0, 100.0, 55.0], [100.0, 100.0, 60.0]])
    p = participant_paths(w, opens, closes, entry_cost=0.01, daily_activity=0.001)
    assert p[0, 0] == pytest.approx(0.05 - 0.01) and p[0, 2] == pytest.approx(0.0 - 0.01 - 0.002)
    assert p[1, 2] == pytest.approx(0.2 - 0.01 - 0.002)


def test_participant_with_missing_asset_is_renormalised():
    w = np.array([[0.5, 0.5]])
    opens = np.array([[100.0, np.nan]])
    closes = np.array([[110.0, np.nan]])
    assert participant_paths(w, opens, closes, 0.0, 0.0)[0, 0] == pytest.approx(0.10)


def test_percentile_from_top_counts_participants_ahead():
    field = np.array([0.1, 0.2, 0.3, 0.4, np.nan])
    assert percentile_from_top(0.25, field) == pytest.approx(50.0)
    assert percentile_from_top(0.5, field) == 0.0 and percentile_from_top(0.0, field) == 100.0
