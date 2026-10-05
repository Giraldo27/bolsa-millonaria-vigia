import numpy as np
import pandas as pd
import pytest

from src.sweep import evaluate, override, pick_and_validate, smooth
from tests.helpers import make_cfg


def test_override_deep_copies_and_sets_nested_paths():
    cfg = make_cfg()
    new = override(cfg, assets__TSLA__tp=0.2, setups__A__time_stop=5)
    assert new["assets"]["TSLA"]["tp"] == 0.2 and cfg["assets"]["TSLA"]["tp"] == 0.08
    assert new["setups"]["A"]["time_stop"] == 5 and cfg["setups"]["A"]["time_stop"] == 3


def test_evaluate_splits_by_entry_date():
    t = pd.DataFrame({"entry_date": pd.to_datetime(["2020-01-02", "2021-05-05", "2022-01-03"]),
                      "ret_net": [0.01, 0.03, -0.02], "net_pnl": [1.0, 3.0, -2.0]})
    e = evaluate(t, pd.Timestamp("2022-01-01"))
    assert e["n_is"] == 2 and e["n_oos"] == 1
    assert e["exp_is"] == pytest.approx(0.02) and e["exp_oos"] == pytest.approx(-0.02)


def test_evaluate_handles_empty():
    e = evaluate(pd.DataFrame(columns=["entry_date", "ret_net", "net_pnl"]), pd.Timestamp("2022-01-01"))
    assert e["n_is"] == 0 and np.isnan(e["exp_oos"])


def test_smooth_penalises_isolated_peak():
    g = pd.DataFrame(np.zeros((5, 5)))
    g.iloc[2, 2] = 1.0
    s = smooth(g)
    assert s.iloc[2, 2] == pytest.approx(1 / 9) and s.iloc[0, 0] == pytest.approx(0.0)


def _grid(exp_is, exp_oos, n=100):
    rows = []
    for i in range(5):
        for j in range(5):
            rows.append(dict(tp=i, sl=j, n_is=n, n_oos=n, exp_is=exp_is[i][j], exp_oos=exp_oos[i][j]))
    return pd.DataFrame(rows)


def test_pick_rejects_in_sample_spike_that_fails_out_of_sample():
    is_ = np.full((5, 5), -0.01); oos = np.full((5, 5), -0.01)
    is_[3][3] = 0.05                                    # pico aislado dentro de muestra
    oos[3][3] = -0.03                                   # y malo fuera de muestra
    r = pick_and_validate(_grid(is_, oos), ["tp", "sl"], (0, 0), ("tp", "sl"))
    assert not r["aceptada"]


def test_pick_accepts_flat_zone_that_holds_out_of_sample():
    is_ = np.full((5, 5), -0.01); oos = np.full((5, 5), -0.01)
    is_[2:5, 2:5] = 0.02; oos[2:5, 2:5] = 0.01          # zona plana mejor en ambos tramos
    r = pick_and_validate(_grid(is_, oos), ["tp", "sl"], (0, 0), ("tp", "sl"))
    assert r["aceptada"] and r["mejora_oos"] >= 0.0025


def test_pick_requires_enough_trades():
    is_ = np.full((5, 5), 0.02); oos = np.full((5, 5), 0.02); oos[0][0] = -0.01
    r = pick_and_validate(_grid(is_, oos, n=10), ["tp", "sl"], (0, 0), ("tp", "sl"))
    assert not r["aceptada"] and "operaciones" in r["motivo"]
