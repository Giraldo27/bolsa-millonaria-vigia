import numpy as np
import pandas as pd
import pytest

from src.events import EventCal, build_event_cal
from tests.helpers import FLAT, N_PRE, candles, dates, drop_then, inputs, make_cfg, run

E = "ECOPETROL"
ENTRY = 96.0 * 1.002            # entrada tras caída a 96 con spread 0,2 %
SL, TP = ENTRY * 0.97, ENTRY * 1.06
SLIP = 0.003


def one_trade(follow, cfg=None, mode="t_a", **kw):
    cfg = cfg or make_cfg()
    res = run(cfg, inputs({E: drop_then(follow)}, **kw), mode)
    assert len(res.trades) >= 1, "no se abrió ninguna operación"
    return res.trades.iloc[0]


# ---------------- salidas ----------------
def test_entry_at_close_with_spread():
    t = one_trade([(97, 98, 96.5, 97)])
    assert t.entry_px == pytest.approx(ENTRY)
    assert t.entry_date == dates(40)[N_PRE]


def test_sl_intraday_has_slippage():
    t = one_trade([(96, 97, 93, 95)])
    assert t.exit_reason == "SL"
    assert t.exit_px == pytest.approx(SL * (1 - SLIP))


def test_gap_below_sl_exits_at_open_with_slippage():
    t = one_trade([(92, 93, 91, 92)])
    assert t.exit_reason == "SL"
    assert t.exit_px == pytest.approx(92 * (1 - SLIP))


def test_tp_exits_at_limit_without_slippage():
    t = one_trade([(97, 102.5, 96.5, 101)])
    assert t.exit_reason == "TP"
    assert t.exit_px == pytest.approx(TP)


def test_gap_above_tp_exits_at_open_without_slippage():
    t = one_trade([(103, 104, 102, 103)])
    assert t.exit_reason == "TP"
    assert t.exit_px == pytest.approx(103)


def test_sl_and_tp_same_candle_assumes_sl_by_default():
    t = one_trade([(96, 102.5, 93, 97)])
    assert t.exit_reason == "SL"


def test_sl_and_tp_same_candle_configurable_to_tp():
    t = one_trade([(96, 102.5, 93, 97)], cfg=make_cfg(same_candle_assumption="TP"))
    assert t.exit_reason == "TP" and t.exit_px == pytest.approx(TP)


def test_trailing_applies_from_next_candle_not_same_candle():
    # vela 1: High toca el gatillo (100,04) y el Low (97,0) queda bajo el nuevo SL (97,63) pero sobre el SL viejo
    # => NO sale en esa vela. Vela 2: Low 97,5 <= 97,63 => sale al SL de trailing con slippage.
    t = one_trade([(98, 100.5, 97.0, 100), (99, 99.5, 97.5, 98)])
    assert t.exit_reason == "SL_trailing"
    assert t.sessions == 2
    assert t.exit_px == pytest.approx(ENTRY * 1.015 * (1 - SLIP))


def test_time_stop_setup_a_is_3_sessions_at_close():
    t = one_trade([(97, 98, 96.5, 97)] * 6)
    assert t.exit_reason == "tiempo" and t.sessions == 3
    assert t.exit_px == pytest.approx(97 * (1 - SLIP))


def test_max_hold_8_sessions_caps_setup_b():
    cal = dates(40)
    px = candles([])                                     # todo plano a 100
    cat = EventCal(b_entry={cal[5]: cal[25]})            # la salida por catalizador queda más allá de 8 sesiones
    res = run(make_cfg(), inputs({E: px}, cat={E: cat}), "t_b")
    t = res.trades.iloc[0]
    assert t.exit_reason == "tiempo" and t.sessions == 8


def test_pre_report_exit_at_close():
    cal = dates(40)
    own = EventCal(pre_exit={cal[N_PRE + 2]})
    t = one_trade([(97, 98, 96.5, 97)] * 6, own={E: own})
    assert t.exit_reason == "reporte" and t.exit_date == cal[N_PRE + 2]


def test_no_entry_on_the_pre_report_session():
    cal = dates(40)
    res = run(make_cfg(), inputs({E: drop_then([])}, own={E: EventCal(pre_exit={cal[N_PRE]})}))
    assert res.trades.empty and res.skipped["reporte_cercano"] == 1


def test_setup_b_buys_two_sessions_before_and_sells_before_report():
    cal = dates(40)
    ev = pd.DataFrame({"report_date": [cal[20]], "timing": ["AMC"]})
    cat = build_event_cal(cal, ev, 2)
    assert cat.b_entry == {cal[18]: cal[20]}             # AMC: sale al cierre del mismo día del reporte
    bmo = build_event_cal(cal, pd.DataFrame({"report_date": [cal[20]], "timing": ["BMO"]}), 2)
    assert bmo.b_entry == {cal[18]: cal[19]}             # BMO: sale al cierre de la sesión anterior
    res = run(make_cfg(), inputs({E: candles([])}, cat={E: cat}), "t_b")
    t = res.trades.iloc[0]
    assert (t.entry_date, t.exit_date, t.exit_reason) == (cal[18], cal[20], "catalizador")


def test_brent_drop_sells_ecopetrol_at_close():
    t = one_trade([(97, 98, 96.5, 97)] * 5, brent={N_PRE + 2: -0.03})
    assert t.exit_reason == "brent_cae" and t.sessions == 2


def test_usdcop_drop_moves_sl_to_entry_minus_2pct_from_next_candle():
    cfg = make_cfg()
    fx = pd.Series(4000.0, index=dates(40))
    nvda = drop_then([(95, 95.5, 94.0, 95), (95, 95.5, 92.0, 94)], drop_close=94.0)   # caída 6 % >= 4 %
    entry = 94.0 * 1.003
    res = run(cfg, inputs({"NVDA": nvda}, fx={"NVDA": fx}, fxret={N_PRE + 1: -0.011}), "t_a")
    t = res.trades.iloc[0]
    assert t.exit_reason == "SL_usdcop"
    assert t.exit_px == pytest.approx(entry * 0.98 * (1 - 0.004))
    assert t.sessions == 2


# ---------------- señales ----------------
def test_setup_a_threshold_and_attack_lowers_it_by_one_point():
    res = run(make_cfg(), inputs({E: drop_then([], drop_close=97.5)}), "t_a")          # -2,5 %
    assert res.trades.empty
    res = run(make_cfg(), inputs({E: drop_then([], drop_close=97.5)}), "t_att")
    assert len(res.trades) == 1
    res = run(make_cfg(), inputs({E: drop_then([], drop_close=96.5)}), "t_a")          # -3,5 %
    assert len(res.trades) == 1


def test_setup_a_invalid_on_day_after_own_report():
    cal = dates(40)
    res = run(make_cfg(), inputs({E: drop_then([])}, own={E: EventCal(reaction={cal[N_PRE]})}))
    assert res.trades.empty


def test_setup_c_buys_next_open_and_time_stop_5():
    px = candles([FLAT] * N_PRE + [(100, 100, 100, 100), (101, 101, 101, 101)])
    res = run(make_cfg(), inputs({E: px}, brent={N_PRE: 0.03}), "t_c")
    t = res.trades.iloc[0]
    assert t.entry_date == dates(40)[N_PRE + 1]
    assert t.entry_ref == 101 and t.exit_reason == "tiempo" and t.sessions == 5


def test_setup_c_not_triggered_below_threshold():
    res = run(make_cfg(), inputs({E: candles([])}, brent={N_PRE: 0.02}), "t_c")
    assert res.trades.empty


def test_fomc_blackout_blocks_mgc_entries_day_before_and_day_of():
    f = 12
    for k, expect_trade in ((f - 1, False), (f, False), (f + 1, True)):
        rows = [FLAT] * k + [(100, 100, 94, 94)]
        res = run(make_cfg(), inputs({"TSLA": candles(rows)}, fx={"TSLA": pd.Series(4000.0, index=dates(40))},
                                     fomc=[f]), "t_a")
        assert (len(res.trades) == 1) == expect_trade, k


def test_fomc_does_not_block_ecopetrol():
    rows = [FLAT] * 11 + [(100, 100, 96, 96)]
    res = run(make_cfg(), inputs({E: candles(rows)}, fomc=[12]), "t_a")
    assert len(res.trades) == 1


# ---------------- cartera ----------------
def test_one_position_per_asset_and_max_positions():
    cfg = make_cfg()
    tsla = drop_then([(94, 95, 93, 94)] * 3, drop_close=94.0)
    ecop = drop_then([(96, 97, 95, 96)] * 3, drop_close=96.0)
    fx = pd.Series(4000.0, index=dates(40))
    res = run(cfg, inputs({"TSLA": tsla, E: ecop}, fx={"TSLA": fx}), "t_a1")
    assert set(res.trades.asset) == {"TSLA"}               # cupo 1: gana TSLA por orden de entrada
    assert res.skipped["sin_cupo"] >= 1


def test_minimum_position_size_enforced():
    res = run(make_cfg(), inputs({E: drop_then([])}), "t_a", capital=10_000_000)   # 40 % = 4M < 5M
    assert res.trades.empty and res.skipped["sin_capital"] == 1


# ---------------- contabilidad ----------------
def test_net_pnl_reconciles_with_cost_breakdown():
    res = run(make_cfg(), inputs({E: drop_then([(97, 102.5, 96.5, 101)])}), "t_a")
    t = res.trades.iloc[0]
    assert t.net_pnl == pytest.approx(t.gross_pnl - t.spread_cost - t.slippage_cost - t.fees)
    assert t.invested_cop == pytest.approx(40_000_000, rel=0.005)   # 40 % del patrimonio tras costos de actividad previos
    assert t.fees == pytest.approx(2 * 40_000_000 * 0.002975, rel=0.05)


def test_usd_asset_pnl_converted_with_fx_at_entry_and_exit():
    cal = dates(40)
    fx = pd.Series(4000.0, index=cal)
    fx.iloc[N_PRE + 2:] = 4400.0                                    # el dólar sube 10 % antes de la salida
    nvda = drop_then([(94, 94.5, 93.5, 94), (100.5, 101, 100, 100.5)], drop_close=94.0)   # TP por gap en la vela 2
    res = run(make_cfg(), inputs({"NVDA": nvda}, fx={"NVDA": fx}), "t_a")
    t = res.trades.iloc[0]
    assert t.fx_entry == 4000 and t.fx_exit == 4400
    assert t.net_pnl > 0.1 * t.invested_cop * 0.9                   # el efecto cambiario suma ~10 %


def test_activity_cost_is_charged_on_sessions_without_purchase():
    res = run(make_cfg(), inputs({E: drop_then([(97, 98, 96.5, 97)] * 3)}), "t_a")
    assert res.activity_days == 40 - 1
    assert res.activity_cost == pytest.approx(39 * 14_875)
    assert res.equity.iloc[0] == pytest.approx(100_000_000, rel=1e-9) or res.equity.iloc[0] < 100_000_000


# ---------------- integridad ----------------
def test_blacklisted_ticker_fails_the_backtest():
    cfg = make_cfg()
    cfg["rules"]["entry_order"] = ["ETB"]
    cfg["costs"]["entry_spread"]["ETB"] = 0.003
    cfg["costs"]["exit_slippage"]["ETB"] = 0.004
    cfg["assets"]["ETB"] = dict(cfg["assets"]["ECOPETROL"])
    cfg["setups"]["A"]["drop_threshold"]["ETB"] = 0.03
    with pytest.raises(AssertionError, match="LISTA NEGRA"):
        run(cfg, inputs({"ETB": drop_then([])}), "t_a")


def test_no_look_ahead_truncating_the_future_does_not_change_the_past():
    rng = np.random.default_rng(7)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    op = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.004, n))
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.008, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.008, n)))
    idx = dates(n)
    df = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close}, index=idx)
    brent = {int(i): float(v) for i, v in zip(rng.integers(0, n, 40), rng.choice([-0.03, 0.03], 40))}
    cfg = make_cfg()
    full = run(cfg, inputs({E: df}, brent=brent), "mantener")
    cut = 250
    part = run(cfg, inputs({E: df.iloc[:cut]}, brent={k: v for k, v in brent.items() if k < cut}), "mantener")
    a = full.trades[full.trades.exit_date <= idx[cut - 1]].reset_index(drop=True)
    b = part.trades[(part.trades.exit_reason != "fin_datos")].reset_index(drop=True)
    assert len(a) > 5
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(full.equity.iloc[:cut - 1], part.equity.iloc[:cut - 1], check_names=False)


def test_activity_cost_can_be_reported_without_deducting_and_purchase_days_are_recorded():
    inp = inputs({E: drop_then([(97, 98, 96.5, 97)] * 3)})
    res = run(make_cfg(), inp, "t_a", deduct_activity=False)
    assert res.activity_days == 39 and res.purchase_days == [dates(40)[N_PRE]]
    assert res.equity.iloc[N_PRE - 1] == pytest.approx(100_000_000)


def test_controller_changes_mode_for_new_entries_only():
    rows = [FLAT] * 10 + [(100, 100, 96, 96)] + [(96, 97, 95.5, 96)] * 4 + [FLAT] * 5 + [(100, 100, 96, 96)] + [(96, 97, 95.5, 96)] * 3
    cfg = make_cfg()
    cfg["modes"]["t_a2"] = {"positions": 2, "size": 0.20, "setups": ["A"]}
    inp = inputs({E: candles(rows, total=40)})
    res = run(cfg, inp, "t_a", controller=lambda i, d, eq: "t_a2" if i == 15 else None)
    inv = res.trades.invested_cop.values
    assert len(inv) == 2 and inv[0] == pytest.approx(40_000_000, rel=0.01) and inv[1] == pytest.approx(20_000_000, rel=0.01)


def test_liquidate_end_false_keeps_position_open_and_marks_to_market():
    inp = inputs({E: drop_then([(97, 98, 96.5, 97)], total=13)})
    res = run(make_cfg(), inp, "t_a", liquidate_end=False)
    assert res.trades.empty and res.exposure.iloc[-1] > 0.3