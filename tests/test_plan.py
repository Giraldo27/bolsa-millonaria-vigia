import numpy as np
import pandas as pd
import pytest

from src.plan import (build_message, contest_sessions, contest_status, drop_incomplete_bar, et_offset_hours, rank_by_volatility,
                      validate_selection)
from tests.helpers import make_cfg

CFG = make_cfg()


def closes(vols: dict[str, float], n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    idx = pd.bdate_range(end="2026-10-02", periods=n)
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(rng.normal(0, v, n))) for t, v in vols.items()}, index=idx)


def test_rank_by_volatility_orders_by_std_and_excludes():
    c = closes({"A": 0.01, "B": 0.03, "C": 0.02})
    r = rank_by_volatility(c, 250)
    assert list(r.index) == ["B", "C", "A"] and r.vol_pct.iloc[0] > 2.0
    assert "B" not in rank_by_volatility(c, 250, exclude=["B"]).index


def test_rank_ignores_assets_without_enough_history():
    c = closes({"A": 0.01, "B": 0.03})
    c.loc[c.index[:100], "B"] = np.nan
    assert list(rank_by_volatility(c, 250).index) == ["A"]


def test_contest_has_23_sessions_without_colombian_holidays():
    s = contest_sessions(CFG)
    assert len(s) == 23 and s[0] == pd.Timestamp("2026-10-05") and s[-1] == pd.Timestamp("2026-11-06")
    assert pd.Timestamp("2026-10-12") not in s and pd.Timestamp("2026-11-02") not in s


def test_contest_status_phases_and_cuts():
    assert contest_status(CFG, pd.Timestamp("2026-10-03")).fase == "antes"
    st = contest_status(CFG, pd.Timestamp("2026-10-05"))
    assert (st.fase, st.sesion, st.sesiones_restantes, st.proximo_corte, st.sesiones_al_corte) == ("durante", 1, 22, 5, 4)
    st = contest_status(CFG, pd.Timestamp("2026-10-15"))        # el 12-oct es festivo: 5,6,7,8,9,13,14,15 => sesión 8
    assert st.sesion == 8 and st.proximo_corte == 10 and st.sesiones_al_corte == 2
    assert contest_status(CFG, pd.Timestamp("2026-11-06")).sesiones_restantes == 0
    assert contest_status(CFG, pd.Timestamp("2026-11-09")).fase == "despues"


def test_us_dst_offsets():
    assert et_offset_hours(pd.Timestamp("2026-10-05 18:00", tz="UTC")) == -4
    assert et_offset_hours(pd.Timestamp("2026-11-05 18:00", tz="UTC")) == -5
    assert et_offset_hours(pd.Timestamp("2026-01-15 18:00", tz="UTC")) == -5


def test_partial_bar_is_dropped_before_us_close_only():
    c = closes({"A": 0.01}, 10)
    c.index = list(c.index[:-1]) + [pd.Timestamp("2026-10-05")]
    now_open = pd.Timestamp("2026-10-05 19:30", tz="UTC")        # 15:30 ET: mercado abierto
    assert len(drop_incomplete_bar(c, now_open)) == 9
    now_closed = pd.Timestamp("2026-10-05 21:00", tz="UTC")      # 17:00 ET: cerrado
    assert len(drop_incomplete_bar(c, now_closed)) == 10


def _msg(today, held=None, price=None, earn=None, pct=None, bought=False):
    rk = rank_by_volatility(closes({"TSLA": 0.03, "NVDA": 0.02, "META": 0.025}), 250)
    pos = pd.DataFrame(held or [], columns=["ticker", "fecha_compra", "precio_compra_usd"])
    return build_message(status=contest_status(CFG, pd.Timestamp(today)), ranking=rk, selected=rk.index[0], positions=pos,
                         prices=price or {}, next_earnings=earn or {}, capital=1e8, percentil=pct, cfg=CFG, bought_today=bought)


def test_message_before_contest_says_what_to_buy_and_when():
    m = _msg("2026-10-03")
    assert "comprar TSLA" in m and "100 %" in m and "$100.000.000" in m and "2026-10-05" in m


def test_message_in_contest_without_position_orders_to_buy_now():
    m = _msg("2026-10-06")
    assert "No tienes posición" in m and "comprar TSLA" in m and "micro-compra" in m


def test_message_with_position_shows_pnl_earnings_and_no_sell_advice():
    m = _msg("2026-10-19", held=[("TSLA", "2026-10-05", 400.0)], price={"TSLA": 340.0}, earn={"TSLA": pd.Timestamp("2026-10-21")},
             pct=18, bought=True)
    assert "-15,0 %" in m and "NO vende" in m
    assert "reporta resultados en 2 día(s)" in m and "mantener" in m
    assert "top 18 %" in m and "micro-compra" not in m


def test_message_warns_not_to_switch_stock():
    m = _msg("2026-10-08", held=[("NVDA", "2026-10-05", 100.0)], price={"NVDA": 110.0})
    assert "NO cambies" in m and "+10,0 %" in m


def test_validate_selection_rejects_blacklist_and_non_mgc():
    validate_selection("TSLA", CFG)
    with pytest.raises(AssertionError):
        validate_selection("ETB", CFG)
    with pytest.raises(AssertionError):
        validate_selection("ECOPETROL", CFG)
