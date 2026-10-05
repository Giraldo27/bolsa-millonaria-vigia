"""Arma los insumos del motor (Inputs) a partir de los datos reales cacheados."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .backtest import AssetData, Inputs
from .data_loader import currency_of, load_earnings, load_macro, load_prices, load_rate_events
from .events import build_event_cal
from .universe import catalyst_events


def build_inputs(cfg: dict[str, Any], assets: list[str] | None = None, refresh: bool = False) -> Inputs:
    assets = assets or cfg["universe"]["system_assets"]
    usdcop = load_macro("usdcop", cfg, refresh)["Close"]
    earnings = load_earnings(cfg, list(set(assets) | {"MSFT", "GOOGL", "META"} | set(cfg["adr_proxy"])), refresh)
    entry_before = cfg["setups"]["B"]["entry_sessions_before_report"]
    data: dict[str, AssetData] = {}
    for a in assets:
        px = load_prices(a, cfg, refresh)[["Open", "High", "Low", "Close"]].dropna()
        cur = currency_of(a, cfg)
        assert cur in ("USD", "COP"), f"{a}: moneda {cur} no soportada por el motor"
        own = earnings[earnings["ticker"] == a]
        data[a] = AssetData(
            ohlc=px, fx=usdcop if cur == "USD" else None,
            own=build_event_cal(px.index, own, entry_before),
            cat=build_event_cal(px.index, catalyst_events(a, earnings, cfg), entry_before))
    assert usdcop.between(1_000, 8_000).all() and usdcop.pct_change().abs().max() < 0.15, "USD/COP con ticks erróneos"
    for a, ad in data.items():
        worst = ad.ohlc["Close"].pct_change().abs().max()
        assert worst < 0.5, f"{a}: salto de cierre a cierre de {worst:.0%}; revisar datos"
    brent = load_macro("brent", cfg, refresh)["Close"].pct_change()
    fx_ret = usdcop.pct_change()
    fomc = list(load_rate_events(cfg)["fecha_decision"])
    activity = data["ECOPETROL"].ohlc.index if "ECOPETROL" in data else next(iter(data.values())).ohlc.index
    return Inputs(assets=data, brent_ret=brent, fx_ret=fx_ret, fomc=fomc, activity_cal=pd.DatetimeIndex(activity))
