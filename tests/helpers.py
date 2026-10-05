"""Constructores de escenarios sintéticos para los tests del motor."""
from __future__ import annotations

import copy

import pandas as pd

from src.backtest import AssetData, Backtester, Inputs
from src.config import load_config
from src.events import EventCal

N_PRE = 10          # velas planas antes de la caída que dispara el Setup A
FLAT = (100.0, 100.0, 100.0, 100.0)


def make_cfg(**over) -> dict:
    cfg = copy.deepcopy(load_config())
    cfg["modes"]["t_a"] = {"positions": 2, "size": 0.40, "setups": ["A"]}
    cfg["modes"]["t_b"] = {"positions": 2, "size": 0.40, "setups": ["B"]}
    cfg["modes"]["t_c"] = {"positions": 2, "size": 0.40, "setups": ["C"]}
    cfg["modes"]["t_a1"] = {"positions": 1, "size": 0.40, "setups": ["A"]}
    cfg["modes"]["t_att"] = {"positions": 2, "size": 0.40, "setups": ["A"], "attack": True}
    for k, v in over.items():
        cfg["rules"][k] = v
    return cfg


def dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2024-01-02", periods=n)


def candles(rows: list[tuple], total: int = 40) -> pd.DataFrame:
    rows = list(rows) + [FLAT] * (total - len(rows))
    idx = dates(total)
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def drop_then(follow: list[tuple], drop_close: float = 96.0, total: int = 40) -> pd.DataFrame:
    """10 velas planas a 100, una vela de caída (cierra en drop_close) y luego las velas indicadas."""
    drop = (100.0, 100.0, min(drop_close, 100.0), drop_close)
    return candles([FLAT] * N_PRE + [drop] + follow, total)


def inputs(assets: dict[str, pd.DataFrame], fx: dict | None = None, brent: dict | None = None,
           fxret: dict | None = None, fomc: list | None = None, own: dict | None = None, cat: dict | None = None
           ) -> Inputs:
    any_idx = next(iter(assets.values())).index
    data = {}
    for a, df in assets.items():
        data[a] = AssetData(ohlc=df, fx=(fx or {}).get(a), own=(own or {}).get(a, EventCal()),
                            cat=(cat or {}).get(a, EventCal()))
    z = pd.Series(0.0, index=any_idx)
    for d, v in (brent or {}).items():
        z = z.copy(); z.loc[any_idx[d]] = v
    br = z
    f = pd.Series(0.0, index=any_idx)
    for d, v in (fxret or {}).items():
        f.loc[any_idx[d]] = v
    return Inputs(assets=data, brent_ret=br, fx_ret=f, fomc=[any_idx[i] for i in (fomc or [])],
                  activity_cal=any_idx)


def run(cfg, inp, mode="t_a", **kw):
    first = next(iter(inp.assets.values())).ohlc.index[0]
    return Backtester(cfg, inp, mode, start=kw.pop("start", first), **kw).run()
