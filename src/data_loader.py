"""Descarga y caché (parquet) de precios, macro, fechas de resultados y eventos FOMC."""
from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from .config import group_of

warnings.filterwarnings("ignore")
OHLCV = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def yahoo_symbol(ticker: str, cfg: dict[str, Any]) -> str:
    over = cfg["yahoo_symbol"]
    if ticker in over:
        return over[ticker]
    return f"{ticker}.CL" if group_of(ticker, cfg) in ("local", "blacklist") else ticker


def _download(symbol: str, start: str) -> pd.DataFrame:
    df = yf.download(symbol, start=start, auto_adjust=False, progress=False, threads=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if df.empty:
        return df
    df = df[[c for c in OHLCV if c in df.columns]].dropna(subset=["Close"])
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "Date"
    return df[~df.index.duplicated()].sort_index()


def _cached(name: str, symbol: str, cfg: dict[str, Any], refresh: bool) -> pd.DataFrame:
    f: Path = cfg["paths"]["cache_dir"] / f"{name}.parquet"
    if f.exists() and not refresh:
        return pd.read_parquet(f)
    df = _download(symbol, cfg["data"]["start"])
    if not df.empty:
        df.to_parquet(f)
    return df


def clean_ohlc(df: pd.DataFrame, max_wick: float = 0.12, spike_log: float = 0.4) -> tuple[pd.DataFrame, int]:
    """Corrige ticks erróneos de Yahoo y devuelve (df, n_barras_corregidas):
    1) Cierre que salta más de spike_log (log) y revierte al día siguiente => barra plana al cierre previo.
    2) Open vacío/0 => Open = Close.
    3) High (Low) a más de max_wick por encima (debajo) de max (min) de Open/Close => se recorta a ese nivel.
    Además se arregla High<max(O,C) y Low>min(O,C)."""
    df = df.copy()
    lr = np.log(df["Close"] / df["Close"].shift(1))
    spike = (lr.abs() > spike_log) & ((lr + lr.shift(-1)).abs() < 0.1)
    if spike.any():
        prev = df["Close"].shift(1)
        for c in ("Open", "High", "Low", "Close"):
            df.loc[spike, c] = prev[spike]
        df.loc[spike, "Volume"] = 0
    bad_open = ~(df["Open"] > 0)
    df.loc[bad_open, "Open"] = df.loc[bad_open, "Close"]       # Open vacío/0 => se asume apertura = cierre previo del día
    hi_ref, lo_ref = df[["Open", "Close"]].max(axis=1), df[["Open", "Close"]].min(axis=1)
    ext_hi = df["High"] > hi_ref * (1 + max_wick)
    ext_lo = df["Low"] < lo_ref * (1 - max_wick)
    df["High"] = df["High"].where(~ext_hi & (df["High"] >= hi_ref), hi_ref)   # también arregla High<max(O,C)
    df["Low"] = df["Low"].where(~ext_lo & (df["Low"] <= lo_ref), lo_ref)      # y Low>min(O,C)
    return df, int((ext_hi | ext_lo | spike).sum())


def load_prices(ticker: str, cfg: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    df = _cached(f"px_{ticker}", yahoo_symbol(ticker, cfg), cfg, refresh)
    if df.empty:
        return df
    df, n = clean_ohlc(df)
    df.attrs["bars_corrected"] = n
    return df


def clean_close(close: pd.Series, tol: float, window: int = 11) -> tuple[pd.Series, int]:
    """Quita ticks erróneos: cierres que se alejan más de tol (relativo) de la mediana móvil centrada de `window`
    sesiones se sustituyen por el último valor válido. Devuelve (serie, n_corregidos)."""
    med = close.rolling(window, center=True, min_periods=3).median()
    bad = (close / med - 1).abs() > tol
    return close.mask(bad).ffill().bfill(), int(bad.sum())


def load_macro(name: str, cfg: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    df = _cached(f"macro_{name}", cfg["macro"][name], cfg, refresh)
    tol = cfg["data"].get("clean_macro", {}).get(name)
    if tol and not df.empty:
        df = df.copy()
        df["Close"], n = clean_close(df["Close"], tol)
        df.attrs["bars_corrected"] = n
    return df


def load_adr(ticker: str, cfg: dict[str, Any], refresh: bool = False) -> pd.DataFrame:
    return _cached(f"adr_{ticker}", cfg["adr_proxy"][ticker], cfg, refresh)


def currency_of(ticker: str, cfg: dict[str, Any]) -> str:
    """Moneda de cotización según Yahoo (cacheada en data/cache/currency.json)."""
    f: Path = cfg["paths"]["cache_dir"] / "currency.json"
    cache = json.loads(f.read_text()) if f.exists() else {}
    if ticker not in cache:
        try:
            cache[ticker] = yf.Ticker(yahoo_symbol(ticker, cfg)).history_metadata.get("currency", "UNKNOWN")
        except Exception:
            cache[ticker] = "UNKNOWN"
        f.write_text(json.dumps(cache, indent=1))
    return cache[ticker]


def load_universe_prices(cfg: dict[str, Any], tickers: list[str], refresh: bool = False) -> dict[str, pd.DataFrame]:
    out = {}
    for t in tickers:
        df = load_prices(t, cfg, refresh)
        if not df.empty:
            out[t] = df
    return out


# ---------------- Fechas de resultados ----------------
def _fetch_earnings_yf(symbol: str) -> pd.DataFrame:
    try:
        e = yf.Ticker(symbol).get_earnings_dates(limit=100)
    except Exception:
        return pd.DataFrame()
    if e is None or e.empty:
        return pd.DataFrame()
    e = e.reset_index()
    e = e.rename(columns={e.columns[0]: "report_datetime"})
    e["reported"] = e["Reported EPS"].notna() if "Reported EPS" in e else True
    return e[["report_datetime", "reported"]]


def classify_timing(ts: pd.Timestamp) -> str:
    """BMO = antes de la apertura, AMC = después del cierre (hora de Nueva York). Sin hora => AMC (conservador)."""
    h = ts.hour + ts.minute / 60
    if h == 0:
        return "AMC"
    return "BMO" if h < 12 else "AMC"


def load_earnings(cfg: dict[str, Any], tickers: list[str], refresh: bool = False) -> pd.DataFrame:
    """Columnas: ticker, report_date (naive), timing (BMO/AMC), reported, source. ECOPETROL usa el ADR EC."""
    f: Path = cfg["paths"]["cache_dir"] / "earnings_yf.parquet"
    if f.exists() and not refresh:
        base = pd.read_parquet(f)
    else:
        rows = []
        proxies = cfg["adr_proxy"]
        for t in tickers:
            if t in cfg["universe"]["local"] and t not in proxies:
                continue                       # Yahoo no publica resultados de acciones .CL
            sym = proxies.get(t, yahoo_symbol(t, cfg))
            e = _fetch_earnings_yf(sym)
            if e.empty:
                continue
            e["ticker"] = t
            e["source"] = f"yfinance:{sym}"
            rows.append(e)
        base = pd.concat(rows, ignore_index=True)
        base["report_datetime"] = pd.to_datetime(base["report_datetime"], utc=True).dt.tz_convert("America/New_York")
        base["timing"] = base["report_datetime"].map(classify_timing)
        base["report_date"] = base["report_datetime"].dt.tz_localize(None).dt.normalize()
        base = base.drop(columns="report_datetime")
        base.to_parquet(f)
    manual = pd.read_csv(cfg["paths"]["data_dir"] / "earnings_dates.csv")
    if not manual.empty:
        manual["report_date"] = pd.to_datetime(manual["report_datetime"]).dt.normalize()
        manual["reported"] = manual["report_date"] <= pd.Timestamp.today().normalize()
        manual["timing"] = manual["timing"].fillna("AMC")
        base = pd.concat([base, manual[["ticker", "report_date", "timing", "reported", "source"]]])
    base = base.drop_duplicates(["ticker", "report_date"], keep="last").sort_values(["ticker", "report_date"])
    return base.reset_index(drop=True)


def load_rate_events(cfg: dict[str, Any]) -> pd.DataFrame:
    df = pd.read_csv(cfg["paths"]["data_dir"] / "rate_events.csv", parse_dates=["fecha_decision"])
    return df


def assert_not_blacklisted(tickers: list[str], cfg: dict[str, Any]) -> None:
    bad = set(tickers) & set(cfg["universe"]["blacklist"])
    assert not bad, f"LISTA NEGRA: se intentó operar {sorted(bad)}"
