"""Fase 1: métricas por activo y ranking del universo permitido (sin look-ahead: sólo describe el pasado)."""
from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from .config import group_of
from .data_loader import currency_of, yahoo_symbol


def atr_pct(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """ATR de Wilder dividido por el cierre, en %. TR = max(H-L, |H-Cprev|, |L-Cprev|)."""
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr / df["Close"] * 100


def fx_to_cop(ticker: str, df: pd.DataFrame, currency: str, usdcop: pd.Series, gbpusd: pd.Series) -> pd.Series:
    idx = df.index
    u = usdcop.reindex(idx, method="ffill")
    if currency == "COP":
        return pd.Series(1.0, index=idx)
    if currency == "USD":
        return u
    if currency in ("GBp", "GBX"):
        return u * gbpusd.reindex(idx, method="ffill") / 100
    if currency == "GBP":
        return u * gbpusd.reindex(idx, method="ffill")
    return pd.Series(np.nan, index=idx)


def catalyst_events(ticker: str, earnings: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Eventos de catalizador del activo: propios, o (NVDA) el primero de cada temporada de los proxies."""
    r = cfg["ranking"]
    prox = r["catalyst_proxy"].get(ticker)
    if prox is None:
        return earnings[earnings["ticker"] == ticker][["report_date", "timing"]].reset_index(drop=True)
    e = earnings[earnings["ticker"].isin(prox)].sort_values("report_date")
    new = e["report_date"].diff().dt.days.fillna(999) > r["cluster_gap_days"]
    return e[new][["report_date", "timing"]].reset_index(drop=True)


def pre_report_moves(df: pd.DataFrame, events: pd.DataFrame, entry_before: int, last_date: pd.Timestamp) -> pd.Series:
    """Retorno bruto cierre(D-entry_before) -> cierre de la última sesión antes de que salga el reporte."""
    idx, close, out = df.index, df["Close"].values, []
    for _, ev in events.iterrows():
        if ev["report_date"] > last_date:
            continue
        iD = idx.searchsorted(ev["report_date"])
        i_in = iD - entry_before
        i_out = iD - 1 if ev["timing"] == "BMO" else iD
        if i_in < 0 or i_out >= len(idx) or i_out <= i_in:
            continue
        out.append(close[i_out] / close[i_in] - 1)
    return pd.Series(out, dtype=float)


def window_events_per_year(df: pd.DataFrame, events: pd.DataFrame, cfg: dict[str, Any]) -> tuple[float, int]:
    cw, first = cfg["ranking"]["contest_window"], max(2015, df.index[0].year)
    last = df.index[-1].year - 1                      # años completos
    years = max(last - first + 1, 0)
    if years == 0 or events.empty:
        return (np.nan if events.empty else 0.0), 0
    d = events["report_date"]
    md = d.dt.month * 100 + d.dt.day
    lo, hi = cw["start_month"] * 100 + cw["start_day"], cw["end_month"] * 100 + cw["end_day"]
    n = int(((md >= lo) & (md <= hi) & d.dt.year.between(first, last)).sum())
    return n / years, n


def compute_metrics(prices: dict[str, pd.DataFrame], earnings: pd.DataFrame, usdcop: pd.Series, gbpusd: pd.Series,
                    cfg: dict[str, Any]) -> pd.DataFrame:
    rows = []
    ry = cfg["data"]["recent_years"]
    for t, df in prices.items():
        last = df.index[-1]
        rec = df[df.index > last - pd.DateOffset(years=ry)]
        cur = currency_of(t, cfg)
        fx = fx_to_cop(t, rec, cur, usdcop, gbpusd)
        traded = (rec["Close"] * rec["Volume"] * fx)
        ev = catalyst_events(t, earnings, cfg)
        per_year, n_win = window_events_per_year(df, ev, cfg)
        mv = pre_report_moves(df, ev, cfg["ranking"]["pre_report_entry_sessions_before"], last)
        rows.append(dict(
            ticker=t, grupo=group_of(t, cfg), simbolo=yahoo_symbol(t, cfg), moneda=cur, desde=df.index[0].date(),
            barras_corregidas=df.attrs.get("bars_corrected", 0),
            atr_pct=atr_pct(df, cfg["ranking"]["atr_period"]).reindex(rec.index).mean(),
            valor_cop_mm=traded.mean() / 1e6, pct_vol0=(rec["Volume"].fillna(0) == 0).mean() * 100,
            eventos_ventana_ano=per_year, eventos_ventana_n=n_win,
            pre_n=len(mv), pre_media_pct=mv.mean() * 100 if len(mv) else np.nan,
            pre_mediana_pct=mv.median() * 100 if len(mv) else np.nan,
            pre_acierto_pct=(mv > 0).mean() * 100 if len(mv) else np.nan,
        ))
    return pd.DataFrame(rows).set_index("ticker")


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True)


def rank_universe(m: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Puntaje 0-100 = suma ponderada de percentiles. Si faltan datos de reportes se puntúa 0 (conservador)
    y se calcula también una versión neutral (0,5) para comprobar que la conclusión no depende de eso."""
    w, ls = cfg["ranking"]["weights"], cfg["ranking"]["liquidez_split"]
    m = m.copy()
    m["p_vol"] = _pct_rank(m["atr_pct"])
    m["p_liq"] = ls["valor_negociado"] * _pct_rank(np.log10(m["valor_cop_mm"].clip(lower=1e-6))) \
        + ls["dias_sin_volumen"] * _pct_rank(-m["pct_vol0"])
    m["p_cat"] = _pct_rank(m["eventos_ventana_ano"])
    m["p_pre"] = _pct_rank(m["pre_media_pct"].where(m["pre_n"] >= 8))   # mínimo 8 eventos para fiarse
    for k, fill in (("", 0.0), ("_neutral", 0.5)):
        m[f"score{k}"] = 100 * (w["volatilidad"] * m["p_vol"] + w["liquidez"] * m["p_liq"]
                                + w["catalizadores"] * m["p_cat"].fillna(fill)
                                + w["pre_reporte"] * m["p_pre"].fillna(fill))
    m["rank"] = m["score"].rank(ascending=False, method="min").astype(int)
    m["rank_neutral"] = m["score_neutral"].rank(ascending=False, method="min").astype(int)
    return m.sort_values("score", ascending=False)


def correlation_matrix(prices: dict[str, pd.DataFrame], tickers: list[str], since: str = "2022-01-01") -> pd.DataFrame:
    r = pd.concat({t: prices[t]["Close"].pct_change() for t in tickers}, axis=1).loc[since:].dropna()
    return r.corr()


def best_triples(m: pd.DataFrame, corr_prices: dict[str, pd.DataFrame], top_n: int = 10, since: str = "2022-01-01"):
    """Mejores tríos entre los top_n por puntaje: media de puntajes penalizada por la correlación media del trío."""
    cand = list(m.index[:top_n])
    c = correlation_matrix(corr_prices, cand, since)
    res = []
    for tri in itertools.combinations(cand, 3):
        sc = m.loc[list(tri), "score"].mean()
        rho = np.mean([c.loc[a, b] for a, b in itertools.combinations(tri, 2)])
        res.append(dict(trio="+".join(tri), score_medio=sc, corr_media=rho, score_ajustado=sc * (1 - 0.5 * max(rho, 0))))
    return pd.DataFrame(res).sort_values("score_ajustado", ascending=False)
