"""Lógica pura del plan final (sin red ni archivos): ranking por volatilidad, calendario del concurso y mensajes."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .data_loader import assert_not_blacklisted


def rank_by_volatility(closes: pd.DataFrame, window: int, exclude: list[str] | None = None) -> pd.DataFrame:
    """Ranking de acciones por desviación estándar de sus retornos diarios (ventana en sesiones). Sólo cierres ya
    publicados. Devuelve columnas vol_pct (diaria, en %) y mom60 (informativo)."""
    c = closes.drop(columns=[x for x in (exclude or []) if x in closes.columns]).dropna(axis=1, thresh=window + 1)
    r = c.pct_change().iloc[-window:]
    out = pd.DataFrame({"vol_pct": r.std() * 100, "mom60": c.iloc[-1] / c.iloc[-61] - 1})
    return out.sort_values("vol_pct", ascending=False)


def et_offset_hours(utc: pd.Timestamp) -> int:
    """Desfase de Nueva York respecto a UTC (horario de verano de EE. UU.: 2.º domingo de marzo a 1.er domingo de noviembre)."""
    y = utc.year
    mar = pd.Timestamp(y, 3, 1) + pd.offsets.Week(weekday=6) * (1 if pd.Timestamp(y, 3, 1).weekday() == 6 else 2)
    nov = pd.Timestamp(y, 11, 1) + pd.offsets.Week(weekday=6) * (0 if pd.Timestamp(y, 11, 1).weekday() == 6 else 1)
    return -4 if (mar + pd.Timedelta(hours=7)) <= utc.tz_localize(None) < (nov + pd.Timedelta(hours=6)) else -5


def drop_incomplete_bar(closes: pd.DataFrame, now_utc: pd.Timestamp) -> pd.DataFrame:
    """Quita la barra de hoy si el mercado de EE. UU. todavía no cerró (a las 14:30 de Bogotá la barra es parcial)."""
    et = now_utc.tz_localize(None) + pd.Timedelta(hours=et_offset_hours(now_utc))
    if closes.index[-1].normalize() == et.normalize() and (et.hour, et.minute) < (16, 5):
        return closes.iloc[:-1]
    return closes


def contest_sessions(cfg: dict[str, Any]) -> pd.DatetimeIndex:
    """Sesiones del concurso: días hábiles entre inicio y fin, sin festivos colombianos."""
    c = cfg["contest"]
    days = pd.bdate_range(c["start"], c["end"])
    return days.difference(pd.DatetimeIndex(c["holidays"]))


@dataclass
class Status:
    fecha: pd.Timestamp
    fase: str                      # 'antes' | 'durante' | 'despues'
    sesion: int                    # nº de sesión (1..23) si durante
    sesiones_restantes: int
    proximo_corte: int | None
    sesiones_al_corte: int | None


def contest_status(cfg: dict[str, Any], today: pd.Timestamp) -> Status:
    ses = contest_sessions(cfg)
    today = pd.Timestamp(today).normalize()
    if today < ses[0]:
        return Status(today, "antes", 0, len(ses), None, None)
    if today > ses[-1]:
        return Status(today, "despues", len(ses), 0, None, None)
    k = int(ses.searchsorted(today, side="right"))        # sesiones jugadas incluyendo hoy (si hoy es sesión)
    cuts = [c for c in cfg["contest"]["weekly_cuts"] if c >= k]
    nxt = cuts[0] if cuts else None
    return Status(today, "durante", k, len(ses) - k, nxt, (nxt - k) if nxt else None)


def pct(x: float, signed: bool = True) -> str:
    s = f"{x * 100:+.1f}" if signed else f"{x * 100:.1f}"
    return s.replace(".", ",") + " %"


def cop(x: float) -> str:
    return "$" + f"{x:,.0f}".replace(",", ".")


def build_message(*, status: Status, ranking: pd.DataFrame, selected: str, positions: pd.DataFrame, prices: dict[str, float],
                  next_earnings: dict[str, pd.Timestamp], capital: float, percentil: float | None, cfg: dict[str, Any],
                  bought_today: bool) -> str:
    """Mensaje de Telegram (texto plano). Siempre dice QUÉ HACER hoy."""
    p = cfg["plan"]
    L: list[str] = [f"📊 Bolsa Millonaria — {status.fecha:%d/%m/%Y}"]
    top = ", ".join(f"{t} {r.vol_pct:.2f} %" for t, r in ranking.head(3).iterrows()).replace(".", ",")
    L.append(f"Más volátiles (σ diaria, {p['vol_window']} sesiones): {top}")
    held = positions["ticker"].tolist() if len(positions) else []
    if status.fase == "antes":
        n_days = len(pd.bdate_range(status.fecha + pd.Timedelta(days=1), cfg["contest"]["start"]))
        L += [f"⏳ Faltan {n_days} día(s) hábil(es) para el inicio ({cfg['contest']['start']}).",
              f"✅ PLAN: el {cfg['contest']['start']} comprar {selected} con el 100 % del capital ({cop(capital)}) en cuanto abra el mercado (orden a mercado)."]
    elif status.fase == "despues":
        L.append("🏁 El concurso terminó. Si el reglamento no obliga a liquidar, no hay nada más que hacer.")
    else:
        L.append(f"Sesión {status.sesion} de {status.sesion + status.sesiones_restantes} · quedan {status.sesiones_restantes}.")
        if not held:
            L.append(f"🔴 No tienes posición. ACCIÓN: comprar {selected} con el 100 % del capital ({cop(capital)}) YA (orden a mercado).")
        for _, r in positions.iterrows():
            t = r["ticker"]
            px = prices.get(t)
            if px and r["precio_compra_usd"] > 0:
                ret = px / r["precio_compra_usd"] - 1
                L.append(f"📈 {t}: {px:.2f} USD · {pct(ret)} desde tu compra ({r['precio_compra_usd']:.2f}).")
                if ret <= p["loss_alert"]:
                    L.append(f"⚠️ Caída de {pct(ret)}: el plan NO vende (las pruebas muestran que los stops reducen la probabilidad de ganar). "
                             "Decide tú si aceptas este riesgo.")
            e = next_earnings.get(t)
            if e is not None:
                d = (e.normalize() - status.fecha).days
                if 0 <= d <= p["earnings_alert_days"]:
                    L.append(f"📅 {t} reporta resultados en {d} día(s) ({e:%d/%m}). ACCIÓN: mantener (el plan sostiene a través del reporte; mayor varianza = más opciones de ganar).")
            if t != selected:
                L.append(f"ℹ️ El ranking ahora señala {selected}, pero NO cambies: la acción se elige el primer día y se mantiene.")
        if percentil is not None:
            L.append(f"🏆 Tu percentil informado: top {percentil:.0f} %."
                     + (f" Corte semanal en {status.sesiones_al_corte} sesión(es)." if status.sesiones_al_corte is not None else "")
                     + " Regla: mantener 100 %; reducir al ir liderando baja la probabilidad de ganar.")
        if status.sesiones_restantes == 0:
            L.append("🏁 Último día: no es necesario vender si el concurso valora a mercado (confirma el reglamento).")
        if not bought_today:
            L.append("🧾 Actividad (15 % del puntaje): hoy aún no tienes compra. Haz tu micro-compra antes de las 14:40.")
    L.append("— Sólo alertas: la orden la envías tú en trii.")
    return "\n".join(L)


def validate_selection(selected: str, cfg: dict[str, Any]) -> None:
    assert_not_blacklisted([selected], cfg)
    assert selected in cfg["universe"]["mgc"], f"{selected} no es una acción MGC permitida"
