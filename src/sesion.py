"""¿De CUÁNDO es el movimiento que se está mostrando? Y ¿qué pasó fuera de horario?

Error que corrige (6-oct-2026): a las 6 de la mañana el bot decía "NUCO sube 13,2 % hoy" y "el Colcap sube 2,9 %", cuando eso había pasado AYER; antes de que
abra la bolsa repetía la sesión anterior como si fuera la de hoy, y además no miraba lo que la acción hacía antes de abrir o después de cerrar en Nueva York.

  · `movimiento(f, simbolo)`: el último cambio diario con su fecha, y si esa sesión es la de HOY o la última que hubo ("ayer").
  · `fuera_de_horario(f, simbolo)`: para lo que cotiza en Nueva York, el movimiento antes de abrir (pre-mercado, desde las 4:00 a. m. de allá) o después de
    cerrar (hasta las 8:00 p. m.), frente al último cierre. Es lo que anticipa cómo abrirá: en trii las acciones de EE. UU. siguen ese precio (en pesos).
Nunca lanzan: sin datos devuelven None."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

NY = "America/New_York"


def movimiento(f: Any, simbolo: str, ahora: dt.datetime | None = None) -> dict[str, Any] | None:
    """{precio, previo, fecha, es_hoy}: último cierre (o último precio si la sesión está abierta), el cierre anterior y de qué día es.
    `es_hoy` = la barra es de la fecha de hoy (hora de Bogotá/Nueva York: misma fecha durante el día de bolsa)."""
    def bajar() -> pd.DataFrame:
        d = f.yf.download(simbolo, period="7d", interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if d is None or d.empty:
            raise RuntimeError("Yahoo devolvió vacío")
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        return d[["Close"]].dropna()
    try:
        d = f._con_cache(f"sesion_mov|{simbolo}", 1.5, bajar, f"sesión {simbolo}")
        if d is None or len(d) < 2:
            return None
        hoy = (ahora or dt.datetime.now(dt.timezone(dt.timedelta(hours=-5)))).date()
        fecha = pd.Timestamp(d.index[-1]).date()
        precio, previo = float(d["Close"].iloc[-1]), float(d["Close"].iloc[-2])
        if precio <= 0 or previo <= 0:
            return None
        return dict(precio=precio, previo=previo, fecha=fecha, es_hoy=fecha >= hoy)
    except Exception:                                                                  # noqa: BLE001
        return None


def fuera_de_horario(f: Any, simbolo: str, ahora: dt.datetime | None = None) -> dict[str, Any] | None:
    """{estado: 'pre' | 'post', precio, cierre, cambio, hora}: movimiento fuera del horario normal de Nueva York frente al último cierre.
    None si el mercado está abierto, si no hay operaciones fuera de horario recientes o si el símbolo no cotiza allá."""
    def bajar() -> pd.DataFrame:
        h = f.yf.Ticker(simbolo).history(period="5d", interval="5m", prepost=True)
        if h is None or h.empty:
            raise RuntimeError("Yahoo devolvió vacío")
        return h[["Close"]].dropna()
    try:
        h = f._con_cache(f"sesion_ext|{simbolo}", 2, bajar, f"fuera de horario {simbolo}")
        if h is None or len(h) < 10:
            return None
        idx = pd.DatetimeIndex(h.index)
        idx = idx.tz_localize("UTC").tz_convert(NY) if idx.tz is None else idx.tz_convert(NY)
        c = pd.Series(h["Close"].to_numpy(), index=idx)
        ahora_ny = pd.Timestamp(ahora or dt.datetime.now(dt.timezone.utc)).tz_convert(NY)
        ultima = c.index[-1]
        if (ahora_ny - ultima).total_seconds() > 20 * 3600:                            # lo último es viejo (fin de semana, festivo): no hay nada "fuera de horario" vigente
            return None
        minutos = ultima.hour * 60 + ultima.minute
        regular = c[(c.index.hour * 60 + c.index.minute >= 570) & (c.index.hour * 60 + c.index.minute < 960)]        # 9:30–16:00
        if regular.empty:
            return None
        if 570 <= minutos < 960 and ultima.date() == ahora_ny.date() and 570 <= ahora_ny.hour * 60 + ahora_ny.minute < 965:
            return None                                                                # mercado abierto: no es "fuera de horario"
        estado = "pre" if minutos < 570 else "post"
        previas = regular[regular.index.date < ultima.date()] if estado == "pre" else regular[regular.index.date <= ultima.date()]
        if previas.empty or minutos in range(570, 960):
            return None
        cierre = float(previas.iloc[-1])
        precio = float(c.iloc[-1])
        if cierre <= 0 or abs(precio / cierre - 1) > 0.6:
            return None
        return dict(estado=estado, precio=precio, cierre=cierre, cambio=precio / cierre - 1, hora=ultima.tz_convert("America/Bogota").strftime("%H:%M"))
    except Exception:                                                                  # noqa: BLE001
        return None
