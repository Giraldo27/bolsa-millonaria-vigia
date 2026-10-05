"""Cartera de varias acciones: TRM, valoración con precios actuales y movimiento esperado del conjunto.

Aproximaciones declaradas: la TRM sale de Yahoo (COP=X) y puede diferir de la que usa trii; los movimientos esperados de la cartera usan los retornos en la moneda de
cada acción (sin el efecto del dólar). Tu ranking oficial lo da siempre trii (/rank)."""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

import pandas as pd

from . import concurso as C
from .config import group_of
from .data_sources import FuentesDatos
from .relevo import calcular_mee

TRM_MIN, TRM_MAX = 2_000.0, 8_000.0                                   # una TRM fuera de este rango es un tick erróneo de Yahoo (ya pasó con ×10 y ×100)


def moneda(ticker: str, cfg: dict[str, Any]) -> str:
    try:
        return "COP" if group_of(ticker, cfg) == "local" else "USD"
    except KeyError:                                                                   # acción desconocida: la validación del estado la rechaza con un mensaje claro
        return "USD"


def trm(f: FuentesDatos, cfg: dict[str, Any] | None = None) -> float | None:
    """Pesos por dólar. Yahoo en vivo; si falla o da un valor absurdo, el último cierre guardado. None si no hay nada confiable."""
    def vivo() -> float:
        v = float(f.yf.Ticker("COP=X").fast_info["last_price"])
        if not TRM_MIN <= v <= TRM_MAX:
            raise RuntimeError("TRM fuera de rango")
        return v
    try:
        v = f._con_cache("trm|COP", 30, vivo, "TRM")
        if v and TRM_MIN <= float(v) <= TRM_MAX:
            return float(v)
    except Exception:                                                                  # noqa: BLE001
        pass
    try:
        from .data_loader import load_macro
        c = load_macro("usdcop", cfg or {"paths": {}, "macro": {}})["Close"].dropna()
        v = float(c.iloc[-1])
        return v if TRM_MIN <= v <= TRM_MAX else None
    except Exception:                                                                  # noqa: BLE001
        return None


def monto_cop(ticker: str, cantidad: float, precio: float, tasa: float | None, cfg: dict[str, Any]) -> float | None:
    """Valor en pesos de una compra. Acciones de la BVC ya están en pesos; las de EE. UU. necesitan la TRM (None si no hay)."""
    if moneda(ticker, cfg) == "COP":
        return float(cantidad * precio)
    return float(cantidad * precio * tasa) if tasa else None


def valorar(f: FuentesDatos, cartera: list[dict[str, Any]], cfg: dict[str, Any], tasa: float | None = None) -> list[dict[str, Any]]:
    """Cada posición con su precio actual, rentabilidad en su moneda y valor aproximado en pesos. Una acción sin cotización queda con precio None (no se inventa)."""
    tasa = tasa if tasa is not None else trm(f, cfg)
    filas = []
    for p in cartera:
        t = p["ticker"]
        try:
            q = f.cotizacion(t)
        except Exception:                                                              # noqa: BLE001
            q = None
        precio = q["precio"] if q else None
        mon = moneda(t, cfg)
        if precio and p.get("cantidad"):
            valor = p["cantidad"] * precio * (tasa if (mon == "USD" and tasa) else (1.0 if mon == "COP" else float("nan")))
        elif precio and p.get("monto_cop"):                                            # vino de /pos: sin cantidad, se escala por el cambio de precio
            valor = p["monto_cop"] * precio / p["precio"]
        else:
            valor = float("nan")
        filas.append(dict(ticker=t, cantidad=p.get("cantidad"), precio_compra=p["precio"], precio=precio, moneda=mon, costo_cop=p.get("monto_cop"), valor_cop=valor,
                          rent=(precio / p["precio"] - 1) if precio else None,
                          cambio_hoy=(precio / q["cierre_previo"] - 1) if (q and q.get("cierre_previo")) else None))
    total = sum(x["valor_cop"] for x in filas if x["valor_cop"] == x["valor_cop"])
    for x in filas:
        x["peso"] = (x["valor_cop"] / total) if (total and x["valor_cop"] == x["valor_cop"]) else None
    return filas


def rent_total(filas: list[dict[str, Any]]) -> float | None:
    """Rentabilidad del conjunto en pesos (valor actual ÷ costo − 1) con las posiciones que tienen ambos datos; None si falta alguno."""
    ok = [x for x in filas if x["valor_cop"] == x["valor_cop"] and x.get("costo_cop")]
    if not ok or len(ok) < len(filas):
        return None
    return sum(x["valor_cop"] for x in ok) / sum(x["costo_cop"] for x in ok) - 1


def mee_cartera(f: FuentesDatos, filas: list[dict[str, Any]], ahora: dt.datetime, cfg: dict[str, Any]) -> tuple[float | None, str]:
    """Movimiento esperado hasta el próximo corte del CONJUNTO: variación diaria de 20 días de la cartera ponderada por su valor actual (retornos en la moneda de
    cada acción, días en común), con el mismo reloj que usa el sistema para una sola acción. None si faltan datos."""
    corte = C.corte_vigente(ahora, cfg)
    pesos = {x["ticker"]: x["peso"] for x in filas if x.get("peso")}
    if corte is None or len(pesos) < 2:
        return None, "sin_datos"
    series = {}
    for t in pesos:
        d = f.diario(t, 300)
        if d is None or d.empty:
            return None, "sin_datos"
        series[t] = d["Close"].pct_change()
    r = pd.concat(series, axis=1, sort=True).dropna().tail(cfg["mee"]["respaldo_sigma_dias"])
    if len(r) < 10:
        return None, "sin_datos"
    port = sum(r[t] * w for t, w in pesos.items()) / sum(pesos.values())
    sigma = float(port.std(ddof=1))
    mee, _ = calcular_mee(None, C.dias_calendario_al_corte(ahora, cfg), sigma, C.sesiones_restantes(ahora, cfg), False, "sin_datos", cfg["mee"]["respaldo"])
    return mee, "σ20 de la cartera"


__all__ = ["moneda", "trm", "monto_cop", "valorar", "rent_total", "mee_cartera", "math"]
