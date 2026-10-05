"""Formato colombiano: punto de miles, coma decimal, COP."""
from __future__ import annotations

import math


def _sw(s: str) -> str:
    return s.replace(",", "§").replace(".", ",").replace("§", ".")


def num(x: float | None, d: int = 1) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n. d."
    return _sw(f"{x:,.{d}f}")


def pct(x: float | None, d: int = 1, sign: bool = False) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n. d."
    return _sw(f"{x * 100:{'+' if sign else ''},.{d}f}") + " %"


def cop(x: float | None, d: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n. d."
    s = _sw(f"{abs(x):,.{d}f}")
    return ("-" if x < 0 else "") + "$ " + s


def cop_m(x: float | None, d: int = 1) -> str:
    """Millones de pesos: $ 12,3 M."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n. d."
    s = _sw(f"{abs(x) / 1e6:,.{d}f}")
    return ("-" if x < 0 else "") + "$ " + s + " M"


def cop_big(x: float | None) -> str:
    """Montos grandes legibles: billones (10^12), mil millones (10^9) o millones (10^6)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n. d."
    if abs(x) >= 1e12:
        return "$ " + _sw(f"{x / 1e12:,.1f}") + " billones"
    if abs(x) >= 1e9:
        return "$ " + _sw(f"{x / 1e9:,.1f}") + " mil millones"
    return "$ " + _sw(f"{x / 1e6:,.1f}") + " millones"


def cls(x: float | None) -> str:
    """Clase CSS según el signo (verde ganancia / rojo pérdida)."""
    if x is None or (isinstance(x, float) and math.isnan(x)) or x == 0:
        return "neu"
    return "pos" if x > 0 else "neg"
