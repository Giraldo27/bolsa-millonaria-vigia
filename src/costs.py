"""Costos de trii: comisión escalonada, spread de entrada, slippage de salida y tamaño de la orden."""
from __future__ import annotations

from typing import Any


def commission(amount_cop: float, costs: dict[str, Any]) -> float:
    """Operación <= 5.000.000: fija (14.875). Mayor: 0,2975 % del monto (0,25 % + IVA)."""
    if amount_cop <= 0:
        return 0.0
    if amount_cop <= costs["fixed_threshold_cop"]:
        return float(costs["fixed_fee_cop"])
    return amount_cop * costs["pct_fee"]


def gross_from_budget(budget_cop: float, costs: dict[str, Any]) -> float:
    """Monto bruto V de la compra tal que V + comisión(V) = presupuesto (la comisión sale del mismo presupuesto)."""
    v_fixed = budget_cop - costs["fixed_fee_cop"]
    if v_fixed <= costs["fixed_threshold_cop"]:
        return max(v_fixed, 0.0)
    return budget_cop / (1 + costs["pct_fee"])


def entry_price(ref_price: float, asset: str, costs: dict[str, Any]) -> float:
    """Se compra por encima de la referencia (cierre o apertura): paga el spread."""
    return ref_price * (1 + costs["entry_spread"][asset])


def market_exit_price(ref_price: float, asset: str, costs: dict[str, Any]) -> float:
    """Salida a mercado (SL, tiempo, reporte, Brent): se vende por debajo de la referencia."""
    return ref_price * (1 - costs["exit_slippage"][asset])
