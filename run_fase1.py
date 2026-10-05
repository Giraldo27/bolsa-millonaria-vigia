"""Fase 1: datos + validación de la selección de activos. Uso: python run_fase1.py [--refresh]"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import all_tickers, load_config
from src.data_loader import load_earnings, load_macro, load_universe_prices
from src.universe import best_triples, compute_metrics, correlation_matrix, rank_universe

pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 100, "display.float_format", "{:,.2f}".format)


def main() -> None:
    cfg = load_config()
    refresh = "--refresh" in sys.argv
    tick = all_tickers(cfg)
    prices = load_universe_prices(cfg, tick, refresh)
    missing = [t for t in tick if t not in prices]
    usdcop, gbpusd = (load_macro(k, cfg, refresh)["Close"] for k in ("usdcop", "gbpusd"))
    for k in ("brent", "ndx", "icolcap"):
        load_macro(k, cfg, refresh)
    earnings = load_earnings(cfg, tick, refresh)
    print(f"Sin datos: {missing}")
    print("Reportes por ticker:", earnings.groupby("ticker").size().to_dict())

    m = rank_universe(compute_metrics(prices, earnings, usdcop, gbpusd, cfg), cfg)
    m.to_csv(cfg["paths"]["outputs_dir"] / "fase1_ranking.csv")
    cols = ["grupo", "moneda", "atr_pct", "valor_cop_mm", "pct_vol0", "eventos_ventana_ano", "pre_n", "pre_media_pct",
            "pre_acierto_pct", "score", "rank", "rank_neutral"]
    print(m[cols].to_string())

    sysa = cfg["universe"]["system_assets"]
    print("\nActivos del sistema:\n", m.loc[sysa, cols].to_string())
    print("\nCorrelación diaria 2022+ (moneda local de cotización):\n",
          correlation_matrix(prices, sysa).round(2).to_string())
    print("\nMejores tríos (top 10 por puntaje):\n", best_triples(m, prices).head(8).to_string(index=False))
    for t in sysa:
        print(f"\nAperturas {t}: eventos de catálogo (últimos 6):")
        from src.universe import catalyst_events
        print(catalyst_events(t, earnings, cfg).tail(6).to_string(index=False))


if __name__ == "__main__":
    main()
