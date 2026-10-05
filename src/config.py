"""Carga de config.yaml y utilidades de rutas."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


CLAVES_ENTORNO = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "FINNHUB_API_KEY", "ALPHAVANTAGE_API_KEY")


def limpiar_valor(v: str) -> str:
    """Quita lo que se cuela al copiar una clave: BOM invisible (\\ufeff), espacios, saltos de línea y comillas."""
    return v.replace("﻿", "").strip().strip("\"'").strip()


def sanear_entorno(env: dict[str, str] | None = None) -> dict[str, str]:
    """Limpia in situ las claves del entorno. Un BOM al inicio de un secreto (pasó al subirlo a GitHub desde PowerShell) rompe el token de Telegram o la clave de
    Finnhub sin dar un error claro. Devuelve el mismo diccionario (por omisión os.environ)."""
    import os
    e = os.environ if env is None else env
    for k in CLAVES_ENTORNO:
        if k in e and e[k]:
            e[k] = limpiar_valor(e[k])
    return e


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    sanear_entorno()
    with open(path or ROOT / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k, v in cfg["paths"].items():
        p = ROOT / v
        p.mkdir(parents=True, exist_ok=True)
        cfg["paths"][k] = p
    return cfg


def all_tickers(cfg: dict[str, Any], include_blacklist: bool = False) -> list[str]:
    u = cfg["universe"]
    out = list(u["mgc"]) + list(u["etf"]) + list(u["local"])
    if include_blacklist:
        out += list(u["blacklist"])
    return out


def group_of(ticker: str, cfg: dict[str, Any]) -> str:
    u = cfg["universe"]
    for g in ("mgc", "etf", "local", "blacklist"):
        if ticker in u[g]:
            return g
    raise KeyError(ticker)
