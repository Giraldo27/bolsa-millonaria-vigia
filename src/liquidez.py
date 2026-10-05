"""Liquidez REAL en trii: cuánto se negocia cada acción en la Bolsa de Colombia, que es donde trii ejecuta tus órdenes.

El error que corrige (5-oct-2026): las acciones de EE. UU. que muestra trii (TSLACO, NVDACO, METACO…) son las del Mercado Global Colombiano y se negocian
MUY poco aquí (TSLACO movió 42 acciones en todo un día), aunque en Nueva York muevan miles de millones. Medir la liquidez con el volumen de Nueva York
decía "muy líquida" de algo que en trii casi no tiene compradores: cada entrada y salida cuesta caro. Aquí se mide con el libro local (`<TICKER>CO.CL`).

Niveles: buena (se puede entrar y salir sin mover el precio) · justa (sólo con orden límite y paciencia) · mala (evítala: salir puede costar mucho) ·
sin_dato (Yahoo no trae un historial fiable de esa acción en Colombia: se dice, no se inventa)."""
from __future__ import annotations

from typing import Any

import pandas as pd

from .config import group_of
from .data_loader import yahoo_symbol

BUENA, JUSTA, MALA, SIN_DATO = "buena", "justa", "mala", "sin_dato"


def simbolo_trii(ticker: str, cfg: dict[str, Any]) -> str | None:
    """Símbolo de Yahoo del libro donde trii negocia la acción: `<T>.CL` para las locales y `<T>CO.CL` para las globales (NUCO ya trae el CO)."""
    try:
        g = group_of(ticker, cfg)
    except KeyError:
        return None
    if g == "local":
        return yahoo_symbol(ticker, cfg)
    if g == "mgc":
        return (ticker if ticker.endswith("CO") else ticker + "CO") + ".CL"
    return None


def clasificar(valor_diario: pd.Series, volumen: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Nivel de liquidez a partir del valor negociado diario (pesos) de las últimas sesiones. Usa la MEDIANA (un día de bloque grande no engaña) y cuenta los
    días sin negociación. Con historial lleno de ceros no se concluye "mala": Yahoo no lo trae completo (pasa con NUCO) y se reporta sin_dato."""
    c = cfg["liquidez"]
    v, vol = valor_diario.tail(c["dias"]), volumen.tail(c["dias"])
    if len(v) < c["dias"] // 2:
        return dict(nivel=SIN_DATO, mediana_mm=None, hoy_mm=None, acciones_hoy=None, dias_sin_negociar=None, dias=len(v))
    sin = int((vol <= 0).sum())
    mediana, hoy = float(v.median()) / 1e6, float(v.iloc[-1]) / 1e6
    base = dict(mediana_mm=mediana, hoy_mm=hoy, acciones_hoy=float(vol.iloc[-1]), dias_sin_negociar=sin, dias=len(v))
    if sin > len(v) // 2:
        return dict(base, nivel=SIN_DATO if hoy >= c["buena_cop_mm"] else MALA)                 # historial incompleto pero hoy se negoció mucho: no se sabe
    if mediana >= c["buena_cop_mm"] and sin <= c["max_dias_sin_negociar"]:
        return dict(base, nivel=BUENA)
    return dict(base, nivel=JUSTA if (mediana >= c["justa_cop_mm"] and sin <= c["max_dias_sin_negociar"]) else MALA)


def medir(f: Any, ticker: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Liquidez de la acción en trii, con caché de 1 hora. Nunca lanza: si no hay datos devuelve nivel sin_dato."""
    sym = simbolo_trii(ticker, cfg)
    vacio = dict(ticker=ticker, simbolo=sym, nivel=SIN_DATO, mediana_mm=None, hoy_mm=None, acciones_hoy=None, dias_sin_negociar=None, dias=0)
    if sym is None:
        return vacio

    def bajar() -> pd.DataFrame:
        d = f.yf.download(sym, period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if d is None or d.empty:
            raise RuntimeError("Yahoo devolvió vacío")
        return d[["Close", "Volume"]].dropna(subset=["Close"])
    try:
        d = f._con_cache(f"liquidez|{sym}", 60, bajar, f"liquidez {ticker}")
        if d is None or len(d) == 0:
            return vacio
        return dict(vacio, **clasificar(d["Close"] * d["Volume"].fillna(0), d["Volume"].fillna(0), cfg))
    except Exception:                                                                  # noqa: BLE001 — sin datos no se inventa
        return vacio


def frase(l: dict[str, Any], orden_cop: float | None = None) -> str:
    """La liquidez en palabras sencillas (texto plano; quien llama le pone el formato)."""
    t = l["ticker"]
    if l["nivel"] == SIN_DATO:
        hoy = f" Hoy negoció cerca de $ {l['hoy_mm']:,.0f} millones.".replace(",", ".") if l.get("hoy_mm") else ""
        return f"💧 Liquidez de {t} en trii: no tengo un historial fiable para medirla.{hoy} Mira en trii cuántas acciones se han negociado hoy antes de operar, y usa orden límite."
    med = f"$ {l['mediana_mm']:,.0f} millones".replace(",", ".")
    acc = f"{l['acciones_hoy']:,.0f}".replace(",", ".")
    parte = ""
    if orden_cop and l["mediana_mm"]:
        pct = orden_cop / (l["mediana_mm"] * 1e6) * 100
        cuanto = "menos del 1%" if pct < 1 else f"el {pct:,.0f}%"
        parte = f" Una orden de $ {orden_cop / 1e6:,.0f} millones sería {cuanto} de lo que se negocia en un día.".replace(",", ".")
    if l["nivel"] == BUENA:
        return f"💧 Liquidez de {t} en trii: buena (se negocian unos {med} al día).{parte}"
    if l["nivel"] == JUSTA:
        return (f"💧 Liquidez de {t} en trii: JUSTA (unos {med} al día; hoy {acc} acciones).{parte} Sólo con orden límite y sin apuro: "
                "una orden a mercado te compra caro y te vende barato.")
    sin = f" y {l['dias_sin_negociar']} de los últimos {l['dias']} días no se negoció" if l.get("dias_sin_negociar") else ""
    return (f"💧 Liquidez de {t} en trii: MALA (unos {med} al día{sin}; hoy {acc} acciones).{parte} Entrar es fácil, salir no: "
            "puedes quedarte sin comprador o vender muy por debajo. Mejor no operarla.")
