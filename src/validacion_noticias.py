"""Fase 3 · Historia de noticias (Finnhub, ≈ 1 año) para medir qué tan bueno es el proxy "volumen ≥ 2× + hueco de apertura".

Finnhub (plan gratuito) sólo devuelve noticias de empresa del último año; Alpha Vantage no tiene clave. Por eso el estudio largo (2015–2026) usa el proxy
y este módulo sólo sirve para CONTRASTARLO con noticias reales donde existen. Usa el mismo cliente (con límite de ritmo) y las mismas funciones de
puntuación del sistema en vivo."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .data_sources import FuentesDatos, NEWS_SYMBOL
from .semaforo import analizar_noticias

TOPE_FINNHUB = 200                              # si un tramo devuelve ≥ esto se parte a la mitad (Finnhub limita ≈ 250 por respuesta)


def _tramo(f: FuentesDatos, simbolo: str, desde: dt.date, hasta: dt.date, carpeta: Path) -> list[dict[str, Any]]:
    """Noticias de [desde, hasta] (ambos incluidos) con caché en disco; si el tramo llega al tope, se divide para no perder titulares."""
    archivo = carpeta / f"{desde.isoformat()}_{hasta.isoformat()}.json"
    if archivo.exists():
        return json.loads(archivo.read_text(encoding="utf-8"))
    items = f._reintentar(lambda: f._finnhub("company-news", symbol=simbolo, **{"from": desde.isoformat(), "to": hasta.isoformat()}), "Finnhub historia") or []
    if len(items) >= TOPE_FINNHUB and desde < hasta:
        mitad = desde + (hasta - desde) // 2
        items = _tramo(f, simbolo, desde, mitad, carpeta) + _tramo(f, simbolo, mitad + dt.timedelta(days=1), hasta, carpeta)
    else:
        items = [{"datetime": x["datetime"], "headline": x["headline"]} for x in items if x.get("headline") and x.get("datetime")]
    archivo.write_text(json.dumps(items), encoding="utf-8")
    return items


def descargar_historia(f: FuentesDatos, ticker: str, desde: dt.date, hasta: dt.date, carpeta: Path, dias_tramo: int = 7) -> pd.DataFrame:
    """Todos los titulares de `ticker` entre las fechas, deduplicados. Columnas: ts (UTC), titulo."""
    carpeta = carpeta / ticker
    carpeta.mkdir(parents=True, exist_ok=True)
    simbolo = NEWS_SYMBOL.get(ticker, ticker)
    filas: list[dict[str, Any]] = []
    ini = desde
    while ini <= hasta:
        fin = min(ini + dt.timedelta(days=dias_tramo - 1), hasta)
        filas += _tramo(f, simbolo, ini, fin, carpeta)
        ini = fin + dt.timedelta(days=1)
    if not filas:
        return pd.DataFrame(columns=["ts", "titulo"])
    d = pd.DataFrame(filas)
    d["ts"] = pd.to_datetime(d["datetime"], unit="s", utc=True)
    d = d.rename(columns={"headline": "titulo"})[["ts", "titulo"]].drop_duplicates().sort_values("ts").reset_index(drop=True)
    return d


def cruce(metricas: pd.DataFrame, diario: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Compara, día por día, el color con NOTICIAS REALES contra el color con el PROXY (volumen + hueco) y contra la cota 'amplia' (toda caída fuerte
    trae mala noticia). `metricas` sale de `validacion.metricas_vectorizadas`; `diario` de `analisis_diario` (mismo índice)."""
    from .validacion import clasificar_vectorizado
    idx = diario.index
    m = metricas.loc[idx]
    real = diario["negativa"].astype(bool)
    s = cfg["semaforo"]
    out = pd.DataFrame(index=idx)
    out["color_real"] = clasificar_vectorizado(m, cfg, negativa=real)
    out["color_proxy"] = clasificar_vectorizado(m, cfg)
    out["color_amplio"] = clasificar_vectorizado(m, cfg, negativa=pd.Series(True, index=idx))
    out["propia_con_volumen"] = (m["z"] <= s["z_caida"]) & (m["z_res"] <= s["z_res_caida"]) & (m["vol_rel"] >= s["vol_rel_min"])
    out["negativa_real"], out["n24"], out["sentimiento"] = real, diario["n24"], diario["sentimiento"]
    return out


def resumen_cruce(c: pd.DataFrame) -> dict[str, Any]:
    """Cifras para el informe: cuántos días, cuántos ROJO con noticias reales vs con el proxy y cuánto coinciden."""
    ok = c.dropna(subset=["color_real"])
    rr, rp = ok["color_real"] == "ROJO", ok["color_proxy"] == "ROJO"
    amp = ok["propia_con_volumen"]
    return dict(dias=int(len(ok)), propia_con_volumen=int(amp.sum()), con_noticia_negativa=int((amp & ok["negativa_real"]).sum()),
                rojo_real=int(rr.sum()), rojo_proxy=int(rp.sum()), ambos=int((rr & rp).sum()), solo_real=int((rr & ~rp).sum()), solo_proxy=int((~rr & rp).sum()),
                recall_proxy=float((rr & rp).sum() / rr.sum()) if rr.sum() else float("nan"),
                precision_proxy=float((rr & rp).sum() / rp.sum()) if rp.sum() else float("nan"),
                dias_con_noticias=int((ok["n24"] > 0).sum()))


def analisis_diario(noticias: pd.DataFrame, fechas: pd.DatetimeIndex, cfg: dict[str, Any], puntuador: Callable[[str], float]) -> pd.DataFrame:
    """Para cada sesión, lo que vería el radar de las 19:30 (Bogotá = 00:30 UTC del día siguiente): titulares de las últimas 24 h, sentimiento medio y si
    hay 'noticia negativa' según la MISMA regla del semáforo (`semaforo.analizar_noticias`). Columnas: n24, sentimiento, negativa."""
    ts = noticias["ts"].to_numpy(dtype="datetime64[ns]")
    titulos = noticias["titulo"].tolist()
    filas = []
    for d in fechas:
        ahora = (pd.Timestamp(d).tz_localize("UTC") + pd.Timedelta(days=1, minutes=30))
        a, b = np.searchsorted(ts, (ahora - pd.Timedelta(hours=24)).tz_convert(None).to_datetime64()), np.searchsorted(ts, ahora.tz_convert(None).to_datetime64(),
                                                                                                                         side="right")
        lista = [{"ts": str(noticias["ts"].iloc[i]), "titulo": titulos[i], "idioma": "en", "proveedor": "finnhub"} for i in range(a, b)]
        r = analizar_noticias(lista, None, ahora.to_pydatetime(), cfg, puntuador)
        filas.append(dict(n24=r["n24"], sentimiento=r["sentimiento"], negativa=bool(r["negativa"])))
    return pd.DataFrame(filas, index=fechas)
