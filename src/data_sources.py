"""Fuentes de datos en vivo: yfinance (precios diarios e intradía, ^NDX, opciones, reportes, noticias) y Finnhub /
Alpha Vantage (noticias y reportes). Incluye caché en disco con TTL, reintentos con backoff y control de límites.

Todo lo que no se pueda obtener con datos gratuitos devuelve vacío y queda registrado en `FuentesDatos.avisos`;
NUNCA se inventan valores."""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import json
import xml.etree.ElementTree as ET
import math
import os
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .config import ROOT
from .data_loader import clean_ohlc, yahoo_symbol

UTC = dt.timezone.utc

# símbolo para noticias/reportes en Finnhub (sólo cubre EE. UU. y ADR; las acciones .CL no tienen cobertura)
NEWS_SYMBOL = {"NUCO": "NU", "ECOPETROL": "EC", "PFBCOLOM": "CIB", "BCOLOMBIA": "CIB"}


class FuenteError(RuntimeError):
    """Una fuente falló después de los reintentos."""


def con_reintentos(fn: Callable[[], Any], intentos: int = 3, espera_s: float = 1.5, factor: float = 2.0,
                   dormir: Callable[[float], None] = time.sleep, nombre: str = "fuente") -> Any:
    """Ejecuta fn() con reintentos y backoff exponencial. Lanza FuenteError si se agotan los intentos."""
    ultimo: Exception | None = None
    for i in range(intentos):
        try:
            return fn()
        except FuenteError:
            raise                                   # error definitivo (clave inválida, sin permiso): no se reintenta
        except Exception as e:                      # noqa: BLE001 — cualquier error de red/API se reintenta
            ultimo = e
            if i < intentos - 1:
                dormir(espera_s * (factor ** i))
    raise FuenteError(f"{nombre}: {type(ultimo).__name__}: {str(ultimo)[:160]}") from ultimo


class CacheDisco:
    """Caché en disco con TTL en minutos. DataFrames en parquet; el resto en JSON."""

    def __init__(self, carpeta: Path, ahora: Callable[[], dt.datetime] = lambda: dt.datetime.now(UTC)):
        self.dir = Path(carpeta)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ahora = ahora

    def _ruta(self, clave: str, ext: str) -> Path:
        h = hashlib.sha1(clave.encode()).hexdigest()[:16]
        seguro = "".join(c if c.isalnum() else "_" for c in clave)[:40]
        return self.dir / f"{seguro}_{h}.{ext}"

    def leer(self, clave: str, ttl_min: float) -> Any | None:
        for ext in ("parquet", "json"):
            p = self._ruta(clave, ext)
            if p.exists() and (self.ahora().timestamp() - p.stat().st_mtime) / 60 <= ttl_min:
                try:
                    return pd.read_parquet(p) if ext == "parquet" else json.loads(p.read_text(encoding="utf-8"))
                except Exception:               # noqa: BLE001 — caché dañada: se ignora
                    return None
        return None

    def leer_vencida(self, clave: str) -> Any | None:
        """Último valor guardado aunque el TTL haya vencido (respaldo cuando la fuente falla)."""
        return self.leer(clave, ttl_min=1e12)

    def escribir(self, clave: str, valor: Any) -> None:
        if isinstance(valor, pd.DataFrame):
            valor.to_parquet(self._ruta(clave, "parquet"))
        else:
            self._ruta(clave, "json").write_text(json.dumps(valor, ensure_ascii=False, default=str), encoding="utf-8")


class Limitador:
    """Ventana deslizante de N llamadas por minuto (Finnhub: 60/min)."""

    def __init__(self, por_minuto: int, reloj: Callable[[], float] = time.monotonic, dormir: Callable[[float], None] = time.sleep):
        self.n, self.reloj, self.dormir = por_minuto, reloj, dormir
        self.marcas: deque[float] = deque()
        self.lock = threading.Lock()

    def esperar(self) -> None:
        with self.lock:
            ahora = self.reloj()
            while self.marcas and ahora - self.marcas[0] >= 60:
                self.marcas.popleft()
            if len(self.marcas) >= self.n:
                self.dormir(60 - (ahora - self.marcas[0]) + 0.05)
                self.marcas.popleft()
            self.marcas.append(self.reloj())


@dataclass
class Noticia:
    ts: str                    # ISO UTC
    titulo: str
    resumen: str = ""
    fuente: str = ""
    url: str = ""
    proveedor: str = ""
    ticker: str = ""
    sentimiento_proveedor: float | None = None     # sólo Alpha Vantage lo entrega
    idioma: str = "en"                             # 'es' para titulares de acciones locales (Google News en español)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalizar_titulo(t: str) -> str:
    return " ".join("".join(c.lower() if c.isalnum() else " " for c in t).split())


def deduplicar(noticias: list[Noticia]) -> list[Noticia]:
    visto, out = set(), []
    for n in sorted(noticias, key=lambda x: x.ts, reverse=True):
        k = normalizar_titulo(n.titulo)
        if k and k not in visto:
            visto.add(k)
            out.append(n)
    return out


def _utc(ts: Any) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").isoformat() if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts, tz="UTC").isoformat()


def noticias_de_yfinance(items: list[dict[str, Any]], ticker: str) -> list[Noticia]:
    """Acepta el formato nuevo ({'content': {...}}) y el antiguo ({'title','providerPublishTime'})."""
    out = []
    for it in items or []:
        c = it.get("content", it)
        titulo = c.get("title") or ""
        pub = c.get("pubDate") or c.get("providerPublishTime")
        if not titulo or pub is None:
            continue
        ts = _utc(pd.Timestamp(pub, unit="s") if isinstance(pub, (int, float)) else pub)
        url = (c.get("canonicalUrl") or {}).get("url") or c.get("link") or ""
        prov = (c.get("provider") or {}).get("displayName") or c.get("publisher") or "Yahoo"
        out.append(Noticia(ts, titulo, c.get("summary") or "", prov, url, "yfinance", ticker))
    return out


def noticias_de_finnhub(items: list[dict[str, Any]], ticker: str) -> list[Noticia]:
    out = []
    for x in items or []:
        if not x.get("headline") or not x.get("datetime"):
            continue
        out.append(Noticia(dt.datetime.fromtimestamp(int(x["datetime"]), UTC).isoformat(), x["headline"], x.get("summary", ""),
                           x.get("source", ""), x.get("url", ""), "finnhub", ticker))
    return out


def noticias_de_rss(contenido: bytes | str, ticker: str, proveedor: str, idioma: str = "en") -> list[Noticia]:
    """Parsea un feed RSS (Yahoo Finance o Google News). Google agrega " - Medio" al final del titular: se separa para deduplicar."""
    try:
        raiz = ET.fromstring(contenido)
    except ET.ParseError as e:
        raise RuntimeError(f"RSS inválido: {e}") from e
    out = []
    for it in raiz.iter("item"):
        titulo, fecha = (it.findtext("title") or "").strip(), it.findtext("pubDate")
        if not titulo or not fecha:
            continue
        medio = (it.findtext("source") or "").strip()
        if proveedor == "google_rss" and " - " in titulo:
            titulo, _, resto = titulo.rpartition(" - ")
            medio = medio or resto
        try:
            ts = email.utils.parsedate_to_datetime(fecha).astimezone(UTC).isoformat()
        except (TypeError, ValueError):
            continue
        out.append(Noticia(ts, titulo, (it.findtext("description") or "")[:200] if proveedor == "yahoo_rss" else "", medio or "Yahoo",
                           (it.findtext("link") or ""), proveedor, ticker, None, idioma))
    return out


def noticias_de_alphavantage(data: dict[str, Any], ticker: str) -> list[Noticia]:
    out = []
    for x in (data or {}).get("feed", []):
        try:
            ts = dt.datetime.strptime(x["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=UTC).isoformat()
        except (KeyError, ValueError):
            continue
        sent = next((float(s["ticker_sentiment_score"]) for s in x.get("ticker_sentiment", []) if s.get("ticker") == ticker), None)
        out.append(Noticia(ts, x.get("title", ""), x.get("summary", ""), x.get("source", ""), x.get("url", ""), "alphavantage", ticker, sent))
    return [n for n in out if n.titulo]


def iv_atm(calls: pd.DataFrame, puts: pd.DataFrame, spot: float, oi_min: int, spread_rel_max: float, vecinos: int = 2
           ) -> dict[str, Any]:
    """IV ATM = promedio de la IV de calls y puts en los strikes más cercanos al spot con cotización válida
    (bid y ask > 0, spread relativo razonable, open interest suficiente). calidad: ok | degradada | sin_datos."""
    vals, usados = [], []
    for df in (calls, puts):
        if df is None or df.empty:
            continue
        d = df.assign(dist=(df["strike"] - spot).abs()).sort_values("dist").head(vecinos + 1)
        for _, r in d.iterrows():
            iv, bid, ask = r.get("impliedVolatility"), r.get("bid"), r.get("ask")
            oi = r.get("openInterest")
            if not (iv and 0.05 < iv < 5.0):
                continue
            if bid and ask and bid > 0 and ask > 0 and (ask - bid) / ((ask + bid) / 2) <= spread_rel_max and (oi or 0) >= oi_min:
                vals.append(float(iv))
                usados.append(float(r["strike"]))
    if not vals:
        return {"iv": None, "calidad": "sin_datos", "strikes": []}
    return {"iv": float(np.mean(vals)), "calidad": "ok" if len(vals) >= 2 else "degradada", "strikes": sorted(set(usados))}


def elegir_vencimiento(vencimientos: list[str], fecha_corte: dt.date, dias_min: int = 1) -> str | None:
    """Primer vencimiento posterior al corte; si no hay, el último disponible."""
    if not vencimientos:
        return None
    limite = fecha_corte + dt.timedelta(days=dias_min)
    for v in sorted(vencimientos):
        if pd.Timestamp(v).date() >= limite:
            return v
    return sorted(vencimientos)[-1]


@dataclass
class FuentesDatos:
    cfg: dict[str, Any]
    env: dict[str, str] = field(default_factory=lambda: dict(os.environ))
    yf: Any = None                                    # módulo yfinance (inyectable para pruebas)
    http: Any = None                                  # módulo/sesión con .get (requests)
    ahora: Callable[[], dt.datetime] = lambda: dt.datetime.now(UTC)
    dormir: Callable[[float], None] = time.sleep
    avisos: list[str] = field(default_factory=list)
    cache_dir: Path | None = None                    # por defecto fuentes.cache_dir del config (las pruebas usan una carpeta temporal)
    proveedores_estado: dict[str, bool] = field(default_factory=dict)   # último resultado de cada proveedor de noticias (para el aviso de salud)

    def __post_init__(self) -> None:
        if self.yf is None:
            import yfinance
            self.yf = yfinance
        if self.http is None:
            import requests
            self.http = requests
        f = self.cfg["fuentes"]
        self.cache = CacheDisco(self.cache_dir or ROOT / f["cache_dir"], self.ahora)
        self.ttl = f["ttl_min"]
        self.lim_finnhub = Limitador(f["finnhub"]["limite_por_min"], dormir=self.dormir)

    # ---------- utilidades ----------
    def _reintentar(self, fn: Callable[[], Any], nombre: str) -> Any:
        r = self.cfg["fuentes"]["reintentos"]
        return con_reintentos(fn, r["intentos"], r["espera_s"], r["factor"], self.dormir, nombre)

    def libro_bvc(self) -> dict[str, dict[str, Any]] | None:
        """Lo negociado en la Bolsa de Colombia, acción por acción (locales y globales): {símbolo: precio, acciones y valor de la última sesión, promedios}.
        Son los datos de la bolsa con 15 minutos de retraso, tomados de un tablero público (la página de la bolsa exige una llave y no se fuerza)."""
        c = (self.cfg.get("liquidez") or {}).get("libro_bvc") or {}
        if not c.get("activo"):
            return None

        def bajar() -> dict[str, dict[str, Any]]:
            cols = ["name", "close", "volume", "Value.Traded", "average_volume_10d_calc", "average_volume_30d_calc", "average_volume_60d_calc"]
            cuerpo = {"filter": [{"left": "exchange", "operation": "equal", "right": "BVC"}], "columns": cols, "range": [0, 600]}
            r = self.http.post(c["url"], json=cuerpo, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}")
            out = {}
            for x in r.json().get("data") or []:
                nombre, precio, vol, valor, p10, p30, p60 = x["d"]
                if not nombre or not precio or precio <= 0:
                    continue
                mm = lambda acciones: None if acciones is None else float(acciones) * float(precio) / 1e6      # noqa: E731 — acciones → millones de pesos
                out[nombre] = dict(simbolo=nombre, precio=float(precio), acciones_ult=float(vol or 0), valor_ult_mm=float(valor or 0) / 1e6,
                                   prom10_mm=mm(p10), prom30_mm=mm(p30), prom60_mm=mm(p60))
            if len(out) < 20:
                raise RuntimeError("tabla de la bolsa incompleta")
            return out
        try:
            return self._con_cache("libro_bvc", c.get("cache_min", 2), bajar, "libro de la Bolsa de Colombia")
        except Exception:                                                              # noqa: BLE001 — sin esta fuente se sigue con la otra
            return None

    def _con_cache(self, clave: str, ttl: float, fn: Callable[[], Any], nombre: str) -> Any | None:
        """Cache fresca → fn() con reintentos → cache vencida como último recurso (con aviso). None si no hay nada."""
        v = self.cache.leer(clave, ttl)
        if v is not None:
            return v
        try:
            v = self._reintentar(fn, nombre)
            if v is not None and not (isinstance(v, (pd.DataFrame, list, dict)) and len(v) == 0):
                self.cache.escribir(clave, v)
            return v
        except FuenteError as e:
            viejo = self.cache.leer_vencida(clave)
            self.avisos.append(f"{e} — " + ("se usa caché vencida" if viejo is not None else "sin datos"))
            return viejo

    def _finnhub(self, ruta: str, **params: Any) -> Any:
        key = (self.env.get("FINNHUB_API_KEY") or "").replace("﻿", "").strip()
        if not key:
            raise FuenteError("Finnhub: falta FINNHUB_API_KEY en .env")
        c = self.cfg["fuentes"]["finnhub"]

        def llamar() -> Any:
            self.lim_finnhub.esperar()
            r = self.http.get(f"{c['base_url']}/{ruta}", params={**params, "token": key}, timeout=c["timeout_s"])
            if r.status_code == 429:
                raise RuntimeError("límite de Finnhub (429)")
            if r.status_code in (401, 403):
                raise FuenteError(f"Finnhub {r.status_code}: {r.text[:80]}")
            r.raise_for_status()
            return r.json()
        return llamar()

    # ---------- precios ----------
    def _limpiar_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
        df = df[cols].dropna(subset=["Close"])
        return df[~df.index.duplicated()].sort_index()

    def diario(self, ticker: str, dias: int | None = None) -> pd.DataFrame:
        """OHLCV diario (moneda de cotización). Índice sin zona horaria."""
        dias = dias or self.cfg["fuentes"]["yahoo"]["dias_diario"]
        sym = yahoo_symbol(ticker, self.cfg)

        def bajar() -> pd.DataFrame:
            d = self.yf.download(sym, period=f"{int(dias * 1.5)}d", interval="1d", auto_adjust=False, progress=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d is None or d.empty:
                raise RuntimeError("Yahoo devolvió vacío")
            d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
            d, corregidas = clean_ohlc(self._limpiar_ohlc(d))       # ticks erróneos de Yahoo (frecuentes en acciones locales)
            if corregidas:
                self.avisos.append(f"{ticker}: {corregidas} barras diarias con ticks erróneos corregidas")
            return d.tail(dias)
        out = self._con_cache(f"diario|{sym}|{dias}", self.ttl["diario"], bajar, f"yfinance diario {ticker}")
        return out if out is not None else pd.DataFrame()

    def intradia(self, ticker: str, dias: int | None = None, intervalo: str | None = None) -> pd.DataFrame:
        """Barras intradía (15 min por defecto). Índice en hora de Nueva York para EE. UU. / Bogotá para locales."""
        y = self.cfg["fuentes"]["yahoo"]
        dias, intervalo = dias or y["dias_intradia"], intervalo or y["intervalo_intradia"]
        sym = yahoo_symbol(ticker, self.cfg)

        def bajar() -> pd.DataFrame:
            d = self.yf.download(sym, period=f"{dias}d", interval=intervalo, auto_adjust=False, progress=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d is None or d.empty:
                raise RuntimeError("Yahoo devolvió vacío (¿mercado cerrado o símbolo sin intradía?)")
            return self._limpiar_ohlc(d)
        out = self._con_cache(f"intradia|{sym}|{dias}|{intervalo}", self.ttl["intradia"], bajar, f"yfinance intradía {ticker}")
        return out if out is not None else pd.DataFrame()

    def ndx_diario(self, dias: int | None = None) -> pd.DataFrame:
        dias = dias or self.cfg["fuentes"]["yahoo"]["dias_diario"]

        def bajar() -> pd.DataFrame:
            d = self.yf.download("^NDX", period=f"{int(dias * 1.5)}d", interval="1d", auto_adjust=False, progress=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
            return self._limpiar_ohlc(d).tail(dias)
        out = self._con_cache(f"ndx|diario|{dias}", self.ttl["ndx"] * 12, bajar, "yfinance ^NDX diario")
        return out if out is not None else pd.DataFrame()

    def ndx_intradia(self) -> pd.DataFrame:
        y = self.cfg["fuentes"]["yahoo"]

        def bajar() -> pd.DataFrame:
            d = self.yf.download("^NDX", period=f"{y['dias_intradia']}d", interval=y["intervalo_intradia"], auto_adjust=False, progress=False, threads=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d is None or d.empty:
                raise RuntimeError("vacío")
            return self._limpiar_ohlc(d)
        out = self._con_cache("ndx|intradia", self.ttl["ndx"], bajar, "yfinance ^NDX intradía")
        return out if out is not None else pd.DataFrame()

    def cotizacion(self, ticker: str) -> dict[str, Any] | None:
        """Último precio y cierre previo. yfinance primero; Finnhub /quote como respaldo (sólo EE. UU.)."""
        sym = yahoo_symbol(ticker, self.cfg)

        def yf_q() -> dict[str, Any]:
            fi = self.yf.Ticker(sym).fast_info
            return {"precio": float(fi["last_price"]), "cierre_previo": float(fi["previous_close"]), "fuente": "yfinance"}

        def fh_q() -> dict[str, Any]:
            q = self._finnhub("quote", symbol=NEWS_SYMBOL.get(ticker, ticker))
            if not q.get("c"):
                raise RuntimeError("Finnhub sin cotización")
            return {"precio": float(q["c"]), "cierre_previo": float(q["pc"]), "fuente": "finnhub"}
        v = self._con_cache(f"cotizacion|{sym}", self.ttl["cotizacion"], yf_q, f"yfinance cotización {ticker}")
        if v is None and ticker not in self.cfg["universe"]["local"]:
            v = self._con_cache(f"cotizacion_fh|{ticker}", self.ttl["cotizacion"], fh_q, f"Finnhub cotización {ticker}")
        return v

    # ---------- opciones ----------
    def opciones_iv(self, ticker: str, fecha_corte: dt.date) -> dict[str, Any]:
        """IV ATM del primer vencimiento posterior al corte. Las acciones locales no tienen opciones → sin_datos."""
        if ticker in self.cfg["universe"]["local"]:
            return {"iv": None, "calidad": "sin_datos", "motivo": "sin opciones listadas (acción local)"}
        sym = yahoo_symbol(ticker, self.cfg)
        o = self.cfg["fuentes"]["opciones"]

        def bajar() -> dict[str, Any]:
            tk = self.yf.Ticker(sym)
            ven = list(tk.options or [])
            v = elegir_vencimiento(ven, fecha_corte, o["dias_min_tras_corte"])
            if v is None:
                raise RuntimeError("sin vencimientos")
            spot = float(tk.fast_info["last_price"])
            ch = tk.option_chain(v)
            r = iv_atm(ch.calls, ch.puts, spot, o["oi_min"], o["spread_rel_max"])
            return {**r, "vencimiento": v, "spot": spot, "ticker": ticker}
        v = self._con_cache(f"opciones|{sym}|{fecha_corte}", self.ttl["opciones"], bajar, f"yfinance opciones {ticker}")
        return v if v is not None else {"iv": None, "calidad": "sin_datos", "motivo": "fuente no disponible"}

    # ---------- reportes ----------
    def reportes(self, ticker: str, dias_adelante: int = 60) -> list[dict[str, Any]]:
        """Próximas fechas de resultados (yfinance + Finnhub). Cada elemento: fecha, hora (bmo/amc/dmh/''), fuente."""
        if ticker in self.cfg["universe"]["local"] and ticker not in NEWS_SYMBOL:
            return []
        hoy = self.ahora().date()
        hasta = hoy + dt.timedelta(days=dias_adelante)
        sym = NEWS_SYMBOL[ticker] if ticker in NEWS_SYMBOL else yahoo_symbol(ticker, self.cfg)    # ADR o acción de EE. UU.
        out: list[dict[str, Any]] = []

        def yf_e() -> list[dict[str, Any]]:
            d = self.yf.Ticker(sym).get_earnings_dates(limit=8)
            res = []
            for ts in d.index:
                f = pd.Timestamp(ts)
                hora = "amc" if f.hour >= 12 else "bmo"
                res.append({"fecha": f.date().isoformat(), "hora": hora, "fuente": "yfinance"})
            return res

        def fh_e() -> list[dict[str, Any]]:
            j = self._finnhub("calendar/earnings", symbol=NEWS_SYMBOL.get(ticker, ticker), **{"from": hoy.isoformat(), "to": hasta.isoformat()})
            return [{"fecha": x["date"], "hora": x.get("hour", ""), "fuente": "finnhub"} for x in j.get("earningsCalendar", [])]
        for fn, nombre, key in ((yf_e, f"yfinance reportes {ticker}", f"rep_yf|{sym}"), (fh_e, f"Finnhub reportes {ticker}", f"rep_fh|{ticker}")):
            v = self._con_cache(key, self.ttl["reportes"], fn, nombre)
            out += v or []
        futuros = [x for x in out if hoy <= pd.Timestamp(x["fecha"]).date() <= hasta]
        # una fila por fecha; si ambas fuentes coinciden se conserva la hora de Finnhub
        por_fecha: dict[str, dict[str, Any]] = {}
        for x in sorted(futuros, key=lambda r: r["fuente"] != "finnhub"):
            por_fecha.setdefault(x["fecha"], x)
        return sorted(por_fecha.values(), key=lambda r: r["fecha"])

    # ---------- noticias ----------
    def noticias(self, ticker: str, dias: int = 3, limite: int | None = -1) -> list[Noticia] | None:
        """Titulares de los últimos `dias` días: Finnhub (principal) → yfinance → Alpha Vantage (sólo si siguen vacíos).
        Devuelve None cuando NO hay datos fiables (sin cobertura gratuita, sin clave o fuentes caídas) y [] cuando la fuente
        respondió pero no hubo titulares: el semáforo distingue las dos situaciones. `limite=-1` usa el máximo del config."""
        simbolo = NEWS_SYMBOL.get(ticker, ticker)
        local = ticker in self.cfg["universe"]["local"]
        hoy = self.ahora().date()
        desde = hoy - dt.timedelta(days=dias)
        n_cfg = self.cfg["fuentes"]["noticias"]
        res: list[Noticia] = []
        respondio = False
        poco = lambda: len(res) < n_cfg["min_titulares_respaldo"]                  # noqa: E731 — ¿hace falta un respaldo?

        for prov in n_cfg["proveedores"]:
            ok: bool | None = None                                                 # True = respondió, False = falló, None = no se consultó
            if prov == "finnhub" and (not local or ticker in NEWS_SYMBOL):
                v = self._con_cache(f"news_fh|{simbolo}|{dias}", self.ttl["noticias"],
                                    lambda: self._finnhub("company-news", symbol=simbolo, **{"from": desde.isoformat(), "to": hoy.isoformat()}),
                                    f"Finnhub noticias {ticker}")
                ok = v is not None
                res += noticias_de_finnhub(v or [], ticker)
                respondio = respondio or ok
            elif prov == "yfinance" and not local and poco():
                v = self._con_cache(f"news_yf|{simbolo}", self.ttl["noticias"], lambda: self.yf.Ticker(simbolo).news or [], f"yfinance noticias {ticker}")
                ok = v is not None
                res += noticias_de_yfinance(v or [], ticker)
                respondio = respondio or bool(v)                                   # hoy yfinance devuelve [] siempre: no cuenta como respuesta
            elif prov == "yahoo_rss" and not local and poco():
                v = self._con_cache(f"news_yrss|{simbolo}", self.ttl["noticias"], lambda: self._rss_yahoo(simbolo), f"Yahoo RSS {ticker}")
                ok = v is not None
                res += noticias_de_rss(v, ticker, "yahoo_rss") if v else []
                respondio = respondio or bool(v)                                   # un feed vacío no cuenta como respuesta fiable
            elif prov == "google_rss" and (local or poco()):
                q = n_cfg["consultas_google"].get(ticker) if local else f"{simbolo} stock"
                if q is None and local:
                    q = f"{ticker} acción Colombia"
                v = self._con_cache(f"news_grss|{q}", self.ttl["noticias"], lambda: self._rss_google(q, "es" if local else "en"),
                                    f"Google News RSS {ticker}")
                ok = v is not None
                res += noticias_de_rss(v, ticker, "google_rss", "es" if local else "en") if v else []
                respondio = respondio or ok                                        # Google respondió (aunque sin titulares): cobertura parcial
            elif prov == "alphavantage" and not res and self.env.get("ALPHAVANTAGE_API_KEY"):
                av = self._alphavantage(simbolo, ticker, desde)
                res += av
                ok = bool(av)
                respondio = respondio or ok
            if ok is not None:
                self.proveedores_estado[prov] = ok
        if not respondio:
            if local and ticker not in NEWS_SYMBOL:
                self.avisos.append(f"{ticker}: sin noticias (Google News no respondió)")
            return None
        lim = pd.Timestamp(dt.datetime.combine(desde, dt.time.min), tz="UTC")
        res = deduplicar([n for n in res if pd.Timestamp(n.ts) >= lim])
        tope = n_cfg["max_titulares"] if limite == -1 else limite
        return res[:tope] if tope else res

    def traducir(self, texto: str) -> str | None:
        """Traduce un titular del inglés al español (MyMemory, gratuito, con límite diario). Caché de 14 días por titular. Devuelve
        None si la traducción no está disponible (límite agotado, sin internet): el mensaje muestra entonces el titular original."""
        t = self.cfg["fuentes"].get("traduccion", {})
        if not t.get("activo") or not texto.strip():
            return None
        clave = f"trad|{texto}"
        v = self.cache.leer(clave, t["ttl_dias"] * 24 * 60)
        if v is not None:
            return v or None
        f = self.cache.dir / "trad_contador.json"
        hoy = self.ahora().date().isoformat()
        c = json.loads(f.read_text()) if f.exists() else {}
        if c.get("fecha") != hoy:
            c = {"fecha": hoy, "n": 0}
        if c["n"] + len(texto) > t["limite_diario_chars"]:
            self.avisos.append("Traducción: límite diario agotado (se muestran los titulares en inglés)")
            return None
        try:
            r = self.http.get("https://api.mymemory.translated.net/get", params={"q": texto[:480], "langpair": "en|es"}, timeout=t["timeout_s"])
            r.raise_for_status()
            j = r.json()
            out = j["responseData"]["translatedText"]
            if j.get("responseStatus") != 200 or "MYMEMORY WARNING" in out.upper():
                raise RuntimeError("MyMemory sin cupo")
        except Exception:                                                                  # noqa: BLE001 — sin traducción se muestra el original
            return None
        c["n"] += len(texto)
        f.write_text(json.dumps(c))
        self.cache.escribir(clave, out)
        return out

    def _rss(self, url: str, params: dict[str, Any]) -> bytes:
        r = self.http.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=self.cfg["fuentes"]["noticias"]["rss_timeout_s"])
        r.raise_for_status()
        contenido = r.content if hasattr(r, "content") else r.text.encode()
        ET.fromstring(contenido)                                                   # valida que sea XML (si no, se reintenta y se registra el fallo)
        return contenido.decode("utf-8", "replace") if isinstance(contenido, bytes) else contenido

    def _rss_yahoo(self, simbolo: str) -> str:
        return self._rss("https://feeds.finance.yahoo.com/rss/2.0/headline", {"s": simbolo, "region": "US", "lang": "en-US"})

    def _rss_google(self, consulta: str, idioma: str) -> str:
        cuando = self.cfg["fuentes"]["noticias"]["google_cuando"]
        p = ({"hl": "es-419", "gl": "CO", "ceid": "CO:es-419"} if idioma == "es" else {"hl": "en-US", "gl": "US", "ceid": "US:en"})
        return self._rss("https://news.google.com/rss/search", {"q": f"{consulta} when:{cuando}", **p})

    def base_noticias_diaria(self, ticker: str, dias: int = 30, tramo: int = 7) -> float | None:
        """Promedio de titulares por día en los últimos `dias` días, SIN contar las últimas 24 h. Consulta Finnhub por tramos de
        `tramo` días (una consulta devuelve como máximo ~250 titulares). Caché de 12 h. None si no hay datos."""
        if ticker in self.cfg["universe"]["local"] and ticker not in NEWS_SYMBOL:
            return None
        simbolo = NEWS_SYMBOL.get(ticker, ticker)
        hoy = self.ahora().date()

        def contar() -> dict[str, Any]:
            total, fin = 0, hoy - dt.timedelta(days=1)
            ini = hoy - dt.timedelta(days=dias)
            while ini < fin:
                hasta = min(ini + dt.timedelta(days=tramo - 1), fin)
                total += len(self._finnhub("company-news", symbol=simbolo, **{"from": ini.isoformat(), "to": hasta.isoformat()}))
                ini = hasta + dt.timedelta(days=1)
            return {"n": total, "dias": (fin - (hoy - dt.timedelta(days=dias))).days + 1}
        v = self._con_cache(f"news_base|{simbolo}|{dias}|{hoy}", 12 * 60, contar, f"Finnhub base de noticias {ticker}")
        return (v["n"] / v["dias"]) if v and v.get("dias") else None

    def _alphavantage(self, simbolo: str, ticker: str, desde: dt.date) -> list[Noticia]:
        a = self.cfg["fuentes"]["alphavantage"]
        f = self.cache.dir / "av_contador.json"
        hoy = self.ahora().date().isoformat()
        c = json.loads(f.read_text()) if f.exists() else {}
        if c.get("fecha") != hoy:
            c = {"fecha": hoy, "n": 0}
        if c["n"] >= a["limite_por_dia"]:
            self.avisos.append("Alpha Vantage: límite diario agotado")
            return []
        c["n"] += 1
        f.write_text(json.dumps(c))

        def llamar() -> dict[str, Any]:
            r = self.http.get(a["base_url"], params={"function": "NEWS_SENTIMENT", "tickers": simbolo, "limit": 50, "apikey": self.env["ALPHAVANTAGE_API_KEY"],
                                                      "time_from": desde.strftime("%Y%m%dT0000")}, timeout=a["timeout_s"])
            r.raise_for_status()
            j = r.json()
            if "feed" not in j:
                raise RuntimeError(str(j)[:100])
            return j
        v = self._con_cache(f"news_av|{simbolo}|{desde}", self.ttl["noticias"], llamar, f"Alpha Vantage {ticker}")
        return noticias_de_alphavantage(v or {}, simbolo)

    # ---------- salud ----------
    def salud(self, ticker: str = "TSLA") -> list[dict[str, Any]]:
        """Prueba cada fuente con una consulta mínima. Sirve para /estado y para --dry-run."""
        corte = self.ahora().date() + dt.timedelta(days=5)
        pruebas: list[tuple[str, Callable[[], str]]] = [
            ("Precios diarios (yfinance)", lambda: f"{len(d := self.diario(ticker, 30))} filas, último {d.index[-1].date()}"),
            ("Intradía 15 min (yfinance)", lambda: f"{len(d := self.intradia(ticker))} barras, última {d.index[-1]}"),
            ("^NDX diario (yfinance)", lambda: f"último {self.ndx_diario(30).index[-1].date()}"),
            ("Opciones / IV ATM (yfinance)", lambda: self._fmt_iv(self.opciones_iv(ticker, corte))),
            ("Reportes (yfinance + Finnhub)", lambda: f"próximo {self.reportes(ticker)[0]['fecha']}"),
            ("Noticias (Finnhub/yfinance)", lambda: f"{len(self.noticias(ticker, 3) or [])} titulares en 3 días"),
        ]
        out = []
        for nombre, fn in pruebas:
            try:
                out.append({"fuente": nombre, "ok": True, "detalle": fn()})
            except Exception as e:                 # noqa: BLE001
                out.append({"fuente": nombre, "ok": False, "detalle": f"{type(e).__name__}: {str(e)[:100]}"})
        return out

    @staticmethod
    def _fmt_iv(r: dict[str, Any]) -> str:
        if r.get("iv") is None:
            raise RuntimeError(r.get("motivo", "IV no disponible"))
        return f"IV {r['iv'] * 100:.1f} % (venc. {r['vencimiento']}, calidad {r['calidad']})"
