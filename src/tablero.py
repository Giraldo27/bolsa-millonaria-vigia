"""Lógica del dashboard (app.py) sin pantalla: carga de datos y tablas listas para mostrar. Así se prueba sin abrir Streamlit.

Privacidad: la app NO necesita el token de Telegram ni tu posición. Si no encuentra data/state.json (p. ej. en Streamlit Community Cloud) trabaja con datos
manuales: la calculadora de la Regla Maestra pide tu rentabilidad y la del corte."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from . import concurso as C
from . import formato as F
from .catalizadores import catalizadores, tickers_con_reporte
from .construir import ResultadoMotor, ejecutar_motor
from .informe_html import porque_descartado
from .regla_maestra import ContextoRegla, Decision, regla_maestra
from .relevo import EntradaBanco
from .semaforo import EMOJI, titulares_puntuados
from .servicio import Contexto, leer_estado
from .state import universo_permitido


@dataclass
class DatosTablero:
    ahora: dt.datetime
    activo: str
    r: ResultadoMotor
    cats: list[dict[str, Any]]
    noticias: list[dict[str, Any]] | None
    base_rank: list[dict[str, Any]] | None
    intradia: pd.DataFrame | None                                    # sesión más reciente en unidades de σ
    estado: dict[str, Any] = field(default_factory=dict)             # rentabilidades guardadas, cambios usados… ({} si no hay state.json)
    permitidos: set[str] = field(default_factory=set)
    avisos: list[str] = field(default_factory=list)


def serie_sigma(intr: pd.DataFrame | None, cierre_previo: float, sigma20: float) -> pd.DataFrame | None:
    """Barras de la última sesión como 'cuántas σ se ha movido el precio frente al cierre previo' (−2 = caída anormal). None si no hay datos."""
    if intr is None or intr.empty or not sigma20 or sigma20 <= 0 or not cierre_previo:
        return None
    ultimo = intr.index[-1].date()
    s = intr[intr.index.date == ultimo]
    if s.empty:
        return None
    return pd.DataFrame({"hora": s.index.strftime("%H:%M"), "precio": s["Close"].to_numpy(float),
                         "sigma": (s["Close"].to_numpy(float) / cierre_previo - 1) / sigma20})


def cargar(ctx: Contexto) -> DatosTablero:
    """Calcula todo lo que muestra el tablero con los datos más recientes. Las partes secundarias fallan por separado (se avisa, no se cae)."""
    est = leer_estado(ctx)
    r = ejecutar_motor(ctx.f, est, ctx.cfg, ctx.ahora, ctx.puntuador, ctx.motor)
    avisos: list[str] = []
    try:
        cats = catalizadores(ctx.f, ctx.cfg, ctx.ahora, tickers_con_reporte(ctx.cfg), ctx.cfg["radar"]["dias_catalizadores"])
    except Exception:                                                                    # noqa: BLE001
        cats = []
        avisos.append("No pude consultar el calendario de fechas importantes.")
    try:
        n = ctx.f.noticias(est.activo, 3)
        noticias = titulares_puntuados(n, ctx.cfg, ctx.puntuador, 30) if n is not None else None
    except Exception:                                                                    # noqa: BLE001
        noticias = None
    try:
        from .seleccion import ranking_base
        brank = ranking_base(ctx.f, ctx.cfg, ctx.ahora, est.activo)
    except Exception:                                                                    # noqa: BLE001
        brank = None
        avisos.append("No pude armar la comparación de base.")
    try:
        sig = serie_sigma(ctx.f.intradia(est.activo), r.ent.cierre_previo, r.res.sigma20)
    except Exception:                                                                    # noqa: BLE001
        sig = None
    guardado = {"mia": est.rent.get("mia"), "umbral": est.rent.get("umbral"), "cambios_usados": est.cambios_usados(), "cambios_max": ctx.cfg["estado"]["max_cambios"],
                "cambio_hoy": est.cambio_hoy(ctx.ahora.date()), "posicion": est.d["posicion"]}
    return DatosTablero(ctx.ahora, est.activo, r, cats, noticias, brank, sig, guardado, universo_permitido(ctx.cfg), avisos)


# ------------------------------------------------------------------ tablas
def df_banco(r: ResultadoMotor) -> pd.DataFrame:
    filas = []
    for i, x in enumerate(r.banco, 1):
        c = x.c
        filas.append({"#": i, "Acción": c.ticker, "Mercado": "BVC" if c.grupo == "local" else "EE. UU.", "Semáforo": f"{EMOJI[c.color]} {F.NOMBRE_COLOR[c.color].split(' (')[0]}",
                      "Se espera que se mueva (±%)": round(c.mee * 100, 1), "Veces tu acción": round(c.mee / r.mee, 2) if r.mee else None,
                      "Parecido a tu acción": F._parecido(c.corr, r.activo) or "n. d.", "Valor negociado/día (millones)": round(c.valor_negociado_mm) if c.valor_negociado_mm else None,
                      "Motivos": "; ".join(m for m in x.razones if not m.startswith("MEE"))})
    return pd.DataFrame(filas)


def df_descartados(r: ResultadoMotor, cfg: dict[str, Any], permitidos: set[str]) -> pd.DataFrame:
    en = {x.c.ticker for x in r.banco}
    filas = [{"Acción": c.ticker, "Mercado": "BVC" if c.grupo == "local" else "EE. UU.", "Se espera que se mueva (±%)": round(c.mee * 100, 1) if c.mee else None,
              "Por qué no": porque_descartado(c, r.activo, cfg, permitidos, set(r.excluidos))} for c in sorted(r.candidatos, key=lambda c: -(c.mee or 0)) if c.ticker not in en]
    return pd.DataFrame(filas)


def df_catalizadores(cats: list[dict[str, Any]]) -> pd.DataFrame:
    icono = {"corte": "🏁", "macro": "🏦", "reporte": "📊"}
    return pd.DataFrame([{"Fecha": F.fecha(x["fecha"]) + (f" {x['hora']}" if x.get("hora") else ""), "Qué pasa": f"{icono.get(x['tipo'], '•')} {x['detalle']}"
                          + (" (fecha por confirmar)" if x.get("verificada") is False else "")} for x in cats])


def df_noticias(items: list[dict[str, Any]] | None, ahora: dt.datetime, traductor: Any = None, max_n: int = 12, max_trad: int = 8) -> pd.DataFrame:
    if not items:
        return pd.DataFrame()
    filas = []
    ahora_utc = ahora.astimezone(dt.timezone.utc)
    for i, k in enumerate(items[:max_n]):
        es = k.get("idioma", "en") == "es"
        trad = traductor(k["titulo"]) if (traductor and not es and i < max_trad) else None
        filas.append({"Tono": F.tono(k["puntaje"], k.get("palabras")), "Titular": trad or k["titulo"] + ("" if es else "  (en inglés)"), "Fuente": k.get("proveedor", ""),
                      "Cuándo": F.hace(str(k["ts"]), ahora_utc)})
    return pd.DataFrame(filas)


def df_base(filas: list[dict[str, Any]] | None, top: int = 12) -> pd.DataFrame:
    if not filas:
        return pd.DataFrame()
    return pd.DataFrame([{"#": i, "Acción": x["ticker"] + (" ← tu base" if x["es_base"] else ""), "Mercado": "BVC" if x["grupo"] == "local" else "EE. UU.",
                          "Se espera que se mueva hasta el final (±%)": round(x["mee"] * 100, 1), "Veces la base": round(x["vs_base"], 2) if x["vs_base"] else None,
                          "Volatilidad anual 60 días (%)": round(x["real"] * 100), "Volatilidad de opciones (%)": round(x["iv"] * 100) if x["iv"] else None,
                          "¿Supera el umbral?": "⚠️ sí" if x["supera"] else ""} for i, x in enumerate(filas[:top], 1)])


# ------------------------------------------------------------------ calculadora de la Regla Maestra
def calc_regla(d: DatosTablero, cfg: dict[str, Any], mia: float, objetivo: float, cambios_usados: int | None = None, cambio_hoy: bool = False
               ) -> tuple[Decision, list[EntradaBanco]]:
    """Misma Regla Maestra del bot (no una copia de la fórmula) con tus números: rentabilidad propia y objetivo en puntos porcentuales (6,5 = +6,5 %)."""
    r = d.r
    usados = d.estado.get("cambios_usados", 0) if cambios_usados is None else cambios_usados
    final = C.semana_final(d.ahora, cfg)
    ctx = ContextoRegla(mia=mia, objetivo=objetivo, activo=r.activo, color_actual=r.res.color, mee_actual=r.mee, cambios_restantes=cfg["estado"]["max_cambios"] - usados,
                        cambio_hoy=cambio_hoy, semana_final=final, soy_primero=final and mia >= objetivo, ultimo_dia=C.ultimo_dia_operable(d.ahora, cfg))
    return regla_maestra(ctx, r.candidatos, cfg)


def evaluados(dec: Decision) -> pd.DataFrame:
    return pd.DataFrame([{"Candidato": x["ticker"], "Se espera que se mueva (±%)": round(x["mee"] * 100, 1), "Veces tu acción": round(x["ratio"], 2),
                          "¿Alcanza?": "✅ sí" if x["cumple"] else "no"} for x in dec.evaluados])
