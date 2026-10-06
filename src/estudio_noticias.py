"""Estudio de eventos de las noticias de la BVC: ¿qué pasó con la acción después de anuncios parecidos?

Usa la "Información relevante" histórica de la Superfinanciera (lo que cada emisor publicó, con fecha y hora) y los precios diarios. Para cada anuncio:
  · día de reacción = el mismo día si se publicó con el mercado abierto; la sesión siguiente si se publicó después del cierre (lo más común);
  · reacción = retorno de ese día menos el del mercado (ICOLCAP), y cuántas veces lo normal fue (z);
  · después = retorno frente al mercado desde donde TÚ podrías comprar (la apertura siguiente si salió de noche; el cierre del día si salió en horario)
    hasta 1, 3 y 5 sesiones más tarde.
Con eso responde dos preguntas, sin adornos: (1) qué tipos de anuncio mueven de verdad la acción (magnitud de la reacción) y (2) si después de la reacción
el precio SIGUE en la misma dirección (que es lo que haría rentable comprar tras la noticia) o no.

Funciones puras (reciben los datos ya descargados). Las descargas viven en run_estudio_noticias.py."""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from . import noticias_bvc as N


def dia_de_reaccion(ts_bogota: dt.datetime, fechas: pd.DatetimeIndex, hora_cierre: int = 15) -> tuple[int, bool] | None:
    """(posición en `fechas` de la sesión en que el mercado pudo reaccionar, ¿se publicó con el mercado abierto?). None si cae fuera de la historia."""
    d = pd.Timestamp(ts_bogota.date())
    en_horario = ts_bogota.weekday() < 5 and 8 <= ts_bogota.hour < hora_cierre and d in fechas
    i = int(fechas.searchsorted(d if (en_horario or ts_bogota.hour < 8) else d + pd.Timedelta(days=1)))
    return (i, en_horario) if 0 < i < len(fechas) else None


def medir(evento: dict[str, Any], px: pd.DataFrame, ref: pd.Series, horizontes: list[int], sigma_dias: int = 20) -> dict[str, Any] | None:
    """Reacción y lo que vino después para un anuncio. `px` = OHLC diario de la acción; `ref` = cierres del mercado (mismas fechas o más)."""
    pos = dia_de_reaccion(evento["ts"], px.index)
    if pos is None:
        return None
    i, en_horario = pos
    if i < sigma_dias + 2:
        return None
    c, o = px["Close"], px["Open"]
    r_m = ref.reindex(px.index).ffill().pct_change()
    ar = c.pct_change() - r_m                                                          # retorno diario frente al mercado
    sigma = float(ar.iloc[i - sigma_dias: i].std(ddof=1))
    if not sigma or sigma != sigma or c.iloc[i] != c.iloc[i]:
        return None
    reac = float(ar.iloc[i])
    hueco = float(o.iloc[i] / c.iloc[i - 1] - 1) if o.iloc[i] == o.iloc[i] and o.iloc[i] > 0 else None
    out = dict(evento, fecha=px.index[i].date(), en_horario=en_horario, reaccion=reac, z=reac / sigma, sigma=sigma, hueco=hueco,
               hueco_z=(hueco / sigma) if (hueco is not None and not en_horario) else None)
    base_m = ref.reindex(px.index).ffill()
    for h in horizontes:
        # despues_h: comprando al CIERRE del día de reacción (ya se conoce la reacción completa: sin trampa) hasta h sesiones después.
        j = i + h
        ok = j < len(px) and c.iloc[i] == c.iloc[i] and base_m.iloc[i] == base_m.iloc[i]
        out[f"despues_{h}"] = float(c.iloc[j] / c.iloc[i] - 1) - float(base_m.iloc[j] / base_m.iloc[i] - 1) if ok else None
        # apertura_h (sólo anuncios fuera de horario): comprando en la APERTURA siguiente (sólo se conoce el hueco) hasta el cierre de la sesión h-ésima (h=1: ese día).
        k = i + h - 1
        ok = (not en_horario) and k < len(px) and hueco is not None and base_m.iloc[i - 1] == base_m.iloc[i - 1]
        out[f"apertura_{h}"] = float(c.iloc[k] / o.iloc[i] - 1) - float(base_m.iloc[k] / base_m.iloc[i - 1] - 1) if ok else None      # el mercado, de cierre a cierre
    return out


def _ic(x: np.ndarray, n: int = 2000, semilla: int = 2026) -> tuple[float, float]:
    if len(x) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    m = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def resumir(filas: list[dict[str, Any]], horizontes: list[int], campo: str = "despues") -> dict[str, Any]:
    """n, reacción típica y, por horizonte, media / mediana / % de veces que subió / IC95 de la media de lo que vino después (`campo`: despues | apertura)."""
    if not filas:
        return {"n": 0}
    out: dict[str, Any] = {"n": len(filas), "z_abs_mediana": float(np.median([abs(f["z"]) for f in filas])),
                           "pct_fuerte": float(np.mean([abs(f["z"]) >= 2 for f in filas])), "reaccion_media": float(np.mean([f["reaccion"] for f in filas])),
                           "abs_media": float(np.mean([abs(f["reaccion"]) for f in filas]))}       # cuánto se movió la acción ese día, sin importar el sentido
    for h in horizontes:
        x = np.array([f[f"{campo}_{h}"] for f in filas if f.get(f"{campo}_{h}") is not None], dtype=float)
        if len(x) == 0:
            continue
        lo, hi = _ic(x)
        out[f"h{h}"] = dict(n=int(len(x)), media=float(x.mean()), mediana=float(np.median(x)), pct_pos=float((x > 0).mean()), ic_lo=lo, ic_hi=hi)
    return out


def depurar(medidos: list[dict[str, Any]], pesos: dict[str, float]) -> list[dict[str, Any]]:
    """Un solo anuncio por emisor y día de reacción (el de mayor peso): varios comunicados del mismo día son el mismo hecho."""
    mejor: dict[tuple[str, dt.date], dict[str, Any]] = {}
    for m in medidos:
        k = (m["emisor"], m["fecha"])
        if k not in mejor or pesos.get(m["cat"], 0) > pesos.get(mejor[k]["cat"], 0):
            mejor[k] = m
    return sorted(mejor.values(), key=lambda m: (m["fecha"], m["emisor"]))


def tabla(medidos: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """La tabla que usa el bot: por tipo de anuncio y por dirección (según la reacción del precio: ≥ z_confirma veces lo normal)."""
    e = cfg["noticias_bvc"]["estudio"]
    hs, zc = e["horizontes"], cfg["noticias_bvc"]["z_confirma"]
    cats: dict[str, Any] = {}
    for cat in sorted({m["cat"] for m in medidos}):
        de_cat = [m for m in medidos if m["cat"] == cat]
        cats[f"{cat}|todos"] = resumir(de_cat, hs)
        cats[f"{cat}|+1"] = resumir([m for m in de_cat if m["z"] >= zc], hs)
        cats[f"{cat}|-1"] = resumir([m for m in de_cat if m["z"] <= -zc], hs)
    con_tipo = [m for m in medidos if m["cat"] != "otros"]
    noche = [m for m in con_tipo if m.get("hueco_z") is not None]

    def texto(s: int) -> list[dict[str, Any]]:
        return [m for m in noche if m["sentido_texto"] == s]
    return {"hecho": dt.date.today().isoformat(), "desde": e["desde"], "n_minimo": e["n_minimo"], "horizontes": hs, "n_total": len(medidos),
            "categorias": cats,
            # comprando al cierre del día de reacción, según cómo reaccionó el precio ese día
            "global": {"todos": resumir(con_tipo, hs), "+1": resumir([m for m in con_tipo if m["z"] >= zc], hs),
                       "-1": resumir([m for m in con_tipo if m["z"] <= -zc], hs),
                       "sin_tipo": resumir([m for m in medidos if m["cat"] == "otros"], hs)},
            # anuncios fuera de horario, comprando en la apertura siguiente: según el hueco con que abrió y según lo que decía el texto
            "apertura": {"todos": resumir(noche, hs, "apertura"),
                         "hueco+": resumir([m for m in noche if m["hueco_z"] >= zc], hs, "apertura"),
                         "hueco-": resumir([m for m in noche if m["hueco_z"] <= -zc], hs, "apertura"),
                         "sin_hueco": resumir([m for m in noche if abs(m["hueco_z"]) < zc], hs, "apertura"),
                         "texto+": resumir(texto(1), hs, "apertura"), "texto-": resumir(texto(-1), hs, "apertura"), "texto0": resumir(texto(0), hs, "apertura")}}


def eventos_de(items: list[N.Item], emisor: str, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Registros de la Superfinanciera de un emisor → eventos clasificados (los que no son de ningún tipo quedan como 'otros': sirven de comparación)."""
    zona = dt.timezone(dt.timedelta(hours=cfg["concurso"]["zona_horaria_utc"]))
    out = []
    for it in items:
        c = N.clasificar(N.texto_para_clasificar(it), cfg)
        out.append(dict(emisor=emisor, ts=it.ts.astimezone(zona), cat=c.cat if c else "otros", sentido_texto=c.sentido if c else 0, peso=c.peso if c else 0.0,
                        titulo=it.titulo[:160], tema=it.tema.split(" #")[0]))
    return out
