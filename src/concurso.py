"""Calendario oficial del concurso (reglas de trii): sesiones, semanas, cortes y horario bvc. Funciones puras."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

TIPO_CORTE, TIPO_FINAL = "corte", "final"


def bogota(cfg: dict[str, Any]) -> dt.timezone:
    return dt.timezone(dt.timedelta(hours=cfg["concurso"]["zona_horaria_utc"]))


def ahora_bogota(cfg: dict[str, Any], utc_now: dt.datetime | None = None) -> dt.datetime:
    utc_now = utc_now or dt.datetime.now(dt.timezone.utc)
    return utc_now.astimezone(bogota(cfg))


def _d(x: str | dt.date | pd.Timestamp) -> dt.date:
    return pd.Timestamp(x).date()


def festivos(cfg: dict[str, Any]) -> set[dt.date]:
    return {_d(f) for f in cfg["concurso"]["festivos"]}


def sesiones(cfg: dict[str, Any]) -> list[dt.date]:
    """Días de bolsa del concurso (lunes a viernes, sin festivos colombianos)."""
    c = cfg["concurso"]
    fest = festivos(cfg)
    return [d.date() for d in pd.bdate_range(c["inicio"], c["fin"]) if d.date() not in fest]


def es_dia_de_bolsa(d: dt.date, cfg: dict[str, Any]) -> bool:
    return d.weekday() < 5 and d not in festivos(cfg)


def semana_concurso(d: dt.date | str, cfg: dict[str, Any]) -> int | None:
    """1..5 según la semana calendario (lunes a viernes) del concurso; None fuera del concurso."""
    d = _d(d)
    ini, fin = _d(cfg["concurso"]["inicio"]), _d(cfg["concurso"]["fin"])
    if d < ini or d > fin + dt.timedelta(days=2):
        return None
    lunes0 = ini - dt.timedelta(days=ini.weekday())
    return (d - dt.timedelta(days=d.weekday()) - lunes0).days // 7 + 1


def clave_semana(d: dt.date | str, cfg: dict[str, Any]) -> str:
    s = semana_concurso(d, cfg)
    return f"S{s}" if s else "fuera"


def horario(d: dt.date, cfg: dict[str, Any]) -> tuple[dt.time, dt.time] | None:
    """(apertura, cierre) en hora de Bogotá, o None si ese día no hay mercado."""
    if not es_dia_de_bolsa(d, cfg):
        return None
    for h in cfg["concurso"]["horario"]:
        if _d(h["desde"]) <= d <= _d(h["hasta"]):
            return dt.time.fromisoformat(h["apertura"]), dt.time.fromisoformat(h["cierre"])
    return None


def mercado_abierto(ahora: dt.datetime, cfg: dict[str, Any]) -> bool:
    h = horario(ahora.date(), cfg)
    return bool(h and h[0] <= ahora.time() < h[1])


def cortes(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out = [dict(c, fecha=_d(c["fecha"]), tipo=TIPO_CORTE) for c in cfg["concurso"]["cortes"]]
    f = cfg["concurso"]["final"]
    out.append(dict(f, fecha=_d(f["fecha"]), tipo=TIPO_FINAL, top_pct=None, liga=f["meta"]))
    return out


def proximo_corte(d: dt.date | str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Primer corte (o el final) con fecha >= d. Tras el último día devuelve None."""
    d = _d(d)
    for c in cortes(cfg):
        if c["fecha"] >= d:
            return c
    return None


def corte_vigente(ahora: dt.datetime, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """Corte que sigue pendiente en este instante: si hoy es el día del corte y el mercado ya cerró, el vigente es el siguiente."""
    c = proximo_corte(ahora.date(), cfg)
    if c is not None and c["fecha"] == ahora.date():
        h = horario(c["fecha"], cfg)
        if h and ahora.time() >= h[1]:
            return proximo_corte(c["fecha"] + dt.timedelta(days=1), cfg)
    return c


def sesiones_restantes(ahora: dt.datetime, cfg: dict[str, Any]) -> int:
    """Sesiones que faltan hasta el corte, incluyendo la de hoy si todavía no cerró (D de la Regla Maestra para locales)."""
    c = corte_vigente(ahora, cfg)
    if c is None:
        return 0
    n = 0
    for d in sesiones(cfg):
        if d < ahora.date() or d > c["fecha"]:
            continue
        if d == ahora.date():
            h = horario(d, cfg)
            if not h or ahora.time() >= h[1]:
                continue
        n += 1
    return n


def dias_calendario_al_corte(ahora: dt.datetime, cfg: dict[str, Any]) -> float:
    """Días calendario hasta el cierre del corte (para IV × √(días/365)); mínimo 0,5."""
    c = corte_vigente(ahora, cfg)
    if c is None:
        return 0.0
    return max((c["fecha"] - ahora.date()).days, 0.5)


def proxima_sesion(ahora: dt.datetime, cfg: dict[str, Any]) -> dt.date | None:
    """Primera sesión del concurso posterior a hoy."""
    return next((d for d in sesiones(cfg) if d > ahora.date()), None)


def corte_manana(ahora: dt.datetime, cfg: dict[str, Any]) -> bool:
    """True si el corte vigente es mañana (la próxima sesión) o hoy mismo antes del cierre: ahí ROJO se trata como NEGRO."""
    c = corte_vigente(ahora, cfg)
    if c is None:
        return False
    return c["fecha"] == ahora.date() or c["fecha"] == proxima_sesion(ahora, cfg)


def ultimo_dia_operable(ahora: dt.datetime, cfg: dict[str, Any]) -> bool:
    """True si hoy (con el mercado abierto) o la próxima sesión es la última del concurso: momento de decidir si se asegura el #1."""
    ses = sesiones(cfg)
    if not ses:
        return False
    ultima = ses[-1]
    return (ahora.date() == ultima and mercado_abierto(ahora, cfg)) or proxima_sesion(ahora, cfg) == ultima


def semana_final(ahora: dt.datetime, cfg: dict[str, Any]) -> bool:
    """Semana 5: hay que cazar o defender el #1."""
    return semana_concurso(ahora.date(), cfg) == cfg["concurso"]["final"]["semana"]


def ultimo_dia(ahora: dt.datetime, cfg: dict[str, Any]) -> bool:
    return ahora.date() == _d(cfg["concurso"]["fin"])


def fase(ahora: dt.datetime, cfg: dict[str, Any]) -> str:
    c = cfg["concurso"]
    if ahora.date() < _d(c["inicio"]):
        return "antes"
    if ahora.date() > _d(c["fin"]):
        return "despues"
    return "durante"


def hora_orden_manana(d: dt.date, cfg: dict[str, Any]) -> str:
    """Hora sugerida para ejecutar órdenes en días de decisión (15 min después de abrir)."""
    h = horario(d, cfg)
    if not h:
        return "-"
    return (dt.datetime.combine(d, h[0]) + dt.timedelta(minutes=15)).strftime("%H:%M")
