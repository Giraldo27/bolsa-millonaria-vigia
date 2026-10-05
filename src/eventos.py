"""Novedades que el bot avisa SOLO, además de los cambios de color: noticia nueva importante, cambio de opinión de la Regla Maestra, fechas del día
(reporte de resultados, evento macro, corte mañana) y subida fuerte. Cada tipo recuerda lo que ya avisó (en state.json, clave `eventos`) para no repetir.

Reglas anti-spam: la primera vez sólo se toma una "línea base" de noticias (no se avisa de lo que ya existía), cada tipo tiene su enfriamiento y hay un tope
de avisos por corrida (`monitor.eventos` en config.yaml). Funciones sin red: reciben los datos ya calculados."""
from __future__ import annotations

import datetime as dt
import hashlib
from typing import Any

import pandas as pd

from . import concurso as C
from .regla_maestra import CAMBIAR, VENDER_TODO, Decision

UTC = dt.timezone.utc


def _hash(titulo: str) -> str:
    return hashlib.sha1(titulo.lower().strip()[:200].encode("utf-8")).hexdigest()[:12]


def _ts(valor: Any) -> pd.Timestamp:
    t = pd.Timestamp(valor)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _minutos(desde_iso: str | None, ahora: dt.datetime) -> float:
    return 1e9 if not desde_iso else (ahora - dt.datetime.fromisoformat(desde_iso)).total_seconds() / 60


def es_importante(k: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Titular que vale un aviso: claramente negativo (tono ≤ umbral, o palabra de riesgo sin tono positivo) o claramente positivo."""
    c = cfg["monitor"]["eventos"]["noticias"]
    p = k["puntaje"]
    return bool(p <= cfg["semaforo"]["sentimiento_negativo"] or (k.get("palabras") and p < 0.05) or p >= c["positiva_min"])


def noticias_nuevas(ev: dict[str, Any], items: list[dict[str, Any]], ahora: dt.datetime, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Titulares importantes que NO se habían visto, publicados dentro de la ventana. Actualiza `ev`.
    · Primera corrida: línea base (se marcan como vistos todos los actuales, no se avisa de nada).
    · Si hay uno importante pero el último aviso de noticias fue hace poco, NO se marca como visto: se avisará después (mientras siga dentro de la ventana)."""
    c = cfg["monitor"]["eventos"]["noticias"]
    actuales = {_hash(k["titulo"]): k for k in items}
    if "noticias_vistas" not in ev:
        ev["noticias_vistas"] = list(actuales)[:400]
        return []
    vistas = set(ev["noticias_vistas"])
    limite = ahora.astimezone(UTC) - dt.timedelta(hours=c["ventana_h"])
    importantes: list[tuple[str, dict[str, Any]]] = []
    marcar = []
    for h, k in actuales.items():
        if h in vistas:
            continue
        if _ts(k["ts"]) >= limite and es_importante(k, cfg):
            importantes.append((h, k))
        else:
            marcar.append(h)                                                                       # vieja o intrascendente: se da por vista sin avisar
    salida: list[dict[str, Any]] = []
    if importantes and _minutos(ev.get("noticias_ultimo_aviso"), ahora) >= c["enfriamiento_min"]:
        importantes.sort(key=lambda x: -abs(x[1]["puntaje"]))
        salida = [k for _, k in importantes[: c["max_por_aviso"]]]
        marcar += [h for h, _ in importantes]
        ev["noticias_ultimo_aviso"] = ahora.isoformat(timespec="minutes")
    ev["noticias_vistas"] = (ev["noticias_vistas"] + marcar)[-400:]
    return salida


def cambio_de_decision(ev: dict[str, Any], dec: Decision, ahora: dt.datetime, cfg: dict[str, Any]) -> str | None:
    """'nueva' si la Regla Maestra pasó a recomendar CAMBIAR/VENDER_TODO (o cambió de candidato), 'retirada' si dejó de recomendarlo, None si no hay novedad.
    Sin ranking o sin movimiento esperado no se concluye nada (no se toca lo guardado)."""
    if dec.codigo in ("sin_ranking", "sin_mee"):
        return None
    prev = ev.get("decision", {})
    ahora_ac = dec.accion in (CAMBIAR, VENDER_TODO)
    antes_ac = prev.get("accion") in (CAMBIAR, VENDER_TODO)
    ev["decision"] = {"accion": dec.accion, "candidato": dec.candidato or "", "avisado": prev.get("avisado")}
    enfr = cfg["monitor"]["eventos"]["decision"]["enfriamiento_min"]
    if ahora_ac and (not antes_ac or prev.get("candidato") != (dec.candidato or "") or prev.get("accion") != dec.accion):
        if _minutos(prev.get("avisado"), ahora) < enfr and antes_ac:                                # cambió de candidato hace poco: no marear
            return None
        ev["decision"]["avisado"] = ahora.isoformat(timespec="minutes")
        return "nueva"
    if antes_ac and not ahora_ac and prev.get("avisado"):
        ev["decision"]["avisado"] = None
        return "retirada"
    return None


def avisos_calendario(ev: dict[str, Any], activo: str, reportes: list[dict[str, Any]] | None, ahora: dt.datetime, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Fechas que merecen un aviso (una sola vez cada una): reporte de resultados de TU acción hoy o en la próxima sesión, evento macro de hoy y corte mañana."""
    hecho = set(ev.setdefault("calendario_avisado", []))
    hoy = ahora.date()
    prox = C.proxima_sesion(ahora, cfg)
    out: list[dict[str, Any]] = []

    def nuevo(clave: str) -> bool:
        if clave in hecho:
            return False
        hecho.add(clave)
        return True
    por_fecha: dict[dt.date, dict[str, Any]] = {}
    for r in reportes or []:
        por_fecha.setdefault(pd.Timestamp(r["fecha"]).date(), r)
    for f, r in por_fecha.items():
        if f == hoy or (prox is not None and f == prox):
            if nuevo(f"rep|{activo}|{f}"):
                out.append(dict(tipo="reporte", ticker=activo, fecha=f, hora=r.get("hora", ""), es_hoy=f == hoy))
    for m in cfg.get("eventos_macro", []):
        if pd.Timestamp(m["fecha"]).date() == hoy and nuevo(f"macro|{m['fecha']}|{m['evento']}"):
            out.append(dict(tipo="macro", evento=m["evento"], hora=m.get("hora", ""), verificada=m.get("verificada", True)))
    if C.corte_manana(ahora, cfg) and nuevo(f"corte|{hoy}"):
        c = C.corte_vigente(ahora, cfg)
        out.append(dict(tipo="corte", corte=c, es_hoy=c["fecha"] == hoy))
    ev["calendario_avisado"] = sorted(hecho)[-80:]
    return out


def alza_fuerte(ev: dict[str, Any], z: float | None, ahora: dt.datetime, cfg: dict[str, Any]) -> bool:
    """True una sola vez por día si el activo sube con z ≥ monitor.eventos.alza_fuerte_z (la caída fuerte ya la cubre el semáforo)."""
    if z is None or z != z or z < cfg["monitor"]["eventos"]["alza_fuerte_z"] or ev.get("alza_fecha") == ahora.date().isoformat():
        return False
    ev["alza_fecha"] = ahora.date().isoformat()
    return True


def estado_base(ev: dict[str, Any], veredicto: str, mejores: list[dict[str, Any]]) -> str | None:
    """Novedad de la comparación de base: 'revisar' si ahora hay una acción que supera a la base (o cambió cuál es), 'ok' si ya no la hay; None si igual que antes."""
    nuevo = {"veredicto": veredicto, "primera": mejores[0]["ticker"] if mejores else ""}
    prev = ev.get("base", {})
    ev["base"] = nuevo
    if veredicto == "sin_datos" or prev == nuevo:
        return None
    if veredicto == "revisar":
        return "revisar"
    return "ok" if prev.get("veredicto") == "revisar" else None
