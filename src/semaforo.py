"""Semáforo por activo: ¿la caída de hoy es ruido (mercado) o un cambio real del activo?

VERDE (normal) · AZUL (cae el mercado, no el activo) · AMARILLO (caída propia sin confirmación de noticias/volumen)
· ROJO (caída propia + noticia negativa + volumen) · NEGRO (ROJO que no se recuperó al cierre del 2.º día).

Todo son funciones puras: reciben datos y devuelven un resultado; el estado (episodio ROJO) lo guarda quien llama."""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from .sentimiento import palabras_clave, puntuar_es, texto_de

VERDE, AZUL, AMARILLO, ROJO, NEGRO = "VERDE", "AZUL", "AMARILLO", "ROJO", "NEGRO"
COLORES = (VERDE, AZUL, AMARILLO, ROJO, NEGRO)
EMOJI = {VERDE: "🟢", AZUL: "🔵", AMARILLO: "🟡", ROJO: "🔴", NEGRO: "⚫"}
GRAVEDAD = {c: i for i, c in enumerate(COLORES)}


@dataclass
class EntradaSemaforo:
    ticker: str
    fecha: dt.date
    precio: float                           # último precio (o el cierre de hoy si es_cierre)
    cierre_previo: float
    cierres: pd.Series                      # cierres diarios COMPLETOS hasta ayer
    volumenes: pd.Series                    # volúmenes diarios COMPLETOS hasta ayer
    volumen_dia: float | None               # volumen de hoy proyectado al día completo (None si no es fiable)
    ndx_cierres: pd.Series
    ndx_precio: float | None
    ndx_cierre_previo: float | None
    noticias: list[Any] | None              # None = sin datos de noticias (sin clave, fuente caída o sin cobertura)
    base_noticias_diaria: float | None      # promedio de titulares por día en los últimos 30 días
    ahora_utc: dt.datetime
    es_cierre: bool = False                 # el mercado ya cerró y 'precio' es el cierre de hoy
    corte_manana: bool = False
    episodio: dict[str, Any] | None = None  # episodio ROJO vigente (viene de state.json)
    precio_apertura: float | None = None    # apertura de hoy (para el hueco de apertura del respaldo sin noticias)
    cobertura_parcial: bool = False         # noticias sólo por RSS (acciones locales): la ausencia de titulares no prueba nada


@dataclass
class ResultadoSemaforo:
    ticker: str
    color: str                              # color efectivo (ROJO pasa a NEGRO si el corte es mañana)
    color_calculado: str                    # color antes de la excepción del corte
    z: float
    z_res: float
    vol_rel: float | None
    beta: float
    sigma20: float
    sigma_res: float
    r_hoy: float
    resid5: float
    umbral5: float
    noticia_negativa: bool
    sentimiento: float | None
    n_24h: int
    base_diaria: float | None
    ratio_noticias: float | None
    claves: list[dict[str, Any]]
    sin_noticias: bool
    motor_sentimiento: str
    motivo: str
    accion: str
    episodio: dict[str, Any] | None
    advertencias: list[str] = field(default_factory=list)
    proxy_noticias: bool = False            # sin noticias: se presumió noticia negativa por volumen anormal + hueco de apertura
    cobertura_parcial: bool = False
    r_mercado: float = 0.0                  # parte del retorno de hoy que explica el mercado (beta × retorno del Nasdaq-100)
    resid_hoy: float = 0.0                  # parte propia del activo (retorno − mercado)
    fecha: dt.date | None = None            # día de la sesión evaluada (antes de abrir es la de AYER: el mensaje debe decirlo)


# ------------------------------------------------------------------ volumen intradía
def perfil_volumen(intradia: pd.DataFrame, dias_previos: int = 20) -> pd.Series:
    """Fracción acumulada típica del volumen diario por hora del día (HH:MM de apertura de cada barra), con los días completos previos."""
    d = intradia.copy()
    d["dia"] = d.index.normalize()
    d["hm"] = d.index.strftime("%H:%M")
    dias = sorted(d["dia"].unique())[:-1][-dias_previos:]
    d = d[d["dia"].isin(dias)]
    if d.empty:
        return pd.Series(dtype=float)
    tot = d.groupby("dia")["Volume"].transform("sum")
    d["frac"] = d["Volume"] / tot
    return d.groupby("hm")["frac"].mean().sort_index().cumsum()


def proyectar_volumen(vol_acum: float, ultima_barra_hm: str, perfil: pd.Series, minimo: float = 0.08) -> float | None:
    """Proyecta el volumen del día completo: vol_acum / fracción típica acumulada hasta esa hora. None si la fracción es muy pequeña."""
    if perfil.empty or ultima_barra_hm not in perfil.index:
        return None
    frac = float(perfil.loc[ultima_barra_hm])
    return vol_acum / frac if frac >= minimo else None


# ------------------------------------------------------------------ métricas
def calcular_metricas(e: EntradaSemaforo, cfg: dict[str, Any]) -> dict[str, Any]:
    s = cfg["semaforo"]
    adv: list[str] = []
    r_d = e.cierres.pct_change().dropna()
    sigma20 = float(r_d.tail(s["sigma_dias"]).std(ddof=1))
    r_hoy = e.precio / e.cierre_previo - 1
    z = r_hoy / sigma20 if sigma20 > 0 else float("nan")

    rn = e.ndx_cierres.pct_change().dropna()
    j = pd.concat([r_d, rn], axis=1, keys=["a", "n"], sort=True).dropna().tail(s["beta_dias"])
    if len(j) >= 30 and j["n"].var() > 0:
        beta = float(j["a"].cov(j["n"]) / j["n"].var())
        resid = j["a"] - beta * j["n"]
    else:                                      # calendarios distintos (acciones locales) o poca historia: sin ajuste de mercado
        beta, resid = 0.0, r_d.tail(s["beta_dias"])
        adv.append("residual aproximado: sin beta contra el Nasdaq-100 (pocas observaciones alineadas)")
    sigma_res = float(resid.std(ddof=1))
    if e.ndx_precio and e.ndx_cierre_previo:
        rn_hoy = e.ndx_precio / e.ndx_cierre_previo - 1
    else:
        rn_hoy = 0.0
        adv.append("sin dato del Nasdaq-100 de hoy: se asume 0 %")
    resid_hoy = r_hoy - beta * rn_hoy
    z_res = resid_hoy / sigma_res if sigma_res > 0 else float("nan")
    n = s["resid_acum_dias"]
    resid5 = float(resid.tail(n - 1).sum() + resid_hoy)
    sigma5 = sigma_res * math.sqrt(n)

    prom = float(e.volumenes.tail(s["vol_dias"]).mean()) if len(e.volumenes) else float("nan")
    vol_rel = (e.volumen_dia / prom) if (e.volumen_dia is not None and prom and prom > 0) else None
    if vol_rel is None:
        adv.append("volumen relativo no disponible (se trata como < 1,5: no puede haber ROJO por volumen)")
    return dict(r_hoy=r_hoy, sigma20=sigma20, z=z, beta=beta, sigma_res=sigma_res, z_res=z_res, resid5=resid5,
                umbral5=s["resid_acum_umbral"] * sigma5, vol_rel=vol_rel, advertencias=adv, r_mercado=beta * rn_hoy, resid_hoy=resid_hoy)


# ------------------------------------------------------------------ noticias
def analizar_noticias(noticias: list[Any] | None, base_diaria: float | None, ahora_utc: dt.datetime, cfg: dict[str, Any],
                      puntuador: Callable[[str], float]) -> dict[str, Any]:
    """Titulares de las últimas 24 h: cuántos, sentimiento medio, palabras clave y si hay 'noticia negativa'.
    Noticia negativa = (≥ min_titulares) y [sentimiento medio < umbral, o algún titular NO positivo con palabra clave negativa]."""
    s = cfg["semaforo"]
    if noticias is None:
        return dict(disponible=False, n24=0, sentimiento=None, negativa=False, ratio=None, claves=[], base=base_diaria)
    lim = pd.Timestamp(ahora_utc) - pd.Timedelta(hours=24)
    lim = lim.tz_localize("UTC") if lim.tzinfo is None else lim.tz_convert("UTC")
    items = []
    for n in noticias:
        ts = pd.Timestamp(n.ts if hasattr(n, "ts") else n["ts"])
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        if ts < lim:
            continue
        txt = texto_de(n, cfg["sentimiento"]["usar_resumen"])
        idioma = getattr(n, "idioma", n.get("idioma", "en") if isinstance(n, dict) else "en")
        es = idioma == "es"
        p = puntuar_es(txt, cfg) if es else puntuador(txt)                       # español: léxico propio (VADER sólo entiende inglés)
        titulo = n.titulo if hasattr(n, "titulo") else n["titulo"]
        kw = palabras_clave(titulo, s["palabras_negativas_es" if es else "palabras_negativas"])
        items.append(dict(titulo=titulo, puntaje=p, palabras=kw, ts=str(ts), idioma=idioma,
                          proveedor=getattr(n, "proveedor", n.get("proveedor", "") if isinstance(n, dict) else "")))
    n24 = len(items)
    sent = float(np.mean([i["puntaje"] for i in items])) if items else None
    for i in items:                                                               # "Tesla BEATS delivery estimates": la clave no cuenta
        lista = s.get("palabras_positivas_anulan_es" if i["idioma"] == "es" else "palabras_positivas_anulan", [])
        i["anulada"] = bool(lista) and bool(palabras_clave(i["titulo"], lista))
    kw_neg = [i for i in items if i["palabras"] and not i["anulada"] and i["puntaje"] < 0.05]   # y tampoco en titulares de tono positivo
    negativa = n24 >= s["min_titulares_24h"] and ((sent is not None and sent < s["sentimiento_negativo"]) or bool(kw_neg))
    claves = sorted(items, key=lambda i: (not (i["palabras"] and not i["anulada"] and i["puntaje"] < 0.05), i["puntaje"]))[: s["titulares_clave"]]
    ratio = (n24 / base_diaria) if base_diaria else None
    return dict(disponible=True, n24=n24, sentimiento=sent, negativa=negativa, ratio=ratio, claves=claves, base=base_diaria)


def titulares_puntuados(noticias: list[Any], cfg: dict[str, Any], puntuador: Callable[[str], float], max_n: int = 20) -> list[dict[str, Any]]:
    """Titulares (más recientes primero) con su sentimiento y palabras clave, para el comando /noticias."""
    out = []
    for n in sorted(noticias, key=lambda x: x.ts, reverse=True)[:max_n]:
        es = getattr(n, "idioma", "en") == "es"
        txt = texto_de(n, cfg["sentimiento"]["usar_resumen"])
        out.append(dict(titulo=n.titulo, puntaje=puntuar_es(txt, cfg) if es else puntuador(txt), ts=str(n.ts), proveedor=n.proveedor, idioma="es" if es else "en",
                        palabras=palabras_clave(n.titulo, cfg["semaforo"]["palabras_negativas_es" if es else "palabras_negativas"])))
    return out


# ------------------------------------------------------------------ clasificación
def clasificar_base(m: dict[str, Any], negativa: bool, cfg: dict[str, Any]) -> tuple[str, str]:
    """Color 'del día' (sin memoria de episodios) y su motivo."""
    s = cfg["semaforo"]
    z, zr, vr = m["z"], m["z_res"], m["vol_rel"]
    vol_ok = vr is not None and vr >= s["vol_rel_min"]
    if m["resid5"] <= m["umbral5"] and negativa:
        return ROJO, f"residual acumulado de {s['resid_acum_dias']} días ≤ {s['resid_acum_umbral']}·σ con noticias negativas"
    if not z <= s["z_caida"]:
        return VERDE, f"movimiento normal (z = {z:+.2f} > {s['z_caida']})"
    if zr > s["z_res_azul"]:
        return AZUL, f"cae con el mercado, no por el activo (z = {z:+.2f}, z_res = {zr:+.2f} > {s['z_res_azul']})"
    if zr <= s["z_res_caida"]:
        if negativa and vol_ok:
            return ROJO, f"caída propia (z_res = {zr:+.2f}) con noticia negativa y volumen {vr:.1f}× el normal"
        falta = "sin noticia negativa" if not negativa else f"volumen {('%.1f×' % vr) if vr is not None else 'no disponible'} < {s['vol_rel_min']}×"
        return AMARILLO, f"caída propia (z_res = {zr:+.2f}) pero {falta}"
    return s["zona_gris"], f"caída parcialmente propia (z_res = {zr:+.2f} entre {s['z_res_caida']} y {s['z_res_azul']}): zona gris → {s['zona_gris']}"


def avanzar_episodio(ep: dict[str, Any] | None, color: str, fecha: dt.date, cierre_previo: float, precio: float, es_cierre: bool,
                     n_sesiones: int, sent_negativo: bool, cfg: dict[str, Any]) -> tuple[dict[str, Any] | None, str, str]:
    """Memoria del ROJO. Devuelve (episodio nuevo o None, color, nota).
    NEGRO = ROJO que al cierre de la 2.ª sesión no recuperó ≥ 50 % de la caída del día 1 y sigue con sentimiento negativo."""
    rec_min = cfg["semaforo"]["recuperacion_negro"]
    if ep is None:
        if color == ROJO:
            ep = {"inicio": fecha.isoformat(), "cierre_previo": float(cierre_previo), "cierre_dia1": float(precio) if es_cierre else None,
                  "estado": ROJO}
            return ep, ROJO, "episodio ROJO iniciado"
        return None, color, ""
    ep = dict(ep)
    if n_sesiones <= 1:                                     # sigue el día 1
        if es_cierre and ep.get("cierre_dia1") is None:
            ep["cierre_dia1"] = float(precio)
        return ep, (ROJO if ep["estado"] == ROJO else color), "día 1 del episodio ROJO"
    dia1 = ep.get("cierre_dia1")
    if dia1 is None:
        return ep, ep["estado"], "sin cierre del día 1: no se puede medir la recuperación"
    caida = ep["cierre_previo"] - dia1
    recuperacion = 1.0 if caida <= 0 else (precio - dia1) / caida
    if ep["estado"] == NEGRO:
        if recuperacion >= rec_min:
            return None, color, f"el activo recuperó {recuperacion:.0%} de la caída: sale de NEGRO"
        return ep, NEGRO, f"sigue sin recuperar (recuperó {recuperacion:.0%} de la caída del día 1)"
    if es_cierre:                                           # decisión al cierre del 2.º día (o del primer cierre posterior)
        if recuperacion >= rec_min:
            return None, color, f"recuperó {recuperacion:.0%} de la caída: el ROJO se descarta"
        if sent_negativo:
            ep["estado"] = NEGRO
            return ep, NEGRO, f"al cierre del 2.º día sólo recuperó {recuperacion:.0%} (< {rec_min:.0%}) y el sentimiento sigue negativo"
        return None, color, f"recuperó {recuperacion:.0%} pero el sentimiento ya no es negativo: no es NEGRO"
    return ep, ROJO, "ROJO: la decisión a NEGRO se toma al cierre"


def contar_sesiones(inicio: str | dt.date, fecha: dt.date, cierres: pd.Series) -> int:
    """Sesiones transcurridas desde el inicio del episodio hasta hoy (1 = el mismo día del ROJO)."""
    ini = pd.Timestamp(inicio)
    previas = [d for d in cierres.index if ini <= pd.Timestamp(d) < pd.Timestamp(fecha)]
    return len(previas) + 1


def accion_sugerida(color: str, corte_manana: bool, sin_noticias: bool) -> str:
    base = {
        VERDE: "Mantener. Movimiento normal.",
        AZUL: "Mantener. Cae el mercado, no tu activo: es ruido.",
        AMARILLO: "Vigilar y no vender por pánico. Revisa el banco de relevo; la Regla Maestra decide si cambiar.",
        ROJO: "Alerta: caída propia con noticia negativa. Revisa la Regla Maestra y el banco de relevo; decide esta noche.",
        NEGRO: "El activo no se recuperó. Es información: la validación histórica (Fase 3) no mostró que las acciones grandes sigan cayendo después de NEGRO; "
               "decide con la Regla Maestra y tu ranking, no por el color.",
    }[color]
    if corte_manana and color == NEGRO:
        base += " (El corte es mañana: ROJO se trata como NEGRO.)"
    if sin_noticias:
        base += " Sin datos de noticias: el semáforo no puede confirmar ROJO."
    return base


def evaluar(e: EntradaSemaforo, cfg: dict[str, Any], puntuador: Callable[[str], float], motor: str = "vader") -> ResultadoSemaforo:
    m = calcular_metricas(e, cfg)
    nt = analizar_noticias(e.noticias, e.base_noticias_diaria, e.ahora_utc, cfg, puntuador)
    # RESPALDO: si no hay NINGUNA fuente de noticias, una caída propia con volumen anormal y hueco de apertura se presume noticia negativa
    proxy = False
    px = cfg["semaforo"].get("proxy_sin_noticias", {})
    if not nt["disponible"] and px.get("activo") and e.precio_apertura:
        gap = e.precio_apertura / e.cierre_previo - 1
        vr = m["vol_rel"]
        if vr is not None and vr >= px["vol_rel_min"] and m["sigma20"] > 0 and gap <= -px["gap_sigma"] * m["sigma20"]:
            nt = dict(nt, negativa=True)
            proxy = True
    color_dia, motivo = clasificar_base(m, nt["negativa"], cfg)
    if proxy:
        motivo += " [SIN NOTICIAS: se presume noticia negativa por volumen anormal y hueco de apertura]"
    ep = e.episodio
    n_ses = contar_sesiones(ep["inicio"], e.fecha, e.cierres) if ep else 1
    sent_neg = bool(nt["negativa"] or (nt["sentimiento"] is not None and nt["sentimiento"] < cfg["semaforo"]["sentimiento_negativo"]))
    ep2, color, nota = avanzar_episodio(ep, color_dia, e.fecha, e.cierre_previo, e.precio, e.es_cierre, n_ses, sent_neg, cfg)
    if nota:
        motivo = f"{motivo}. {nota}"
    efectivo = color
    if color == ROJO and e.corte_manana and cfg["semaforo"]["rojo_es_negro_si_corte_manana"]:
        efectivo = NEGRO
        motivo += ". El corte es mañana: ROJO se trata como NEGRO"
    adv = list(m["advertencias"])
    if not nt["disponible"]:
        adv.append("sin datos de noticias" + (" (se usó el proxy de volumen y hueco de apertura)" if proxy else ""))
    if e.cobertura_parcial:
        adv.append("noticias con cobertura parcial (RSS en español): la ausencia de titulares no descarta una mala noticia")
    return ResultadoSemaforo(
        ticker=e.ticker, color=efectivo, color_calculado=color, z=m["z"], z_res=m["z_res"], vol_rel=m["vol_rel"], beta=m["beta"],
        sigma20=m["sigma20"], sigma_res=m["sigma_res"], r_hoy=m["r_hoy"], resid5=m["resid5"], umbral5=m["umbral5"],
        noticia_negativa=nt["negativa"], sentimiento=nt["sentimiento"], n_24h=nt["n24"], base_diaria=nt["base"], ratio_noticias=nt["ratio"],
        claves=nt["claves"], sin_noticias=not nt["disponible"], motor_sentimiento=motor, motivo=motivo,
        accion=accion_sugerida(efectivo, e.corte_manana, not nt["disponible"] and not proxy), episodio=ep2, advertencias=adv,
        proxy_noticias=proxy, cobertura_parcial=e.cobertura_parcial, r_mercado=m["r_mercado"], resid_hoy=m["resid_hoy"], fecha=e.fecha)
