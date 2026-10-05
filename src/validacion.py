"""Fase 3 · Estudio de eventos del semáforo (2015–2026).

¿Qué pasa con una acción DESPUÉS de cada color? Se calculan las mismas métricas del semáforo real (z, z_res, volumen relativo, residual de 5 días)
de forma vectorizada —una prueba verifica que coinciden con `semaforo.evaluar` día por día— y se mide el retorno residual (acción menos
beta × Nasdaq-100) a 1, 5, 10 y 20 sesiones.

Reglas para no engañarnos:
  · SIN mirar al futuro: la señal es el cierre del día t, la orden se da esa noche y se ejecuta a la APERTURA del día t+1; el retorno futuro
    va de esa apertura al cierre del día t+h. La beta es la conocida el día t.
  · SIN historia de noticias (Finnhub sólo da ~1 año): la "noticia negativa" se reemplaza por el proxy del propio sistema (volumen ≥ 2× y hueco de
    apertura ≤ −1σ). El NEGRO histórico supone que el sentimiento sigue negativo en el día 2 (cota superior de la frecuencia de NEGRO).
  · Intervalos bootstrap por BLOQUES de mes (los eventos se agrupan en crisis; remuestrear días sueltos daría intervalos demasiado optimistas)."""
from __future__ import annotations

import datetime as dt
from typing import Any

import numpy as np
import pandas as pd

from .semaforo import AMARILLO, AZUL, NEGRO, ROJO, VERDE, avanzar_episodio

CLASES = (VERDE, AZUL, AMARILLO, ROJO, NEGRO)
WARMUP = 80                                    # sesiones iniciales sin métricas completas (σ20, beta de 60 días)


# ------------------------------------------------------------------ métricas vectorizadas
def metricas_vectorizadas(px: pd.DataFrame, ndx: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Réplica vectorizada de `semaforo.calcular_metricas` para TODOS los días. `px` tiene Open/Close/Volume; `ndx` tiene Open/Close.
    Cada fila usa sólo datos hasta ese cierre (σ, beta y volumen medio se calculan con los días ANTERIORES, igual que el sistema en vivo)."""
    s = cfg["semaforo"]
    c, o, v = px["Close"], px["Open"], px["Volume"]
    r = c.pct_change()
    rn = ndx["Close"].pct_change().reindex(px.index)
    sigma20 = r.shift(1).rolling(s["sigma_dias"]).std(ddof=1)
    a, n = r.shift(1), rn.shift(1)
    par = a.notna() & n.notna()
    a, n = a.where(par), n.where(par)
    nb = s["beta_dias"]
    cov = a.rolling(nb, min_periods=30).cov(n)
    var_n = n.rolling(nb, min_periods=30).var(ddof=1)
    var_a = a.rolling(nb, min_periods=30).var(ddof=1)
    beta = cov / var_n
    sigma_res = np.sqrt((var_a - cov ** 2 / var_n).clip(lower=0))
    resid_hoy = r - beta * rn
    z = r / sigma20
    z_res = resid_hoy / sigma_res
    k = s["resid_acum_dias"]
    resid5 = r.rolling(k).sum() - beta * rn.rolling(k).sum()
    umbral5 = s["resid_acum_umbral"] * sigma_res * np.sqrt(k)
    vol_rel = v / v.shift(1).rolling(s["vol_dias"]).mean()
    gap = o / c.shift(1) - 1
    d = pd.DataFrame(dict(r=r, rn=rn, sigma20=sigma20, beta=beta, sigma_res=sigma_res, z=z, z_res=z_res, resid_hoy=resid_hoy, resid5=resid5,
                          umbral5=umbral5, vol_rel=vol_rel, gap=gap))
    p = s["proxy_sin_noticias"]
    d["negativa"] = (vol_rel >= p["vol_rel_min"]) & (sigma20 > 0) & (gap <= -p["gap_sigma"] * sigma20)        # proxy de "noticia negativa"
    return d


def clasificar_vectorizado(m: pd.DataFrame, cfg: dict[str, Any], negativa: pd.Series | None = None) -> pd.Series:
    """Réplica vectorizada de `semaforo.clasificar_base` (color 'del día', sin memoria de episodios). `negativa` por defecto es el proxy."""
    s = cfg["semaforo"]
    neg = m["negativa"] if negativa is None else negativa
    vol_ok = m["vol_rel"] >= s["vol_rel_min"]
    z, zr = m["z"], m["z_res"]
    rojo5 = (m["resid5"] <= m["umbral5"]) & neg
    verde = ~(z <= s["z_caida"])
    azul = zr > s["z_res_azul"]
    propia = zr <= s["z_res_caida"]
    color = np.select([rojo5, verde, azul, propia & neg & vol_ok, propia], [ROJO, VERDE, AZUL, ROJO, AMARILLO], default=s["zona_gris"])
    out = pd.Series(color, index=m.index, dtype=object)
    out[m[["z", "z_res", "sigma20", "sigma_res"]].isna().any(axis=1)] = None                               # sin historia suficiente
    return out


# ------------------------------------------------------------------ episodios ROJO → NEGRO (misma máquina de estados que el sistema real)
def episodios(px: pd.DataFrame, color_dia: pd.Series, vol_rel: pd.Series, cfg: dict[str, Any], negro_sentimiento: str = "siempre") -> pd.DataFrame:
    """Recorre la historia día a día con `semaforo.avanzar_episodio` (cierre de cada día = medición oficial del radar).
    negro_sentimiento: 'siempre' = se supone que el sentimiento sigue negativo en el día 2 (NEGRO = ROJO que no recuperó el 50 %);
                       'volumen_dia2' = además exige volumen ≥ 1,5× en esa sesión (la noticia sigue "viva").
    Devuelve: color (efectivo al cierre), nuevo_rojo, nuevo_negro, en_negro."""
    n = len(px)
    cierre = px["Close"].to_numpy()
    colores = color_dia.to_numpy(dtype=object)
    vr = vol_rel.to_numpy()
    fechas = px.index
    vol_min = cfg["semaforo"]["vol_rel_min"]
    out_color = np.empty(n, dtype=object)
    nuevo_rojo, nuevo_negro, en_negro = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
    ep, i_ini, previo = None, -1, None
    for i in range(n):
        c = colores[i]
        if c is None:
            out_color[i] = None
            continue
        if ep is None and c != ROJO:
            out_color[i] = c
            previo = c
            continue
        if ep is None:
            i_ini = i
        n_ses = i - i_ini + 1
        sent_neg = True if negro_sentimiento == "siempre" else bool(vr[i] >= vol_min)
        ep2, color, _ = avanzar_episodio(ep, c, fechas[i].date() if hasattr(fechas[i], "date") else fechas[i], float(cierre[i - 1]), float(cierre[i]), True,
                                         n_ses, sent_neg, cfg)
        if ep is None and ep2 is not None:
            nuevo_rojo[i] = True
        if color == NEGRO and previo != NEGRO:
            nuevo_negro[i] = True
        en_negro[i] = color == NEGRO
        out_color[i], previo, ep = color, color, ep2
    return pd.DataFrame(dict(color=out_color, nuevo_rojo=nuevo_rojo, nuevo_negro=nuevo_negro, en_negro=en_negro), index=px.index)


# ------------------------------------------------------------------ retornos futuros (sin mirar al futuro)
def retornos_futuros(px: pd.DataFrame, ndx: pd.DataFrame, beta: pd.Series, horizontes: list[int]) -> pd.DataFrame:
    """Para cada señal en el cierre del día t: retorno de la APERTURA de t+1 al CIERRE de t+h (R_h) y retorno residual AR_h = R_h − beta_t × retorno del
    Nasdaq-100 en el mismo tramo. NaN si la ventana se sale de la muestra."""
    o_sig = px["Open"].shift(-1)
    nd_o = ndx["Open"].reindex(px.index).shift(-1)
    nd_c = ndx["Close"].reindex(px.index)
    out = {}
    for h in horizontes:
        rh = px["Close"].shift(-h) / o_sig - 1
        nh = nd_c.shift(-h) / nd_o - 1
        out[f"R{h}"] = rh
        out[f"AR{h}"] = rh - beta * nh
    return pd.DataFrame(out, index=px.index)


# ------------------------------------------------------------------ panel completo de un activo
def estudio_activo(ticker: str, px: pd.DataFrame, ndx: pd.DataFrame, cfg: dict[str, Any], inicio: str, negro_sentimiento: str = "siempre",
                   negativa_modo: str = "proxy") -> pd.DataFrame:
    """Una fila por sesión con métricas, color del día, color efectivo (con episodios), marcas de eventos y retornos futuros.
    negativa_modo: 'proxy' (volumen ≥ 2× + hueco de apertura, el respaldo real del sistema) o 'amplio' (cota superior de ROJO: se supone que SIEMPRE
    hay noticia negativa cuando el precio y el volumen cumplen; es lo que ocurriría si toda caída fuerte viniera con mala noticia)."""
    m = metricas_vectorizadas(px, ndx, cfg)
    if negativa_modo == "amplio":
        m["negativa"] = True
    cd = clasificar_vectorizado(m, cfg)
    cd.iloc[:WARMUP] = None
    ep = episodios(px, cd, m["vol_rel"], cfg, negro_sentimiento)
    h = cfg["validacion"]["horizontes"]
    f = retornos_futuros(px, ndx, m["beta"], h)
    d = pd.concat([m, cd.rename("color_dia"), ep, f], axis=1)
    d.insert(0, "ticker", ticker)
    return d.loc[pd.Timestamp(inicio):]


def eventos_por_clase(d: pd.DataFrame) -> dict[str, pd.Series]:
    """Máscaras de eventos (señal al cierre del día). VERDE = línea base (todos los días normales). ROJO = primer día de cada episodio.
    NEGRO = día en que se declara NEGRO (cierre del 2.º día). AZUL/AMARILLO = cada día con ese color."""
    return {VERDE: d["color_dia"] == VERDE, AZUL: d["color_dia"] == AZUL, AMARILLO: d["color_dia"] == AMARILLO,
            ROJO: d["nuevo_rojo"], NEGRO: d["nuevo_negro"]}


# ------------------------------------------------------------------ estadística
def bootstrap_por_bloques(valores: np.ndarray, bloques: np.ndarray, n_boot: int, semilla: int, nivel: float = 0.95) -> tuple[float, float, float, float]:
    """Media, intervalo (nivel) y p-valor bilateral de la media, remuestreando BLOQUES (p. ej. meses) con reemplazo.
    p-valor = 2 × min(P(media* ≤ 0), P(media* ≥ 0)) con piso 1/n_boot."""
    ids, inv = np.unique(bloques, return_inverse=True)
    sumas = np.bincount(inv, weights=valores, minlength=len(ids))
    cuentas = np.bincount(inv, minlength=len(ids)).astype(float)
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, len(ids), size=(n_boot, len(ids)))
    medias = sumas[idx].sum(axis=1) / cuentas[idx].sum(axis=1)
    lo, hi = np.quantile(medias, [(1 - nivel) / 2, 1 - (1 - nivel) / 2])
    p = max(2 * min((medias <= 0).mean(), (medias >= 0).mean()), 1.0 / n_boot)
    return float(valores.mean()), float(lo), float(hi), float(min(p, 1.0))


def resumir(eventos: pd.DataFrame, col: str, cfg: dict[str, Any], semilla_extra: int = 0) -> dict[str, Any]:
    """Estadísticos de la columna `col` (p. ej. 'AR5') sobre los eventos: n, media, mediana, % positivos, IC95 por bloques de mes y p-valor."""
    e = eventos.dropna(subset=[col])
    n = len(e)
    if n == 0:
        return dict(n=0, media=np.nan, mediana=np.nan, pct_pos=np.nan, ic_lo=np.nan, ic_hi=np.nan, p=np.nan)
    b = cfg["validacion"]["bootstrap"]
    bloques = e.index.to_period("M").astype(str).to_numpy() if isinstance(e.index, pd.DatetimeIndex) else e["mes"].to_numpy()
    media, lo, hi, p = bootstrap_por_bloques(e[col].to_numpy(float), bloques, b["n"], b["semilla"] + semilla_extra, b["nivel"])
    return dict(n=n, media=media, mediana=float(e[col].median()), pct_pos=float((e[col] > 0).mean()), ic_lo=lo, ic_hi=hi, p=p)


def agregar_exceso(panel: pd.DataFrame, horizontes: list[int]) -> pd.DataFrame:
    """Agrega AR{h}x = AR{h} − promedio de AR{h} de esa misma acción en TODOS sus días: lo que ocurre después de la señal MENOS la deriva normal
    de la acción (TSLA o NVDA suben mucho en promedio; sin esta resta, cualquier clase parecería 'buena' en ellas)."""
    p = panel.copy()
    for h in horizontes:
        p[f"AR{h}x"] = p[f"AR{h}"] - p.groupby("ticker")[f"AR{h}"].transform("mean")
    return p


def tabla_eventos(panel: pd.DataFrame, cfg: dict[str, Any], grupos: dict[str, list[str]]) -> pd.DataFrame:
    """Tabla larga grupo × clase × horizonte × medida. Medidas: 'AR' (retorno residual), 'ARx' (residual menos la deriva normal de la acción) y
    'R' (retorno bruto). Además `salir_neto` = −AR − costo del cambio: es > 0 sólo si cambiar de activo compensa lo que pierde la acción
    después de la señal (se supone que el reemplazo tiene residual 0)."""
    v = cfg["validacion"]
    costo = cfg["costos_v5"]["costo_cambio"]
    p = agregar_exceso(panel, v["horizontes"])
    filas = []
    for gname, tick in grupos.items():
        sub = p[p["ticker"].isin(tick)]
        masks = eventos_por_clase(sub)
        for clase in CLASES:
            ev = sub[masks[clase]]
            for h in v["horizontes"]:
                for medida, col in (("AR", f"AR{h}"), ("ARx", f"AR{h}x"), ("R", f"R{h}")):
                    r = resumir(ev, col, cfg, semilla_extra=h)
                    filas.append(dict(grupo=gname, clase=clase, h=h, medida=medida, **r, salir_neto=(-r["media"] - costo) if medida == "AR" else np.nan,
                                      n_acciones=int(ev.dropna(subset=[col])["ticker"].nunique())))
    return pd.DataFrame(filas)


def criterio_negro(tabla: pd.DataFrame, cfg: dict[str, Any], grupo: str = "grandes") -> dict[str, Any]:
    """Regla fijada ANTES de ver los resultados (config: validacion.criterio_negro): NEGRO 'muestra deriva' en las acciones grandes si, en TODOS los
    horizontes indicados, el IC del retorno residual DESPUÉS de NEGRO queda por completo por debajo de 0 (la caída sigue, con 95 % de confianza)."""
    c = cfg["validacion"]["criterio_negro"]
    t = tabla[(tabla["grupo"] == grupo) & (tabla["clase"] == NEGRO) & (tabla["medida"] == "AR") & (tabla["h"].isin(c["horizontes"]))].set_index("h")
    detalle = {int(h): dict(n=int(r["n"]), media=float(r["media"]), ic_lo=float(r["ic_lo"]), ic_hi=float(r["ic_hi"]), p=float(r["p"]),
                            salir_neto=float(r["salir_neto"])) for h, r in t.iterrows()}
    hay = len(detalle) == len(c["horizontes"]) and all(x["n"] >= c["n_minimo"] for x in detalle.values())
    deriva = bool(hay and all(x["ic_hi"] < 0 for x in detalle.values()))
    rentable = bool(hay and all(x["salir_neto"] > 0 for x in detalle.values()))
    return dict(deriva=deriva, rentable_neto=rentable, datos_suficientes=hay, detalle=detalle, horizontes=c["horizontes"])
