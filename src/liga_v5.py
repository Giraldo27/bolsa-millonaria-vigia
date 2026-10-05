"""Fase 3 · Simulador de ligas con las reglas OFICIALES del concurso (cortes de 50 %, 33 %, 10 %, 10 % y final #1).

Cada VENTANA histórica de 25 sesiones (5-oct → 6-nov equivale a 25 sesiones de EE. UU.: 5 por semana) es un concurso: se sortean rivales sintéticos,
se juegan los cortes del viernes y se mide si mi estrategia sobrevive. Se prueban tres estrategias:
  (a) mantener la base (97 % TSLA) hasta el final,
  (b) base + Regla Maestra (cambia de activo sólo si el múltiplo de MEE compensa el costo),
  (c) (b) + cambio por NEGRO (la Regla Maestra baja el listón cuando mi activo está en NEGRO).

Supuestos que NO son datos (ver ASSUMPTIONS.md): el número real de participantes es desconocido (se evalúa P(#1) y P(top 30) para varios tamaños);
los rivales no operan (comprar y mantener 2–4 activos al azar, con un 30 % concentrado en TSLA o NVDA); "top X %" se mide contra TODO el campo
(alternativa: contra los sobrevivientes); el MEE de la historia se calcula con la volatilidad de 20 días (no hay IV histórica); sin noticias, el color
usa el proxy de volumen + hueco de apertura."""
from __future__ import annotations

import copy
import datetime as dt
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binom

from .regla_maestra import CAMBIAR, VENDER_TODO, ContextoRegla, regla_maestra
from .relevo import Candidato, calcular_mee
from .semaforo import NEGRO, ROJO

COSTO_OP_MICRO = 14_875 / 100_000_000          # comisión fija de cada micro-compra sobre un capital de $100 M


# ------------------------------------------------------------------ calendario oficial en "sesiones desde el inicio"
def calendario_concurso(cfg: dict[str, Any]) -> dict[str, Any]:
    """Sesión (0 = lunes de inicio) de cada corte, del final, de las noches de decisión y de las micro-compras. Las sesiones son de EE. UU., que abre los
    festivos colombianos del 12-oct y el 2-nov (el precio de TSLA se mueve aunque trii no deje operar)."""
    c = cfg["concurso"]
    ini = np.datetime64(c["inicio"])

    def idx(f: str) -> int:
        return int(np.busday_count(ini, np.datetime64(f)))
    cortes = [dict(sesion=idx(x["fecha"]), top_pct=x["top_pct"], semana=x["semana"]) for x in c["cortes"]]
    final = idx(c["final"]["fecha"])
    n_ses = final + 1
    ops = [1, 2, 3] + [5 * w + k for w in range(1, 5) for k in range(4)]            # la compra inicial es la 1.ª operación de la semana 1
    return dict(cortes=cortes, final=final, n_ses=n_ses, noches=[idx(d) for d in cfg["radar"]["noches_decision"]], ops=ops,
                semana_final_desde=5 * (cfg["concurso"]["cortes"][-1]["semana"]))


def proximo_corte(t: int, cal: dict[str, Any]) -> tuple[int, float | None]:
    """Corte que se juega a partir de la sesión t+1: (sesión, top %). El final devuelve (sesión final, None)."""
    for k in cal["cortes"]:
        if k["sesion"] >= t + 1:
            return k["sesion"], float(k["top_pct"])
    return cal["final"], None


# ------------------------------------------------------------------ panel
@dataclass
class PanelLiga:
    fechas: pd.DatetimeIndex
    tickers: list[str]                 # todo el universo permitido (rivales y base), columnas de O y C
    O: np.ndarray                      # (T × A) aperturas en COP
    C: np.ndarray                      # (T × A) cierres en COP
    mgc: list[str]                     # acciones con semáforo y MEE (candidatas a relevo)
    sigma: np.ndarray                  # (T × M) σ de 20 días (USD, sólo datos anteriores)
    color: np.ndarray                  # (T × M) color efectivo al cierre (con episodios ROJO → NEGRO, memoria de TODA la historia), dtype object
    color_dia: np.ndarray              # (T × M) color del día, sin memoria de episodios (así ve el sistema en vivo a los candidatos)
    close_usd: np.ndarray              # (T × M) cierres en USD (para reconstruir los episodios desde el inicio de cada ventana)
    vol_rel: np.ndarray                # (T × M)
    z: np.ndarray                      # (T × M)
    valor_mm: np.ndarray               # (T × M) valor negociado diario medio de 20 días, millones de USD
    ret: np.ndarray                    # (T × M) retornos diarios en USD (para la correlación de 60 días)

    def col(self, t: str) -> int:
        return self.tickers.index(t)


def construir_panel(cfg: dict[str, Any], fechas: pd.DatetimeIndex, apertura: pd.DataFrame, cierre: pd.DataFrame, estudios: dict[str, pd.DataFrame],
                    precios_usd: dict[str, pd.DataFrame]) -> PanelLiga:
    """Junta los precios en COP (todo el universo) con las métricas del semáforo de las acciones MGC, alineados a `fechas`."""
    mgc = [t for t in cfg["universe"]["mgc"] if t in estudios]
    cols = list(apertura.columns)
    reidx = lambda s: s.reindex(fechas)                                                               # noqa: E731
    sig = np.column_stack([reidx(estudios[t]["sigma20"]).to_numpy(float) for t in mgc])
    col = np.column_stack([reidx(estudios[t]["color"]).to_numpy(object) for t in mgc])
    cdia = np.column_stack([reidx(estudios[t]["color_dia"]).to_numpy(object) for t in mgc])
    cusd = np.column_stack([reidx(precios_usd[t]["Close"]).to_numpy(float) for t in mgc])
    vr = np.column_stack([reidx(estudios[t]["vol_rel"]).to_numpy(float) for t in mgc])
    zz = np.column_stack([reidx(estudios[t]["z"]).to_numpy(float) for t in mgc])
    vm = np.column_stack([reidx((precios_usd[t]["Close"] * precios_usd[t]["Volume"]).shift(1).rolling(20).mean() / 1e6).to_numpy(float) for t in mgc])
    rr = np.column_stack([reidx(precios_usd[t]["Close"].pct_change()).to_numpy(float) for t in mgc])
    return PanelLiga(fechas, cols, apertura.reindex(fechas).to_numpy(float), cierre.reindex(fechas).to_numpy(float), mgc, sig, col, cdia, cusd, vr, zz, vm, rr)


# ------------------------------------------------------------------ rivales sintéticos
def pesos_rivales(n: int, n_activos: int, rng: np.random.Generator, concentrados: float, idx_conc: list[int], k_min: int = 2, k_max: int = 4,
                  peso_conc_min: float = 0.70) -> np.ndarray:
    """Matriz (n × activos) de pesos. Una fracción `concentrados` pone entre `peso_conc_min` y 100 % (al azar) en TSLA o NVDA (mitad y mitad) y reparte
    el resto en 1–3 activos al azar (con peso_conc_min = 1 pone TODO: el peor caso para mí, porque duplican mi cartera con más exposición que mi 97 %).
    El resto de los rivales elige entre k_min y k_max activos al azar con pesos iguales."""
    w = np.zeros((n, n_activos))
    n_conc = int(round(n * concentrados))
    if n_conc:
        elegido = rng.choice(idx_conc, n_conc)
        wc = rng.uniform(peso_conc_min, 1.0, n_conc) if peso_conc_min < 1.0 else np.ones(n_conc)
        w[np.arange(n_conc), elegido] = wc
        if peso_conc_min < 1.0:
            sc = rng.random((n_conc, n_activos))
            sc[np.arange(n_conc), elegido] = np.inf                                                      # el activo concentrado no se repite en el resto
            kk = rng.integers(1, 4, n_conc)
            rango = np.argsort(np.argsort(sc, axis=1), axis=1)
            w[:n_conc] += (rango < kk[:, None]) * ((1 - wc) / kk)[:, None]
    m = n - n_conc
    if m:
        k = rng.integers(k_min, k_max + 1, m)
        orden = np.argsort(rng.random((m, n_activos)), axis=1)                                          # permutación al azar de cada fila
        rango = np.argsort(orden, axis=1)                                                                # posición de cada activo en su permutación
        w[n_conc:] = (rango < k[:, None]) / k[:, None]
    return w


def recorridos_rivales(w: np.ndarray, O0: np.ndarray, C: np.ndarray, costo_entrada: float, costo_acum: np.ndarray) -> np.ndarray:
    """Rentabilidad acumulada (n × sesiones) de rivales que compran a la apertura de la sesión 0 y mantienen. Los activos sin precios completos en la ventana
    se descartan y los pesos se renormalizan (un rival sin ningún activo válido queda en NaN)."""
    ok = ~np.isnan(O0) & ~np.isnan(C).any(axis=0)
    wa = w * ok
    tot = wa.sum(axis=1)
    rets = np.nan_to_num(C[:, ok] / O0[ok] - 1)
    cam = (wa[:, ok] @ rets.T) / np.where(tot > 0, tot, np.nan)[:, None]
    return cam - costo_entrada - costo_acum[None, :]


# ------------------------------------------------------------------ mi estrategia, noche a noche
def _corr60(ret: np.ndarray, g: int, j: int) -> np.ndarray:
    """Correlación de 60 días (hasta la sesión g incluida) de cada acción MGC con la acción j. NaN si falta historia."""
    x = ret[g - 59: g + 1]
    xj = x[:, j]
    if np.isnan(xj).any():
        return np.full(x.shape[1], np.nan)
    xc = x - x.mean(axis=0)
    yc = xj - xj.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (xc * yc[:, None]).sum(axis=0) / np.sqrt((xc ** 2).sum(axis=0) * (yc ** 2).sum())


def simular_estrategia(p: PanelLiga, s: int, cfg: dict[str, Any], cal: dict[str, Any], rivales: np.ndarray, vivos: np.ndarray, base: str,
                       decidir: bool, costo_entrada: float, costo_acum: np.ndarray) -> dict[str, Any]:
    """Mi cartera (97 % en `base`, 3 % en efectivo) durante la ventana que empieza en la sesión s del panel. Si `decidir`, cada noche de decisión
    aplica la Regla Maestra con el ranking de los rivales y ejecuta el cambio a la apertura siguiente (con el costo del cambio).
    `vivos` es la matriz (sesiones × rivales) de quién sigue en carrera cada noche (para el objetivo de la semana 5 y la ventaja sobre el #2).
    Devuelve el recorrido de rentabilidad acumulada y los eventos (cambios, venta total)."""
    T = cal["n_ses"]
    peso = cfg["posicion_base"]["peso"]
    costo_cambio = cfg["costos_v5"]["costo_cambio"]
    costo_salida = cfg["costos_v5"]["slippage_mgc"] + cfg["costos_v5"]["comision_pct"]
    O, C = p.O[s: s + T], p.C[s: s + T]
    fechas = p.fechas[s: s + T]
    cash = 1.0 - peso
    activo = base
    j = p.col(activo)
    unidades = peso * (1 - costo_entrada) / O[0, j]
    precio_entrada = O[0, j]
    pendiente: tuple[str, str] | None = None
    usados, ultimo_cambio, vendido = 0, -1, False
    camino = np.empty(T)
    eventos: list[dict[str, Any]] = []
    noches = set(cal["noches"]) if decidir else set()
    cfg_ctx = cfg
    for t in range(T):
        if pendiente is not None:                                                                      # orden de anoche, a la apertura de hoy (validada al decidir)
            tipo, destino = pendiente
            valor = unidades * O[t, p.col(activo)]
            if tipo == "VENDER":
                cash += valor * (1 - costo_salida)
                unidades, vendido = 0.0, True
                eventos.append(dict(t=t, tipo="venta_total", de=activo))
            else:
                valor *= 1 - costo_cambio
                jn = p.col(destino)
                unidades, precio_entrada = valor / O[t, jn], O[t, jn]
                eventos.append(dict(t=t, tipo="cambio", de=activo, a=destino))
                activo, usados, ultimo_cambio = destino, usados + 1, t
            pendiente = None
        camino[t] = cash + unidades * C[t, p.col(activo)] - 1.0 - costo_acum[t]
        if t in noches and not vendido and t < T - 1:
            pendiente = _decidir(p, s, t, T, cfg_ctx, cal, rivales, vivos, activo, precio_entrada, C[t, p.col(activo)], camino[t], usados,
                                 ultimo_cambio == t, fechas)
    return dict(camino=camino, eventos=eventos, cambios=usados, vendio=vendido)


def _color_propio(p: PanelLiga, s: int, g: int, m: int, cfg: dict[str, Any]) -> str | None:
    """Color efectivo de mi activo al cierre de la sesión g, con la máquina de episodios ROJO → NEGRO corriendo SÓLO desde el inicio de la ventana s."""
    from .validacion import episodios
    ini = s - 1                                                                                          # un día antes, sólo para tener el cierre previo
    idx = p.fechas[ini: g + 1]
    cd = pd.Series(p.color_dia[ini: g + 1, m], index=idx, dtype=object)
    cd.iloc[0] = None                                                                                    # lo anterior a la ventana no cuenta
    px = pd.DataFrame({"Close": p.close_usd[ini: g + 1, m]}, index=idx)
    vr = pd.Series(p.vol_rel[ini: g + 1, m], index=idx)
    return episodios(px, cd, vr, cfg, cfg["validacion"]["negro_sentimiento"])["color"].iloc[-1]


def _decidir(p: PanelLiga, s: int, t: int, T: int, cfg: dict[str, Any], cal: dict[str, Any], rivales: np.ndarray, vivos: np.ndarray, activo: str,
             precio_entrada: float, precio: float, mia: float, usados: int, cambio_hoy: bool, fechas: pd.DatetimeIndex) -> tuple[str, str] | None:
    """Regla Maestra de esta noche con la información disponible AL CIERRE de la sesión t (nada del futuro)."""
    g = s + t
    corte_s, top = proximo_corte(t, cal)
    final = t >= cal["semana_final_desde"]
    campo = rivales[:, t]
    campo = campo[~np.isnan(campo)]
    viv = rivales[vivos[t] & ~np.isnan(rivales[:, t]), t]
    mejor = float(viv.max()) if len(viv) else float(campo.max())
    segundo = float(np.partition(viv, -2)[-2]) if len(viv) >= 2 else None
    if final or top is None:
        objetivo = mejor * 100                                                                           # semana 5: el objetivo es la rentabilidad del #1
    else:
        objetivo = float(np.quantile(campo, 1 - top / 100)) * 100                                       # umbral que marca hoy el corte
    soy_primero = mia >= mejor
    jm = p.mgc.index(activo)
    d_corte = max((fechas[min(corte_s, T - 1)] - fechas[t]).days, 1)
    corte_manana = corte_s == t + 1
    sesiones = max(corte_s - t, 1)                                                                       # sesiones hasta el corte (calcular_mee exige > 0)

    def con_excepcion(c: str | None) -> str:
        if c == ROJO and corte_manana and cfg["semaforo"]["rojo_es_negro_si_corte_manana"]:
            return NEGRO                                                                                 # excepción del plan: el corte es mañana
        return c if c is not None else "VERDE"
    # MI activo: episodios ROJO → NEGRO reconstruidos SÓLO desde el inicio de la ventana (el sistema en vivo arranca sin memoria el 5-oct).
    # CANDIDATOS: color del día, sin memoria (en vivo se evalúan sin episodio previo, así que nunca están en NEGRO salvo por la excepción del corte).
    color_mio = con_excepcion(_color_propio(p, s, g, jm, cfg))
    mee_a, _ = calcular_mee(None, d_corte, p.sigma[g, jm], sesiones, False, "sin_datos", cfg["mee"]["respaldo"])
    corr = _corr60(p.ret, g, jm)
    cands = []
    for m, tk in enumerate(p.mgc):
        if tk == activo or np.isnan(p.sigma[g, m]) or np.isnan(p.C[g, p.col(tk)]) or np.isnan(p.O[g + 1, p.col(tk)]):
            continue                                                                                     # sin precio hoy o sin apertura mañana: no es operable (nada del futuro más allá)
        mee, met = calcular_mee(None, d_corte, p.sigma[g, m], sesiones, False, "sin_datos", cfg["mee"]["respaldo"])
        cands.append(Candidato(tk, "mgc", mee, met, con_excepcion(p.color_dia[g, m]), None if np.isnan(corr[m]) else float(corr[m]), None, float(p.vol_rel[g, m]),
                               float(p.z[g, m]), None, float(p.valor_mm[g, m]) if not np.isnan(p.valor_mm[g, m]) else None, False))
    ctx = ContextoRegla(mia=mia * 100, objetivo=objetivo, activo=activo, color_actual=color_mio, mee_actual=mee_a, cambios_restantes=cfg["estado"]["max_cambios"] - usados,
                        cambio_hoy=cambio_hoy, semana_final=final, soy_primero=bool(soy_primero and final),
                        ventaja_pp=((mia - segundo) * 100) if (soy_primero and segundo is not None) else None, ultimo_dia=(t == T - 2),
                        rend_desde_entrada_pp=(precio / precio_entrada - 1) * 100)
    dec, _ = regla_maestra(ctx, cands, cfg)
    if dec.accion == CAMBIAR and dec.candidato:
        return ("CAMBIAR", dec.candidato)
    if dec.accion == VENDER_TODO:
        return ("VENDER", "")
    return None


# ------------------------------------------------------------------ cortes
def vivos_por_noche(rivales: np.ndarray, cal: dict[str, Any], base_pct: str = "campo") -> np.ndarray:
    """Matriz (sesiones × rivales): True si el rival sigue en carrera DURANTE la noche t (ya pasó todos los cortes con sesión < t)."""
    T, N = cal["n_ses"], rivales.shape[0]
    vivos = np.zeros((T, N), bool)
    actual = ~np.isnan(rivales[:, 0])
    cortes = {k["sesion"]: k["top_pct"] for k in cal["cortes"]}
    for t in range(T):
        vivos[t] = actual
        if t in cortes:
            campo = rivales[:, t]
            ref = campo[actual] if base_pct == "supervivientes" else campo[~np.isnan(campo)]
            actual = actual & (campo >= np.quantile(ref, 1 - cortes[t] / 100))
    return vivos


def jugar_cortes(rivales: np.ndarray, mis: dict[str, np.ndarray], cal: dict[str, Any], base_pct: str = "campo") -> tuple[dict[str, Any], np.ndarray]:
    """Aplica los cortes del viernes a los rivales y a cada una de mis estrategias.
    base_pct='campo': "top X %" se mide contra TODOS los participantes; 'supervivientes': sólo contra los que siguen vivos.
    Devuelve ({estrategia: {pasa: [bool×4], vivo_final, q_final, pct_cortes}}, vivos_rivales_al_final)."""
    vivos = np.ones(rivales.shape[0], bool) & ~np.isnan(rivales[:, 0])
    res = {k: dict(pasa=[], pct=[], vivo=True) for k in mis}
    for corte in cal["cortes"]:
        t, top = corte["sesion"], corte["top_pct"]
        campo = rivales[:, t]
        ref = campo[vivos] if base_pct == "supervivientes" else campo[~np.isnan(campo)]
        umbral = np.quantile(ref, 1 - top / 100)
        for k, cam in mis.items():
            ok = res[k]["vivo"] and cam[t] >= umbral
            res[k]["pasa"].append(bool(ok))
            res[k]["pct"].append(float((campo[~np.isnan(campo)] > cam[t]).mean()))                      # fracción del campo que va por delante
            res[k]["vivo"] = bool(ok)
        vivos = vivos & (campo >= umbral)
    tf = cal["final"]
    N = rivales.shape[0]
    for k, cam in mis.items():
        if res[k]["vivo"]:
            adelante = int((vivos & (rivales[:, tf] > cam[tf])).sum())
            res[k]["q_final"] = (adelante + 0.5) / (N + 1)                                              # fracción del campo por delante (suavizada)
        else:
            res[k]["q_final"] = float("nan")
    return res, vivos


def correr_liga(p: PanelLiga, cfg: dict[str, Any], desde: str, base: str = "TSLA", n_rivales: int = 4000, concentrados: float = 0.30, semilla: int = 2026,
                base_pct: str = "campo", paso: int = 1, estrategias: tuple[str, ...] = ("a", "b", "c"), peso_conc_min: float = 0.70) -> pd.DataFrame:
    """Juega TODAS las ventanas de 25 sesiones que empiezan en `desde` o después (cada una es un concurso completo). Una fila por ventana × estrategia."""
    cal = calendario_concurso(cfg)
    T = cal["n_ses"]
    v = cfg["costos_v5"]
    costo_entrada = v["comision_pct"] + v["spread_mgc"]
    costo_acum = np.cumsum(np.isin(np.arange(T), cal["ops"])) * COSTO_OP_MICRO
    cfgs = {"a": cfg, "b": copy.deepcopy(cfg), "c": copy.deepcopy(cfg)}
    cfgs["b"]["regla_maestra"]["cambio_por_negro"] = False
    cfgs["c"]["regla_maestra"]["cambio_por_negro"] = True
    idx_conc = [p.col("TSLA"), p.col("NVDA")]
    i0 = int(p.fechas.searchsorted(pd.Timestamp(desde)))
    filas = []
    for s in range(i0, len(p.fechas) - T + 1, paso):
        rng = np.random.default_rng(semilla + s)
        w = pesos_rivales(n_rivales, len(p.tickers), rng, concentrados, idx_conc, peso_conc_min=peso_conc_min)
        R = recorridos_rivales(w, p.O[s], p.C[s: s + T], costo_entrada, costo_acum)
        vivos = vivos_por_noche(R, cal, base_pct)
        mis, extra = {}, {}
        for k in estrategias:
            r = simular_estrategia(p, s, cfgs[k], cal, R, vivos, base, k != "a", costo_entrada, costo_acum)
            mis[k], extra[k] = r["camino"], r
        res, _ = jugar_cortes(R, mis, cal, base_pct)
        for k in estrategias:
            f = res[k]
            filas.append(dict(inicio=p.fechas[s], estrategia=k, pasa1=f["pasa"][0], pasa2=f["pasa"][1], pasa3=f["pasa"][2], pasa4=f["pasa"][3], vivo_final=f["vivo"],
                              q_final=f["q_final"], ret_final=float(mis[k][-1]), cambios=extra[k]["cambios"], vendio=extra[k]["vendio"],
                              frac_adelante_c1=f["pct"][0], mediana_campo=float(np.nanmedian(R[:, -1])), mejor_campo=float(np.nanmax(R[:, -1]))))
    return pd.DataFrame(filas)


def correr_bases(p: PanelLiga, cfg: dict[str, Any], desde: str, bases: list[str], n_rivales: int = 4000, concentrados: float = 0.30, semilla: int = 2026,
                 base_pct: str = "campo", paso: int = 1, peso_conc_min: float = 0.70) -> pd.DataFrame:
    """¿Y si la base fuera OTRA acción? Mantener cada acción de `bases` (97 %, sin Regla Maestra) en cada ventana, contra LOS MISMOS rivales (la comparación es justa).
    Una fila por ventana × base; las acciones sin precios completos en la ventana se omiten."""
    cal = calendario_concurso(cfg)
    T = cal["n_ses"]
    v = cfg["costos_v5"]
    costo_entrada = v["comision_pct"] + v["spread_mgc"]
    costo_acum = np.cumsum(np.isin(np.arange(T), cal["ops"])) * COSTO_OP_MICRO
    idx_conc = [p.col("TSLA"), p.col("NVDA")]
    i0 = int(p.fechas.searchsorted(pd.Timestamp(desde)))
    filas = []
    for s in range(i0, len(p.fechas) - T + 1, paso):
        rng = np.random.default_rng(semilla + s)
        w = pesos_rivales(n_rivales, len(p.tickers), rng, concentrados, idx_conc, peso_conc_min=peso_conc_min)
        R = recorridos_rivales(w, p.O[s], p.C[s: s + T], costo_entrada, costo_acum)
        vivos = vivos_por_noche(R, cal, base_pct)
        mis = {}
        for b in bases:
            j = p.col(b)
            if np.isnan(p.O[s, j]) or np.isnan(p.C[s: s + T, j]).any():
                continue
            mis[b] = simular_estrategia(p, s, cfg, cal, R, vivos, b, False, costo_entrada, costo_acum)["camino"]
        res, _ = jugar_cortes(R, mis, cal, base_pct)
        for b, cam in mis.items():
            f = res[b]
            filas.append(dict(inicio=p.fechas[s], base=b, pasa1=f["pasa"][0], pasa2=f["pasa"][1], pasa3=f["pasa"][2], pasa4=f["pasa"][3], vivo_final=f["vivo"],
                              q_final=f["q_final"], ret_final=float(cam[-1])))
    return pd.DataFrame(filas)


def resumen_bases(df: pd.DataFrame, n_real: int) -> pd.DataFrame:
    """Por base: probabilidades de seguir vivo en cada corte, P(top 30) y P(#1) para `n_real` participantes, y distribución de la rentabilidad final."""
    filas = []
    for b, g in df.groupby("base"):
        p1, pt = probabilidades_finales(g["q_final"].to_numpy(float), g["vivo_final"].to_numpy(bool), n_real)
        filas.append(dict(base=b, ventanas=len(g), pasa1=g.pasa1.mean(), pasa2=g.pasa2.mean(), pasa3=g.pasa3.mean(), pasa4=g.pasa4.mean(), p1=p1, top30=pt,
                          ret_media=g.ret_final.mean(), ret_mediana=g.ret_final.median(), ret_p10=g.ret_final.quantile(.1), ret_p90=g.ret_final.quantile(.9),
                          pct_pierde=float((g.ret_final < 0).mean())))
    return pd.DataFrame(filas).sort_values(["pasa4", "pasa3"], ascending=False).reset_index(drop=True)


def resumen_liga(df: pd.DataFrame, cfg: dict[str, Any], n_reales: list[int]) -> pd.DataFrame:
    """Probabilidades por estrategia: pasar cada corte (acumulado), llegar vivo al final, P(top 30 de Diamante) y P(#1) para varios tamaños del campo real.
    Intervalo 95 % de cada probabilidad con bootstrap por BLOQUES de mes de inicio (las ventanas se solapan mucho)."""
    from .validacion import bootstrap_por_bloques
    b = cfg["validacion"]["bootstrap"]
    filas = []
    for k, g in df.groupby("estrategia"):
        bloques = g["inicio"].dt.to_period("M").astype(str).to_numpy()
        fila: dict[str, Any] = dict(estrategia=k, ventanas=len(g))
        for col in ("pasa1", "pasa2", "pasa3", "pasa4"):
            m, lo, hi, _ = bootstrap_por_bloques(g[col].to_numpy(float), bloques, b["n"], b["semilla"], b["nivel"])
            fila[col], fila[col + "_lo"], fila[col + "_hi"] = m, lo, hi
        for n in n_reales:
            p1, pt = probabilidades_finales(g["q_final"].to_numpy(float), g["vivo_final"].to_numpy(bool), n)
            fila[f"p1_{n}"], fila[f"top30_{n}"] = p1, pt
        fila.update(ret_media=float(g["ret_final"].mean()), ret_mediana=float(g["ret_final"].median()), ret_p10=float(g["ret_final"].quantile(0.10)),
                    ret_p90=float(g["ret_final"].quantile(0.90)), cambios_medios=float(g["cambios"].mean()), pct_con_cambio=float((g["cambios"] > 0).mean()))
        filas.append(fila)
    return pd.DataFrame(filas)


def comparar_estrategias(df: pd.DataFrame, cfg: dict[str, Any], mejor: str, base: str, n_real: int) -> dict[str, Any]:
    """Diferencia PAREADA (misma ventana, mismos rivales) entre dos estrategias: `mejor` − `base`, con intervalo 95 % por bloques de mes de inicio.
    Medidas: probabilidad de pasar cada corte, P(#1) y P(top 30) para `n_real` participantes, y rentabilidad final. Una diferencia cuyo intervalo cruza 0 NO
    es una mejora demostrada."""
    from .validacion import bootstrap_por_bloques
    b = cfg["validacion"]["bootstrap"]
    x = df[df.estrategia == mejor].set_index("inicio").sort_index()
    y = df[df.estrategia == base].set_index("inicio").sort_index()
    assert x.index.equals(y.index), "las dos estrategias deben compartir ventanas"
    bloques = x.index.to_period("M").astype(str).to_numpy()

    def por_ventana(g: pd.DataFrame, col: str) -> np.ndarray:
        if col in ("p1", "top30"):
            q, vivo = g["q_final"].to_numpy(float), g["vivo_final"].to_numpy(bool)
            qv = np.where(vivo, q, 1.0)
            return np.where(vivo, (1 - qv) ** n_real if col == "p1" else binom.cdf(29, n_real, qv), 0.0)
        return g[col].to_numpy(float)
    out: dict[str, Any] = {}
    for col in ("pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30", "ret_final"):
        d = por_ventana(x, col) - por_ventana(y, col)
        m, lo, hi, p = bootstrap_por_bloques(d, bloques, b["n"], b["semilla"], b["nivel"])
        out[col] = dict(dif=m, lo=lo, hi=hi, p=p)
    out["ventanas"] = int(len(x))
    return out


def probabilidades_finales(q: np.ndarray, vivo: np.ndarray, n_real: int, top: int = 30) -> tuple[float, float]:
    """Con q = fracción del campo que va por delante de mí (sólo entre los vivos), P(#1) = (1−q)^N y P(top 30) = P(Binomial(N, q) < 30), promediado
    sobre las ventanas (las que no sobreviven al corte clave aportan 0)."""
    qv = np.where(vivo, q, 1.0)
    p1 = np.where(vivo, (1 - qv) ** n_real, 0.0)
    pt = np.where(vivo, binom.cdf(top - 1, n_real, qv), 0.0)
    return float(p1.mean()), float(pt.mean())

