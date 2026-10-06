"""Une las fuentes en vivo con el motor de decisión: arma las entradas del semáforo, el MEE, el banco de relevo y la Regla Maestra.

Lo usan monitor.py, radar.py, bot.py y la app. Un activo sin datos suficientes se omite (nunca se inventa un valor)."""
from __future__ import annotations

import datetime as dt
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from . import concurso as C
from .config import group_of
from .data_sources import FuentesDatos
from .regla_maestra import ContextoRegla, Decision, regla_maestra
from .relevo import Candidato, EntradaBanco, calcular_mee, reporte_antes_del_corte
from .semaforo import EntradaSemaforo, ResultadoSemaforo, evaluar, perfil_volumen, proyectar_volumen
from .state import Estado, universo_permitido


class DatosInsuficientes(RuntimeError):
    """No hay datos suficientes de un activo para evaluarlo."""


def _cierre_min(ticker: str, cfg: dict[str, Any]) -> int:
    h, m = cfg["fuentes"]["cierre_bolsa"][group_of(ticker, cfg)].split(":")
    return int(h) * 60 + int(m)


def evaluar_activo(f: FuentesDatos, ticker: str, ahora: dt.datetime, cfg: dict[str, Any], estado: Estado | None,
                   puntuador: Callable[[str], float], motor: str = "vader", con_base_noticias: bool = False
                   ) -> tuple[ResultadoSemaforo, EntradaSemaforo, dict[str, Any]]:
    """Semáforo de un activo con los datos más recientes. Si el mercado está abierto usa el último precio y proyecta el volumen del
    día; si ya cerró (o es fin de semana) evalúa la última sesión completa (es_cierre=True)."""
    daily, intr = f.diario(ticker, 300), f.intradia(ticker)
    ndx_d, ndx_i = f.ndx_diario(300), f.ndx_intradia()
    if daily.empty or intr.empty or ndx_d.empty:
        raise DatosInsuficientes(f"{ticker}: sin datos de precios")
    ultima = intr.index[-1]
    fecha = ultima.date()
    es_cierre = (ultima.hour * 60 + ultima.minute) >= _cierre_min(ticker, cfg) - 15
    sesion = intr[intr.index.date == fecha]
    hist = daily[daily.index < pd.Timestamp(fecha)]
    if len(hist) < 65:
        raise DatosInsuficientes(f"{ticker}: historia diaria insuficiente ({len(hist)} sesiones)")
    precio = float(sesion["Close"].iloc[-1])
    if es_cierre:
        fila = daily[daily.index == pd.Timestamp(fecha)]
        vol_dia = float(fila["Volume"].iloc[0]) if len(fila) and fila["Volume"].iloc[0] > 0 else float(sesion["Volume"].sum())
    else:                                                   # la última barra está incompleta: se proyecta con las completas
        completas = sesion.iloc[:-1]
        vol_dia = proyectar_volumen(float(completas["Volume"].sum()), completas.index[-1].strftime("%H:%M"), perfil_volumen(intr)) if len(completas) else None
    hist_ndx = ndx_d[ndx_d.index < pd.Timestamp(fecha)]
    ndx_hoy = ndx_i[ndx_i.index.date == fecha] if not ndx_i.empty else ndx_i
    noticias = f.noticias(ticker, 2)
    base = f.base_noticias_diaria(ticker) if con_base_noticias else None
    ep = estado.d["semaforo"].get(ticker, {}).get("episodio") if estado else None
    ent = EntradaSemaforo(
        ticker=ticker, fecha=fecha, precio=precio, cierre_previo=float(hist["Close"].iloc[-1]), cierres=hist["Close"], volumenes=hist["Volume"],
        volumen_dia=vol_dia, ndx_cierres=hist_ndx["Close"], ndx_precio=float(ndx_hoy["Close"].iloc[-1]) if len(ndx_hoy) else None,
        ndx_cierre_previo=float(hist_ndx["Close"].iloc[-1]) if len(hist_ndx) else None, noticias=noticias, base_noticias_diaria=base,
        ahora_utc=ahora.astimezone(dt.timezone.utc), es_cierre=es_cierre, corte_manana=C.corte_manana(ahora, cfg), episodio=ep,
        precio_apertura=float(sesion["Open"].iloc[0]), cobertura_parcial=(group_of(ticker, cfg) == "local" and noticias is not None))
    res = evaluar(ent, cfg, puntuador, motor)
    return res, ent, {"fecha": fecha, "es_cierre": es_cierre, "ultima_barra": str(ultima)}


def guardar_resultado(estado: Estado, res: ResultadoSemaforo, ahora: dt.datetime, avisado: bool = False) -> None:
    """Persiste el color y el episodio ROJO del activo (la memoria que necesita NEGRO)."""
    estado.set_semaforo(res.ticker, res.color, ahora, avisado)
    s = estado.d["semaforo"][res.ticker]
    s["color_calculado"], s["episodio"] = res.color_calculado, res.episodio
    estado.guardar()


def mee_de(f: FuentesDatos, ticker: str, sigma20: float, ahora: dt.datetime, cfg: dict[str, Any]) -> tuple[float | None, str, dict[str, Any]]:
    corte = C.corte_vigente(ahora, cfg)
    if corte is None:
        return None, "sin_datos", {}
    iv = f.opciones_iv(ticker, corte["fecha"])
    mee, metodo = calcular_mee(iv.get("iv"), C.dias_calendario_al_corte(ahora, cfg), sigma20, C.sesiones_restantes(ahora, cfg),
                               ticker in cfg["universe"]["local"], iv.get("calidad", "sin_datos"), cfg["mee"]["respaldo"])
    return mee, metodo, iv


def universo_candidatos(cfg: dict[str, Any]) -> list[str]:
    b = cfg["banco"]
    orden = [t for g in b["grupos_candidatos"] for t in cfg["universe"][g]]
    perm = universo_permitido(cfg)
    return [t for t in orden if t in perm and t not in b["excluir_duplicados"]]


def _corr(a: pd.Series, b: pd.Series, dias: int) -> float | None:
    j = pd.concat([a.pct_change(), b.pct_change()], axis=1, keys=["a", "b"], sort=True).dropna().tail(dias)
    return float(j["a"].corr(j["b"])) if len(j) >= 30 else None


def construir_candidato(f: FuentesDatos, ticker: str, ahora: dt.datetime, cfg: dict[str, Any], actual_cierres: pd.Series | None,
                        puntuador: Callable[[str], float], motor: str) -> Candidato | None:
    corte = C.corte_vigente(ahora, cfg)
    if corte is None:
        return None
    try:
        res, ent, _ = evaluar_activo(f, ticker, ahora, cfg, None, puntuador, motor)
    except DatosInsuficientes:
        return None
    mee, metodo, _ = mee_de(f, ticker, res.sigma20, ahora, cfg)
    grupo = group_of(ticker, cfg)
    liquidez = float((ent.cierres * ent.volumenes).tail(20).mean() / 1e6)
    if grupo == "local":                                                               # en la BVC manda la liquidez real de trii: mediana y días sin negociar
        from .liquidez import clasificar
        lq = clasificar(ent.cierres * ent.volumenes, ent.volumenes, cfg)
        from .liquidez import BUENA, afinar
        lq = afinar(f, ticker, lq, cfg)                                                # + lo negociado en la bolsa y que se negocie todo el día, no a ratos
        liquidez = (lq["mediana_mm"] or 0.0) if lq["nivel"] == BUENA else 0.0             # sólo "buena" entra a las recomendaciones; justa, mala o sin dato quedan fuera
    else:                                                                              # acción de EE. UU.: NO vale su liquidez en Nueva York, vale la de trii (libro de Colombia)
        from .liquidez import BUENA, medir
        lq = medir(f, ticker, cfg)
        liquidez = (lq["mediana_mm"] or 0.0) if lq["nivel"] == BUENA else 0.0             # (queda en millones de PESOS negociados en trii; sin dato o poca liquidez = 0)
    return Candidato(
        ticker=ticker, grupo=grupo, mee=mee, metodo=metodo, color=res.color,
        corr=_corr(ent.cierres, actual_cierres, cfg["banco"]["corr_dias"]) if actual_cierres is not None else None,
        reporte_antes_corte=reporte_antes_del_corte(f.reportes(ticker), ahora.date(), corte["fecha"]),
        vol_rel=res.vol_rel, z=res.z, sentimiento=res.sentimiento, valor_negociado_mm=liquidez, sin_noticias=res.sin_noticias,
        r5=float(ent.precio / ent.cierres.iloc[-5] - 1) if len(ent.cierres) >= 5 and ent.cierres.iloc[-5] > 0 else None)


def candidatos_banco(f: FuentesDatos, cfg: dict[str, Any], ahora: dt.datetime, actual: str, actual_cierres: pd.Series | None,
                     puntuador: Callable[[str], float], motor: str) -> list[Candidato]:
    tickers = [t for t in universo_candidatos(cfg) if t != actual]
    with ThreadPoolExecutor(max_workers=cfg["banco"]["max_hilos"]) as ex:
        res = list(ex.map(lambda t: construir_candidato(f, t, ahora, cfg, actual_cierres, puntuador, motor), tickers))
    return [c for c in res if c is not None]


def contexto_regla(estado: Estado, res: ResultadoSemaforo, mee_actual: float | None, precio_actual: float, ahora: dt.datetime,
                   cfg: dict[str, Any]) -> ContextoRegla:
    r = estado.rent
    final = C.semana_final(ahora, cfg)
    mia, primero, segundo = r.get("mia"), r.get("primero"), r.get("segundo")
    soy_primero = bool(final and mia is not None and primero is not None and mia >= primero)
    entrada = estado.d["posicion"].get("precio_entrada")
    return ContextoRegla(
        mia=mia, objetivo=r.get("umbral"), activo=estado.activo, color_actual=res.color, mee_actual=mee_actual,
        cambios_restantes=estado.cambios_restantes(), cambio_hoy=estado.cambio_hoy(ahora.date()), semana_final=final,
        soy_primero=soy_primero, ventaja_pp=(mia - segundo) if (soy_primero and segundo is not None and mia is not None) else None,
        ultimo_dia=C.ultimo_dia_operable(ahora, cfg), rend_desde_entrada_pp=((precio_actual / entrada - 1) * 100) if entrada else None,
        tenidos=frozenset(estado.tenidos()))


@dataclass
class ResultadoMotor:
    ahora: dt.datetime
    activo: str
    res: ResultadoSemaforo
    ent: EntradaSemaforo
    mee: float | None
    metodo_mee: str
    iv: dict[str, Any]
    candidatos: list[Candidato]
    banco: list[EntradaBanco]
    decision: Decision
    corte: dict[str, Any] | None
    excluidos: set[str] = field(default_factory=set)
    avisos: list[str] = field(default_factory=list)


def decidir(f: FuentesDatos, estado: Estado, res: ResultadoSemaforo, ent: EntradaSemaforo, cands: list[Candidato], ahora: dt.datetime,
            cfg: dict[str, Any]) -> ResultadoMotor:
    """MEE de mi activo + Regla Maestra con los candidatos ya calculados (no vuelve a consultar fuentes que estén en caché)."""
    mee, metodo, iv = mee_de(f, estado.activo, res.sigma20, ahora, cfg)
    if len(estado.d.get("cartera", [])) >= 2:                                                   # varias acciones: se compara contra el movimiento esperado del CONJUNTO
        try:
            from . import cartera as CA
            m_c, met_c = CA.mee_cartera(f, CA.valorar(f, estado.d["cartera"], cfg), ahora, cfg)
            if m_c:
                mee, metodo, iv = m_c, met_c, {}
        except Exception:                                                                       # noqa: BLE001 — si falla, se usa la acción principal (como antes)
            f.avisos.append("No pude calcular el movimiento esperado de la cartera: se usa el de la acción principal")
    ctx = contexto_regla(estado, res, mee, ent.precio, ahora, cfg)
    dec, banco = regla_maestra(ctx, cands, cfg)
    excl = set(cfg["regla_maestra"]["excluir_semana_final"]) if (ctx.semana_final and not ctx.soy_primero) else set()
    return ResultadoMotor(ahora, estado.activo, res, ent, mee, metodo, iv, cands, banco, dec, C.corte_vigente(ahora, cfg), excl,
                          list(dict.fromkeys(f.avisos)))


def ejecutar_motor(f: FuentesDatos, estado: Estado, cfg: dict[str, Any], ahora: dt.datetime, puntuador: Callable[[str], float],
                   motor: str = "vader", con_banco: bool = True) -> ResultadoMotor:
    """Todo el motor en un paso: semáforo de mi activo → MEE → banco de relevo → Regla Maestra."""
    res, ent, _ = evaluar_activo(f, estado.activo, ahora, cfg, estado, puntuador, motor, con_base_noticias=True)
    cands = candidatos_banco(f, cfg, ahora, estado.activo, ent.cierres, puntuador, motor) if con_banco else []
    return decidir(f, estado, res, ent, cands, ahora, cfg)
