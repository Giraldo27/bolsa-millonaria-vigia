"""Regla Maestra: ¿cambiar de activo? Sólo si la varianza extra del candidato compensa el costo del cambio (≈ 1,3 %).

g = objetivo − mi rentabilidad + 1  (puntos porcentuales)
Si g ≤ 0 → mantener. Si g > 0 → cambiar si MEE_candidato / MEE_actual ≥ (g + 1,3) / g.
Con mi activo en NEGRO el múltiplo baja a (g + 1,3) / (g + d).

Cada decisión lleva un `codigo` del motivo (para explicarla en palabras sencillas) y los `datos` que hacen falta para el texto."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .relevo import EntradaBanco, construir_banco, Candidato
from .semaforo import NEGRO

MANTENER, CAMBIAR, VENDER_TODO, SIN_DATOS = "MANTENER", "CAMBIAR", "VENDER_TODO", "SIN_DATOS"


@dataclass
class ContextoRegla:
    mia: float | None                        # mi rentabilidad acumulada (puntos %, p. ej. 6.5)
    objetivo: float | None                   # umbral del próximo corte; en la semana 5, la rentabilidad del #1
    activo: str
    color_actual: str
    mee_actual: float | None                 # fracción
    cambios_restantes: int
    cambio_hoy: bool
    semana_final: bool = False
    soy_primero: bool = False                # semana 5: mi rentabilidad >= la del #1
    ventaja_pp: float | None = None          # puntos sobre el #2 (sólo si soy #1 y se conoce el #2)
    ultimo_dia: bool = False                 # hoy o mañana es la última sesión del concurso
    rend_desde_entrada_pp: float | None = None
    tenidos: frozenset[str] = frozenset()    # acciones que ya tienes (cartera): nunca se recomiendan como cambio


@dataclass
class Decision:
    accion: str
    g: float | None = None
    multiplo: float | None = None
    candidato: str | None = None
    ratio: float | None = None
    razones: list[str] = field(default_factory=list)
    evaluados: list[dict[str, Any]] = field(default_factory=list)
    modo_negro: bool = False
    codigo: str = ""                         # sin_ranking | g_cero | negro_adelante | primero_mantener | ultimo_dia_mantener | vender_todo |
                                             # limite_dia | limite_total | sin_mee | banco_vacio | ninguno_cumple | cambiar
    mee_actual: float | None = None
    datos: dict[str, Any] = field(default_factory=dict)


def multiplo_requerido(g: float, costo: float = 1.3, d: float | None = None) -> float:
    """(g + costo) / g, o (g + costo) / (g + d) en modo NEGRO."""
    return (g + costo) / (g if d is None else g + d)


def g_de(mia: float, objetivo: float, cfg: dict[str, Any]) -> float:
    return objetivo - mia + cfg["regla_maestra"]["margen_g_pp"]


def _fmt(x: float) -> str:
    return f"{x:.2f}".replace(".", ",")


def regla_maestra(ctx: ContextoRegla, candidatos: list[Candidato], cfg: dict[str, Any]) -> tuple[Decision, list[EntradaBanco]]:
    """Devuelve (decisión, banco usado)."""
    rm = cfg["regla_maestra"]
    excluir = set(rm["excluir_semana_final"]) if (ctx.semana_final and not ctx.soy_primero) else set()
    banco = construir_banco(candidatos, ctx.activo, cfg, excluir | set(ctx.tenidos))     # el banco se muestra aunque todavía no haya ranking; lo que ya tienes no es "relevo"
    if ctx.mia is None or ctx.objetivo is None:
        return Decision(SIN_DATOS, razones=["Faltan tu rentabilidad y el objetivo: usa /rank <mi_rent> <objetivo>."], codigo="sin_ranking",
                        mee_actual=ctx.mee_actual), banco
    g = g_de(ctx.mia, ctx.objetivo, cfg)
    adelante = ctx.mia - ctx.objetivo                       # puntos por encima del objetivo (negativo = voy atrás)
    dec = Decision(MANTENER, g=g, mee_actual=ctx.mee_actual, datos={"adelante": adelante, "mia": ctx.mia, "objetivo": ctx.objetivo,
                                                                    "excluidos": sorted(excluir)})

    # --- último día: asegurar el #1 ---
    if ctx.ultimo_dia and ctx.soy_primero:
        if ctx.ventaja_pp is not None and ctx.ventaja_pp >= rm["ultimo_dia_ventaja_vender_pp"]:
            dec.accion, dec.codigo = VENDER_TODO, "vender_todo"
            dec.datos["ventaja"] = ctx.ventaja_pp
            dec.razones.append(f"Vas #1 con ventaja de {_fmt(ctx.ventaja_pp)} pp (≥ {rm['ultimo_dia_ventaja_vender_pp']:g}): vende todo en la apertura con orden límite.")
            return dec, banco
        dec.codigo = "ultimo_dia_mantener"
        dec.datos["ventaja"] = ctx.ventaja_pp
        dec.razones.append("Vas #1 pero sin ventaja ≥ 8 pp confirmada sobre el #2: mantener (tus perseguidores probablemente tienen activos parecidos).")
        return dec, banco

    negro = ctx.color_actual == NEGRO and rm["cambio_por_negro"]
    dec.modo_negro = negro

    # --- ¿hay motivo para evaluar un cambio? ---
    if negro:
        if adelante > rm["ventaja_negro_pp"]:
            if ctx.rend_desde_entrada_pp is None or ctx.rend_desde_entrada_pp > rm["caida_desde_entrada_pp"]:
                dec.codigo = "negro_adelante"
                dec.datos["rend_entrada"] = ctx.rend_desde_entrada_pp
                dec.razones.append(f"Tu activo está en NEGRO pero vas {_fmt(adelante)} pp por encima del objetivo (> {rm['ventaja_negro_pp']:g}): "
                                   f"sólo se cambia si acumula {rm['caida_desde_entrada_pp']:g} % desde la entrada.")
                return dec, banco
            dec.razones.append(f"NEGRO y el activo acumula {_fmt(ctx.rend_desde_entrada_pp)} % desde la entrada (≤ {rm['caida_desde_entrada_pp']:g} %): se evalúa el cambio.")
        g_ef = max(g, 0.0)
        mult = multiplo_requerido(g_ef, rm["costo_cambio_pp"], rm["d_negro"])
        dec.razones.append(f"Modo NEGRO: múltiplo requerido = (g + {rm['costo_cambio_pp']:g}) / (g + {rm['d_negro']:g}) = {_fmt(mult)}×")
    else:
        if ctx.semana_final and ctx.soy_primero:
            dec.codigo = "primero_mantener"
            dec.razones.append("Semana final y vas #1: mantener (imita; sólo se vende el último día con ventaja ≥ 8 pp).")
            return dec, banco
        if g <= 0:
            dec.codigo = "g_cero"
            dec.razones.append(f"g = {_fmt(g)} ≤ 0: vas por encima del objetivo ({_fmt(adelante)} pp). Mantener (imita al campo).")
            return dec, banco
        mult = multiplo_requerido(g, rm["costo_cambio_pp"])
        dec.razones.append(f"g = {_fmt(g)} pp → múltiplo requerido (g + {rm['costo_cambio_pp']:g}) / g = {_fmt(mult)}×")
    dec.multiplo = mult

    # --- límites ---
    if ctx.cambio_hoy:
        dec.codigo = "limite_dia"
        dec.razones.append("Ya hiciste un cambio hoy (máximo 1 por día): mantener.")
        return dec, banco
    if ctx.cambios_restantes <= 0:
        dec.codigo = "limite_total"
        dec.razones.append(f"Ya usaste los {cfg['estado']['max_cambios']} cambios permitidos: mantener.")
        return dec, banco
    if not ctx.mee_actual or ctx.mee_actual <= 0:
        dec.accion, dec.codigo = SIN_DATOS, "sin_mee"
        dec.razones.append(f"Sin MEE de {ctx.activo}: no se puede aplicar la regla.")
        return dec, banco
    if excluir:
        dec.razones.append(f"Semana final sin ir #1: {', '.join(sorted(excluir))} excluidos como candidatos nuevos.")
    if not banco:
        dec.codigo = "banco_vacio"
        dec.razones.append("El banco de relevo está vacío (ningún candidato líquido y fuera de ROJO/NEGRO).")
        return dec, banco

    # --- primer candidato (en el orden del banco) que cumple el múltiplo ---
    for e in banco:
        ratio = e.c.mee / ctx.mee_actual
        dec.evaluados.append({"ticker": e.c.ticker, "mee": e.c.mee, "ratio": ratio, "cumple": ratio >= mult})
    ok = next((x for x in dec.evaluados if x["cumple"]), None)
    mejor = max(dec.evaluados, key=lambda x: x["ratio"])
    if ok:
        dec.accion, dec.candidato, dec.ratio, dec.codigo = CAMBIAR, ok["ticker"], ok["ratio"], "cambiar"
        dec.razones.append(f"{ok['ticker']}: MEE {_fmt(ok['mee'] * 100)} % ÷ {_fmt(ctx.mee_actual * 100)} % = {_fmt(ok['ratio'])}× ≥ {_fmt(mult)}× → CAMBIAR.")
    else:
        dec.ratio, dec.codigo = mejor["ratio"], "ninguno_cumple"
        dec.datos["mejor"] = mejor["ticker"]
        dec.razones.append(f"Mejor ratio: {mejor['ticker']} {_fmt(mejor['ratio'])}× < {_fmt(mult)}× requerido → MANTENER.")
    return dec, banco
