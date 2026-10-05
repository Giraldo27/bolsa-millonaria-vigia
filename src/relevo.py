"""MEE (movimiento esperado hasta el próximo corte) y banco de relevo (los mejores candidatos para reemplazar mi activo)."""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .semaforo import NEGRO, ROJO
from .state import universo_permitido


def calcular_mee(iv: float | None, dias_calendario: float, sigma_diaria: float | None, sesiones: int,
                 es_local: bool = False, calidad_iv: str = "ok", respaldo: str = "consistente") -> tuple[float | None, str]:
    """MEE como fracción (0,065 = 6,5 %).
    · Con IV ATM del vencimiento posterior al corte: IV × √(días calendario / 365).
    · Respaldo (sin IV, o acciones locales), según `respaldo`:
        'consistente': σ_diaria × √252 × √(días calendario / 365)  (mismo reloj que la IV: comparación justa)
        'literal'    : σ_diaria × √(sesiones hasta el corte)       (fórmula del enunciado)"""
    if not es_local and iv and calidad_iv != "sin_datos" and dias_calendario > 0:
        return float(iv * math.sqrt(dias_calendario / 365)), "IV"
    if sigma_diaria and sesiones > 0 and dias_calendario > 0:
        if respaldo == "literal":
            return float(sigma_diaria * math.sqrt(sesiones)), "σ20"
        return float(sigma_diaria * math.sqrt(252) * math.sqrt(dias_calendario / 365)), "σ20"
    return None, "sin_datos"


@dataclass
class Candidato:
    ticker: str
    grupo: str                              # mgc | etf | local
    mee: float | None
    metodo: str = "IV"
    color: str = "VERDE"
    corr: float | None = None               # correlación de 60 días con mi activo
    reporte_antes_corte: str | None = None  # fecha ISO del reporte entre hoy y el corte
    vol_rel: float | None = None
    z: float | None = None
    sentimiento: float | None = None
    valor_negociado_mm: float | None = None  # millones de USD (mgc/etf) o de COP (local) por día
    sin_noticias: bool = False


@dataclass
class EntradaBanco:
    c: Candidato
    razones: list[str] = field(default_factory=list)
    desempate: bool = False


def liquidez_ok(c: Candidato, cfg: dict[str, Any]) -> bool:
    b = cfg["banco"]
    if c.valor_negociado_mm is None:
        return False
    return c.valor_negociado_mm >= (b["liquidez_min_cop_mm_locales"] if c.grupo == "local" else b["liquidez_min_usd_mm"])


def _desempate(c: Candidato, cfg: dict[str, Any]) -> bool:
    """vol_rel ≥ 1,5 sin caída, o sentimiento positivo."""
    b = cfg["banco"]
    volumen = c.vol_rel is not None and c.vol_rel >= b["vol_rel_desempate"] and (c.z is None or c.z > cfg["semaforo"]["z_caida"])
    return bool(volumen or (c.sentimiento is not None and c.sentimiento > 0))


def _razones(c: Candidato, cfg: dict[str, Any], desempate: bool) -> list[str]:
    r = [f"MEE {c.mee * 100:.1f} % ({c.metodo})"]
    r.append(f"correlación {c.corr:.2f}" if c.corr is not None else "correlación n. d.")
    if c.reporte_antes_corte:
        r.append(f"reporta {pd.Timestamp(c.reporte_antes_corte):%d/%m} (antes del corte)")
    if desempate:
        if c.vol_rel is not None and c.vol_rel >= cfg["banco"]["vol_rel_desempate"]:
            r.append(f"volumen {c.vol_rel:.1f}× sin caída")
        if c.sentimiento is not None and c.sentimiento > 0:
            r.append(f"sentimiento {c.sentimiento:+.2f}")
    if c.sin_noticias:
        r.append("sin datos de noticias")
    return r


def construir_banco(candidatos: list[Candidato], actual: str, cfg: dict[str, Any], excluir: set[str] | None = None) -> list[EntradaBanco]:
    """Filtros: universo permitido (sin lista negra), líquido, con MEE, que no esté en ROJO/NEGRO y distinto de mi activo.
    Orden: MEE (empatados los que están a menos de `tolerancia_mee` del mejor) → menor correlación → catalizador antes del corte →
    desempate (volumen sin caída o sentimiento positivo). Devuelve el top N con sus motivos."""
    permitidos = universo_permitido(cfg)
    excluir = set(excluir or ())
    def filtrar(exigir: bool) -> list[Candidato]:
        return [c for c in candidatos
                if c.ticker in permitidos and c.ticker != actual and c.ticker not in excluir and c.mee and c.mee > 0
                and c.color not in (ROJO, NEGRO) and liquidez_ok(c, cfg) and not (exigir and c.sin_noticias)]
    exigir = cfg["banco"].get("exigir_noticias", False)
    ok = filtrar(exigir)
    if exigir and len(ok) < cfg["banco"].get("relajar_si_menos_de", 0):
        ok = filtrar(False)            # RESPALDO: si las fuentes de noticias están caídas no se deja el banco vacío; quedan marcados "sin datos de noticias"
    tol = cfg["banco"]["tolerancia_mee"]
    restantes = sorted(ok, key=lambda c: c.mee, reverse=True)
    out: list[EntradaBanco] = []
    while restantes and len(out) < cfg["banco"]["top_n"]:
        mejor = restantes[0].mee
        grupo = [c for c in restantes if c.mee >= mejor * (1 - tol)]
        grupo.sort(key=lambda c: (c.corr if c.corr is not None else 1.0, c.reporte_antes_corte is None, not _desempate(c, cfg), -c.mee))
        for c in grupo:
            d = _desempate(c, cfg)
            out.append(EntradaBanco(c, _razones(c, cfg, d), d))
        restantes = [c for c in restantes if c not in grupo]
    return out[: cfg["banco"]["top_n"]]


def reporte_antes_del_corte(reportes: list[dict[str, Any]], hoy: dt.date, corte: dt.date) -> str | None:
    """Primera fecha de reporte entre mañana y el día del corte (incluido)."""
    for r in sorted(reportes, key=lambda x: x["fecha"]):
        f = pd.Timestamp(r["fecha"]).date()
        if hoy <= f <= corte:
            return f.isoformat()
    return None
