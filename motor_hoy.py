"""Corre el motor completo con datos en vivo y lo muestra en español (no guarda nada en state.json).
Uso: python motor_hoy.py [--mia 6 --objetivo 11] [--activo TSLA] [--sin-banco]"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from dotenv import load_dotenv

from src import concurso as C
from src.config import ROOT, load_config
from src.construir import ejecutar_motor
from src.data_sources import FuentesDatos
from src.semaforo import EMOJI
from src.sentimiento import crear_puntuador
from src.state import Estado


def pct(x: float | None, d: int = 1) -> str:
    return "n. d." if x is None else f"{x * 100:.{d}f} %".replace(".", ",")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mia", type=float)
    ap.add_argument("--objetivo", type=float)
    ap.add_argument("--primero", type=float)
    ap.add_argument("--segundo", type=float)
    ap.add_argument("--activo", default=None)
    ap.add_argument("--sin-banco", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    ahora = C.ahora_bogota(cfg)
    est = Estado(cfg, Path(tempfile.mkdtemp()) / "demo_state.json")              # copia de trabajo: no toca state.json
    real = Estado(cfg)
    est.d = real.copia()
    if a.activo:
        est.d["posicion"]["activo"] = a.activo.upper()
    if a.mia is not None and a.objetivo is not None:
        est.d["rentabilidad"].update(mia=a.mia, umbral=a.objetivo, primero=a.primero, segundo=a.segundo)
    f = FuentesDatos(cfg)
    puntuador, motor = crear_puntuador(cfg)
    r = ejecutar_motor(f, est, cfg, ahora, puntuador, motor, con_banco=not a.sin_banco)
    s = r.res
    L = [f"🧭 Motor de decisión — {ahora:%d/%m/%Y %H:%M} (Bogotá) · sentimiento: {s.motor_sentimiento}", ""]
    L.append(f"{EMOJI[s.color]} {r.activo}: {s.color}" + (f"  (calculado: {s.color_calculado})" if s.color != s.color_calculado else ""))
    L.append(f"   retorno del día {pct(s.r_hoy, 2)} · z = {s.z:+.2f} · z_res = {s.z_res:+.2f} · vol_rel = " + (f"{s.vol_rel:.2f}×" if s.vol_rel else "n. d.")
             + f" · beta {s.beta:.2f} · σ20 {pct(s.sigma20, 2)}")
    L.append(f"   residual 5 días {pct(s.resid5, 2)} (umbral {pct(s.umbral5, 2)})")
    L.append(f"   noticias 24 h: {s.n_24h} titulares" + (f" · {s.ratio_noticias:.1f}× el promedio diario ({s.base_diaria:.1f}/día)" if s.ratio_noticias else "")
             + (f" · sentimiento medio {s.sentimiento:+.2f}" if s.sentimiento is not None else " · SIN DATOS DE NOTICIAS") + f" · negativa: {'sí' if s.noticia_negativa else 'no'}")
    for k in s.claves:
        L.append(f"     • «{k['titulo'][:90]}» ({k['puntaje']:+.2f}{', ' + '/'.join(k['palabras']) if k['palabras'] else ''})")
    L.append(f"   motivo: {s.motivo}")
    L.append(f"   acción: {s.accion}")
    if s.advertencias:
        L.append("   ⚠️ " + "; ".join(s.advertencias))
    c = r.corte
    L += ["", f"MEE de {r.activo} hasta el corte del {c['fecha']:%d/%m} ({c['liga']}): {pct(r.mee, 2)} (método {r.metodo_mee}"
          + (f", IV {r.iv['iv'] * 100:.1f} % venc. {r.iv.get('vencimiento')}" if r.iv.get('iv') else "") + ")"]
    if r.banco:
        L += ["", f"Banco de relevo ({len(r.candidatos)} candidatos evaluados):"]
        for i, e in enumerate(r.banco, 1):
            L.append(f"  {i}. {e.c.ticker:<9} {EMOJI[e.c.color]}  " + " · ".join(e.razones))
    d = r.decision
    L += ["", f"⚖️ Regla Maestra → {d.accion}" + (f" {d.candidato}" if d.candidato else "")]
    L += [f"   {x}" for x in d.razones]
    if d.evaluados:
        L.append("   ratios: " + ", ".join(f"{x['ticker']} {x['ratio']:.2f}×{'✓' if x['cumple'] else ''}" for x in d.evaluados))
    if r.avisos:
        sin_cob = [x.split(":")[0] for x in r.avisos if x.endswith("sin cobertura gratuita de noticias")]
        otros = [x for x in r.avisos if not x.endswith("sin cobertura gratuita de noticias")]
        L += ["", "Avisos de fuentes:"]
        if sin_cob:
            L.append(f"  ⚠️ {len(sin_cob)} acciones locales sin cobertura gratuita de noticias ({', '.join(sin_cob[:6])}…): su semáforo no puede llegar a ROJO")
        L += [f"  ⚠️ {x}" for x in otros]
    print("\n".join(L))


if __name__ == "__main__":
    main()
