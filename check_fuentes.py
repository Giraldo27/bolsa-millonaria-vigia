"""Verifica las fuentes de datos con consultas reales y muestra un resumen en español.
Uso: python check_fuentes.py [--telegram]   (--telegram envía el resumen a tu chat)"""
from __future__ import annotations

import sys

import pandas as pd
from dotenv import load_dotenv

from src import concurso as C
from src.config import ROOT, load_config
from src.data_sources import FuentesDatos
from src.notify import enviar
from src.state import Estado

CANDIDATAS = ["TSLA", "NVDA", "META", "UBER", "NUCO"]


def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    f = FuentesDatos(cfg)
    ahora = C.ahora_bogota(cfg)
    corte = C.corte_vigente(ahora, cfg) or C.cortes(cfg)[0]
    L = [f"🔎 Verificación de fuentes — {ahora:%d/%m/%Y %H:%M} (Bogotá)", ""]
    for s in f.salud("TSLA"):
        L.append(f"{'✅' if s['ok'] else '❌'} {s['fuente']}: {s['detalle']}")
    L += ["", f"IV ATM (vencimiento posterior al corte del {corte['fecha']:%d/%m}):"]
    for t in CANDIDATAS:
        o = f.opciones_iv(t, corte["fecha"])
        L.append(f"  {t}: " + (f"{o['iv'] * 100:.1f} % · venc. {o['vencimiento']} · {o['calidad']}" if o.get("iv") else f"sin dato ({o.get('motivo', 'cotización inválida')})"))
    L += ["", "Reportes próximos (primer dato):"]
    for t in CANDIDATAS + ["ECOPETROL"]:
        r = f.reportes(t)
        L.append(f"  {t}: " + (", ".join(f"{x['fecha']} {x['hora'] or '?'} ({x['fuente']})" for x in r[:2]) if r else "sin fecha"))
    L += ["", "Noticias (últimos 3 días, Finnhub/yfinance):"]
    for t in ["TSLA", "ECOPETROL", "ISA"]:
        n = f.noticias(t, 3)
        L.append(f"  {t}: {len(n)} titulares" + (f" · el más reciente: «{n[0].titulo[:70]}» ({n[0].proveedor})" if n else ""))
    e = Estado(cfg)
    L += ["", f"Estado: posición {e.activo}, cambios usados {e.cambios_usados()}/{cfg['estado']['max_cambios']}, operaciones esta semana {e.ops_semana(ahora.date())}",
          f"Próximo corte: {corte['fecha']:%d/%m} ({corte['liga']}) · sesiones restantes: {C.sesiones_restantes(ahora, cfg)}"]
    if f.avisos:
        L += ["", "Avisos:"] + [f"  ⚠️ {a}" for a in dict.fromkeys(f.avisos)]
    txt = "\n".join(L)
    print(txt)
    if "--telegram" in sys.argv:
        print("Telegram:", "enviado ✅" if enviar(txt, cfg) else "NO enviado ❌")
    return 0


if __name__ == "__main__":
    sys.exit(main())
