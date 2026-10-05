"""Radar nocturno (19:30 de lunes a jueves): resumen completo por Telegram — semáforo, banco de relevo, catalizadores de 10 días hábiles,
Regla Maestra y recordatorio de la micro-compra de mañana. Es la medición oficial del cierre (decide NEGRO).

Uso:  python radar.py                 # genera y envía
      python radar.py --dry-run       # sólo imprime (no toca state.json)
      python radar.py --fecha 2026-10-27   # simula otra fecha a las 19:30 (pruebas; con --dry-run)"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import traceback

from src import concurso as C
from src import formato as F
from src.servicio import alerta_base, crear_contexto, enviar_o_encolar, enviar_pendientes, generar_radar


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fecha", default=None)
    a = ap.parse_args()
    utc = None
    if a.fecha:
        utc = dt.datetime.fromisoformat(a.fecha + "T19:30:00-05:00").astimezone(dt.timezone.utc)
    ctx = crear_contexto(dry=a.dry_run or bool(a.fecha), utc_now=utc)
    if C.fase(ctx.ahora, ctx.cfg) == "despues":
        ctx.salida("El concurso terminó: no hay radar.")
        return 0
    try:
        enviar_pendientes(ctx)
        ok = enviar_o_encolar(ctx, generar_radar(ctx))
        try:                                                                       # novedad de la comparación de base (otra acción superó a la tuya)
            aviso = alerta_base(ctx)
            if aviso:
                enviar_o_encolar(ctx, aviso)
        except Exception as ex:                                                    # noqa: BLE001 — la comparación de base es secundaria: no tumba el radar
            ctx.salida(f"no se pudo revisar la comparación de base: {type(ex).__name__}")
        return 0 if ok else 1
    except Exception as e:                                                         # noqa: BLE001
        traceback.print_exc()
        enviar_o_encolar(ctx, f"⚠️ {F.b('El radar de esta noche falló')} {F.it('(error técnico: ' + type(e).__name__ + ')')}\n\n"
                              "No pude armar el resumen. Si hoy hay que decidir, hazlo a mano en 3 minutos:\n"
                              "• Tu ranking: app de trii\n• Fechas de resultados: Investing.com\n• Cuánto se espera que se mueva: OptionCharts (\"expected move\")\n"
                              "• Noticias: ficha de la acción en Yahoo Finance\nTambién puedes probar /actualizar o /estado aquí en el chat.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
