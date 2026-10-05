"""Monitor: se ejecuta cada 15 min en horario de bolsa. Calcula el semáforo de tu activo (y del banco de relevo) y envía un mensaje por
Telegram SÓLO si cambia el color. NUNCA envía órdenes (trii no tiene API).

Uso:  python monitor.py                 # normal (no hace nada si el mercado de trii está cerrado)
      python monitor.py --dry-run       # imprime en vez de enviar y NO toca state.json
      python monitor.py --forzar        # ignora el horario de bolsa (pruebas)
      python monitor.py --prueba        # fuerza una alerta de prueba con los datos reales (etiquetada PRUEBA)"""
from __future__ import annotations

import argparse
import sys
import traceback

from src import concurso as C
from src import formato as F
from src.construir import decidir
from src.servicio import crear_contexto, correr_monitor, enviar_o_encolar, leer_estado


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--forzar", action="store_true")
    ap.add_argument("--prueba", action="store_true")
    a = ap.parse_args()
    ctx = crear_contexto(dry=a.dry_run)
    try:
        if a.prueba:
            from src.construir import evaluar_activo, candidatos_banco
            est = leer_estado(ctx)
            res, ent, _ = evaluar_activo(ctx.f, est.activo, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor)
            r = decidir(ctx.f, est, res, ent, candidatos_banco(ctx.f, ctx.cfg, ctx.ahora, est.activo, ent.cierres, ctx.puntuador, ctx.motor), ctx.ahora, ctx.cfg)
            hora = C.hora_orden_manana(C.proxima_sesion(ctx.ahora, ctx.cfg) or ctx.ahora.date(), ctx.cfg)
            aviso = F.it("🧪 PRUEBA: esto no es una señal real; solo sirve para que veas cómo llegan los mensajes.")
            ok = enviar_o_encolar(ctx, aviso + "\n\n" + F.msg_alerta_color(res, None, ctx.ahora, r.mee, r.decision, ctx.f.traducir, hora, est.activo))
            return 0 if ok else 1
        correr_monitor(ctx, forzar=a.forzar)
        return 0
    except Exception as e:                                                         # noqa: BLE001 — el monitor nunca debe morir en silencio
        traceback.print_exc()
        try:
            enviar_o_encolar(ctx, f"⚠️ {F.b('El monitor falló esta vez')} {F.it('(error técnico: ' + type(e).__name__ + ')')}\n"
                                  "No es grave: lo intento de nuevo en 15 minutos. Mientras tanto, si ves un movimiento fuerte, mira trii a mano.")
        except Exception:                                                          # noqa: BLE001
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
