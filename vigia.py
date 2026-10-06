"""Vigía: UN solo proceso que hace todo y no depende del "cron" de GitHub.

  · contesta el chat de Telegram al instante (comandos, texto libre y botones);
  · cada 15 segundos revisa las noticias de la BVC y avisa todas las de empresas líquidas (src/noticias_bvc.py);
  · cada 2 minutos revisa la macro (petróleo, dólar, Wall Street, Brasil, oro y titulares macro) y dice qué acciones ganan y cuáles pierden (src/macro.py);
  · cada media hora revisa si conviene un cambio "esta por esta" (src/cambios.py) y, antes de abrir, manda el resumen de la mañana;
  · cada 15 minutos, en horario de bolsa, corre el monitor (semáforo de tu cartera);
  · a las 19:30 de lunes a jueves manda el radar de la noche;
  · respalda el estado cada minuto y medio (en la nube: en la rama `estado`).

En GitHub Actions cada "eslabón" vive 5 h 30 min y al terminar lanza el siguiente (ver .github/workflows/vigia.yml). En tu PC: `python vigia.py` y déjalo abierto.
Telegram sólo permite UN lector por bot: no corras esto a la vez en la nube y en el PC, ni junto con bot.py.

Uso:  python vigia.py                  # hasta que se cumpla vigia.duracion_max_min (o Ctrl+C)
      python vigia.py --minutos 10 --prueba   # prueba corta (sin el mensaje de "ya quedé encendido")
      python vigia.py --ronda          # UNA ronda de noticias en seco (imprime, no envía, no guarda) y sale
NUNCA envía órdenes: sólo avisos y recomendaciones."""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import hashlib
import logging
import os
import signal
import sys
import threading
import time
from typing import Any, Callable

from dotenv import load_dotenv

import bot
from src import concurso as C
from src import formato as F
from src import noticias_bvc as N
from src import servicio as S
from src.config import ROOT, load_config

log = logging.getLogger("vigia")
EN_NUBE = os.environ.get("GITHUB_ACTIONS") == "true"


class Vigia:
    def __init__(self, cfg: dict[str, Any], minutos: float | None = None, prueba: bool = False):
        self.cfg = cfg
        self.prueba = prueba                                                            # prueba corta: no anuncia que quedó encendido de forma permanente
        self.v = cfg["vigia"]
        self.fin = time.monotonic() + 60 * (minutos or self.v["duracion_max_min"])
        self.lector = N.Lector(cfg)
        self.mem = N.Memoria()
        self.sugerencia: dict[str, Any] | None = None
        self.banco: list[Any] | None = None                                             # candidatas de la BVC (se recalcula cada media hora en horario de bolsa)
        self._candado_mem = threading.Lock()                                            # noticias y macro comparten la memoria de titulares
        self.parar = asyncio.Event()
        self.rondas = self.avisos = 0
        self._salud: dict[str, bool] = {}
        self._candado_git = threading.Lock()
        arch = ROOT / cfg["estado"]["archivo"]
        self._marca_estado = hashlib.sha1(arch.read_bytes()).hexdigest() if arch.exists() else ""      # lo que había al arrancar ya está respaldado
        self._ultimo_respaldo = time.monotonic()

    # ---------- tareas (cada una corre en un hilo aparte: nunca bloquean al chat) ----------
    def noticias(self) -> None:
        ctx = S.crear_contexto()
        with self._candado_mem:
            msgs = S.tick_noticias(ctx, self.lector, self.mem, self.sugerencia)
        self.rondas += 1
        self.avisos += len(msgs)
        salud = self.mem.d.get("salud") or {}
        if salud != self._salud:                                                        # sólo se registra cuando cambia (no cada 2 minutos)
            self._salud = dict(salud)
            log.info("fuentes de noticias: %s · titulares en memoria: %d", " ".join(f"{k}={'ok' if v else 'FALLA'}" for k, v in salud.items()), len(self.mem.d["vistos"]))
        if msgs:
            log.info("noticias: %d aviso(s) enviados", len(msgs))
            self.guardar(True)                                                          # lo ya avisado se respalda enseguida: si la máquina muere, no se repite el aviso

    def calentar(self) -> None:
        """Deja calculado lo pesado (liquidez de cada acción, relaciones entre empresas del mismo grupo) para que un aviso importante salga sin esperar."""
        from src import macro as M
        from src.construir import universo_candidatos
        ctx = S.crear_contexto()
        vecinos = M.contagio(ctx.f, ctx.cfg, universo_candidatos(ctx.cfg))
        S.acciones_liquidas(ctx, {t for v in vecinos.values() for t in v})

    def macro(self) -> None:
        """Factores macro y titulares macro (reutiliza los titulares que acaba de leer la ronda de noticias)."""
        ctx = S.crear_contexto()
        with self._candado_mem:
            msgs = S.tick_macro(ctx, self.mem, getattr(self.lector, "ultimos", []))
        if msgs:
            log.info("macro: %d aviso(s) enviados", len(msgs))
            self.guardar(True)

    def manana(self) -> None:
        """Resumen antes de abrir: una vez por día de bolsa, desde la hora configurada y hasta la apertura."""
        ctx = S.crear_contexto()
        ahora, h = ctx.ahora, C.horario(ctx.ahora.date(), self.cfg)
        hh, mm = (int(x) for x in self.cfg["macro_vivo"]["resumen_manana"].split(":"))
        if h is None or not (dt.time(hh, mm) <= ahora.time() < h[0]) or C.fase(ahora, self.cfg) != "durante":
            return
        hoy = ahora.date().isoformat()
        with S.transaccion(ctx) as e:
            marca = e.d.setdefault("vigia", {})
            if marca.get("manana") == hoy:
                return
            marca["manana"] = hoy
        S.enviar_o_encolar(ctx, S.resumen_manana(ctx, self.banco), bot.MENU)

    def monitor(self) -> None:
        ctx = S.crear_contexto()
        ctx.salida = lambda t: log.info("monitor: %s", t)
        S.correr_monitor(ctx)

    def sugerir(self) -> None:
        """La BVC con más movimiento esperado (para el pie de los avisos de noticias). Es pesada (todas las acciones): sólo en horario de bolsa y cada media hora."""
        ctx = S.crear_contexto()
        if not (C.mercado_abierto(ctx.ahora, self.cfg) or self.banco is None):
            return
        self.banco = S.banco_bvc(ctx)
        corte = C.corte_vigente(ctx.ahora, self.cfg)
        self.sugerencia = dict(ticker=self.banco[0].c.ticker, mee=self.banco[0].c.mee, corte=corte["fecha"]) if (self.banco and corte) else None
        if self.cfg["cambios"].get("activo"):                                           # ¿hay un cambio "esta por esta" que valga la pena avisar?
            msgs = S.tick_cambios(ctx, self.banco)
            if msgs:
                log.info("cambios: %d aviso(s) enviados", len(msgs))

    def radar(self) -> None:
        """Radar de la noche: una vez por día, de lunes a jueves, desde la hora configurada (19:30) y hasta las 23:00."""
        ctx = S.crear_contexto()
        ahora = ctx.ahora
        h, m = (int(x) for x in self.cfg["radar"]["hora"].split(":"))
        if ahora.weekday() > 3 or not (dt.time(h, m) <= ahora.time() < dt.time(23, 0)) or C.fase(ahora, self.cfg) == "despues":
            return
        hoy = ahora.date().isoformat()
        with S.transaccion(ctx) as e:
            marca = e.d.setdefault("vigia", {})
            if marca.get("radar") == hoy:
                return
            marca["radar"] = hoy                                                        # se marca ANTES: si el radar falla no se repite en bucle toda la noche
        try:
            S.enviar_pendientes(ctx)
            S.enviar_o_encolar(ctx, S.generar_radar(ctx))
            aviso = S.alerta_base(ctx)
            if aviso:
                S.enviar_o_encolar(ctx, aviso)
        except Exception as ex:                                                         # noqa: BLE001
            log.exception("radar")
            S.enviar_o_encolar(ctx, f"⚠️ {F.b('El radar de esta noche falló')} {F.it('(error técnico: ' + type(ex).__name__ + ')')}\n"
                                    "Puedes pedir lo mismo a mano: /estado, /semaforo y /comprar.")

    def guardar(self, forzar: bool = False) -> None:
        """Respaldo en la nube. Lo que TÚ registras (compras, ventas, ranking: state.json) se sube enseguida; la memoria de titulares vistos, que cambia
        a cada rato, sólo cada `guardar_cada_s` (así la rama `estado` no se llena de miles de versiones)."""
        if not EN_NUBE:
            return
        arch = ROOT / self.cfg["estado"]["archivo"]
        marca = hashlib.sha1(arch.read_bytes()).hexdigest() if arch.exists() else ""     # por contenido, no por fecha: reescribir lo mismo no cuenta como cambio
        if not (forzar or marca != self._marca_estado or time.monotonic() - self._ultimo_respaldo >= self.v["guardar_cada_s"]):
            return
        from nube import sincronizar_estado as NE
        with self._candado_git:
            if NE.subir(ROOT):
                log.info("estado respaldado en la rama 'estado'")
        self._marca_estado, self._ultimo_respaldo = marca, time.monotonic()

    def bienvenida(self) -> None:
        """Una sola vez: avisa que el bot quedó encendido de forma permanente."""
        if self.prueba:
            return
        ctx = S.crear_contexto()
        with S.transaccion(ctx) as e:
            marca = e.d.setdefault("vigia", {})
            if marca.get("anunciado"):
                return
            marca["anunciado"] = ctx.ahora.isoformat(timespec="minutes")
        S.enviar_o_encolar(ctx, f"✅ {F.b('Ya quedé encendido de forma permanente')}\n"
                                "• Te contesto al instante, a cualquier hora, sin que tengas que hacer nada.\n"
                                "• Reviso las noticias de la BVC cada 2 minutos y te aviso solo cuando salga una de alto impacto.\n"
                                "• Ya no hace falta usar comandos: escríbeme normal (" + F.b("compré 300 argos a 21500") + ", " + F.b("voy 3,5") + ") o toca un botón.",
                           bot.MENU)

    # ---------- bucles ----------
    async def bucle(self, nombre: str, cada_s: float, fn: Callable[[], None], espera_inicial: float = 0.0) -> None:
        """Repite `fn` cada `cada_s` segundos (descontando lo que tardó) hasta que se pida parar. Las tareas arrancan escalonadas (`espera_inicial`)."""
        if espera_inicial:
            try:
                await asyncio.wait_for(self.parar.wait(), timeout=espera_inicial)
            except asyncio.TimeoutError:
                pass
        while not self.parar.is_set():
            t0 = time.monotonic()
            try:
                await asyncio.to_thread(fn)
            except Exception:                                                           # noqa: BLE001 — una tarea que falla no tumba a las demás ni al chat
                log.exception("falló la tarea %s", nombre)
            try:
                await asyncio.wait_for(self.parar.wait(), timeout=max(cada_s - (time.monotonic() - t0), 1.0))
            except asyncio.TimeoutError:
                pass

    async def correr(self, token: str, chat: int) -> None:
        app = bot.construir_app(token, chat, self.cfg)
        n = self.cfg["noticias_bvc"]
        try:
            loop = asyncio.get_running_loop()
            for s in (signal.SIGTERM, signal.SIGINT):
                loop.add_signal_handler(s, self.parar.set)
        except (NotImplementedError, RuntimeError):                                     # Windows no tiene add_signal_handler: Ctrl+C llega como KeyboardInterrupt
            pass
        async with app:
            await app.start()
            await app.updater.start_polling(allowed_updates=bot.ACTUALIZACIONES, drop_pending_updates=False, bootstrap_retries=-1)
            log.info("vigía encendido: chat al instante, noticias cada %s s, monitor cada %s min", n["cada_s"], self.v["monitor_cada_min"])
            tareas = [asyncio.create_task(self.bucle("bienvenida", 10 ** 9, self.bienvenida)),
                      asyncio.create_task(self.bucle("calentar", 6 * 3600, self.calentar, 15)),
                      asyncio.create_task(self.bucle("guardar", 30, self.guardar, 45)),
                      asyncio.create_task(self.bucle("monitor", self.v["monitor_cada_min"] * 60, self.monitor, 20)),
                      asyncio.create_task(self.bucle("radar", 120, self.radar, 30)),
                      asyncio.create_task(self.bucle("sugerencia", 1800, self.sugerir, 90))]
            if n["activo"]:
                tareas.append(asyncio.create_task(self.bucle("noticias", n["cada_s"], self.noticias, 5)))
            if self.cfg["macro_vivo"].get("activo"):
                tareas.append(asyncio.create_task(self.bucle("macro", self.cfg["macro_vivo"]["cada_s"], self.macro, 40)))
                tareas.append(asyncio.create_task(self.bucle("resumen de la mañana", 60, self.manana, 150)))
            try:
                await asyncio.wait_for(self.parar.wait(), timeout=max(self.fin - time.monotonic(), 1.0))
            except asyncio.TimeoutError:
                pass
            self.parar.set()
            await asyncio.gather(*tareas, return_exceptions=True)
            await app.updater.stop()
            await app.stop()
        await asyncio.to_thread(self.guardar, True)
        log.info("vigía apagado: %d rondas de noticias, %d avisos", self.rondas, self.avisos)


def ronda_en_seco(cfg: dict[str, Any]) -> int:
    """Una ronda de noticias que no envía ni guarda nada: muestra qué fuentes responden y qué avisaría."""
    import tempfile
    from pathlib import Path
    ctx = S.crear_contexto(dry=True, cfg=cfg)
    mem = N.Memoria(Path(tempfile.mkdtemp()) / "noticias.json")
    mem.d["base"] = "prueba"                                                            # sin línea base: se evalúa todo lo que hay ahora mismo
    lector = N.Lector(cfg)
    t0 = time.time()
    est = S.leer_estado(ctx)
    senales, salud = N.ronda(lector, mem, ctx.f, ctx.ahora, cfg, est.tenidos())
    print(f"Fuentes ({time.time() - t0:.1f} s): " + " · ".join(f"{k} {'✅' if v else '❌'}" for k, v in salud.items()))
    print(f"Titulares leídos: {len(mem.d['vistos'])} · avisos que saldrían ahora: {len(senales)} ({', '.join(s.nivel for s in senales) or 'ninguno'})")
    for s in senales:
        print("─" * 70 + "\n" + F.plano(F.msg_noticia_bvc(s, ctx.ahora, cfg) if s.nivel == "alto" else F.msg_noticia_relevante(s, ctx.ahora, cfg)))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutos", type=float, default=None)
    ap.add_argument("--ronda", action="store_true")
    ap.add_argument("--prueba", action="store_true")
    a = ap.parse_args()
    load_dotenv(ROOT / ".env")
    cfg = load_config()
    if a.ronda:
        return ronda_en_seco(cfg)
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID", file=sys.stderr)
        return 1
    try:
        asyncio.run(Vigia(cfg, a.minutos, a.prueba).correr(token, int(chat)))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
