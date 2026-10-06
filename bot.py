"""Bot de Telegram (python-telegram-bot). Responde a los chats que indique `bot.responder_a` (por defecto, todos). Sólo alertas y recomendaciones: nunca envía órdenes.

Se le puede hablar de tres formas: con comandos (/semaforo), escribiendo normal ("compré 300 argos a 21500", "voy 3,5") o tocando los botones de /menu.

Uso:  python bot.py              # queda escuchando (sólo el chat; para chat + noticias + monitor en un solo proceso usa vigia.py)
      python bot.py --probar     # verifica el token y la conexión y sale
      python bot.py --una-vez    # procesa los mensajes pendientes y sale (modo antiguo de la nube: ver README)"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from typing import Any, Callable

from dotenv import load_dotenv
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.error import BadRequest
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from src import formato as F
from src import servicio as S
from src.config import ROOT, load_config
from src.notify import a_plano

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
for _ruidoso in ("httpx", "httpcore", "telegram", "apscheduler"):      # httpx imprime la URL completa (¡con el token!) en cada petición
    logging.getLogger(_ruidoso).setLevel(logging.WARNING)
log = logging.getLogger("bot")

ACTUALIZACIONES = ["message", "callback_query"]                         # mensajes y toques de botones
MENU = [[("📊 Semáforo", "c:semaforo"), ("💼 Mi cartera", "c:cartera")],
        [("📰 Noticias BVC", "c:noticias"), ("🌍 Macro", "c:macro")],
        [("🛒 Qué comprar", "c:comprar"), ("📍 Cómo voy", "c:estado")],
        [("➕ Compré", "p:compra"), ("➖ Vendí", "p:venta")],
        [("🏁 Mi ranking", "p:rank")]]
PENDIENTE_S = 600                                                       # cuánto espera el bot la respuesta a "¿cuántas compraste?"


def teclado(filas: list[list[tuple[str, str]]] | None) -> InlineKeyboardMarkup | None:
    return InlineKeyboardMarkup([[InlineKeyboardButton(t, callback_data=d) for t, d in fila] for fila in filas]) if filas else None


def _mensaje(update: Update) -> Any:
    """El mensaje al que se contesta: el que escribió el usuario o, si tocó un botón, el mensaje que traía ese botón."""
    m = getattr(update, "message", None)
    if m is None and getattr(update, "callback_query", None) is not None:
        m = update.callback_query.message
    return m


async def responder(update: Update, texto: str, botones: list[list[tuple[str, str]]] | None = None) -> None:
    """Responde con formato HTML; si Telegram lo rechaza (etiqueta rota), reenvía el mismo texto sin formato."""
    m = _mensaje(update)
    extra = {"reply_markup": teclado(botones)} if botones else {}
    try:
        await m.reply_text(texto, parse_mode="HTML", disable_web_page_preview=True, **extra)
    except BadRequest:
        await m.reply_text(a_plano(texto), disable_web_page_preview=True, **extra)


def construir_app(token: str, chat_id: int, cfg: dict | None = None) -> Application:
    """Crea la aplicación. Según `bot.responder_a` en config.yaml contesta a TODOS los chats (por defecto, decisión del usuario: también al grupo)
    o sólo a TELEGRAM_CHAT_ID. Con `bot.escritura_solo_mi_chat: true`, lo que cambia el estado (registrar compras, ventas, ranking) queda reservado a ese chat."""
    cfg = cfg or load_config()
    b = cfg["bot"]
    app = Application.builder().token(token).build()
    filtro = filters.ALL if b.get("responder_a", "todos") == "todos" else filters.Chat(chat_id=chat_id)
    ultimo: dict[int, float] = {}                                                       # última consulta pesada por chat (enfriamiento)
    pendiente: dict[int, tuple[str, str | None, float]] = {}                            # chat → (compra|venta|rank, ticker, hasta cuándo): respuesta a un botón
    comandos: dict[str, Callable[[Update, list[str]], Any]] = {}

    def solo_dueno(chat: int) -> bool:
        return b.get("escritura_solo_mi_chat", False) and chat != chat_id

    def comando(nombre: str, fn: Callable[[S.Contexto, list[str]], str], espera: str | None = None, escribe: bool = False, pesado: bool = False,
                admite_html: bool = True, botones: list[list[tuple[str, str]]] | None = None):
        async def ejecutar(update: Update, args: list[str]) -> None:
            chat = update.effective_chat.id
            if escribe and solo_dueno(chat):
                await responder(update, "🔒 Este comando cambia el seguimiento y sólo lo puede usar el dueño del bot.")
                return
            quiere_html = nombre == "informe" or (admite_html and any(a.lower() == "html" for a in args))
            args = [a for a in args if a.lower() != "html"]
            if quiere_html or pesado:                                                   # evita gastar las consultas de datos a base de repetir
                ahora_m, espera_s = time.monotonic(), b.get("enfriamiento_pesado_s", 45)
                if ahora_m - ultimo.get(chat, -1e9) < espera_s:
                    await responder(update, f"⏳ Acabo de hacer una consulta grande. Espera unos {int(espera_s - (ahora_m - ultimo[chat])) + 1} segundos y vuelve a pedirla.")
                    return
                ultimo[chat] = ahora_m
            if espera:
                await _mensaje(update).reply_text(espera)
            doc: tuple[str, str] | None = None
            try:
                ctx = S.crear_contexto()                         # contexto fresco: hora y datos actuales
                if nombre == "informe":
                    texto, htm = await asyncio.to_thread(S.generar_informe, ctx)
                    doc = (htm, f"informe_{ctx.ahora:%Y%m%d_%H%M}.html")
                else:
                    texto = await asyncio.to_thread(fn, ctx, args)
                    if quiere_html:
                        doc = (await asyncio.to_thread(S.informe_html, ctx), f"informe_{ctx.ahora:%Y%m%d_%H%M}.html")
            except Exception:                                     # noqa: BLE001 — el bot nunca se cae por un error de datos
                log.exception("error en /%s", nombre)
                texto = (f"⚠️ No pude completar /{nombre} ahora (algún dato no está disponible).\n"
                         "Prueba de nuevo en un minuto con /actualizar, o mira trii a mano.")
            partes = _partir(texto)
            for k, parte in enumerate(partes):
                await responder(update, parte, botones if k == len(partes) - 1 else None)
            if doc:
                await _mensaje(update).reply_document(document=doc[0].encode("utf-8"), filename=doc[1],
                                                      caption="📎 Informe completo: ábrelo con el navegador (Chrome, Safari…). Explica cada número y cada paso de la regla.")

        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            await ejecutar(update, list(context.args or []))
        comandos[nombre] = ejecutar
        app.add_handler(CommandHandler(nombre, handler, filters=filtro))

    comando("estado", lambda c, a: S.resp_estado(c))
    comando("semaforo", lambda c, a: S.resp_semaforo(c, a[0] if a else None), "⏳ Revisando tus acciones…")
    comando("detalle", lambda c, a: S.resp_detalle(c, a[0] if a else None), "⏳ Calculando los números técnicos…")
    comando("banco", lambda c, a: S.resp_banco(c), "⏳ Buscando los mejores relevos (tarda cerca de 1 minuto)…", pesado=True)
    comando("comprar", lambda c, a: S.resp_comprar(c), "⏳ Comparando las acciones de la BVC (cerca de 1 minuto)…", pesado=True)
    comando("noticias", lambda c, a: S.resp_noticias(c, " ".join(a) if a else None))
    comando("macro", lambda c, a: S.resp_macro(c), "⏳ Mirando el petróleo, el dólar, Wall Street y Brasil…")
    comando("catalizadores", lambda c, a: S.resp_catalizadores(c), "⏳ Buscando fechas importantes…")
    comando("actualizar", lambda c, a: S.resp_actualizar(c), "🔄 Consultando todo de nuevo con datos frescos (cerca de 1 minuto)…", pesado=True)
    comando("base", lambda c, a: S.resp_base(c), "⏳ Comparando todas las acciones de la BVC (cerca de 1 minuto)…", pesado=True)
    comando("informe", lambda c, a: "", "📝 Armando el informe completo (cerca de 1 minuto)…")
    comando("rank", lambda c, a: S.resp_rank(c, a), escribe=True, admite_html=False)
    comando("pos", lambda c, a: S.resp_pos(c, a), escribe=True, admite_html=False)
    comando("compra", lambda c, a: S.resp_compra(c, a), escribe=True, admite_html=False)
    comando("venta", lambda c, a: S.resp_venta(c, a), escribe=True, admite_html=False)
    comando("deshacer", lambda c, a: S.resp_deshacer(c), escribe=True, admite_html=False)
    comando("cartera", lambda c, a: S.resp_cartera(c), "⏳ Valorando tu cartera…")
    comando("op", lambda c, a: S.resp_op(c), escribe=True, admite_html=False)
    comando("ayuda", lambda c, a: F.AYUDA, admite_html=False, botones=MENU)
    comando("start", lambda c, a: F.AYUDA, admite_html=False, botones=MENU)
    comando("menu", lambda c, a: "👇 " + F.b("¿Qué quieres hacer?"), admite_html=False, botones=MENU)

    def suscripcion(activar: bool):
        async def handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
            try:
                texto = await asyncio.to_thread(S.suscribir, S.crear_contexto(), update.effective_chat.id, activar)
            except Exception:                                     # noqa: BLE001
                log.exception("error en la suscripción")
                texto = "⚠️ No pude cambiar los avisos de este chat ahora. Prueba de nuevo en un minuto."
            await responder(update, texto)
        return handler
    app.add_handler(CommandHandler("avisos", suscripcion(True), filters=filtro))
    app.add_handler(CommandHandler("silencio", suscripcion(False), filters=filtro))

    def tomar_pendiente(chat: int) -> tuple[str, str | None] | None:
        p = pendiente.pop(chat, None)
        return (p[0], p[1]) if p and time.monotonic() <= p[2] else None

    async def texto_libre(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Mensajes sin comando: "compré 300 argos a 21500", "vendí tesla", "voy 3,5", "¿qué compro?"…"""
        chat, escrito = update.effective_chat.id, (update.message.text or "").strip()
        if not escrito:
            return
        pend = tomar_pendiente(chat)
        forzar = pend[0] if pend and pend[0] in ("compra", "venta") else None
        if pend and pend[0] == "compra" and pend[1]:
            escrito = f"{pend[1]} {escrito}"                                            # venía de "✅ La compré" de una noticia: ya se sabe la acción
        if pend and pend[0] == "rank":
            escrito = f"voy {escrito}" if not any(p in escrito.lower() for p in ("voy", "corte", "llevo")) else escrito
        try:
            ctx = S.crear_contexto()
            r = await asyncio.to_thread(S.resp_texto, ctx, escrito, forzar, solo_dueno(chat))
        except Exception:                                         # noqa: BLE001
            log.exception("error entendiendo un mensaje")
            await responder(update, "⚠️ No pude procesar eso ahora. Prueba de nuevo en un minuto.")
            return
        if r.get("consulta"):
            await comandos[r["consulta"]](update, [r["ticker"]] if r.get("ticker") else [])
            return
        if r.get("pendiente"):
            pendiente[chat] = (r["pendiente"], r.get("ticker"), time.monotonic() + PENDIENTE_S)
        if not r.get("texto"):
            if update.effective_chat.type != "private":                                 # en un grupo no se contesta a lo que no es para el bot
                return
            await responder(update, "🤔 No te entendí. Escríbeme por ejemplo " + F.b("compré 300 argos a 21500") + ", " + F.b("vendí tesla") + " o " + F.b("voy 3,5")
                            + " — o toca un botón:", MENU)
            return
        for parte in _partir(r["texto"]):
            await responder(update, parte)

    async def boton(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Toque de un botón. c:<comando> lo ejecuta; p:<compra|venta|rank> pregunta el dato; k:<TICKER> = "la compré"; v:<TICKER> = "la vendí toda"."""
        q = update.callback_query
        try:
            await q.answer()
        except Exception:                                         # noqa: BLE001 — un botón viejo (más de 48 h) no se puede "contestar", pero sí atender
            pass
        tipo, _, dato = (q.data or "").partition(":")
        chat = update.effective_chat.id
        if b.get("responder_a", "todos") != "todos" and chat != chat_id:                # modo cerrado: los botones tampoco responden a otros chats
            return
        if tipo == "c" and dato in comandos:
            await comandos[dato](update, [])
        elif tipo == "p" and dato == "compra":
            pendiente[chat] = ("compra", None, time.monotonic() + PENDIENTE_S)
            await responder(update, "➕ " + F.b("¿Qué compraste?") + "\nEscríbelo como te salga: " + F.b("300 argos a 21500") + " · " + F.b("20 millones de ecopetrol") + " · " + F.b("8 meta")
                            + "\n" + F.it("Si no pones precio, uso el de hoy."))
        elif tipo == "p" and dato == "venta":
            tenidos = sorted(await asyncio.to_thread(lambda: S.leer_estado(S.crear_contexto()).tenidos()))
            if not tenidos:
                await responder(update, "No tienes compras registradas todavía.")
                return
            pendiente[chat] = ("venta", None, time.monotonic() + PENDIENTE_S)
            await responder(update, "➖ " + F.b("¿Cuál vendiste?") + "\nToca una si la vendiste TODA, o escribe por ejemplo " + F.b(f"100 {tenidos[0].lower()}") + " si fue una parte.",
                            [[(f"Vendí todo {t}", f"v:{t}")] for t in tenidos])
        elif tipo == "p" and dato == "rank":
            pendiente[chat] = ("rank", None, time.monotonic() + PENDIENTE_S)
            await responder(update, "🏁 " + F.b("¿Cómo vas?") + "\nEscribe tu rentabilidad (la que muestra trii), por ejemplo " + F.b("3,5") + ".\nSi también sabes la del corte: "
                            + F.b("voy 3,5 y el corte está en 8") + ".")
        elif tipo == "k" and dato:
            pendiente[chat] = ("compra", dato, time.monotonic() + PENDIENTE_S)
            await responder(update, f"✅ {F.b('¿Cuánto compraste de ' + dato + '?')}\nEscribe la cantidad de acciones ({F.b('300')}) o la plata ({F.b('20 millones')}). "
                                    "Si quieres, agrega el precio: " + F.b("300 a 21500") + ".")
        elif tipo == "v" and dato:
            if solo_dueno(chat):
                await responder(update, "🔒 Sólo el dueño del bot puede registrar ventas.")
                return
            await comandos["venta"](update, [dato])

    app.add_handler(CallbackQueryHandler(boton), group=1)
    app.add_handler(MessageHandler(filtro & filters.TEXT & ~filters.COMMAND, texto_libre), group=1)

    async def error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("error de Telegram: %s", type(context.error).__name__)   # sin imprimir el token
    app.add_error_handler(error)
    return app


def _partir(texto: str, largo: int = 3900) -> list[str]:
    from src.notify import partir
    return partir(texto, largo)


async def procesar_pendientes(app: Application, estado_offset: dict) -> int:
    """Modo nube antiguo: lee los mensajes pendientes (getUpdates sin espera), los procesa con los mismos manejadores y guarda hasta dónde llegó (offset).
    No usar a la vez que otro lector del mismo bot: Telegram sólo permite UNO (error 409)."""
    n = 0
    async with app:
        ups = await app.bot.get_updates(offset=estado_offset.get("offset"), timeout=0, allowed_updates=ACTUALIZACIONES)
        for u in ups:
            await app.process_update(u)
            estado_offset["offset"] = u.update_id + 1
            n += 1
    return n


def main() -> int:
    load_dotenv(ROOT / ".env")
    cfg = load_config()                                                                  # también limpia las claves del entorno (BOM, espacios)
    token, chat = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat:
        print("Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en .env", file=sys.stderr)
        return 1
    app = construir_app(token, int(chat), cfg)
    if "--probar" in sys.argv:
        async def probar() -> None:
            async with app:
                me = await app.bot.get_me()
                print(f"✅ Bot conectado: @{me.username}. Handlers: {sum(len(h) for h in app.handlers.values())}. Responde a: {cfg['bot'].get('responder_a', 'todos')}.")
        asyncio.run(probar())
        return 0
    if "--una-vez" in sys.argv:
        import json
        from pathlib import Path
        arch = ROOT / cfg["estado"]["archivo"]
        off_arch = arch.with_name("telegram_offset.json")
        off = json.loads(off_arch.read_text()) if off_arch.exists() else {}
        try:
            n = asyncio.run(procesar_pendientes(app, off))
        finally:
            Path(off_arch).write_text(json.dumps(off))
        print(f"Mensajes procesados: {n}")
        return 0
    log.info("Bot escuchando (responde a: %s). Ctrl+C para salir.", cfg["bot"].get("responder_a", "todos"))
    app.run_polling(drop_pending_updates=True, bootstrap_retries=-1, allowed_updates=ACTUALIZACIONES)         # reintenta indefinidamente si no hay internet al arrancar
    return 0


if __name__ == "__main__":
    sys.exit(main())
