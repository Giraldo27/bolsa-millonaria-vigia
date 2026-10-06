"""Servicio: orquesta fuentes → motor → mensajes para el monitor, el radar y el bot. Con respaldos para que "algo salga mal" no deje ciego
ni mudo al sistema: caché vencida, cadena de noticias, proxy sin noticias, cola de alertas pendientes y avisos de salud.

NUNCA envía órdenes: sólo alertas y recomendaciones (trii no tiene API)."""
from __future__ import annotations

import datetime as dt
import json
import re
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from . import concurso as C
from . import formato as F
from .catalizadores import catalizadores, tickers_con_reporte
from .config import ROOT, load_config
from .construir import (DatosInsuficientes, ResultadoMotor, candidatos_banco, decidir, ejecutar_motor, evaluar_activo, guardar_resultado,
                        mee_de)
from .data_sources import FuentesDatos
from .notify import enviar
from .semaforo import AZUL, EMOJI, GRAVEDAD, NEGRO, ROJO, VERDE, titulares_puntuados
from .sentimiento import crear_puntuador
from .state import Estado, ErrorEstado, universo_permitido

RESPALDO_FUENTE = {
    "precios": "uso los últimos datos guardados; si no hay, no habrá alertas de color hasta que vuelvan. Si ves un movimiento fuerte, mira trii a mano.",
    "noticias": "pruebo otras fuentes de noticias (Yahoo y Google). Si también fallan, sospecho de una mala noticia cuando hay mucho volumen y la acción abre muy abajo.",
    "noticias_bvc": "sigo intentando cada pocos segundos y te aviso cuando vuelvan. Mientras tanto no recibirás avisos de noticias de la BVC; el semáforo de tu cartera sigue funcionando.",
}


@dataclass
class Contexto:
    cfg: dict[str, Any]
    f: FuentesDatos
    puntuador: Callable[[str], float]
    motor: str
    ahora: dt.datetime
    dry: bool = False
    estado_path: Path | None = None            # en dry-run apunta a una copia temporal: nunca se toca el state.json real
    salida: Callable[[str], None] = print
    enviados: list[str] = field(default_factory=list)


def crear_contexto(dry: bool = False, utc_now: dt.datetime | None = None, cfg: dict[str, Any] | None = None) -> Contexto:
    load_dotenv(ROOT / ".env")
    cfg = cfg or load_config()
    f = FuentesDatos(cfg)
    puntuador, motor = crear_puntuador(cfg)
    ruta = None
    if dry:
        tmp = Path(tempfile.mkdtemp()) / "state_dry.json"
        real = ROOT / cfg["estado"]["archivo"]
        if real.exists():
            shutil.copy2(real, tmp)
        ruta = tmp
    return Contexto(cfg, f, puntuador, motor, C.ahora_bogota(cfg, utc_now), dry, ruta)


def leer_estado(ctx: Contexto) -> Estado:
    return Estado(ctx.cfg, ctx.estado_path)


def transaccion(ctx: Contexto):
    return Estado.transaccion(ctx.cfg, ctx.estado_path)


# ------------------------------------------------------------------ envío con respaldo
def enviar_o_encolar(ctx: Contexto, texto: str, botones: list[list[tuple[str, str]]] | None = None) -> bool:
    """Envía por Telegram (con botones opcionales); si falla, guarda el mensaje para reintentarlo en la próxima corrida. En dry-run sólo lo imprime."""
    ctx.enviados.append(texto)
    if ctx.dry:
        ctx.salida("─" * 60 + "\n" + F.plano(texto) + ("\n[" + "] [".join(t for fila in botones for t, _ in fila) + "]" if botones else "") + "\n" + "─" * 60)
        return True
    _a_suscriptores(ctx, texto, botones)
    if (enviar(texto, ctx.cfg, botones=botones) if botones else enviar(texto, ctx.cfg)):
        return True
    with transaccion(ctx) as e:
        e.agregar_pendiente(texto, ctx.ahora)
    return False


def _a_suscriptores(ctx: Contexto, texto: str, botones: list[list[tuple[str, str]]] | None = None) -> None:
    """Copia del aviso automático a los chats que lo pidieron con /avisos (otro celular, un grupo…). Si uno falla no afecta a los demás ni al chat principal."""
    import os
    try:
        chats = [c for c in leer_estado(ctx).d.get("suscriptores", []) if str(c) != (os.environ.get("TELEGRAM_CHAT_ID") or "").strip()]
    except Exception:                                                                  # noqa: BLE001
        return
    for c in chats:
        try:
            env = {**os.environ, "TELEGRAM_CHAT_ID": str(c)}
            enviar(texto, ctx.cfg, env=env, botones=botones) if botones else enviar(texto, ctx.cfg, env=env)
        except Exception:                                                              # noqa: BLE001
            continue


def suscribir_grupo(ctx: Contexto, chat: int) -> bool:
    """Suscribe un grupo a los avisos automáticos si no lo estaba ni pidió silencio. Devuelve True si quedó suscrito ahora."""
    d = leer_estado(ctx).d
    if chat in d.get("suscriptores", []) or chat in d.get("silenciados", []):
        return False
    with transaccion(ctx) as e:
        e.d["suscriptores"] = ([c for c in e.d.get("suscriptores", []) if c != chat] + [chat])[-20:]
    return True


def suscribir(ctx: Contexto, chat: int, activar: bool) -> str:
    """/avisos y /silencio: este chat empieza (o deja) de recibir los avisos automáticos. El chat principal (TELEGRAM_CHAT_ID) los recibe siempre."""
    import os
    if str(chat) == (os.environ.get("TELEGRAM_CHAT_ID") or "").strip():
        return "🔔 Este es el chat principal: ya recibe todos los avisos automáticos (noticias, macro, cambios sugeridos, resumen de la mañana y radar)."
    with transaccion(ctx) as e:
        lista = [c for c in e.d.setdefault("suscriptores", []) if c != chat]
        if activar:
            lista.append(chat)
        e.d["suscriptores"] = lista[-20:]
        e.d["silenciados"] = ([c for c in e.d.get("silenciados", []) if c != chat] + ([] if activar else [chat]))[-50:]      # un grupo en silencio no se vuelve a suscribir solo
    if activar:
        return ("🔔 " + F.b("Listo: este chat también recibirá los avisos automáticos") + "\nNoticias de la BVC con su impacto en %, movimientos macro con las acciones "
                "beneficiadas y afectadas, cambios sugeridos, el resumen de las 8:10 y el radar de las 19:30.\nPara dejar de recibirlos: /silencio")
    return "🔕 Listo: este chat ya no recibirá avisos automáticos. Para volver a activarlos: /avisos"


def enviar_pendientes(ctx: Contexto) -> int:
    """Reintenta las alertas que no se pudieron enviar antes. Devuelve cuántas se entregaron."""
    if ctx.dry:
        return 0
    with transaccion(ctx) as e:
        pend = e.tomar_pendientes()
    if not pend:
        return 0
    falladas, entregadas = [], 0
    for p in pend:
        if enviar(f"⏱ <i>(Aviso atrasado: no se pudo enviar el {p['cuando'][5:16].replace('T', ' a las ')})</i>\n{p['texto']}", ctx.cfg):
            entregadas += 1
        else:
            falladas.append(p)
    if falladas:
        with transaccion(ctx) as e:
            for p in falladas:
                e.agregar_pendiente(p["texto"], dt.datetime.fromisoformat(p["cuando"]))
    return entregadas


# ------------------------------------------------------------------ monitor (cada 15 min en horario de bolsa)
def correr_monitor(ctx: Contexto, forzar: bool = False, evaluar: Callable = evaluar_activo, banco: Callable = candidatos_banco,
                   mee: Callable = mee_de) -> list[str]:
    """Calcula el semáforo de mi activo (y del banco) y avisa SÓLO si cambia el color. Devuelve los mensajes generados."""
    cfg, ahora, f = ctx.cfg, ctx.ahora, ctx.f
    mon = cfg["monitor"]
    enviar_pendientes(ctx)
    if not forzar and not C.mercado_abierto(ahora, cfg):
        ctx.salida(f"[{ahora:%H:%M}] mercado cerrado para trii: nada que hacer (usa --forzar para probar).")
        return []
    est = leer_estado(ctx)
    mensajes: list[str] = []
    salud: list[tuple[str, bool]] = []
    try:
        res, ent, _ = evaluar(f, est.activo, ahora, cfg, est, ctx.puntuador, ctx.motor)
        salud.append(("precios", True))
    except (DatosInsuficientes, Exception) as ex:                                      # noqa: BLE001 — cualquier fallo de datos se cuenta, no se oculta
        salud.append(("precios", False))
        ctx.salida(f"[{ahora:%H:%M}] sin datos de precios de {est.activo}: {type(ex).__name__}: {ex}")
        res = ent = None
    if res is not None:
        salud.append(("noticias", not res.sin_noticias))
    extras: dict[str, Any] = {}                                                        # el resto de tu cartera: sólo se vigila el color de cada acción
    for tk in sorted(est.tenidos() - {est.activo}):
        try:
            extras[tk] = evaluar(f, tk, ahora, cfg, est, ctx.puntuador, ctx.motor)[0]
        except Exception as ex:                                                        # noqa: BLE001 — una acción sin datos no tumba a las demás
            ctx.salida(f"[{ahora:%H:%M}] sin datos de {tk}: {type(ex).__name__}")
    cands = []
    if res is not None and mon["alertar_banco"]:
        try:
            cands = banco(f, cfg, ahora, est.activo, ent.cierres, ctx.puntuador, ctx.motor)
        except Exception as ex:                                                        # noqa: BLE001 — el banco es secundario: no tumba el monitor
            ctx.salida(f"[{ahora:%H:%M}] no se pudo calcular el banco: {type(ex).__name__}")

    with transaccion(ctx) as e:
        for nombre, ok in salud:
            ev = e.registrar_fuente(nombre, ok, ahora, mon["fallos_para_avisar"] if nombre == "precios" else 2, mon["aviso_salud_cada_min"])
            if ev:
                mensajes.append(F.msg_salud(nombre, ev, ahora, RESPALDO_FUENTE[nombre], dict(f.proveedores_estado) if nombre == "noticias" else None))
        dec_color = None
        if res is not None:
            prev = e.color_previo(res.ticker)
            guardar_resultado(e, res, ahora, avisado=False)
            cambia = (prev is None and res.color not in (VERDE, AZUL)) or (prev is not None and prev != res.color)
            if cambia and (prev is None or GRAVEDAD[res.color] > GRAVEDAD[prev] or e.puede_alertar(res.ticker, ahora, mon["enfriamiento_min"])):
                m, dec = None, None
                if GRAVEDAD[res.color] >= GRAVEDAD["AMARILLO"]:                       # sólo vale la pena calcular la decisión si hay algo que decidir
                    r = decidir(f, e, res, ent, cands, ahora, cfg)
                    m, dec = r.mee, r.decision
                    dec_color = dec
                mensajes.append(F.msg_alerta_color(res, prev, ahora, m, dec, ctx.f.traducir, C.hora_orden_manana(C.proxima_sesion(ahora, cfg) or ahora.date(), cfg),
                                                   est.activo))
                guardar_resultado(e, res, ahora, avisado=True)
            mensajes += _alertas_banco(e, cands, ahora, cfg)
        for tk, rx in extras.items():                                                      # otras acciones de tu cartera: aviso de color (sin decisión: la regla trabaja con la principal)
            prev_x = e.color_previo(tk)
            guardar_resultado(e, rx, ahora, avisado=False)
            cambia_x = (prev_x is None and rx.color not in (VERDE, AZUL)) or (prev_x is not None and prev_x != rx.color)
            if cambia_x and (prev_x is None or GRAVEDAD[rx.color] > GRAVEDAD[prev_x] or e.puede_alertar(tk, ahora, mon["enfriamiento_min"])):
                mensajes.append(F.msg_alerta_color(rx, prev_x, ahora, None, None, f.traducir, "", tk))
                guardar_resultado(e, rx, ahora, avisado=True)
        if res is not None:
            if mon.get('eventos', {}).get('activo'):
                mensajes += _eventos(ctx, e, res, ent, cands, ahora, dec_color)
    for m in mensajes:
        enviar_o_encolar(ctx, m)
    if not mensajes:
        ctx.salida(f"[{ahora:%H:%M}] {est.activo}: " + (f"{res.color} sin cambios" if res else "sin datos") + " — sin alertas.")
    return mensajes


def _alertas_banco(e: Estado, cands: list, ahora: dt.datetime, cfg: dict[str, Any]) -> list[str]:
    """Avisa si un candidato del banco (el top 7 anterior o actual) entra o sale de ROJO/NEGRO. Guarda los colores de todos los candidatos."""
    if not cands:
        return []
    from .relevo import construir_banco
    mios = e.tenidos()                                                                 # lo que ya tienes no es un "relevo": se vigila aparte (extras)
    top_actual = [x.c.ticker for x in construir_banco(cands, e.activo, cfg, set(mios))]
    vigilados = set(top_actual) | set(e.d.get("banco_top", []))
    graves = (ROJO, NEGRO)
    msgs = []
    for c in cands:
        if c.ticker in mios:
            continue
        prev = e.color_previo(c.ticker)
        if c.ticker in vigilados and prev is not None and prev != c.color and (prev in graves or c.color in graves):
            msgs.append(F.msg_relevo_color(c.ticker, prev, c.color, graves))
        e.set_semaforo(c.ticker, c.color, ahora)
    e.d["banco_top"] = top_actual
    return msgs


def _eventos(ctx: Contexto, e: Estado, res: Any, ent: Any, cands: list, ahora: dt.datetime, dec_color: Any) -> list[str]:
    """Novedades que se avisan solas (ver eventos.py). Cada tipo falla por separado sin tumbar al monitor; tope de avisos por corrida."""
    from . import eventos as EV
    cfg, f = ctx.cfg, ctx.f
    conf = cfg["monitor"]["eventos"]
    ev = e.d.setdefault("eventos", {})
    hora = C.hora_orden_manana(C.proxima_sesion(ahora, cfg) or ahora.date(), cfg)
    msgs: list[str] = []

    def seguro(nombre: str, fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception as ex:                                                          # noqa: BLE001 — una novedad que falla no tumba el monitor
            ctx.salida(f"[{ahora:%H:%M}] no se pudo revisar «{nombre}»: {type(ex).__name__}")

    def decision() -> None:
        r = decidir(f, e, res, ent, cands, ahora, cfg)
        tipo = EV.cambio_de_decision(ev, r.decision, ahora, cfg)
        if tipo and dec_color is None:                                                   # si la alerta de color ya trajo la decisión, no se repite
            msgs.append(F.msg_decision_cambio(tipo, r.decision, e.activo, hora))

    def calendario() -> None:
        try:
            rep = f.reportes(e.activo)
        except Exception:                                                                # noqa: BLE001
            rep = None
        for a in EV.avisos_calendario(ev, e.activo, rep, ahora, cfg):
            msgs.append(F.msg_calendario(a))

    def noticias() -> None:
        n = f.noticias(e.activo, 2)
        if n:
            nuevas = EV.noticias_nuevas(ev, titulares_puntuados(n, cfg, ctx.puntuador, 60), ahora, cfg)
            if nuevas:
                msgs.append(F.msg_noticia_nueva(e.activo, nuevas, res, ahora, f.traducir))

    if conf["decision"]["activo"] and cands and e.rent.get("mia") is not None and e.rent.get("umbral") is not None:
        seguro("decisión", decision)
    if conf["calendario"]["activo"]:
        seguro("calendario", calendario)
    if conf["noticias"]["activo"]:
        seguro("noticias", noticias)
    if EV.alza_fuerte(ev, res.z, ahora, cfg):
        msgs.append(F.msg_alza(res, ahora))
    return msgs[: conf["max_por_corrida"]]


def alerta_base(ctx: Contexto) -> str | None:
    """Aviso (en el radar nocturno) si cambió la comparación de base: otra acción superó a la tuya, o dejó de superarla."""
    from . import eventos as EV
    from .seleccion import ranking_base, veredicto
    est = leer_estado(ctx)
    filas = ranking_base(ctx.f, ctx.cfg, ctx.ahora, est.activo)
    v, mejores = veredicto(filas, ctx.cfg)
    with transaccion(ctx) as e:
        tipo = EV.estado_base(e.d.setdefault("eventos", {}), v, mejores)
    return F.msg_base_cambio(tipo, filas, mejores, est.activo, ctx.cfg) if tipo else None


# ------------------------------------------------------------------ radar (19:30 de lunes a jueves)
def generar_radar(ctx: Contexto, forzar: bool = False) -> str:
    """Resumen nocturno completo. Es la medición oficial del cierre (la que decide NEGRO), por eso guarda el color y el episodio."""
    cfg, ahora = ctx.cfg, ctx.ahora
    est = leer_estado(ctx)
    r = ejecutar_motor(ctx.f, est, cfg, ahora, ctx.puntuador, ctx.motor)
    with transaccion(ctx) as e:
        guardar_resultado(e, r.res, ahora, avisado=False)
        e.registrar_fuente("precios", True, ahora, 3, 60)
        e.registrar_fuente("noticias", not r.res.sin_noticias, ahora, 2, 60)
    cats = catalizadores(ctx.f, cfg, ahora, tickers_con_reporte(cfg), cfg["radar"]["dias_catalizadores"])
    return armar_radar(est, r, cats, cfg, ahora, ctx.f.traducir)


def armar_radar(est: Estado, r: ResultadoMotor, cats: list[dict[str, Any]], cfg: dict[str, Any], ahora: dt.datetime,
                traductor: F.Traductor | None = None) -> str:
    """Resumen nocturno en 5 secciones numeradas, cada una con su explicación en palabras sencillas."""
    c = r.corte
    sem = C.semana_concurso(ahora.date(), cfg)
    noche = ahora.date().isoformat() in cfg["radar"]["noches_decision"]
    hora = C.hora_orden_manana(C.proxima_sesion(ahora, cfg) or ahora.date(), cfg)
    L = [f"🌙 {F.b('RADAR DE LA NOCHE')}  {F.it(f'{F.fecha(ahora)} {ahora:%H:%M} (hora Bogotá)')}"]
    if c:
        top = f"pasa el top {c['top_pct']}%" if c.get("top_pct") else "final: gana el #1"
        L.append(f"📍 {'Semana ' + str(sem) + ' de 5' if sem else 'Antes del concurso'} · próximo corte: {F.fecha(c['fecha'])} ({top}) · faltan {C.sesiones_restantes(ahora, cfg)} días de bolsa")
        if C.corte_manana(ahora, cfg):
            L.append(f"⚠️ {F.b('¡El corte es mañana!')} Hoy un ROJO se trata como NEGRO.")
    L.append("🧭 Hoy es noche de decisión: mira la sección 2 y decide si cambias." if noche
             else "🧭 Hoy no hay decisión programada; igual te muestro qué diría la regla.")
    L += ["", F.b("1️⃣ Tu acción") + f" — {EMOJI[r.res.color]} {F.esc(r.res.ticker)}: {F.NOMBRE_COLOR[r.res.color]}", *F.cuerpo_semaforo(r.res, traductor)]
    L += ["", F.b("2️⃣ ¿Te conviene cambiar?"), *([] if r.decision.codigo == "sin_ranking" else [F.linea_ranking(est.rent)]),
          *F.bloque_decision(r.decision, r.activo, hora)[1:]]
    L += ["", F.b("3️⃣ Relevos posibles"), F.msg_banco(r.banco, r.activo, r.mee, r.excluidos, top=5, titulo=False).split("\n", 1)[-1]
          if r.banco else F.msg_banco(r.banco, r.activo, r.mee)]
    destacados = {r.activo} | {e.c.ticker for e in r.banco[:5]}
    L += ["", F.b("4️⃣ Qué viene"), F.msg_catalizadores(cats, cfg["radar"]["dias_catalizadores"], max_items=6, destacados=destacados).split("\n", 1)[-1]
          if cats else F.msg_catalizadores(cats, cfg["radar"]["dias_catalizadores"])]
    L += ["", F.b("5️⃣ Tus tareas de mañana"), F.recordatorio_actividad(est, cfg, ahora),
          f"🔁 Cambios de acción usados: {est.cambios_usados()} de {cfg['estado']['max_cambios']}."]
    ruido = ("sin cobertura", "ticks erróneos", "Traducción", "intradía")                 # fallos menores de acciones candidatas: no merecen alarma
    relevantes = [x for x in r.avisos if not any(m in x for m in ruido)]
    if relevantes:
        L += ["", f"⚠️ {F.it('Algunas fuentes de datos fallaron hoy; el sistema usó su plan B. Más datos: /detalle')}"]
    L.append(F.it("¿Dudas con los colores? /ayuda · ¿Quieres el detalle completo? /informe (archivo HTML)"))
    return "\n".join(L)


# ------------------------------------------------------------------ respuestas del bot
def purgar_cache(ctx: Contexto) -> int:
    n = 0
    for p in ctx.f.cache.dir.glob("*"):
        if p.name != "av_contador.json":
            p.unlink(missing_ok=True)
            n += 1
    return n


def resp_estado(ctx: Contexto) -> str:
    est = leer_estado(ctx)
    res = mee = None
    try:
        res, ent, _ = evaluar_activo(ctx.f, est.activo, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor)
        mee = mee_de(ctx.f, est.activo, res.sigma20, ctx.ahora, ctx.cfg)[0]
    except Exception:                                                                  # noqa: BLE001 — el estado se muestra aunque fallen los datos
        pass
    sal = {k: bool(v.get("fallos", 0) == 0) for k, v in est.d["salud"].items()}
    txt = F.msg_estado(est, ctx.cfg, ctx.ahora, res, mee, sal)
    if est.d["cartera"]:
        try:
            txt += "\n\n" + _cartera_txt(ctx, est)
        except Exception:                                                              # noqa: BLE001 — el estado se muestra aunque falle la valoración
            txt += "\n\n💼 No pude valorar tu cartera ahora; /cartera lo intenta de nuevo."
    if res is None:
        txt += "\n⚠️ No pude calcular el semáforo ahora (datos no disponibles). Mira trii a mano si ves un movimiento fuerte."
    if est.d["alertas_pendientes"]:
        txt += f"\n⏱ Tengo {len(est.d['alertas_pendientes'])} avisos sin enviar; los reintento en la próxima revisión."
    return txt


def resp_semaforo(ctx: Contexto, ticker: str | None = None) -> str:
    """Sin ticker: el semáforo de TODAS las acciones de tu cartera (la principal primero). Con ticker: el de esa acción."""
    est = leer_estado(ctx)
    if ticker:
        lista = [resolver(ctx, ticker, est.tenidos())]
    else:
        otras = sorted(est.tenidos() - {est.activo})
        lista = [est.activo] + otras
    partes = []
    for t in lista:
        if t not in universo_permitido(ctx.cfg) and t != est.activo:
            partes.append(f"❌ {F.esc(t)} no está permitida en el concurso (o está en la lista negra).")
            continue
        try:
            res, _, _ = evaluar_activo(ctx.f, t, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor, con_base_noticias=True)
            partes.append(F.msg_semaforo(res, ctx.ahora, ctx.f.traducir))
        except Exception as ex:                                                          # noqa: BLE001 — una acción sin datos no impide ver las demás
            partes.append(f"⚠️ {F.b(F.esc(t))}: no pude calcular su semáforo ahora ({F.esc(type(ex).__name__)}). Prueba /actualizar.")
    avisos = avisos_liquidez(ctx, [t for t in lista if t in universo_permitido(ctx.cfg)])
    avisos = lineas_fuera_de_horario(ctx, lista) + avisos
    if any(t in ctx.cfg["universe"]["mgc"] for t in lista):
        avisos.append(F.it("El semáforo de las acciones de EE. UU. mira su precio en Nueva York (en dólares). En trii las ves en pesos y se negocian aparte: "
                           "su precio allá también cambia con el dólar y con lo poco que se negocian."))
    return "\n\n➖➖➖\n\n".join(partes) + ("\n\n" + "\n".join(avisos) if avisos else "")


def resp_banco(ctx: Contexto) -> str:
    est = leer_estado(ctx)
    r = ejecutar_motor(ctx.f, est, ctx.cfg, ctx.ahora, ctx.puntuador, ctx.motor)
    hora = C.hora_orden_manana(C.proxima_sesion(ctx.ahora, ctx.cfg) or ctx.ahora.date(), ctx.cfg)
    return "\n".join([F.msg_banco(r.banco, r.activo, r.mee, r.excluidos), "", *F.bloque_decision(r.decision, r.activo, hora)])


def resp_detalle(ctx: Contexto, ticker: str | None = None) -> str:
    """Los números técnicos del semáforo de tu acción principal, o de la que indiques (/detalle META)."""
    est = leer_estado(ctx)
    t = resolver(ctx, ticker, est.tenidos()) if ticker else est.activo
    if t not in universo_permitido(ctx.cfg) and t != est.activo:
        return f"❌ {F.esc(t)} no está permitida en el concurso (o está en la lista negra)."
    res, ent, _ = evaluar_activo(ctx.f, t, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor)
    mee, _, iv = mee_de(ctx.f, t, res.sigma20, ctx.ahora, ctx.cfg)
    return F.msg_detalle(res, mee, iv, ctx.cfg)


def solo_bvc(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("modo", {}).get("solo_bvc"))


def resolver(ctx: Contexto, texto: str, tenidos: set[str] | None = None) -> str:
    """'argos', 'nuco', 'Bancolombia' → el ticker del sistema. Si no lo reconoce, devuelve el texto en mayúsculas (el aviso de 'no permitida' lo da quien llama)."""
    from .entender import buscar_ticker
    return buscar_ticker(texto, ctx.cfg, tenidos)[0] or texto.strip().upper()


def senales_recientes(horas: float = 24, ahora: dt.datetime | None = None) -> list[dict[str, Any]]:
    """Noticias de alto impacto que el vigía avisó en las últimas `horas`."""
    from . import noticias_bvc as N
    ahora = ahora or dt.datetime.now(dt.timezone.utc)
    lim = (ahora.astimezone(dt.timezone.utc) - dt.timedelta(hours=horas)).isoformat(timespec="minutes")
    return [x for x in N.Memoria().d["senales"] if x["ts"] >= lim]


def resp_noticias(ctx: Contexto, ticker: str | None = None) -> str:
    """Sin ticker: las noticias fuertes de la BVC de las últimas 24 horas (las que avisó el vigía). Con ticker: los titulares de esa acción."""
    if not ticker:
        return F.msg_noticias_bvc_recientes(senales_recientes(24, ctx.ahora), None, ctx.ahora)
    est = leer_estado(ctx)
    t = resolver(ctx, ticker, est.tenidos())
    if t not in universo_permitido(ctx.cfg):
        return f"❌ {F.esc(t)} no está permitida en el concurso (o está en la lista negra)."
    if solo_bvc(ctx.cfg) and t not in ctx.cfg["universe"]["local"] and t not in est.tenidos():
        return f"🇨🇴 Ahora trabajo sólo con acciones de la BVC. {F.esc(t)} es de EE. UU. y no la tienes en cartera. Prueba /noticias ECOPETROL."
    n = ctx.f.noticias(t, 3)
    return F.msg_noticias(t, n, titulares_puntuados(n or [], ctx.cfg, ctx.puntuador), ctx.cfg["bot"]["titulares_max"], ctx.f.traducir)


def buscar_noticias(ctx: Contexto, horas: float = 12, lector: Any = None) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    """Consulta AHORA todas las fuentes (la Superfinanciera primero) y devuelve las noticias de las últimas `horas` de empresas de la BVC con liquidez buena
    en trii (o que tengas), ya analizadas: tipo, sentido, impacto estimado en %. No toca la memoria del vigía ni manda avisos: es para mirar."""
    from . import noticias_bvc as N
    from .analisis_noticia import analizar
    lector = lector or N.Lector(ctx.cfg)
    items, salud = lector.todo(ctx.ahora, completo=True)
    est = leer_estado(ctx)
    emisores = N.emisores_cfg(ctx.cfg)
    estudio = N.cargar_estudio()
    ahora_utc = ctx.ahora.astimezone(dt.timezone.utc)
    limite = ahora_utc - dt.timedelta(hours=horas)
    liquidas = acciones_liquidas(ctx, {t for e in emisores for t in e.tickers})
    solo_frases = {**ctx.cfg, "analisis_noticias": {**ctx.cfg.get("analisis_noticias", {}), "leer_articulo": False, "usar_modelo": False}}      # rápido: sin descargar artículos
    filas: list[dict[str, Any]] = []
    vistos: dict[str, list[list[str]]] = {}
    for it in sorted(items, key=lambda i: (i.origen != "sfc", -i.ts.timestamp())):                      # lo oficial primero: si un medio repite el hecho, queda el oficial
        if it.ts < limite or it.ts > ahora_utc + dt.timedelta(minutes=10):
            continue
        for e in N.emisores_de(it, emisores):
            ticker = next((t for t in e.tickers if t in est.tenidos()), None) or next((t for t in e.tickers if t in liquidas), None)
            if ticker is None:
                continue                                                               # sin liquidez buena en trii: no se muestra
            if N.es_repetida(it.titulo, vistos.get(e.nombre, []), ctx.cfg["noticias_bvc"]["parecido_repetida"]):
                continue
            vistos.setdefault(e.nombre, []).append(sorted(N.palabras_clave(it.titulo)))
            c = N.clasificar(N.texto_para_clasificar(it), ctx.cfg)
            cat, sentido = (c.cat, c.sentido) if c else ("otra", 0)
            if sentido == 0:
                sentido = analizar(e.nombre, it.titulo, it.resumen, "", solo_frases, None, {})["sentido"]
            base = N.impacto_estimado(cat, 1, estudio, ctx.cfg)
            filas.append(dict(ts=it.ts, emisor=e.nombre, ticker=ticker, titulo=it.titulo, fuente=it.fuente, oficial=it.origen == "sfc", cat=cat, sentido=sentido,
                              impacto=(sentido * base if sentido else base) if base is not None else None, tengo=ticker in est.tenidos()))
    return sorted(filas, key=lambda x: -x["ts"].timestamp()), salud


def resp_nuevas(ctx: Contexto) -> str:
    """/nuevas: actualiza las noticias ahora mismo y muestra lo último de cada empresa de la BVC."""
    filas, salud = buscar_noticias(ctx)
    return F.msg_noticias_actualizadas(filas, salud, ctx.ahora, 12)


def resp_catalizadores(ctx: Contexto) -> str:
    cats = catalizadores(ctx.f, ctx.cfg, ctx.ahora, tickers_con_reporte(ctx.cfg), ctx.cfg["radar"]["dias_catalizadores"])
    return F.msg_catalizadores(cats, ctx.cfg["radar"]["dias_catalizadores"])


def resp_actualizar(ctx: Contexto) -> str:
    n = purgar_cache(ctx)
    est = leer_estado(ctx)
    r = ejecutar_motor(ctx.f, est, ctx.cfg, ctx.ahora, ctx.puntuador, ctx.motor)
    with transaccion(ctx) as e:
        guardar_resultado(e, r.res, ctx.ahora, avisado=False)
    hora = C.hora_orden_manana(C.proxima_sesion(ctx.ahora, ctx.cfg) or ctx.ahora.date(), ctx.cfg)
    return "\n".join([f"🔄 {F.b('Datos actualizados')} {F.it(f'(borré {n} datos guardados y volví a consultar todo)')}", "",
                      F.msg_estado(est, ctx.cfg, ctx.ahora, r.res, r.mee), "", *F.bloque_decision(r.decision, r.activo, hora), "",
                      F.msg_banco(r.banco, r.activo, r.mee, r.excluidos, top=5, titulo=False)])


def resp_base(ctx: Contexto) -> str:
    from .seleccion import ranking_base, veredicto
    est = leer_estado(ctx)
    filas = ranking_base(ctx.f, ctx.cfg, ctx.ahora, est.activo)
    v, mejores = veredicto(filas, ctx.cfg)
    from . import liquidez as LQ
    try:                                                                               # la base también tiene que poder negociarse en trii: si no, no se dice "mantén"
        base_liquida = LQ.medir(ctx.f, est.activo, ctx.cfg)["nivel"] == LQ.BUENA
    except Exception:                                                                  # noqa: BLE001 — sin dato no se aprueba
        base_liquida = False
    return F.msg_base(filas, v, mejores, est.activo, ctx.cfg, base_liquida=base_liquida)


def generar_informe(ctx: Contexto) -> tuple[str, str]:
    """(mensaje corto en HTML de Telegram, informe HTML completo). Sólo se llama cuando alguien lo pide en el bot: los avisos automáticos no lo adjuntan."""
    from .informe_html import armar_informe
    est = leer_estado(ctx)
    r = ejecutar_motor(ctx.f, est, ctx.cfg, ctx.ahora, ctx.puntuador, ctx.motor)
    try:
        cats = catalizadores(ctx.f, ctx.cfg, ctx.ahora, tickers_con_reporte(ctx.cfg), ctx.cfg["radar"]["dias_catalizadores"])
    except Exception:                                                                  # noqa: BLE001 — el informe sale aunque falle el calendario
        cats = []
    try:
        n = ctx.f.noticias(est.activo, 3)
        punt = titulares_puntuados(n, ctx.cfg, ctx.puntuador, 20) if n is not None else None
    except Exception:                                                                  # noqa: BLE001
        punt = None
    try:
        from .seleccion import ranking_base
        brank = ranking_base(ctx.f, ctx.cfg, ctx.ahora, est.activo)
    except Exception:                                                                  # noqa: BLE001
        brank = None
    hora = C.hora_orden_manana(C.proxima_sesion(ctx.ahora, ctx.cfg) or ctx.ahora.date(), ctx.cfg)
    corto = "\n".join([F.msg_semaforo(r.res, ctx.ahora, ctx.f.traducir), "", *F.bloque_decision(r.decision, r.activo, hora), "",
                       F.it("📎 Te mando también el informe completo (ábrelo con el navegador).")])
    return corto, armar_informe(est, r, cats, punt, ctx.cfg, ctx.ahora, ctx.f.traducir, universo_permitido(ctx.cfg), brank)


def informe_html(ctx: Contexto) -> str:
    return generar_informe(ctx)[1]


def numero(s: str) -> float:
    """Acepta '6,5', '6.5', '+6,5', '-3', '6,5%'."""
    t = s.strip().replace("%", "").replace(" ", "")
    if "," in t and "." in t:                                                         # 1.234,5 → 1234.5
        t = t.replace(".", "").replace(",", ".")
    else:
        t = t.replace(",", ".")
    return float(t)


def monto(s: str) -> float:
    """Acepta '97000000', '97.000.000', '97M', '97,5M', '97mm'."""
    t = s.strip().lower().replace("$", "").replace(" ", "")
    mult = 1.0
    for suf, m in (("mm", 1e6), ("m", 1e6), ("k", 1e3)):
        if t.endswith(suf):
            t, mult = t[: -len(suf)], m
            break
    if mult == 1.0:
        t = t.replace(".", "").replace(",", "")
        return float(t)
    return numero(t) * mult


def guardar_rank(ctx: Contexto, mia: float | None, objetivo: float | None, segundo: float | None = None) -> str:
    """Guarda tu rentabilidad y/o la del corte. Lo que no digas se toma de lo último guardado ("voy 3,5" no obliga a repetir el corte)."""
    rm = ctx.cfg["regla_maestra"]
    try:
        with transaccion(ctx) as e:
            mia = e.rent.get("mia") if mia is None else mia
            objetivo = e.rent.get("umbral") if objetivo is None else objetivo
            if mia is None or objetivo is None:
                falta = "tu rentabilidad" if mia is None else "la rentabilidad del corte"
                ejemplo = "voy 3,5" if mia is None else "el corte está en 8"
                return (f"{F.b('Cómo usar /rank')}\nMe falta {falta}. Escríbela así: {F.b(ejemplo)} — o las dos juntas: {F.b('/rank 6,5 11')} "
                        "(\"voy +6,5% y para pasar el corte hay que tener +11%\").\nEn la semana 5 el segundo número es la rentabilidad del #1.")
            e.actualizar_rank(mia, objetivo, ctx.ahora, segundo=segundo)
            g = objetivo - mia + rm["margen_g_pp"]
    except (ValueError, ErrorEstado) as ex:
        return f"❌ No pude guardar el ranking: {F.esc(ex)}\nEjemplo: {F.b('/rank 6,5 11')}"
    mult = (g + rm["costo_cambio_pp"]) / g if g > 0 else None
    return F.msg_rank_guardado(mia, objetivo, g, mult, C.semana_final(ctx.ahora, ctx.cfg) and segundo is None and mia >= objetivo)


def resp_rank(ctx: Contexto, args: list[str]) -> str:
    """/rank 6,5 11 (mi rentabilidad y la del corte). Con un solo número actualiza tu rentabilidad y conserva el corte que ya habías dado."""
    try:
        vals = [numero(a) for a in args[:3]]
    except ValueError as ex:
        return f"❌ No pude guardar el ranking: {F.esc(ex)}\nEjemplo: {F.b('/rank 6,5 11')}"
    return guardar_rank(ctx, vals[0] if vals else None, vals[1] if len(vals) > 1 else None, vals[2] if len(vals) > 2 else None)


def resp_pos(ctx: Contexto, args: list[str]) -> str:
    if len(args) < 3:
        return (f"{F.b('Cómo usar /pos')}\nCuando cambies de acción, dime cuál compraste, cuánta plata y a qué precio. "
                f"Ejemplo: {F.b('/pos META 95M 720,5')} (META, $95 millones, a 720,5).")
    try:
        t, m, p = args[0].upper(), monto(args[1]), numero(args[2])
        with transaccion(ctx) as e:
            res = e.registrar_cambio(t, m, p, ctx.ahora)
            return (f"✅ {F.b(F.esc(res))}\n💼 Ahora tienes {F.b(t)}: {F.cop(m)} comprados a {F.n(p, 2)}.\n"
                    f"🔁 Cambios de acción que te quedan: {e.cambios_restantes()}.")
    except (ValueError, ErrorEstado) as ex:
        return f"❌ {F.esc(ex)}\nEjemplo: {F.b('/pos META 95M 720,5')}"


def avisos_liquidez(ctx: Contexto, tickers: list[str], montos: dict[str, float] | None = None, todos: bool = False) -> list[str]:
    """Una línea por acción cuya liquidez EN TRII no es buena (o por todas, con `todos`). Se revisa antes de dar cualquier resultado: una acción que casi no
    se negocia en trii cuesta cara de comprar y de vender, por muy líquida que sea en Nueva York."""
    from . import liquidez as LQ
    out = []
    for t in tickers:
        l = LQ.medir(ctx.f, t, ctx.cfg)
        if todos or l["nivel"] in (LQ.JUSTA, LQ.MALA):
            out.append(F.esc(LQ.frase(l, (montos or {}).get(t), ctx.cfg)))
    return out


def lineas_fuera_de_horario(ctx: Contexto, tickers: list[str]) -> list[str]:
    """Para tus acciones de EE. UU.: cómo van ANTES de abrir o DESPUÉS de cerrar en Nueva York (frente al último cierre). Es lo que anticipa cómo abrirán."""
    from . import sesion as SE
    from .data_loader import yahoo_symbol
    out = []
    for t in tickers:
        if t not in ctx.cfg["universe"]["mgc"]:
            continue
        fx = SE.fuera_de_horario(ctx.f, yahoo_symbol(t, ctx.cfg), ctx.ahora)
        if fx:
            momento = "antes de abrir (pre-mercado)" if fx["estado"] == "pre" else "después del cierre"
            out.append(f"🌙 {F.b(t)} {momento}: {F.b(F.pct(fx['cambio'], 1, True))} frente al último cierre, a las {fx['hora']} (Nueva York, US$ {F.n(fx['precio'], 2)}).")
    return out


def _cartera_txt(ctx: Contexto, est: Estado, titulo: bool = True) -> str:
    from . import cartera as CA
    tasa = CA.trm(ctx.f, ctx.cfg)
    filas = CA.valorar(ctx.f, est.d["cartera"], ctx.cfg, tasa)
    colores = {t: est.color_previo(t) for t in est.tenidos()}
    montos = {x["ticker"]: x["valor_cop"] for x in filas if x["valor_cop"] == x["valor_cop"]}
    avisos = lineas_fuera_de_horario(ctx, [x["ticker"] for x in filas]) + avisos_liquidez(ctx, [x["ticker"] for x in filas], montos)
    return F.msg_cartera(filas, CA.rent_total(filas), tasa, colores, titulo) + ("\n" + "\n".join(avisos) if avisos else "")


def resp_cartera(ctx: Contexto) -> str:
    return _cartera_txt(ctx, leer_estado(ctx))


USO_COMPRA = ("{t}\nDímelo como te salga, por ejemplo:\n• {b1}\n• {b2}\n• {b3}\nSi no pones precio uso el de hoy. No importa si el precio va en pesos o en dólares, "
              "ni si pones el total en vez del precio: lo detecto y te muestro lo que entendí.")


def uso_compra() -> str:
    return USO_COMPRA.format(t=F.b("Cómo usar /compra"), b1=F.b("compré 300 argos a 21500"), b2=F.b("compré 20 millones de ecopetrol"), b3=F.b("/compra TSLA 55 380,5"))


def registrar_compra(ctx: Contexto, i: Any, confirmado: bool = False) -> str:
    """Registra una compra ya entendida (`entender.Intencion`): resuelve precio en pesos/dólares o total-en-vez-de-precio contra la cotización de hoy,
    y SIEMPRE devuelve lo que entendió. Si el precio queda muy lejos del de hoy y no cuadra con ninguna lectura, NO guarda y pregunta."""
    from . import cartera as CA
    from .entender import aclarar_compra
    t = i.ticker
    try:
        perm = universo_permitido(ctx.cfg)
        if t not in perm:
            Estado(ctx.cfg, ctx.estado_path)._validar_titulo(t)                        # da el mensaje exacto (lista negra / no permitida)
        try:
            q = ctx.f.cotizacion(t)
        except Exception:                                                              # noqa: BLE001
            q = None
        mon = CA.moneda(t, ctx.cfg)
        tasa = CA.trm(ctx.f, ctx.cfg)
        a = aclarar_compra(i, mon, q["precio"] if q else None, tasa)
        if not a["ok"]:
            if a["motivo"] == "sin_precio":
                return f"❌ No pude obtener el precio de hoy de {F.esc(t)}. Dime el precio de tu compra: {F.b('compré ' + f'{i.cantidad or 10:g} ' + t.lower() + ' a PRECIO')}"
            if a["motivo"] == "sin_trm":
                return "❌ No pude obtener la TRM ahora. Agrega el valor total en pesos al final, por ejemplo: /compra META 8 720,5 23M"
            return uso_compra()
        if a["monto_cop"] is None:
            return "❌ No pude obtener la TRM ahora. Agrega el valor total en pesos al final, por ejemplo: /compra META 8 720,5 23M"
        simb = "US$ " if mon == "USD" else "$ "
        ya = next((p for p in leer_estado(ctx).d["cartera"] if p["ticker"] == t and p.get("cantidad")), None)
        if ya and not confirmado and abs(ya["cantidad"] - a["cantidad"]) < 1e-9:           # misma cantidad que ya tienes: casi siempre es una repetición, no otra compra
            return (f"🤔 {F.b('Ya tienes ' + f'{ya['cantidad']:g} ' + t + ' registradas')}\n¿Compraste {a['cantidad']:g} MÁS (quedarías con {ya['cantidad'] + a['cantidad']:g})?\n"
                    f"• Si sí, repítelo y agrega la palabra {F.b('confirmo')}.\n• Si sólo querías decirme lo que tienes, escribe: {F.b('tengo ' + f'{a['cantidad']:g} ' + t.lower())}")
        if a["dudoso"] and not confirmado:
            return (f"🤔 {F.b('Antes de guardar, confirma el precio')}\nMe dijiste {simb}{F.n(a['precio'], 2)} por acción de {F.esc(t)}, pero hoy cotiza cerca de {simb}{F.n(q['precio'], 2)}.\n"
                    f"• Si está bien, repítelo y agrega la palabra {F.b('confirmo')}.\n• Si prefieres el precio de hoy, escribe: {F.b('compré ' + f'{a['cantidad']:g} ' + t.lower())}")
        with transaccion(ctx) as e:
            res = e.registrar_compra(t, a["cantidad"], a["precio"], ctx.ahora, a["monto_cop"])
            act = ctx.cfg["concurso"]["actividad"]
            ops, principal = e.ops_semana(ctx.ahora.date()), e.activo
        L = [f"✅ {F.b(F.esc(res))}", f"{F.esc(t)}: {a['cantidad']:g} acciones a {simb}{F.n(a['precio'], 2)} (≈ $ {F.n(a['monto_cop'] / 1e6, 1)} millones)."]
        if a["notas"]:
            L.append("🧠 " + F.it("Lo que entendí: " + "; ".join(a["notas"]) + "."))
        L += [f"🧾 Cuenta como operación de actividad: llevas {ops} de {act['ops_por_semana']} esta semana.",
              f"🎯 Acción principal que vigila el sistema: {F.b(principal)} (la de mayor monto)."]
        if a["aproximado"]:
            L.append("⚠️ " + F.it("Precio APROXIMADO (el de hoy): la rentabilidad de esta acción no es exacta."))
        if solo_bvc(ctx.cfg) and t not in ctx.cfg["universe"]["local"]:
            L.append("🇨🇴 " + F.it("No es de la BVC: la vigilo porque la tienes, pero mis recomendaciones de compra son sólo de la BVC."))
        L += avisos_liquidez(ctx, [t], {t: a["monto_cop"]}, todos=True)
        L.append(F.it("¿Quedó mal? Escribe: me equivoqué"))
        return "\n".join(L) + "\n\n" + _cartera_txt(ctx, leer_estado(ctx))
    except (ValueError, ErrorEstado) as ex:
        return f"❌ {F.esc(ex)}\nEjemplo: {F.b('compré 8 meta a 720,5')}"


def resp_compra(ctx: Contexto, args: list[str]) -> str:
    """/compra TICKER cantidad [precio] [monto_en_pesos]. También acepta nombres ('argos'), plata ('20M') y lo mismo que el texto libre."""
    if len(args) < 2:
        return uso_compra()
    from .entender import Intencion, entender
    libre = entender("compre " + " ".join(args), ctx.cfg, forzar="compra")
    try:
        cant = monto(args[1]) if args[1].lower().rstrip(".").endswith(("m", "mm", "k")) else numero(args[1])
    except ValueError:
        return f"❌ No entendí la cantidad «{F.esc(args[1])}».\nEjemplo: {F.b('/compra META 8 720,5')}"
    es_plata = args[1].lower().endswith(("m", "mm", "k")) or cant >= 1e6
    try:
        precio = None
        if len(args) > 2 and args[2].lower() != "confirmo":
            crudo = args[2].strip().lstrip("$")
            precio = libre.precio if re.fullmatch(r"\d{1,3}(\.\d{3})+(,\d+)?", crudo) and libre.precio else numero(crudo)      # "21.500" son veintiún mil quinientos
        extra = monto(args[3]) if len(args) > 3 and args[3].lower() != "confirmo" else None
    except ValueError as ex:
        return f"❌ {F.esc(ex)}\nEjemplo: {F.b('/compra META 8 720,5')}"
    i = Intencion("compra", ticker=libre.ticker or args[0].upper(), cantidad=None if es_plata else cant, precio=precio, monto=extra or (cant if es_plata else None))
    return registrar_compra(ctx, i, confirmado=any(a.lower() == "confirmo" for a in args))


def registrar_venta(ctx: Contexto, i: Any) -> str:
    try:
        with transaccion(ctx) as e:
            cant = i.cantidad
            if cant is None and i.fraccion is not None and i.fraccion < 1.0:
                p = next((x for x in e.d["cartera"] if x["ticker"] == i.ticker), None)
                cant = (p["cantidad"] * i.fraccion) if (p and p.get("cantidad")) else None
            res = e.registrar_venta(i.ticker, cant, ctx.ahora)
            a = ctx.cfg["concurso"]["actividad"]
            ops = e.ops_semana(ctx.ahora.date())
        return (f"✅ {F.b(F.esc(res))}\n🧾 Cuenta como operación de actividad: llevas {ops} de {a['ops_por_semana']} esta semana.\n"
                + F.it("¿Quedó mal? Escribe: me equivoqué") + "\n\n" + _cartera_txt(ctx, leer_estado(ctx)))
    except (ValueError, ErrorEstado) as ex:
        return f"❌ {F.esc(ex)}"


def resp_venta(ctx: Contexto, args: list[str]) -> str:
    """/venta TICKER [cantidad]: registra que vendiste todo (o una parte) de una acción de tu cartera."""
    if not args:
        return f"{F.b('Cómo usar /venta')}\nDímelo como te salga: {F.b('vendí argos')} (todo), {F.b('vendí 20 tesla')} o {F.b('vendí la mitad de meta')}."
    from .entender import Intencion
    try:
        cant = numero(args[1]) if len(args) > 1 else None
    except ValueError as ex:
        return f"❌ {F.esc(ex)}"
    return registrar_venta(ctx, Intencion("venta", ticker=resolver(ctx, args[0], leer_estado(ctx).tenidos()), cantidad=cant, fraccion=None if cant else 1.0))


def registrar_tenencias(ctx: Contexto, partes: list[Any]) -> str:
    """"Tengo 8 meta que valen 19.120.000, tengo 300 argos…": deja cada acción con esa cantidad (no suma ni cuenta como operación). El valor que digas se toma
    como referencia de precio; si no lo dices, el precio de hoy. Lo que no menciones no se toca."""
    from . import cartera as CA
    tasa = CA.trm(ctx.f, ctx.cfg)
    lineas, errores = [], []
    try:
        with transaccion(ctx) as e:
            e.guardar_punto("el ajuste de tu cartera", ctx.ahora)
            for p in partes:
                try:
                    mon = CA.moneda(p.ticker, ctx.cfg)
                    conv = (tasa if mon == "USD" else 1.0)
                    if p.monto and conv:
                        precio, mc = p.monto / p.cantidad / conv, p.monto
                    else:
                        previa = next((x for x in e.d["cartera"] if x["ticker"] == p.ticker), None)
                        q = None
                        try:
                            q = ctx.f.cotizacion(p.ticker)
                        except Exception:                                              # noqa: BLE001
                            pass
                        precio = previa["precio"] if previa else (q["precio"] if q else None)
                        if precio is None:
                            errores.append(f"{p.ticker}: no pude obtener su precio; dime cuánto valen en total.")
                            continue
                        mc = CA.monto_cop(p.ticker, p.cantidad, precio, tasa, ctx.cfg) or (previa or {}).get("monto_cop")
                    lineas.append("• " + F.esc(e.fijar_posicion(p.ticker, p.cantidad, precio, ctx.ahora, mc, punto=False)))
                except ErrorEstado as ex:
                    errores.append(f"{p.ticker}: {ex}")
    except ErrorEstado as ex:
        return f"❌ {F.esc(ex)}"
    if not lineas:
        return "❌ " + F.esc(" ".join(errores) or "No pude registrar nada.")
    est = leer_estado(ctx)
    otras = sorted(est.tenidos() - {p.ticker for p in partes})
    L = [f"✅ {F.b('Cartera ajustada')} " + F.it("(no cuenta como operación)"), *lineas, *["⚠️ " + F.esc(x) for x in errores]]
    if otras:
        L.append(F.it("No toqué: " + ", ".join(otras) + ". Si ya no tienes alguna, escribe por ejemplo: vendí " + otras[0].lower()))
    L.append(F.it("¿Quedó mal? Escribe: me equivoqué"))
    return "\n".join(L) + "\n\n" + _cartera_txt(ctx, est)


def resp_liquidez(ctx: Contexto, args: list[str]) -> str:
    """/liquidez [ACCIÓN]: el filtro ANTES de comprar. Con una acción dice si es apta y por qué; sin argumento, revisa todo lo que tienes."""
    from . import liquidez as LQ
    est = leer_estado(ctx)
    if args:
        tickers = [resolver(ctx, " ".join(args), est.tenidos())]
        if tickers[0] not in universo_permitido(ctx.cfg):
            return f"❌ {F.esc(tickers[0])} no está permitida en el concurso (o está en la lista negra), o no reconocí el nombre."
    else:
        tickers = sorted(est.tenidos())
        if not tickers:
            return f"{F.b('Cómo usar /liquidez')}\nEscribe la acción que estás pensando comprar: {F.b('/liquidez ecopetrol')}. Te digo si pasa el filtro antes de que metas la plata."
    L = [f"💧 {F.b('Filtro de liquidez en trii')} " + F.it("(lo que de verdad se negocia en la Bolsa de Colombia)")]
    for t in tickers:
        l = LQ.medir(ctx.f, t, ctx.cfg)
        apta, corto = LQ.veredicto(l, t in ctx.cfg["universe"]["local"])
        L += ["", F.b(t) + " — " + F.esc(corto), F.esc(LQ.frase(l, None, ctx.cfg))]
    c = ctx.cfg["liquidez"]
    mil = lambda v: f"{v:,}".replace(",", ".")                                          # noqa: E731
    L += ["", F.it(f"Para aprobar una acción pido: día normal de al menos $ {mil(c['buena_cop_mm'])} millones, estable en 3 meses (al menos {mil(c['buena_60_cop_mm'])}), "
                   f"días flojos de al menos {mil(c['buena_dia_flojo_cop_mm'])}, ningún día sin negociar y que sea de la BVC."),
          F.it("Fuentes: historial diario de cada acción en Colombia, cruzado con lo negociado en la Bolsa de Colombia (15 minutos de retraso). "
               "Si no coinciden, me quedo con la más prudente." if getattr(ctx.f, "libro_bvc", lambda: None)() else
               "Ahora mismo no pude leer lo negociado en la Bolsa de Colombia: uso sólo el historial diario.")]
    return "\n".join(L)


def resp_deshacer(ctx: Contexto) -> str:
    try:
        with transaccion(ctx) as e:
            que = e.deshacer()
        return f"↩️ {F.b('Listo: deshice ' + que)}.\n\n" + _cartera_txt(ctx, leer_estado(ctx))
    except ErrorEstado as ex:
        return f"❌ {F.esc(ex)}"


def resp_texto(ctx: Contexto, texto: str, forzar: str | None = None, solo_lectura: bool = False) -> dict[str, Any]:
    """Mensaje escrito sin comando. Devuelve {'texto': respuesta} o {'consulta': nombre, 'ticker': …} para que el bot ejecute ese comando (con su aviso de espera).
    `solo_lectura`: el chat no es el del dueño y la escritura está reservada: se entiende el mensaje pero no se registra nada."""
    from .entender import entender
    est = leer_estado(ctx)
    i = entender(texto, ctx.cfg, est.tenidos(), forzar)
    if solo_lectura and i.tipo in ("deshacer", "rank", "compra", "venta", "tengo"):
        return {"texto": "🔒 Sólo el dueño del bot puede registrar compras, ventas o el ranking."}
    if i.tipo == "tengo":
        return {"texto": registrar_tenencias(ctx, i.partes)}
    if i.tipo == "deshacer":
        return {"texto": resp_deshacer(ctx)}
    if i.tipo == "rank":
        return {"texto": guardar_rank(ctx, i.mia, i.objetivo)}
    if i.tipo in ("compra", "venta"):
        if "ticker" in i.falta:
            pista = (" ¿Quisiste decir " + " o ".join(F.b(x) for x in i.sugerencias) + "?") if i.sugerencias else ""
            ej = "compré 300 argos a 21500" if i.tipo == "compra" else "vendí argos"
            return {"texto": f"🤔 No reconocí qué acción {'compraste' if i.tipo == 'compra' else 'vendiste'}.{pista}\nEscríbelo así: {F.b(ej)}", "pendiente": i.tipo}
        if i.tipo == "compra":
            if "cantidad" in i.falta:
                return {"texto": f"¿Cuántas acciones de {F.b(i.ticker)} compraste, o cuánta plata metiste?\nEscribe por ejemplo {F.b('300')} o {F.b('20 millones')}.",
                        "pendiente": "compra", "ticker": i.ticker}
            return {"texto": registrar_compra(ctx, i, confirmado="confirmo" in texto.lower())}
        return {"texto": registrar_venta(ctx, i)}
    if i.tipo == "consulta":
        return {"consulta": i.consulta, "ticker": i.ticker}
    return {"texto": ""}


def resp_comprar(ctx: Contexto) -> str:
    """/comprar: qué acción de la BVC comprar, qué esperar y por cuánto tiempo (las de mayor movimiento esperado hasta el próximo corte + la Regla Maestra)."""
    est = leer_estado(ctx)
    r = ejecutar_motor(ctx.f, est, ctx.cfg, ctx.ahora, ctx.puntuador, ctx.motor)
    hora = C.hora_orden_manana(C.proxima_sesion(ctx.ahora, ctx.cfg) or ctx.ahora.date(), ctx.cfg)
    from . import noticias_bvc as N
    liq = avisos_liquidez(ctx, [r.banco[0].c.ticker], {r.banco[0].c.ticker: float(ctx.cfg["capital"]) * 0.25}, todos=True) if r.banco else []
    return F.msg_que_comprar(r.banco, r.activo, r.mee, r.decision, r.corte, C.sesiones_restantes(ctx.ahora, ctx.cfg), senales_recientes(24, ctx.ahora), hora, ctx.cfg,
                             n_estudio=N.cargar_estudio().get("n_total"), liquidez_txt=liq[0] if liq else None)


def mejor_bvc(ctx: Contexto) -> dict[str, Any] | None:
    """La acción de la BVC con más movimiento esperado hasta el próximo corte (la primera del banco). Se usa como sugerencia al pie de los avisos de noticias."""
    est = leer_estado(ctx)
    corte = C.corte_vigente(ctx.ahora, ctx.cfg)
    if corte is None:
        return None
    from .relevo import construir_banco
    cands = candidatos_banco(ctx.f, ctx.cfg, ctx.ahora, est.activo, None, ctx.puntuador, ctx.motor)
    banco = construir_banco(cands, est.activo, ctx.cfg, set(est.tenidos()))
    return dict(ticker=banco[0].c.ticker, mee=banco[0].c.mee, corte=corte["fecha"]) if banco else None


BOTONES_NOTICIA = [[("🛒 Qué comprar", "c:comprar"), ("💼 Mi cartera", "c:cartera")]]
BOTONES_MACRO = [[("🌍 Macro ahora", "c:macro"), ("💼 Mi cartera", "c:cartera")]]


# ------------------------------------------------------------------ macro y cambios sugeridos
def acciones_liquidas(ctx: Contexto, tickers: Any) -> set[str]:
    """De esas acciones, las que tienen liquidez BUENA en trii (src/liquidez.py). Las demás no se nombran en ningún aviso ni recomendación."""
    from . import liquidez as LQ
    out = set()
    for t in tickers:
        try:
            if LQ.medir(ctx.f, t, ctx.cfg)["nivel"] == LQ.BUENA:
                out.add(t)
        except Exception:                                                              # noqa: BLE001
            continue
    return out


def _tickers_macro(ctx: Contexto, est: Estado) -> list[str]:
    """Acciones a las que se les mide la sensibilidad macro: las de la BVC con liquidez buena en trii + las que tienes."""
    from .construir import universo_candidatos
    return list(dict.fromkeys(sorted(acciones_liquidas(ctx, universo_candidatos(ctx.cfg))) + sorted(est.tenidos())))


def panorama_macro(ctx: Contexto) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """(tablero de factores ahora, {factor: quién se beneficia y quién se afecta hoy}) para los factores que se movieron."""
    from . import macro as M
    est = leer_estado(ctx)
    tab = M.tablero(ctx.f, ctx.cfg, ctx.ahora)
    if not any(x["notable"] for x in tab):
        return tab, {}
    sens = M.sensibilidades(ctx.f, ctx.cfg, _tickers_macro(ctx, est))
    return tab, {x["clave"]: M.impacto(x, sens.get(x["clave"], {}), ctx.cfg, est.tenidos()) for x in tab if x["notable"]}


def resp_macro(ctx: Contexto) -> str:
    tab, imp = panorama_macro(ctx)
    return F.msg_macro_tablero(tab, imp, ctx.ahora)


def tick_macro(ctx: Contexto, mem: Any, items: list[Any] | None = None) -> list[str]:
    """Una revisión macro (cada 2 minutos): avisa si un factor se mueve mucho más de lo normal o si sale un titular macro importante, y en ambos casos
    dice qué acciones se benefician y cuáles se afectan. `items` = titulares ya leídos por el vigía de noticias (no se vuelven a descargar)."""
    from . import macro as M
    if not ctx.cfg["macro_vivo"].get("activo"):
        return []
    tab = M.tablero(ctx.f, ctx.cfg, ctx.ahora)
    fuertes = M.movimientos_nuevos(tab, mem.d, ctx.ahora)
    temas = M.titulares_macro(items or [], mem.d, ctx.ahora, ctx.cfg)
    mem.sucia = True
    if not fuertes and not temas:
        return []
    est = leer_estado(ctx)
    sens = M.sensibilidades(ctx.f, ctx.cfg, _tickers_macro(ctx, est))
    imp = {x["clave"]: M.impacto(x, sens.get(x["clave"], {}), ctx.cfg, est.tenidos()) for x in tab if x["notable"]}
    uno = {x["clave"]: M.impacto(dict(x, cambio=0.01), sens.get(x["clave"], {}), ctx.cfg, est.tenidos(), minimo=0.001) for x in tab}      # efecto por cada 1 % de cada factor
    mensajes = [F.msg_macro_movimiento(x, imp[x["clave"]], ctx.ahora) for x in fuertes]
    mensajes += [F.msg_macro_titular(g, tab, imp, ctx.ahora, uno) for g in temas]
    for m in mensajes:
        enviar_o_encolar(ctx, m, BOTONES_MACRO)
    if not ctx.dry:
        mem.guardar()
    return mensajes


def banco_bvc(ctx: Contexto) -> list[Any]:
    """Candidatas de la BVC (líquidas en trii, sin alerta, que no tienes) de mayor a menor movimiento esperado hasta el próximo corte."""
    est = leer_estado(ctx)
    if C.corte_vigente(ctx.ahora, ctx.cfg) is None:
        return []
    from .relevo import construir_banco
    cands = candidatos_banco(ctx.f, ctx.cfg, ctx.ahora, est.activo, None, ctx.puntuador, ctx.motor)
    return construir_banco(cands, est.activo, ctx.cfg, set(est.tenidos()))


def cambios_sugeridos(ctx: Contexto, banco: list[Any]) -> list[dict[str, Any]]:
    """Cambios "esta por esta" que hoy tienen un motivo que justifica el costo (ver src/cambios.py)."""
    from . import cambios as CB
    from . import cartera as CA
    from . import liquidez as LQ
    est = leer_estado(ctx)
    tenidos = sorted(est.tenidos())
    if not tenidos:
        return []
    liq = {t: LQ.medir(ctx.f, t, ctx.cfg) for t in tenidos}
    try:
        montos = {x["ticker"]: x["valor_cop"] for x in CA.valorar(ctx.f, est.d["cartera"], ctx.cfg) if x["valor_cop"] == x["valor_cop"]}
    except Exception:                                                                  # noqa: BLE001
        montos = {}
    sugs = CB.sugerencias(tenidos, liq, banco, montos, ctx.cfg)
    for s in sugs:
        s["liquidez_a"] = LQ.frase(LQ.medir(ctx.f, s["a"], ctx.cfg), montos.get(s["de"]))
    return sugs


def tick_cambios(ctx: Contexto, banco: list[Any]) -> list[str]:
    """Avisa los cambios sugeridos que no se han avisado en las últimas 24 horas."""
    from . import cambios as CB
    sugs = cambios_sugeridos(ctx, banco)
    memoria = {"cambios_avisados": dict(leer_estado(ctx).d.get("cambios_avisados", {}))}
    antes = json.dumps(memoria, sort_keys=True)
    pend = CB.nuevos(sugs, memoria, ctx.ahora, ctx.cfg)
    if json.dumps(memoria, sort_keys=True) != antes:                                   # sólo se escribe el estado si de verdad cambió algo
        with transaccion(ctx) as e:
            e.d["cambios_avisados"] = memoria["cambios_avisados"]
    mensajes = [F.msg_cambio(s, CB.estudio_relativo(3)) for s in pend]
    for m in mensajes:
        enviar_o_encolar(ctx, m, [[("🛒 Qué comprar", "c:comprar"), ("💼 Mi cartera", "c:cartera")]])
    return mensajes


def resumen_manana(ctx: Contexto, banco: list[Any] | None = None) -> str:
    """Resumen antes de abrir: macro de la noche, calendario de hoy, lo que se negocia poco, cambios sugeridos y la regla de operar con plan."""
    est = leer_estado(ctx)
    tab, imp = panorama_macro(ctx)
    hoy = ctx.ahora.date()
    fechas = [f"🏦 {F.esc(m['evento'])}" + (f" ({F.esc(m['hora'])})" if m.get("hora") else "") + (" " + F.it("(fecha por confirmar)") if m.get("verificada") is False else "")
              for m in ctx.cfg.get("eventos_macro", []) if str(m["fecha"]) == hoy.isoformat()]
    corte = C.corte_vigente(ctx.ahora, ctx.cfg)
    if corte and corte["fecha"] == hoy:
        fechas.append("🏁 " + F.b("Hoy es el corte") + ": al cierre sólo avanzan los mejores del ranking.")
    try:
        cambios = cambios_sugeridos(ctx, banco or [])
    except Exception:                                                                  # noqa: BLE001 — el resumen sale aunque falle una parte
        cambios = []
    try:
        liq = avisos_liquidez(ctx, sorted(est.tenidos()))
    except Exception:                                                                  # noqa: BLE001
        liq = []
    h = C.horario(hoy, ctx.cfg)
    return F.msg_resumen_manana(F.msg_macro_tablero(tab, imp, ctx.ahora), fechas, liq, cambios, f"{h[0]:%H:%M}" if h else "—", est.rent.get("mia") is None)


def tick_noticias(ctx: Contexto, lector: Any, mem: Any, sugerencia: dict[str, Any] | None = None) -> list[str]:
    """Una ronda del vigía de noticias de la BVC (cada 2 minutos): lee las fuentes y avisa SÓLO lo nuevo de alto impacto. Devuelve los mensajes generados."""
    from . import noticias_bvc as N
    est = leer_estado(ctx)
    senales, salud = N.ronda(lector, mem, ctx.f, ctx.ahora, ctx.cfg, est.tenidos())
    mem.ultimos_items = getattr(lector, "ultimos", [])                                 # los titulares recién leídos también sirven para la revisión macro
    mensajes = []
    vecinos: dict[str, dict[str, float]] = {}
    if senales:                                                                        # a qué otras acciones de la BVC les pega (sólo se calcula si hay algo que avisar)
        try:
            from . import macro as M
            from .construir import universo_candidatos
            vecinos = M.contagio(ctx.f, ctx.cfg, universo_candidatos(ctx.cfg))
        except Exception:                                                              # noqa: BLE001 — sin ese cálculo el aviso sale igual, sólo con la acción propia
            vecinos = {}
    liquidas = acciones_liquidas(ctx, {t for v in vecinos.values() for t in v}) if senales else set()
    for s in senales:
        from . import macro as M
        afectadas = M.afectadas_por_noticia(s.ticker, s.impacto, s.sentido, vecinos.get(s.ticker, {}), ctx.cfg, est.tenidos())
        afectadas = [x for x in afectadas if x["propia"] or x["tengo"] or x["ticker"] in liquidas]      # nunca se listan acciones sin liquidez buena en trii
        if s.nivel != "alto":                                                          # relevante u otra noticia de la empresa: aviso corto con su impacto estimado en %
            m = F.msg_noticia_relevante(s, ctx.ahora, ctx.cfg, afectadas)
            mensajes.append(m)
            enviar_o_encolar(ctx, m)
            continue
        botones = BOTONES_NOTICIA
        if s.sentido > 0 and not s.rec["tengo"]:
            botones = [[("✅ La compré", f"k:{s.ticker}")]] + BOTONES_NOTICIA
        m = F.msg_noticia_bvc(s, ctx.ahora, ctx.cfg, sugerencia if (sugerencia and sugerencia["ticker"] != s.ticker) else None, afectadas)
        mensajes.append(m)
        enviar_o_encolar(ctx, m, botones)
    try:                                                                               # ¿alguna noticia ya avisada está moviendo el precio? → segundo aviso
        for x in N.movimientos_tras_noticia(mem, ctx.f, ctx.ahora, ctx.cfg):
            m = F.msg_noticia_mueve_precio(x, ctx.ahora)
            mensajes.append(m)
            enviar_o_encolar(ctx, m, BOTONES_NOTICIA)
    except Exception:                                                                  # noqa: BLE001 — el seguimiento no tumba la ronda
        pass
    if salud:                                                                          # aviso si TODAS las fuentes llevan varias rondas caídas (y cuando vuelven)
        ok, antes = any(salud.values()), int(mem.d.get("rondas_sin_fuentes", 0))
        if not ok or antes:                                                            # con todo bien no se toca state.json (se escribiría en cada ronda sin necesidad)
            with transaccion(ctx) as e:
                ev = e.registrar_fuente("noticias_bvc", ok, ctx.ahora, max(5, int(600 / max(ctx.cfg["noticias_bvc"]["cada_s"], 1))), 180)
            if ev:
                enviar_o_encolar(ctx, F.msg_salud("las noticias de la BVC", ev, ctx.ahora, RESPALDO_FUENTE["noticias_bvc"], salud))
        mem.d["rondas_sin_fuentes"] = 0 if ok else antes + 1
    mem.d["salud"] = salud
    if not ctx.dry:
        mem.guardar()
    return mensajes


def resp_op(ctx: Contexto) -> str:
    with transaccion(ctx) as e:
        e.registrar_op(ctx.ahora)
        a = ctx.cfg["concurso"]["actividad"]
        sem = e.ops_semana(ctx.ahora.date())
        return (f"✅ {F.b('Operación anotada')}. Esta semana llevas {sem} de {a['ops_por_semana']} y en total {e.ops_total()} de {a['minimo_total']}."
                + (" 🎯 ¡Ya cumpliste la meta de esta semana!" if sem >= a["ops_por_semana"] else ""))
