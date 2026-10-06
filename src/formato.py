"""Mensajes de Telegram en lenguaje sencillo (HTML de Telegram: negritas y cursivas). Funciones puras: reciben resultados y devuelven texto.

Reglas de redacción: sin siglas técnicas (nada de z, z_res, MEE, g, beta en los mensajes normales), frases cortas, cada mensaje responde
"¿qué pasó?", "¿qué significa?" y "¿qué hago?". Los números técnicos viven en /detalle."""
from __future__ import annotations

import datetime as dt
import html
from typing import Any, Callable

import pandas as pd

from . import concurso as C
from .notify import a_plano
from .regla_maestra import Decision
from .relevo import EntradaBanco
from .semaforo import AMARILLO, AZUL, EMOJI, NEGRO, ROJO, VERDE, ResultadoSemaforo

DIAS = ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"]
Traductor = Callable[[str], "str | None"]


# ------------------------------------------------------------------ utilidades
def esc(s: Any) -> str:
    return html.escape(str(s), quote=False)


def b(s: Any) -> str:
    return f"<b>{esc(s)}</b>"


def it(s: Any) -> str:
    return f"<i>{esc(s)}</i>"


plano = a_plano                                   # respaldo si Telegram rechaza el formato, y para imprimir en consola


def n(x: float | None, d: int = 1, signo: bool = False) -> str:
    if x is None or x != x:
        return "n. d."
    return (f"{x:+.{d}f}" if signo else f"{x:.{d}f}").replace(".", ",")


def pct(x: float | None, d: int = 1, signo: bool = False) -> str:
    """Fracción → '5,2%' (sin espacio, como se escribe en los mensajes)."""
    return "n. d." if x is None or x != x else n(x * 100, d, signo) + "%"


def puntos(x: float, d: int = 1) -> str:
    return n(x, d) + (" punto" if abs(round(x, d)) == 1 else " puntos")


def fecha(d: dt.date | dt.datetime) -> str:
    return f"{DIAS[d.weekday()]} {d:%d/%m}"


def cop(x: float) -> str:
    return "$" + f"{x:,.0f}".replace(",", ".")


def hace(ts_iso: str, ahora_utc: dt.datetime) -> str:
    m = int((pd.Timestamp(ahora_utc).tz_convert("UTC") - pd.Timestamp(ts_iso).tz_convert("UTC")).total_seconds() // 60)
    if m < 60:
        return f"hace {max(m, 1)} min"
    if m < 60 * 36:
        return f"hace {m // 60} h"
    return f"hace {m // 1440} días"


def tono(puntaje: float, palabras: list[str] | None = None) -> str:
    if puntaje <= -0.3 or (palabras and puntaje < 0.05):
        return "😟"
    return "🙂" if puntaje >= 0.3 else "😐"


def veces(x: float) -> str:
    return n(x, 1) + " veces"


# ------------------------------------------------------------------ semáforo en palabras
EXPLICA_COLOR = {
    VERDE: "Todo normal: el movimiento de hoy está dentro de lo habitual.",
    AZUL: "Tu acción baja, pero casi todo se explica porque el mercado en general está bajando. Es ruido, no una mala señal de la empresa.",
    AMARILLO: "Tu acción cae más de lo normal por su cuenta, pero no veo una mala noticia ni mucho volumen que lo confirme. Conviene vigilar, pero no vender por susto.",
    ROJO: "Tu acción cae fuerte por su cuenta, con mala noticia y mucho volumen. Es la alerta más seria, aunque en el historial las acciones grandes no siguieron "
          "cayendo en promedio después de un caso así. Esta noche el radar mira el cierre.",
    NEGRO: "Pasaron dos sesiones y la acción no recupera lo perdido. Ojo: en el historial (2015–2026) las acciones grandes no siguieron cayendo en promedio después "
           "de un caso así, así que por sí solo no es motivo para vender.",
}
NOMBRE_COLOR = {VERDE: "VERDE (todo normal)", AZUL: "AZUL (cae el mercado)", AMARILLO: "AMARILLO (vigilar)", ROJO: "ROJO (alerta)", NEGRO: "NEGRO (no se recupera)"}
LEYENDA = ("🟢 todo normal · 🔵 cae el mercado, no tu acción · 🟡 cae sola, vigila · 🔴 cae sola con mala noticia · "
           "⚫ no se recupera: ojo, pero no vendas solo por eso")


def que_paso(r: ResultadoSemaforo) -> str:
    mov = r.r_hoy
    t = [f"{esc(r.ticker)} {'sube' if mov >= 0 else 'baja'} {n(abs(mov) * 100, 1)}% hoy."]
    if r.z == r.z and abs(r.z) >= 2:
        t.append(f"Es un movimiento {veces(abs(r.z))} más fuerte de lo normal para esta acción.")
    elif r.z == r.z and abs(r.z) < 1:
        t.append("Está dentro de lo normal para esta acción.")
    else:
        t.append("Es algo más fuerte de lo normal, pero sin llegar a ser raro.")
    if mov < 0 and r.r_mercado <= -0.002:
        t.append(f"El mercado en general explica {pct(r.r_mercado)} de esa baja; lo propio de la acción es {pct(r.resid_hoy)}.")
    elif mov < 0 and r.r_mercado >= 0.002:
        t.append("El mercado sube pero tu acción baja: la caída es propia de la acción.")
    if r.vol_rel is not None and r.vol_rel >= 1.5:
        t.append(f"Se está negociando {veces(r.vol_rel)} el volumen normal (mucha actividad).")
    return " ".join(t)


def bloque_noticias(r: ResultadoSemaforo, traductor: Traductor | None = None, max_n: int = 3) -> list[str]:
    if r.sin_noticias:
        extra = " Como plan B uso el volumen y el hueco de apertura para sospechar una mala noticia." if r.proxy_noticias else ""
        return [f"📰 {b('Noticias')}: hoy no pude consultarlas (fuentes caídas), así que no puedo confirmar si hay una mala noticia.{extra}"]
    if r.n_24h == 0:
        return [f"📰 {b('Noticias')}: no hay titulares en las últimas 24 horas."]
    if r.noticia_negativa:
        general = "negativo"
    elif r.sentimiento is not None and r.sentimiento >= 0.2:
        general = "positivo"
    else:
        general = "mixto, sin una señal clara de problema"
    mostradas = r.claves[:max_n]
    preocupan = any(tono(k["puntaje"], k["palabras"]) == "😟" for k in mostradas)
    L = [f"📰 {b('Noticias')}: {r.n_24h} en las últimas 24 horas, tono general {general}."
         + (f" Estos son los {len(mostradas)} titulares {'más preocupantes' if preocupan else 'destacados'}:" if r.n_24h > len(mostradas) else "")]
    for k in mostradas:
        titulo = k["titulo"]
        trad = traductor(titulo) if (traductor and k.get("idioma", "en") != "es") else None
        L.append(f"  {tono(k['puntaje'], k['palabras'])} {esc(trad or titulo)}" + ("" if trad or k.get("idioma") == "es" else " " + it("(en inglés)")))
    if r.cobertura_parcial:
        L.append(it("  Cobertura parcial: aquí la falta de noticias no descarta una mala noticia."))
    return L


ACCION_COLOR = {
    VERDE: "No hagas nada.",
    AZUL: "No hagas nada: es el mercado, no tu acción.",
    AMARILLO: "No hagas nada ahora. Solo vigila.",
    ROJO: "No vendas por pánico a mitad del día. El radar de las 19:30 mira el cierre y confirma.",
    NEGRO: "No vendas solo por esto. Mira qué dice la Regla Maestra con tu ranking y decide con calma.",
}


def cuerpo_semaforo(r: ResultadoSemaforo, traductor: Traductor | None = None) -> list[str]:
    """Qué pasó / qué significa / noticias / qué hacer (sin título), para reutilizarlo en el mensaje del semáforo y en el radar."""
    L = [f"{b('¿Qué pasó?')} {que_paso(r)}", f"{b('¿Qué significa?')} {EXPLICA_COLOR[r.color]}", "", *bloque_noticias(r, traductor), "",
         f"👉 {b('Qué hacer')}: {ACCION_COLOR[r.color]}"]
    if r.color != r.color_calculado:
        L.append(it("El corte es mañana: por eso un ROJO se trata como NEGRO."))
    return L


def msg_semaforo(r: ResultadoSemaforo, ahora: dt.datetime, traductor: Traductor | None = None) -> str:
    L = [f"{EMOJI[r.color]} {b(r.ticker + ': ' + NOMBRE_COLOR[r.color])}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}", "", *cuerpo_semaforo(r, traductor),
         it("Números técnicos: /detalle · Informe completo en HTML: /informe")]
    return "\n".join(L)


# ------------------------------------------------------------------ decisión (Regla Maestra) en palabras
def texto_decision(d: Decision, activo: str, hora_orden: str = "08:45") -> list[str]:
    """Explica la recomendación de cambio en frases simples."""
    ds = d.datos
    c = d.codigo
    if c == "sin_ranking":
        return ["Todavía no sé cómo vas en el ranking, así que no puedo recomendarte un cambio.",
                f"Escríbeme {b('voy 6,5 y el corte está en 11')} (tu rentabilidad y la del corte, en %) y te digo qué hacer."]
    if c == "g_cero":
        return [f"{b('Mantén ' + activo)}. Vas por encima del objetivo (+{n(ds['adelante'], 1)} puntos): cambiar solo costaría plata.",
                "Cuando vas ganando conviene parecerte a los demás, no diferenciarte."]
    if c == "primero_mantener":
        return [f"{b('Mantén ' + activo)}. Vas de primero: no te cambies; sigue lo que probablemente tienen los demás."]
    if c == "ultimo_dia_mantener":
        return [f"{b('Mantén ' + activo)}. Vas de primero pero sin una ventaja de 8 puntos o más sobre el segundo: no te pases a efectivo."]
    if c == "vender_todo":
        return [f"{b('VENDE TODO mañana en la apertura')} con orden límite: vas de primero con {n(ds['ventaja'], 1)} puntos de ventaja y es el último día.",
                f"Ejecútalo a las {hora_orden}."]
    if c == "negro_adelante":
        r = f" (hoy {n(ds['rend_entrada'], 1)}%)" if ds.get("rend_entrada") is not None else ""
        return [f"{b('Mantén ' + activo)}. Está en NEGRO, pero vas {n(ds['adelante'], 1)} puntos por encima del objetivo.",
                f"Solo se cambia si acumula una pérdida de 15% desde tu compra{r}."]
    if c == "limite_dia":
        return [f"{b('Mantén ' + activo)}. Ya hiciste un cambio hoy y el máximo es uno por día."]
    if c == "limite_total":
        return [f"{b('Mantén ' + activo)}. Ya usaste los 4 cambios permitidos."]
    if c == "sin_mee":
        return ["No pude calcular cuánto se espera que se mueva tu acción, así que hoy no puedo evaluar el cambio."]
    if c == "banco_vacio":
        return [f"{b('Mantén ' + activo)}. No hay ningún candidato bueno para cambiar ahora (todos poco líquidos, en alerta o sin datos)."]
    if c not in ("cambiar", "ninguno_cumple") or d.g is None or d.multiplo is None:
        return [f"{b('Mantén ' + activo)}. Por ahora no hay una razón clara para cambiar."]
    g = d.g
    faltan = g - 1.0
    intro = [f"Te faltan unos {puntos(faltan)} para el objetivo (la regla suma 1 de margen de seguridad: {n(g, 1)})."]
    if d.modo_negro:
        intro.append(f"Tu acción está en NEGRO (no se recupera), así que bajo el listón: basta que el nuevo se mueva {n(d.multiplo, 2)} veces lo que se mueve el tuyo.")
    else:
        intro.append(f"Cambiar cuesta cerca de 1,3%, así que solo vale la pena si el nuevo activo se mueve al menos {n(d.multiplo, 2)} veces lo que se mueve el tuyo.")
    if c == "cambiar":
        return [*intro, f"{b('Recomendación: cambia a ' + d.candidato)}. Se espera que se mueva {n(d.ratio, 2)} veces lo que se mueve {activo}: alcanza.",
                f"Si decides hacerlo: mañana a las {hora_orden}, orden límite, y luego registra con {b('/pos ' + d.candidato + ' <monto> <precio>')}."]
    mejor = ds.get("mejor")
    return [*intro, f"{b('Mantén ' + activo)}. El mejor candidato ({esc(mejor)}) se mueve {n(d.ratio, 2)} veces lo tuyo: no alcanza para pagar el cambio."]


def bloque_decision(d: Decision, activo: str, hora_orden: str = "08:45") -> list[str]:
    return [f"⚖️ {b('¿Cambio de acción?')}", *texto_decision(d, activo, hora_orden)]


# ------------------------------------------------------------------ alertas
def msg_alerta_color(r: ResultadoSemaforo, anterior: str | None, ahora: dt.datetime, mee: float | None = None, decision: Decision | None = None,
                     traductor: Traductor | None = None, hora_orden: str = "08:45", activo: str | None = None) -> str:
    graves = (AMARILLO, ROJO, NEGRO)
    if anterior in graves and r.color in (VERDE, AZUL):
        titulo = f"✅ {b(r.ticker + ' se recuperó')}  {it(f'ahora {NOMBRE_COLOR[r.color]}')}"
        accion = "Buena noticia: no hace falta hacer nada."
    elif r.color in (VERDE, AZUL):
        titulo = f"{EMOJI[r.color]} {b(r.ticker + ': ' + NOMBRE_COLOR[r.color])}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}"
        accion = ACCION_COLOR[r.color]
    else:
        previo = f"antes: {anterior}" if anterior else "primera lectura"
        titulo = f"{EMOJI[r.color]} {b('ALERTA: ' + r.ticker + ' ahora en ' + NOMBRE_COLOR[r.color])}\n{it(f'{previo} · {fecha(ahora)} {ahora:%H:%M}')}"
        accion = ACCION_COLOR[r.color]
    cuerpo = cuerpo_semaforo(r, traductor)
    cuerpo[cuerpo.index(next(x for x in cuerpo if x.startswith("👉")))] = f"👉 {b('Qué hacer')}: {accion}"
    L = [titulo, "", *cuerpo]
    if decision is not None and r.color in graves:
        L += ["", *bloque_decision(decision, activo or r.ticker, hora_orden)]
    L.append(it("Números técnicos: /detalle · Informe completo en HTML: /informe"))
    return "\n".join(L)


def msg_detalle(r: ResultadoSemaforo, mee: float | None, iv: dict[str, Any] | None, cfg: dict[str, Any]) -> str:
    s = cfg["semaforo"]
    L = [f"🔬 {b('Detalle técnico de ' + r.ticker)}", "",
         f"• Retorno de hoy: {pct(r.r_hoy, 2, True)}",
         f"• Caída frente a lo normal (z): {n(r.z, 2, True)}  {it(f'0 es normal; por debajo de {s['z_caida']:g} es raro')}",
         f"• Lo propio de la acción sin el mercado (z_res): {n(r.z_res, 2, True)}  {it(f'por debajo de {s['z_res_caida']:g} es caída propia')}",
         f"• Parte explicada por el mercado: {pct(r.r_mercado, 2, True)} · parte propia: {pct(r.resid_hoy, 2, True)}",
         f"• Sensibilidad al mercado (beta): {n(r.beta, 2)}  {it('1 = se mueve igual que el Nasdaq-100')}",
         f"• Volumen frente al normal: {n(r.vol_rel, 2) + '×' if r.vol_rel is not None else 'n. d.'}  {it(f'ROJO exige {s['vol_rel_min']:g}× o más')}",
         f"• Variación diaria típica (σ 20 días): {pct(r.sigma20, 2)}",
         f"• Suma de lo propio en 5 días: {pct(r.resid5, 2, True)}  {it(f'alerta si baja de {pct(r.umbral5, 2)} con mala noticia')}",
         f"• Noticias 24 h: {r.n_24h}" + (f" · tono medio {n(r.sentimiento, 2, True)}" if r.sentimiento is not None else " · sin datos") + f" · ¿mala noticia?: {'sí' if r.noticia_negativa else 'no'}",
         f"• Método de sentimiento: {r.motor_sentimiento}",
         f"• Por qué este color: {esc(r.motivo)}"]
    if mee is not None:
        L.append(f"• Movimiento esperado hasta el corte (±): {pct(mee, 2)}" + (f"  {it('(según opciones, vencimiento ' + str(iv['vencimiento']) + ')')}" if iv and iv.get("iv") else f"  {it('(según su variación típica)')}"))
    if r.advertencias:
        L.append("⚠️ " + esc("; ".join(r.advertencias)))
    return "\n".join(L)


# ------------------------------------------------------------------ banco de relevo
def _parecido(corr: float | None, actual: str) -> str:
    if corr is None:
        return ""
    if corr < 0.3:
        return f"se mueve distinto a {actual} ✅"
    if corr < 0.6:
        return f"algo parecida a {actual}"
    return f"se mueve casi igual que {actual}"


def conclusion_banco(banco: list[EntradaBanco], mee: float | None, costo: float = 0.013) -> str | None:
    """Lo primero que hay que saber antes de mirar la lista: ¿alguna se mueve más que lo que ya tienes? Si no, cambiar no ayuda a remontar."""
    if not mee or not banco:
        return None
    mejor = max(banco, key=lambda e: e.c.mee).c
    r = mejor.mee / mee
    if r < 0.95:
        return (f"➡️ {b('Ninguna se mueve más que lo que ya tienes')} (la que más, {esc(mejor.ticker)}: {n(r, 1)} veces lo tuyo). Cambiar {b('no')} te ayuda a remontar y cuesta "
                f"cerca de {pct(costo, 1)}. Solo tendría sentido para bajar riesgo o para salir de algo que se negocia poco en trii.")
    if r < 1.15:
        return (f"➡️ {b('Ninguna se mueve claramente más que lo que ya tienes')} ({esc(mejor.ticker)}: {n(r, 1)} veces lo tuyo). Con una diferencia tan pequeña, "
                f"el cambio no paga su costo (cerca de {pct(costo, 1)}).")
    return (f"➡️ {b(mejor.ticker + ' se mueve ' + n(r, 1) + ' veces lo que ya tienes')}: esa es la única razón para cambiar. Si alcanza o no depende de cuánto te falte "
            "en el ranking (mira la recomendación de abajo).")


def msg_banco(banco: list[EntradaBanco], actual: str, mee: float | None, excluidos: set[str] | None = None, top: int = 7, titulo: bool = True) -> str:
    """Relevos posibles: primero la conclusión (¿vale la pena cambiar?), luego las opciones de la que MÁS se mueve a la que menos, con lo que hizo cada una
    esta semana. "Más movimiento" es lo que sirve para remontar en el ranking; no dice hacia dónde."""
    if not banco:
        return f"📋 {b('Relevos posibles')}: hoy no hay ninguno que cumpla (poco líquidos en trii, en alerta o sin datos)."
    L = [f"📋 {b('Relevos posibles para ' + actual)}" + (f" {it('(cuánto puede moverse cada uno hasta el próximo corte)')}" if titulo else "")]
    c0 = conclusion_banco(banco, mee)
    if c0:
        L += [c0, ""]
    orden = sorted(banco[:top], key=lambda e: -e.c.mee)
    for i, e in enumerate(orden, 1):
        c = e.c
        partes = [f"se espera ±{n(c.mee * 100, 1)}% hasta el corte" + (f" ({n(c.mee / mee, 1)} veces lo tuyo)" if mee else "")]
        if c.r5 is not None:
            partes.append(f"esta semana {pct(c.r5, 1, True)}")
        p = _parecido(c.corr, actual)
        if p:
            partes.append(p)
        if c.reporte_antes_corte:
            partes.append(f"reporta resultados el {pd.Timestamp(c.reporte_antes_corte):%d/%m}, antes del corte")
        if c.sin_noticias:
            partes.append("sin datos de noticias")
        L.append(f"{i}. {EMOJI[c.color]} {b(c.ticker)}: " + "; ".join(partes) + ".")
    if any(e.c.r5 is not None for e in orden):
        L.append(it("Entre dos parecidas, mejor la que viene quieta que la que más subió esta semana: en la BVC las que más suben suelen devolver parte."))
    if excluidos:
        L.append(it(f"Excluidos esta semana: {', '.join(sorted(excluidos))} (probablemente los tienen los líderes)."))
    return "\n".join(L)


# ------------------------------------------------------------------ estado
def msg_estado(estado: Any, cfg: dict[str, Any], ahora: dt.datetime, res: ResultadoSemaforo | None = None, mee: float | None = None,
               fuentes_ok: dict[str, bool] | None = None) -> str:
    d = estado.d
    p, r = d["posicion"], d["rentabilidad"]
    corte = C.corte_vigente(ahora, cfg)
    a = cfg["concurso"]["actividad"]
    L = [f"📍 {b('Tu estado')}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}", ""]
    if p.get("precio_entrada"):
        L.append(f"💼 {b('Tu acción')}: {esc(p['activo'])}, {cop(p['monto_cop'])} (la compraste a {n(p['precio_entrada'], 2)} el {p['fecha_entrada']}).")
    else:
        L.append(f"💼 {b('Tu acción')}: {esc(p['activo'])}, {cop(p['monto_cop'])} — {b('aún sin comprar')}. Cuando compres, avísame con /pos {esc(p['activo'])} {int(p['monto_cop'])} <precio>.")
    if res is not None:
        L.append(f"{EMOJI[res.color]} {b('Semáforo')}: {NOMBRE_COLOR[res.color]}. {EXPLICA_COLOR[res.color]}")
    if r.get("mia") is not None and r.get("umbral") is not None:
        falta = r["umbral"] - r["mia"]
        L.append(f"🏁 {b('Ranking')}: vas {n(r['mia'], 1, True)}% y el objetivo del corte es {n(r['umbral'], 1, True)}%. "
                 + (f"Te faltan {puntos(falta)}." if falta > 0 else f"Vas {puntos(-falta)} por encima del objetivo. 👍"))
    else:
        L.append(f"🏁 {b('Ranking')}: no me has dicho cómo vas. Escríbeme {b('voy 6,5 y el corte está en 11')} (tu rentabilidad y la del corte, en %).")
    L.append(f"🔁 {b('Cambios de acción')}: has usado {estado.cambios_usados()} de {cfg['estado']['max_cambios']}" + (" (hoy ya hiciste uno)." if estado.cambio_hoy(ahora.date()) else "."))
    L.append(f"🧾 {b('Operaciones')}: {estado.ops_semana(ahora.date())} de {a['ops_por_semana']} esta semana · {estado.ops_total()} de {a['minimo_total']} mínimas en total.")
    if corte:
        top = f"pasa el top {corte['top_pct']}%" if corte.get("top_pct") else "final: gana el #1"
        sem = C.semana_concurso(ahora.date(), cfg)
        L.append(f"📅 {b('Próximo corte')}: {fecha(corte['fecha'])} ({top}). Faltan {C.sesiones_restantes(ahora, cfg)} días de bolsa"
                 + (f"; es la semana {sem} de 5." if sem else ". El concurso aún no empieza.") + (" ⚠️ " + b("¡El corte es mañana!") if C.corte_manana(ahora, cfg) else ""))
    if fuentes_ok:
        mal = [k for k, v in fuentes_ok.items() if not v]
        L.append("🔌 " + b("Datos") + ": " + ("todo funciona bien ✅" if not mal else "hay problemas con " + ", ".join(mal) + " ⚠️ (el sistema usa su plan B)"))
    return "\n".join(L)


# ------------------------------------------------------------------ catalizadores / noticias / salud / actividad
def msg_catalizadores(items: list[dict[str, Any]], dias: int, max_items: int | None = None, destacados: set[str] | None = None) -> str:
    if not items:
        return f"📅 {b('Qué viene')}: no hay fechas importantes conocidas en los próximos {dias} días hábiles."
    destacados = destacados or set()
    mostrar = items
    if max_items is not None:
        prioridad = [i for i in items if i["tipo"] in ("corte", "macro") or i.get("ticker") in destacados]
        mostrar = prioridad[:max_items] if prioridad else items[:max_items]
    L = [f"📅 {b('Qué viene')} {it(f'(próximos {dias} días hábiles)')}"]
    for x in mostrar:
        icono = {"corte": "🏁", "macro": "🏦", "reporte": "📊"}.get(x["tipo"], "•")
        marca = " " + it("(fecha por confirmar)") if x.get("verificada") is False else ""
        L.append(f"{icono} {b(fecha(x['fecha']))}{(' ' + x['hora']) if x.get('hora') else ''}: {esc(x['detalle'])}{marca}")
    resto = len(items) - len(mostrar)
    if resto > 0:
        L.append(it(f"…y {resto} más. Escribe /catalizadores para verlos todos."))
    return "\n".join(L)


def msg_noticias(ticker: str, noticias: list[Any] | None, analizadas: list[dict[str, Any]], max_n: int, traductor: Traductor | None = None,
                 ahora_utc: dt.datetime | None = None) -> str:
    if noticias is None:
        return f"📰 {b(ticker)}: no pude consultar noticias ahora (fuentes caídas o sin cobertura). Mira la ficha en Yahoo Finance."
    if not noticias:
        return f"📰 {b(ticker)}: no hay noticias recientes."
    ahora_utc = ahora_utc or dt.datetime.now(dt.timezone.utc)
    L = [f"📰 {b('Noticias de ' + ticker)} {it('(las más nuevas primero)')}", it("😟 mala · 😐 neutral · 🙂 buena")]
    for k in analizadas[:max_n]:
        es = k.get("idioma", "en") == "es"
        trad = traductor(k["titulo"]) if (traductor and not es) else None
        L.append(f"{tono(k['puntaje'], k['palabras'])} {it(hace(k['ts'], ahora_utc))}: {esc(trad or k['titulo'])}" + ("" if trad or es else " " + it("(en inglés)")))
    return "\n".join(L)


def msg_salud(nombre: str, evento: str, ahora: dt.datetime, respaldo: str, detalle: dict[str, bool] | None = None) -> str:
    nombre_ok = {"precios": "los precios", "noticias": "las noticias"}.get(nombre, nombre)
    if evento == "CAIDA":
        L = [f"⚠️ {b('Falló una fuente de datos: ' + nombre_ok)}  {it(f'{ahora:%H:%M}')}", "",
             f"{b('¿Qué significa?')} No pude consultar {nombre_ok} en varios intentos seguidos.",
             f"{b('¿Qué hago?')} Nada urgente: {esc(respaldo)}"]
        if detalle:
            L.append(it("Fuentes: " + " · ".join(f"{k} {'✅' if v else '❌'}" for k, v in detalle.items())))
        return "\n".join(L)
    return f"✅ {b('Ya funciona de nuevo: ' + nombre_ok)}  {it(f'{ahora:%H:%M}')}\nTodo vuelve a la normalidad."


def recordatorio_actividad(estado: Any, cfg: dict[str, Any], ahora: dt.datetime) -> str:
    """Qué hacer mañana respecto a la actividad (requisito de 4 operaciones semanales y 15 en total; NO da puntos)."""
    prox = C.proxima_sesion(ahora, cfg)
    if prox is None:
        return "🧾 El concurso terminó: no hay más operaciones."
    hora = C.hora_orden_manana(prox, cfg)
    p = estado.d["posicion"]
    if p.get("precio_entrada") is None and prox == C.sesiones(cfg)[0]:
        return (f"🛒 {b('MAÑANA ' + fecha(prox) + ' a las ' + hora + ': COMPRA INICIAL')}\nCompra {esc(p['activo'])} por {cop(p['monto_cop'])} con orden límite al precio de venta que muestre trii "
                f"(si no se ejecuta en 10 minutos, ajústala).\nEsa compra cuenta como tu operación del día: ese día no necesitas micro-compra.\n"
                f"Después avísame con {b('/pos ' + p['activo'] + ' ' + str(int(p['monto_cop'])) + ' <precio>')}.")
    a = cfg["concurso"]["actividad"]
    sem = C.clave_semana(prox, cfg)
    ops = int(estado.d["operaciones"]["por_semana"].get(sem, 0))
    if ops < a["ops_por_semana"]:
        return (f"🧾 {b('Mañana ' + fecha(prox) + ' toca micro-compra')}: unos {cop(a['micro_compra_cop'])} de {esc(a['activo_micro'])} a las {hora} (la comisión es de unos $14.875).\n"
                f"Llevas {ops} de {a['ops_por_semana']} operaciones esta semana. Si mañana haces un cambio de acción, ese cambio cuenta y no necesitas micro-compra.\n"
                f"Cuando la hagas, avísame con {b('/op')}.")
    return (f"🧾 {b('Mañana no hace falta micro-compra')}: ya tienes las {a['ops_por_semana']} operaciones de esta semana. Total: {estado.ops_total()} de {a['minimo_total']}.")


def msg_noticia_nueva(ticker: str, items: list[dict[str, Any]], res: ResultadoSemaforo | None, ahora: dt.datetime, traductor: Traductor | None = None) -> str:
    """Aviso automático de noticia nueva importante de tu acción."""
    L = [f"📰 {b('Noticia nueva de ' + ticker)}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}", ""]
    ahora_utc = ahora.astimezone(dt.timezone.utc)
    for k in items:
        es = k.get("idioma", "en") == "es"
        trad = traductor(k["titulo"]) if (traductor and not es) else None
        fuente = f"{esc(k.get('proveedor', ''))}, {hace(str(k['ts']), ahora_utc)}" if k.get("ts") else esc(k.get("proveedor", ""))
        L.append(f"{tono(k['puntaje'], k.get('palabras'))} {esc(trad or k['titulo'])}" + ("" if trad or es else " " + it("(en inglés)")) + f"  {it('(' + fuente + ')')}")
    positiva = all(k["puntaje"] > 0 for k in items)
    estado = f" El semáforo sigue en {NOMBRE_COLOR[res.color]}." if res else ""
    L += ["", f"{b('¿Qué significa?')} Un titular por sí solo no cambia el plan.{estado} Si la noticia pesa de verdad, se verá en el precio y el volumen y el semáforo te lo dirá.",
          f"👉 {b('Qué hacer')}: " + ("no persigas el precio por una buena noticia." if positiva else "no vendas solo por un titular.") + f" Para leer más: /noticias {esc(ticker)}"]
    return "\n".join(L)


def msg_decision_cambio(tipo: str, dec: Decision, activo: str, hora_orden: str = "08:45") -> str:
    """La Regla Maestra cambió de opinión: 'nueva' recomienda moverse, 'retirada' ya no."""
    if tipo == "nueva":
        L = [f"⚖️ {b('La Regla Maestra ahora recomienda moverte')}", "", *texto_decision(dec, activo, hora_orden), "",
             it("Es una recomendación: la orden la das tú. Más detalle: /informe")]
    else:
        L = [f"⚖️ {b('La Regla Maestra ya no recomienda cambiar')}", "", *texto_decision(dec, activo, hora_orden)]
    return "\n".join(L)


def msg_calendario(a: dict[str, Any], activo: str = "") -> str:
    """Aviso de una fecha importante: reporte de resultados de tu acción, evento macro del día o corte."""
    if a["tipo"] == "reporte":
        cuando = "hoy" if a["es_hoy"] else f"en la próxima sesión ({fecha(a['fecha'])})"
        hora = {"amc": " después del cierre", "bmo": " antes de abrir", "dmh": " durante el mercado"}.get(a.get("hora", ""), "")
        return (f"📊 {b(esc(a['ticker']) + ' reporta resultados ' + cuando)}{esc(hora)}\n\n{b('¿Qué significa?')} Los resultados suelen mover mucho el precio, para arriba o para abajo, "
                f"y la fecha puede variar un día según la fuente.\n👉 {b('Qué hacer')}: nada especial; está en tu plan. Si el precio se mueve fuerte, el semáforo te avisa.")
    if a["tipo"] == "macro":
        h = f" a las {esc(a['hora'])}" if a.get("hora") else ""
        dudosa = "\n" + it("La fecha de este evento está por confirmar.") if a.get("verificada") is False else ""
        return (f"🏦 {b('Hoy' + h + ': ' + esc(a['evento']))}\n\n{b('¿Qué significa?')} Estos anuncios pueden mover todo el mercado en minutos.\n"
                f"👉 {b('Qué hacer')}: no operes con prisa alrededor del anuncio.{dudosa}")
    c = a["corte"]
    top = f"pasa el top {c['top_pct']}%" if c.get("top_pct") else "final: gana el #1"
    return (f"🏁 {b('El corte es ' + ('hoy' if a.get('es_hoy') else 'mañana'))}: {fecha(c['fecha'])} ({top})\n\n"
            f"{b('¿Qué significa?')} Al cierre solo avanzan los mejores del ranking. Hoy un ROJO se trata como NEGRO.\n"
            f"👉 {b('Qué hacer')}: revisa tu ranking con /rank y la recomendación con /estado.")


def msg_alza(res: ResultadoSemaforo, ahora: dt.datetime) -> str:
    return (f"📈 {b(res.ticker + ' sube fuerte hoy: ' + pct(res.r_hoy, 1, True))}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}\n\n{b('¿Qué pasó?')} {que_paso(res)}\n"
            f"👉 {b('Qué hacer')}: nada. Una subida fuerte no es señal de compra; el plan es mantener.")


def msg_base_cambio(tipo: str, filas: list[dict[str, Any]], mejores: list[dict[str, Any]], base: str, cfg: dict[str, Any]) -> str:
    """Aviso de que cambió la comparación de base (otra acción superó o dejó de superar a la base)."""
    if tipo == "revisar":
        return f"🔎 {b('Novedad en la comparación de base')}\n\n" + msg_base(filas, "revisar", mejores, base, cfg, 5)
    return f"🔎 {b('Ya ninguna acción supera a ' + esc(base))}\n\nLa comparación de base volvió a favorecer tu acción. Detalle: /base"


def msg_cartera(filas: list[dict[str, Any]], rent: float | None, tasa: float | None, colores: dict[str, str] | None = None, titulo: bool = True) -> str:
    """Tu cartera con precios de hoy: cantidad, precio de compra, rentabilidad, peso y semáforo de cada acción."""
    if not filas:
        return f"💼 {b('Tu cartera')}: aún no has registrado compras. Usa /compra TSLA 55 380,5 (acción, cantidad y precio)."
    validos = [x["valor_cop"] for x in filas if x["valor_cop"] == x["valor_cop"]]
    L = [f"💼 {b('Tu cartera')}" + (f" {it('(≈ $ ' + n(sum(validos) / 1e6, 1) + ' millones en total)')}" if validos else "")] if titulo else []
    for x in sorted(filas, key=lambda x: -(x["peso"] or 0)):
        mon = "US$ " if x["moneda"] == "USD" else "$ "
        cant = f"{x['cantidad']:g} acciones" if x.get("cantidad") else "cantidad no registrada"
        px = lambda v: (f"{v:,.2f}".replace(",", "§").replace(".", ",").replace("§", ".")) if v >= 1000 else n(v, 2)      # noqa: E731 — 21.366,13 y no 21366,13
        hoy = (f"hoy {mon}{px(x['precio'])} ({pct(x['rent'], 1, True)})" if x["precio"] else "sin precio ahora")
        peso = f" · {pct(x['peso'], 0)} de la cartera" if x["peso"] is not None else ""
        col = f" {EMOJI[colores[x['ticker']]]}" if colores and colores.get(x["ticker"]) in EMOJI else ""
        L.append(f"• {b(x['ticker'])}{col}: {cant} · compra {mon}{px(x['precio_compra'])} · {hoy}{peso}")
    if rent is not None:
        L.append(f"Rentabilidad del conjunto (aprox.): {b(pct(rent, 1, True))}" + (f" · TRM usada $ {n(tasa, 0)}" if tasa else ""))
    L.append(it("Es aproximada: la TRM de trii puede ser distinta. Tu ranking oficial lo da trii: usa /rank."))
    return "\n".join(L)


def msg_base(filas: list[dict[str, Any]], veredicto: str, mejores: list[dict[str, Any]], base: str, cfg: dict[str, Any], top: int = 8) -> str:
    """¿Hay una base mejor para TODO el concurso? Compara EE. UU. y BVC por el movimiento esperado hasta el final."""
    fin = pd.Timestamp(cfg["concurso"]["final"]["fecha"]).date()
    ventaja = cfg["seleccion_base"]["ventaja_minima"]
    if veredicto == "sin_datos":
        return f"🔎 {b('Mejor base')}: no pude calcular la comparación ahora (faltan datos). Intenta de nuevo en unos minutos."
    L = [f"🔎 {b('¿Hay una base mejor que ' + base + '?')}", it(f"Compara todas las acciones líquidas de EE. UU. y de la BVC por cuánto se espera que se muevan hasta el {fecha(fin)}."), ""]
    if mejores:
        m = mejores[0]
        L.append(f"⚠️ {b('Revisa la base')}: {esc(m['ticker'])} se espera que se mueva {n(m['vs_base'], 2)} veces lo que {esc(base)} (supera el {pct(ventaja, 0)} de ventaja que pido para cambiar).")
    else:
        L.append(f"✅ {b('Mantén ' + base)}: ninguna otra supera su movimiento esperado por el {pct(ventaja, 0)} que se necesita para justificar el cambio.")
    L.append("")
    for i, x in enumerate(filas[:top], 1):
        marca = " ← tu base" if x["es_base"] else ""
        fuente = "opciones + últimos 60 días" if x["iv"] else "últimos 60 días"
        L.append(f"{i}. {b(x['ticker'])} ({'BVC' if x['grupo'] == 'local' else 'EE. UU.'}): ±{n(x['mee'] * 100, 1)}% hasta el final ({n(x['vs_base'], 2)}× la base; {fuente}){marca}")
    L += ["", it("Es el movimiento ESPERADO (sube o baja): mide oportunidad y riesgo por igual. Más movimiento no garantiza ganar.")]
    return "\n".join(L)


def linea_ranking(rent: dict[str, Any]) -> str:
    if rent.get("mia") is None or rent.get("umbral") is None:
        return f"🏁 {b('Ranking')}: no me has dicho cómo vas. Escríbeme {b('voy 6,5 y el corte está en 11')} (tu rentabilidad y la del corte, en %) para que pueda aconsejarte."
    falta = rent["umbral"] - rent["mia"]
    return (f"🏁 {b('Ranking')}: vas {n(rent['mia'], 1, True)}% y el objetivo del corte es {n(rent['umbral'], 1, True)}%. "
            + (f"Te faltan {puntos(falta)}." if falta > 0 else f"Vas {puntos(-falta)} por encima del objetivo. 👍"))


def msg_relevo_color(ticker: str, previo: str, nuevo: str, graves: tuple[str, ...]) -> str:
    if nuevo in graves:
        return (f"⚠️ {b('Un relevo se puso en alerta: ' + ticker)}\nPasó de {NOMBRE_COLOR.get(previo, previo)} a {NOMBRE_COLOR.get(nuevo, nuevo)}. "
                "Ya no lo recomiendo como cambio hasta que se recupere.")
    return f"✅ {b('Un relevo se recuperó: ' + ticker)}\nPasó de {NOMBRE_COLOR.get(previo, previo)} a {NOMBRE_COLOR.get(nuevo, nuevo)}. Puede volver a la lista de relevos."


def msg_rank_guardado(mia: float, obj: float, g: float, multiplo: float | None, pide_ventaja: bool) -> str:
    L = [f"✅ {b('Ranking guardado')}: vas {n(mia, 1, True)}% y el objetivo es {n(obj, 1, True)}%."]
    if g > 0:
        L.append(f"Te faltan {puntos(g - 1.0)} (la regla suma 1 de margen: {n(g, 1)}). Para cambiar de acción, el nuevo tendría que moverse al menos "
                 f"{n(multiplo, 2)} veces lo que se mueve el tuyo.")
    else:
        L.append("Vas por encima del objetivo: lo mejor es mantener lo que tienes.")
    if pide_ventaja:
        L.append(it("Vas de primero: para decidir la venta del último día necesito la ventaja sobre el segundo. Escribe /rank <mi_rent> <del_primero> <del_segundo>."))
    L.append("Mira qué haría la regla con /banco")
    return "\n".join(L)


# ------------------------------------------------------------------ noticias de la BVC (vigía de cada 2 minutos)
def _evidencia_txt(ev: dict[str, Any] | None, costo: float) -> list[str]:
    """Lo que pasó en anuncios parecidos, en una frase, y si alcanzó a pagar el costo de entrar y salir."""
    if not ev:
        return ["No tengo suficientes casos parecidos medidos para decirte qué esperar: trátalo como una moneda al aire."]
    h, casos, media = ev["h"], ev["casos"], ev["media"]
    plazo = "la sesión siguiente" if h == 1 else f"las {h} sesiones siguientes"
    cuando = {("global", "+1"): "la acción ya había subido fuerte ese día", ("global", "-1"): "la acción ya había caído fuerte ese día",
              ("apertura", "texto+"): "el anuncio era positivo y el precio aún no había reaccionado",
              ("apertura", "texto-"): "el anuncio era negativo y el precio aún no había reaccionado"}.get((ev["grupo"], ev["clave"]), "el precio aún no había reaccionado")
    verbo = "subió" if media >= 0 else "bajó"
    L = [f"en {casos} anuncios oficiales (2023–2026) en los que {cuando}, en {plazo} {verbo} en promedio {b(pct(abs(media), 1))} frente al mercado "
         f"(subió {pct(ev['pct_pos'], 0)} de las veces)."]
    if media > 0:
        L.append(f"Entrar y salir cuesta cerca de {pct(costo, 1)}: " + ("alcanza a pagarlo." if ev.get("ic_lo", 0) > costo else "no alcanza a pagarlo."))
    return L


def _precio_txt(p: dict[str, Any] | None, abierto: bool) -> str:
    if p is None:
        return "💹 " + b("Precio") + ": no pude consultarlo ahora."
    z = p.get("z")
    mov = f"{esc(p['ticker'])} {'sube' if p['r_hoy'] >= 0 else 'baja'} {pct(abs(p['r_hoy']), 1)} hoy"
    if z is None or abs(z) < 1:
        extra = "todavía no reacciona" if abierto else "el mercado está cerrado: la reacción se verá en la próxima apertura"
    else:
        extra = f"es {veces(abs(z))} lo normal para esta acción: el precio ya reaccionó"
    return f"💹 {b('Precio')}: {mov} ({extra})." + it(" Precios con unos 15 min de retraso.")


def _impacto_txt(s: Any) -> str:
    """Impacto en porcentaje y con signo: + si la noticia empuja el precio hacia arriba, − si hacia abajo. Es lo que se movió en promedio la acción el día
    de un anuncio de ese tipo (medido), no una promesa; al lado va lo que la acción lleva hoy de verdad."""
    p = s.rec.get("pulso")
    hoy = f" · hoy la acción va {b(pct(p['r_hoy'], 1, True))}" if p else ""
    if s.impacto is None:
        return f"🎯 {b('Impacto estimado')}: {'positivo (+)' if s.sentido > 0 else 'negativo (−)'}, sin una cifra medida para este tipo{hoy}"
    return f"🎯 {b('Impacto estimado')}: {b(pct(s.impacto, 1, True))} {it('(lo que suele mover a la acción una noticia así)')}{hoy}"


def bloque_afectadas(afectadas: list[dict[str, Any]] | None) -> list[str]:
    """'Acciones de la BVC afectadas' por una noticia de empresa: cada una con su porcentaje estimado y su signo (+ sube, − baja)."""
    if not afectadas:
        return []
    def una(x: dict[str, Any]) -> str:
        cifra = pct(x["efecto"], 1, True) if x["efecto"] is not None else ("sube (+)" if x["sentido"] > 0 else "baja (−)")
        return f"{'🟢' if x['sentido'] > 0 else '🔴'} {b(x['ticker'])} {cifra}" + (" (la tienes)" if x["tengo"] else "")
    L = [f"📊 {b('Acciones de la BVC afectadas')}: " + " · ".join(una(x) for x in afectadas)]
    if len(afectadas) > 1:
        L.append(it("Las demás son su otra serie o empresas de su mismo grupo, según cuánto suelen moverse con ella."))
    return L


def msg_noticia_relevante(s: Any, ahora: dt.datetime, cfg: dict[str, Any], afectadas: list[dict[str, Any]] | None = None) -> str:
    """Aviso corto de una noticia RELEVANTE (no llega a alto impacto): qué salió y su impacto estimado en %, con signo."""
    from .noticias_bvc import ETIQUETA
    n0 = s.items[0]
    ahora_utc = ahora.astimezone(dt.timezone.utc)
    tengo = " · la tienes" if s.rec.get("tengo") else ""
    return "\n".join([f"📰 {b(s.emisor + ' (' + s.ticker + ')')} · {esc(ETIQUETA[s.cat])}{tengo}", f"«{esc(n0.titulo[:260])}»",
                      it(f"{', '.join(esc(x) for x in s.fuentes[:2])} · {hace(n0.ts.isoformat(), ahora_utc)}"), _impacto_txt(s), *bloque_afectadas(afectadas),
                      it("Noticia informativa: por sí sola no es motivo para comprar ni vender.")])


QUE_HACER_NOTICIA = {
    "COMPRAR": "Compra {t} ahora, con orden límite (nunca a mercado). En casos parecidos lo que vino después superó el costo.",
    "COMPRAR_MANANA": "Compra {t} en la próxima apertura ({cuando}, a las {hora}) con orden límite, si no abre ya disparada.",
    "NO_PERSEGUIR": "No compres {t} ahora: ya subió, y en casos así lo normal fue que devolviera parte en los días siguientes.",
    "NO_COMPRAR": "No compres {t} solo por esta noticia: en casos parecidos lo que subió después no alcanzó a pagar el costo de entrar y salir.",
    "POCO_LIQUIDA": "No operes {t}: se negocia muy poco y sería difícil vender después.",
    "MANTENER": "Ya tienes {t}: mantenla. No compres más por la noticia.",
    "NO_VENDER_PANICO": "Tienes {t} y ya cayó: no vendas por pánico. En casos así lo normal fue un rebote parcial. Mira /semaforo {t}.",
    "VIGILAR": "Tienes {t}: vigila. Si empieza a caer fuerte con mucho volumen, el semáforo te avisa.",
    "EVITAR": "No tienes {t}: no hagas nada, y no la compres estos días.",
}


def msg_noticia_bvc(s: Any, ahora: dt.datetime, cfg: dict[str, Any], sugerencia: dict[str, Any] | None = None, afectadas: list[dict[str, Any]] | None = None) -> str:
    """Aviso automático de una noticia de alto impacto de una empresa de la BVC: qué pasó, qué hacer según la evidencia, qué esperar y por cuánto tiempo."""
    from .noticias_bvc import ETIQUETA
    rec, n0 = s.rec, s.items[0]
    ahora_utc = ahora.astimezone(dt.timezone.utc)
    fuentes = ", ".join(esc(x) for x in s.fuentes[:3])
    L = [f"🚨 {b('Noticia de alto impacto: ' + s.emisor)}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}",
         f"📰 «{esc(n0.titulo[:300])}»", it(f"{fuentes} · {hace(n0.ts.isoformat(), ahora_utc)}"),
         f"{'📈' if s.sentido > 0 else '📉'} {b('Tipo')}: {esc(ETIQUETA[s.cat])}", _impacto_txt(s),
         *bloque_afectadas(afectadas), _precio_txt(rec.get("pulso"), C.mercado_abierto(ahora, cfg)), ""]
    cuando = fecha(rec["cuando"]) if rec.get("cuando") else ""
    L.append(f"👉 {b('Qué hacer')}: " + QUE_HACER_NOTICIA[rec["accion"]].format(t=esc(rec["ticker"]), cuando=cuando, hora=rec.get("hora", "")))
    L.append(f"📊 {b('Qué esperar')}: " + " ".join(_evidencia_txt(rec.get("evidencia"), rec["costo"])))
    if rec.get("tipico") and rec["tipico"].get("pct_fuerte") is not None:
        L.append(it(f"Los anuncios de este tipo movieron fuerte la acción solo {pct(rec['tipico']['pct_fuerte'], 0)} de las veces."))
    if s.sentido > 0 and not rec["tengo"] and rec["accion"] != "POCO_LIQUIDA" and rec.get("salida") and rec.get("stop"):
        intro = "Mantén" if rec["accion"].startswith("COMPRAR") else "Si aun así decides entrar, que sea corto: mantén"
        L.append(f"⏱ {b('Cuánto tiempo')}: {intro} máximo {rec['sesiones']} sesiones (vende a más tardar el {fecha(rec['salida'])}) y sal antes si cae {pct(rec['stop'], 0)} "
                 f"desde tu compra. En ese plazo esta acción suele moverse ±{pct(rec['rango'], 1)}.")
    if sugerencia:
        L += ["", f"🛒 {b('¿Buscas en qué meter plata?')} La acción de la BVC con más movimiento esperado es {b(sugerencia['ticker'])}: ±{pct(sugerencia['mee'], 1)} hasta el corte del "
                  f"{fecha(sugerencia['corte'])}. Detalle: /comprar"]
    L.append(it("Es una recomendación, no una garantía. La orden la das tú en trii."))
    return "\n".join(L)


def msg_que_comprar(banco: list[EntradaBanco], actual: str, mee_actual: float | None, decision: Decision, corte: dict[str, Any] | None, sesiones: int,
                    senales: list[dict[str, Any]], hora_orden: str, cfg: dict[str, Any], top: int = 5, n_estudio: int | None = None, liquidez_txt: str | None = None) -> str:
    """/comprar: qué acción de la BVC comprar si quieres moverte, qué esperar y por cuánto tiempo. Ordena por movimiento esperado hasta el próximo corte."""
    costo = cfg["noticias_bvc"]["recomendacion"]["costo_ida_vuelta"]
    L = [f"🛒 {b('¿Qué acción de la BVC comprar?')}", ""]
    if not banco or corte is None:
        L.append("Hoy no tengo ninguna candidata que cumpla (poco líquidas, en alerta o sin datos). Prueba de nuevo en unos minutos con /actualizar.")
        return "\n".join(L)
    hasta = f"el corte del {fecha(corte['fecha'])}" if corte.get("top_pct") else f"el final del concurso ({fecha(corte['fecha'])})"
    L.append(f"{b('Las que más pueden moverse')} hasta {hasta} (quedan {sesiones} sesiones):")
    for i, e in enumerate(banco[:top], 1):
        c = e.c
        extra = []
        if mee_actual:
            extra.append(f"{n(c.mee / mee_actual, 1)} veces lo que se mueve tu cartera")
        if c.valor_negociado_mm:
            extra.append(f"negocia ${n(c.valor_negociado_mm / 1000, 1)} mil millones al día")
        L.append(f"{i}. {EMOJI[c.color]} {b(c.ticker)}: se espera ±{pct(c.mee, 1)}" + (" · " + " · ".join(extra) if extra else ""))
    m = banco[0].c
    if len(banco) > 1 and banco[1].c.mee > m.mee:
        L.append(it("Las que se mueven casi igual van empatadas; entre ellas va primero la que menos se parece a lo que ya tienes."))
    if mee_actual and m.mee < mee_actual:
        L.append(it(f"Ojo: todas se mueven menos que tu cartera de hoy (±{pct(mee_actual, 1)}). Pasarte a la BVC baja el riesgo, y también lo que puedes remontar."))
    L += ["", f"👉 {b('Si vas a comprar una')}: {b(m.ticker)}. {b('Qué esperar')}: que se mueva cerca de ±{pct(m.mee, 1)} hasta {hasta}, para arriba o para abajo "
              f"(no sé hacia dónde: más movimiento es más oportunidad y más riesgo). {b('Cuánto tiempo')}: hasta ese corte; entrar y salir cuesta cerca de {pct(costo, 1)}, "
              "así que no la compres para venderla en uno o dos días.",
          *([liquidez_txt] if liquidez_txt else []),
          f"📝 {b('Cómo comprar')}: siempre con orden límite (tú pones el precio máximo), nunca \"a mercado\". Decide antes de abrir la bolsa el precio de entrada y a cuánto sales si sale mal.",
          "", *bloque_decision(decision, actual, hora_orden)]
    if senales:
        L += ["", b("Noticias fuertes de las últimas 24 horas")]
        for x in senales[-4:][::-1]:
            L.append(f"{'📈' if x['sentido'] > 0 else '📉'} {b(x['ticker'])}: {esc(x['titulo'][:110])} {it('(' + esc(x['fuente']) + ')')}")
        medido = f" (medido en {n_estudio:,} anuncios oficiales)".replace(",", ".") if n_estudio else ""
        L.append(it(f"Comprar después de una noticia no pagó el costo en promedio{medido}: úsalas para entender el movimiento, no para perseguirlo."))
    L.append(it("Solo acciones de la BVC que se negocian bien en trii (liquidez revisada). Recomendación, no garantía: la orden la das tú en trii."))
    return "\n".join(L)


def msg_noticias_bvc_recientes(senales: list[dict[str, Any]], salud: dict[str, bool] | None, ahora: dt.datetime) -> str:
    """/noticias sin ticker: las noticias de alto impacto de la BVC que el vigía avisó en las últimas 24 horas."""
    L = [f"📰 {b('Noticias fuertes de la BVC')} {it('(últimas 24 horas)')}"]
    if not senales:
        L.append("No ha salido ninguna de alto impacto. Reviso las fuentes cada 2 minutos y te aviso solo.")
    for x in senales[::-1][:8]:
        cuando = hace(x["ts"] if "+" in x["ts"] or x["ts"].endswith("Z") else x["ts"] + "+00:00", ahora.astimezone(dt.timezone.utc))
        L.append(f"{'📈' if x['sentido'] > 0 else '📉'} {b(x['ticker'])} {it(cuando)}: {esc(x['titulo'][:140])} {it('(' + esc(x['fuente']) + ')')}")
    if salud:
        mal = [k for k, v in salud.items() if not v]
        L.append(it("Fuentes: " + (f"las {len(salud)} responden ✅" if not mal else "sin respuesta de " + ", ".join(mal) + " ⚠️")))
    L.append(it("Para una empresa: /noticias ECOPETROL"))
    return "\n".join(L)


# ------------------------------------------------------------------ macroeconomía y cambios sugeridos
def _mov_factor(x: dict[str, Any]) -> str:
    """'el petróleo (Brent) cae 3,2% (2,1 veces lo normal)'."""
    verbo = "sube" if x["cambio"] > 0 else "cae"
    cuanto = f"{n(abs(x['cambio']), 2)} puntos" if x["en_puntos"] else pct(abs(x["cambio"]), 1)
    return f"{x['nombre']} {verbo} {cuanto}" + (f" ({veces(abs(x['z']))} lo normal)" if abs(x["z"]) >= 1.5 else "")


def _lista_impacto(filas: list[dict[str, Any]], max_n: int = 5) -> str:
    return ", ".join(f"{b(x['ticker'])}{' (la tienes)' if x['tengo'] else ''} {pct(x['efecto'], 1, True)}" for x in filas[:max_n]) + ("…" if len(filas) > max_n else "")


def bloque_impacto(factor: dict[str, Any], imp: dict[str, Any]) -> list[str]:
    """Quién gana y quién pierde con el movimiento de hoy de un factor (según lo medido en los últimos 12 meses)."""
    L = []
    if imp["beneficiadas"]:
        L.append(f"🟢 {b('Beneficiadas')}: {_lista_impacto(imp['beneficiadas'])}")
    if imp["afectadas"]:
        L.append(f"🔴 {b('Afectadas')}: {_lista_impacto(imp['afectadas'])}")
    if not L:
        L.append("Ninguna acción de la BVC ni de tu cartera ha mostrado una relación firme con este factor en los últimos 12 meses.")
    return L


def msg_macro_movimiento(factor: dict[str, Any], imp: dict[str, Any], ahora: dt.datetime) -> str:
    """Aviso automático: un factor macro se está moviendo hoy mucho más de lo normal."""
    L = [f"🌍 {b('Movimiento macro fuerte')}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}", f"{b('¿Qué pasó?')} Hoy {esc(_mov_factor(factor))}.", "",
         b("¿A quién le pega?") + " " + it("(efecto estimado de hoy: lo que cada acción suele moverse con este factor)"), *bloque_impacto(factor, imp)]
    if imp["mias"]:
        L += ["", f"💼 {b('En tu cartera')}: " + ", ".join(f"{b(x['ticker'])} {pct(x['efecto'], 1, True)}" + (" (por el dólar: en trii se ve en pesos)" if x.get("directo") else "") for x in imp["mias"]) + "."]
    L += ["", f"👉 {b('Qué hacer')}: nada por reflejo. Es para que entiendas por qué se mueven tus acciones: cuando lees esto el precio ya lo está recogiendo, y en la BVC "
              "comprar lo que acaba de subir ha salido peor que esperar.", it("Todos los factores ahora: /macro")]
    return "\n".join(L)


def msg_macro_titular(g: dict[str, Any], tab: list[dict[str, Any]], impactos: dict[str, dict[str, Any]], ahora: dt.datetime,
                      por_punto: dict[str, dict[str, Any]] | None = None) -> str:
    """Aviso automático: titular macro importante + cómo están reaccionando los mercados + acciones expuestas."""
    ahora_utc = ahora.astimezone(dt.timezone.utc)
    L = [f"🏦 {b('Noticia macro: ' + g['nombre'])}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}"]
    for x in g["items"]:
        L.append(f"📰 {esc(x.titulo[:220])} {it('(' + esc(x.fuente) + ', ' + hace(x.ts.isoformat(), ahora_utc) + ')')}")
    ligados = [x for x in tab if x["clave"] in g["factores"]]
    movidos = [x for x in ligados if x["notable"]]
    L.append("")
    if movidos:
        L.append(f"{b('Cómo reacciona el mercado ahora')}: " + "; ".join(esc(_mov_factor(x)) for x in movidos) + ".")
        for x in movidos:
            imp = impactos.get(x["clave"])
            if imp and (imp["beneficiadas"] or imp["afectadas"]):
                L += [it(f"Por {x['nombre']}:"), *bloque_impacto(x, imp)]
    else:
        nombres = ", ".join(x["nombre"] for x in ligados) or "los mercados"
        L.append(f"{b('Cómo reacciona el mercado ahora')}: por ahora {esc(nombres)} {'se mueve' if len(ligados) == 1 else 'se mueven'} dentro de lo normal. "
                 "El titular todavía no está moviendo los precios.")
    quietos = [x for x in ligados if not x["notable"]]
    listos = False
    for x in quietos:                                                                  # aunque el factor no se haya movido: a quién le pegaría y cuánto
        imp = (por_punto or {}).get(x["clave"])
        if imp and (imp["beneficiadas"] or imp["afectadas"]):
            if not listos:
                L += ["", f"📊 {b('Acciones de la BVC afectadas')} " + it("(cuánto suele moverse cada una por cada 1% del factor)")]
                listos = True
            sube = _lista_impacto(imp["beneficiadas"], 6)
            baja = _lista_impacto(imp["afectadas"], 6)
            L.append(f"Si {esc(x['nombre'])} sube 1%: " + " · ".join(p for p in (("🟢 " + sube) if sube else "", ("🔴 " + baja) if baja else "") if p)
                     + it(" (si baja 1%, lo contrario)"))
    if not movidos and not listos:
        L.append("Ninguna acción de la BVC ha mostrado una relación firme con estos factores en los últimos 12 meses.")
    L += ["", f"👉 {b('Qué hacer')}: nada por el titular. Si de verdad pesa, lo verás en los precios y te aviso con el movimiento y las acciones afectadas.",
          it("Todos los factores ahora: /macro")]
    return "\n".join(L)


def msg_macro_tablero(tab: list[dict[str, Any]], impactos: dict[str, dict[str, Any]], ahora: dt.datetime, titulo: str = "Macro ahora") -> str:
    """/macro y el resumen de la mañana: cada factor y, para los que se movieron, quién se beneficia y quién se afecta."""
    if not tab:
        return f"🌍 {b(titulo)}: no pude consultar los datos macro ahora. Prueba de nuevo en unos minutos."
    L = [f"🌍 {b(titulo)}  {it(f'{fecha(ahora)} {ahora:%H:%M}')}"]
    for x in sorted(tab, key=lambda x: -abs(x["z"])):
        marca = "🔥" if x["fuerte"] else ("▫️" if not x["notable"] else ("🔺" if x["cambio"] > 0 else "🔻"))
        txt = _mov_factor(x)
        L.append(f"{marca} {esc(txt[0].upper() + txt[1:])}" + ("" if x["notable"] else " — normal"))
    movidos = [x for x in sorted(tab, key=lambda x: -abs(x["z"])) if x["notable"]]
    if not movidos:
        L += ["", "Nada se mueve más de lo normal: hoy la macro no está empujando a tus acciones."]
    for x in movidos[:3]:
        imp = impactos.get(x["clave"])
        if imp and (imp["beneficiadas"] or imp["afectadas"]):
            L += ["", it(f"Por {x['nombre']}:"), *bloque_impacto(x, imp)]
    L += ["", it("Efecto estimado = lo que cada acción suele moverse el mismo día con ese factor (medido en los últimos 12 meses). Es contexto, no una señal de compra.")]
    return "\n".join(L)


def msg_cambio(c: dict[str, Any], rel: dict[str, Any] | None = None) -> str:
    """Aviso automático "cambia esta acción por esta". Sólo sale con un motivo que justifica pagar el costo del cambio (hoy: liquidez mala en trii)."""
    L = [f"🔁 {b('Cambio sugerido: ' + c['de'] + ' → ' + c['a'])}", "",
         f"{b('¿Por qué?')} {esc(c['motivo'])}",
         f"{b('¿Por qué ' + c['a'] + '?')} {esc(c['por_que_a'])}"]
    if c.get("liquidez_a"):
        L.append(esc(c["liquidez_a"]))
    L += ["", f"👉 {b('Cómo hacerlo')}: vende {esc(c['de'])} con orden límite (pon tu precio y ten paciencia: si la vendes a mercado regalas plata) y compra {esc(c['a'])} "
              "también con orden límite. Luego escríbeme: " + b(f"vendí {c['de'].lower()}") + " y " + b(f"compré … {c['a'].lower()}") + "."]
    if rel:
        L.append(it(f"No elijo \"la que más viene subiendo\": en la BVC, las que más subieron en {rel['k']} días rindieron {pct(abs(rel['dif']), 1)} menos que las rezagadas "
                    f"en los {rel['k']} siguientes (medido en {rel['n']} períodos)."))
    L.append(it("Es una recomendación, no una garantía. La orden la das tú en trii."))
    return "\n".join(L)


def msg_resumen_manana(tablero_txt: str, fechas: list[str], avisos_liq: list[str], cambios: list[dict[str, Any]], hora_apertura: str, sin_ranking: bool) -> str:
    """Resumen automático antes de abrir la bolsa: qué pasó de noche, qué hay hoy y qué revisar antes de operar."""
    L = [f"🌅 {b('Antes de abrir')} {it('(la bolsa abre a las ' + hora_apertura + ')')}", "", tablero_txt.replace("Macro ahora", "Qué pasó mientras dormías", 1)]
    if fechas:
        L += ["", b("Hoy en el calendario"), *fechas]
    if avisos_liq:
        L += ["", b("Ojo con lo que se negocia poco"), *avisos_liq]
    if cambios:
        L += ["", b("Cambios sugeridos"), *[f"🔁 {b(c['de'] + ' → ' + c['a'])}: {esc(c['motivo'])}" for c in cambios]]
    L += ["", f"📝 {b('Si vas a operar hoy')}: decide AHORA qué, a qué precio máximo y a cuánto sales si sale mal. Órdenes límite, nunca a mercado. Si no tienes un motivo claro, no operes."]
    if sin_ranking:
        L.append("🏁 Aún no sé cómo vas: escríbeme " + b("voy -0,1 y el corte está en 1,5") + " (con tus números) para decirte si conviene cambiar.")
    return "\n".join(L)


AYUDA = """🤖 <b>Cómo usar este bot</b>
Vigilo tu cartera y las noticias de la <b>Bolsa de Colombia (BVC)</b> y te aviso solo cuando pasa algo importante. <b>Yo nunca compro ni vendo</b>: las órdenes las das tú en trii.

<b>Escríbeme normal, sin comandos</b>
• <i>compré 300 argos a 21500</i> · <i>compré 20 millones de ecopetrol</i>
• <i>vendí tesla</i> · <i>vendí 100 argos</i>
• <i>voy 3,5</i> (tu rentabilidad) · <i>el corte está en 8</i>
• <i>tengo 1200 nuco y 300 argos</i> (para decirme lo que tienes, sin que cuente como compra)
• <i>me equivoqué</i> (deshace lo último que registraste)
No importa si pones el precio en pesos o en dólares, o el total en vez del precio: lo detecto y te muestro lo que entendí.

<b>Lo que puedes preguntar</b> (o toca un botón de /menu)
/comprar — qué acción de la BVC comprar, qué esperar y por cuánto tiempo (sólo las que se negocian bien en trii)
/noticias — noticias fuertes de la BVC (o /noticias ECOPETROL)
/macro — petróleo, dólar, Wall Street, Brasil… y qué acciones ganan o pierden con eso
/semaforo — ¿tus acciones están bien?
/cartera — tus acciones con precios de hoy
/estado — cómo vas en el concurso
/detalle · /catalizadores · /informe · /actualizar — para mirar más a fondo

<b>Los avisos que te mando solo</b> (¿no te llegan en este chat? escribe /avisos)
🚨 noticia de alto impacto de una empresa de la BVC (reviso cada minuto), con su impacto estimado en % (+ sube, − baja)
📰 noticia relevante (aviso corto, también con su impacto en %)
🌍 movimiento o noticia macro fuerte, con las acciones beneficiadas y afectadas
🔁 cambio sugerido (por ejemplo, salir de una acción que casi no se negocia en trii)
🌅 resumen antes de abrir · 🔴 una acción tuya cae fuerte · 🌙 radar de las 19:30

<b>Los colores del semáforo</b>
🟢 todo normal · 🔵 cae el mercado, no tu acción · 🟡 cae sola: vigila
🔴 cae sola con mala noticia: alerta · ⚫ no se recupera: ojo, pero no vendas solo por eso"""
