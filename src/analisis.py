"""Los tres análisis que juntan TODOS los filtros en una sola respuesta (antes había que pedir cada filtro por separado y armar la conclusión a mano):

  · `resp_ganar`      (/ganar)            — ¿con cuál tengo más opción de ganar? Movimiento esperado, semáforo, macro, noticias, costo y liquidez de tu base.
  · `resp_reemplazo`  (/reemplazo NUCO)   — ¿cuál puede reemplazar a esa acción? Las de la BVC que más se le acercan en movimiento, con su liquidez en trii.
  · `resp_revisar`    (/revisar ECOPETROL)— ¿la compro hoy? Precio, macro, noticias del día y liquidez de ESA acción, con lo que tiene a favor y en contra.

Ninguno predice hacia dónde va el precio: el movimiento esperado es para arriba o para abajo, y así se dice siempre."""
from __future__ import annotations

from typing import Any

import pandas as pd

from . import concurso as C
from . import formato as F
from . import liquidez as LQ
from . import servicio as S

NIVEL_TXT = {LQ.BUENA: "✅ liquidez buena", LQ.JUSTA: "🟡 liquidez justa", LQ.MALA: "⛔ liquidez mala", LQ.SIN_DATO: "⛔ liquidez sin comprobar"}


def _costo(cfg: dict[str, Any]) -> float:
    return float(cfg["noticias_bvc"]["recomendacion"]["costo_ida_vuelta"])


def _fin(cfg: dict[str, Any]) -> Any:
    return pd.Timestamp(cfg["concurso"]["final"]["fecha"]).date()


def _filas(ctx: S.Contexto, base: str) -> list[dict[str, Any]]:
    from .seleccion import ranking_base
    return ranking_base(ctx.f, ctx.cfg, ctx.ahora, base, todas=True)


def _valores(ctx: S.Contexto, est: Any) -> dict[str, float]:
    """{acción que tienes: cuánto vale hoy en pesos}. Vacío si no se puede valorar."""
    try:
        from . import cartera as CA
        filas = CA.valorar(ctx.f, est.d["cartera"], ctx.cfg, CA.trm(ctx.f, ctx.cfg))
        return {x["ticker"]: float(x["valor_cop"]) for x in filas if x["valor_cop"] == x["valor_cop"]}
    except Exception:                                                                  # noqa: BLE001
        return {}


def _casi(x: dict[str, Any], cfg: dict[str, Any]) -> bool:
    """Se queda fuera del filtro sólo por continuidad y por poco (hasta 5 puntos): se puede nombrar como alternativa, diciendo que no pasa todo el filtro."""
    c, k = cfg["liquidez"]["continuidad"], x.get("continuidad")
    return bool(x.get("a_ratos") and k and k["mediana"] >= c["min"] - 0.05 and k["flojo"] >= c["min_dia_flojo"] - 0.05)


def _liq_txt(x: dict[str, Any]) -> str:
    k = x.get("continuidad")
    if x["nivel"] == LQ.BUENA:
        return "✅ liquidez buena" + (f" (operaciones en el {k['mediana']:.0%} del día)" if k else "")
    if x.get("a_ratos") and k:
        return f"🟡 liquidez justa: se negocia a ratos ({k['mediana']:.0%} del día)"
    return NIVEL_TXT[x["nivel"]]


def _parte_del_dia(valor: float | None, x: dict[str, Any]) -> str:
    if not valor or not x.get("liquidez_mm"):
        return ""
    p = valor / (x["liquidez_mm"] * 1e6)
    return f" · tu posición sería {'menos del 1%' if p < 0.01 else F.pct(p, 1)} de lo que negocia en un día"


def _consejo_rank(est: Any) -> str:
    mia, corte = est.rent.get("mia"), est.rent.get("umbral")
    if mia is None or corte is None:
        return (f"🏁 No sé cómo vas en el ranking. Escríbeme {F.b('voy 6,5 y el corte está en 11')}: si vas por encima conviene cuidar lo ganado; "
                "si vas por debajo, mantener lo que más se mueve.")
    if mia >= corte:
        return f"🏁 Vas {F.n(mia, 1, True)}% y el corte está en {F.n(corte, 1, True)}%: vas por encima. Cuida lo ganado; no cambies por cambiar."
    return (f"🏁 Vas {F.n(mia, 1, True)}% y el corte está en {F.n(corte, 1, True)}%: vas por debajo. Para alcanzarlo te conviene mantener lo que más se mueve, "
            "no pasarte a algo más quieto.")


# ------------------------------------------------------------------ /reemplazo
def resp_reemplazo(ctx: S.Contexto, args: list[str]) -> str:
    est = S.leer_estado(ctx)
    t = S.resolver(ctx, " ".join(args), est.tenidos()) if args else est.activo
    if t not in S.universo_permitido(ctx.cfg) and t not in est.tenidos():
        return f"❌ No reconocí la acción «{F.esc(' '.join(args))}». Ejemplo: {F.b('/reemplazo nuco')}"
    filas = _filas(ctx, t)
    yo = next((x for x in filas if x["es_base"]), None)
    if yo is None:
        return f"🔁 No pude calcular cuánto se mueve {F.esc(t)} ahora (faltan datos). Intenta de nuevo en unos minutos."
    cands = [x for x in filas if not x["es_base"] and x["grupo"] == "local" and (x["nivel"] == LQ.BUENA or _casi(x, ctx.cfg))][:5]
    if not cands:
        return f"🔁 Ahora mismo ninguna acción de la BVC pasa el filtro de liquidez para reemplazar a {F.esc(t)}."
    valor = _valores(ctx, est).get(t)
    L = [f"🔁 {F.b('¿Cuál puede reemplazar a ' + t + '?')}",
         F.it(f"Las de la BVC que más pueden moverse hasta el final ({F.fecha(_fin(ctx.cfg))}). {t} se espera que se mueva ±{F.n(yo['mee'] * 100, 1)}%."), ""]
    for i, x in enumerate(cands, 1):
        L.append(f"{i}. {F.b(x['ticker'])}: ±{F.n(x['mee'] * 100, 1)}% ({F.n(x['vs_base'], 2)} veces {F.esc(t)}) · {_liq_txt(x)}{_parte_del_dia(valor, x)}")
    cerca, segura = cands[0], next((x for x in cands if x["nivel"] == LQ.BUENA), None)
    L.append("")
    if cerca["vs_base"] >= 1:
        L.append(f"👉 {F.b(cerca['ticker'])} se mueve incluso más que {F.esc(t)}.")
    else:
        L.append(f"👉 Ninguna la iguala. La que más se le acerca es {F.b(cerca['ticker'])}: conserva cerca del {F.pct(cerca['vs_base'], 0)} de su movimiento.")
    if cerca["nivel"] != LQ.BUENA:
        L.append(f"Ojo: {F.esc(cerca['ticker'])} se queda fuera del filtro de liquidez por poco (se negocia a ratos). Cabe, pero sólo con orden límite y paciencia.")
        if segura:
            L.append(f"Si quieres una que pase todo el filtro: {F.b(segura['ticker'])}, aunque sólo conserva cerca del {F.pct(segura['vs_base'], 0)} del movimiento.")
    L += ["", F.b("Antes de cambiar:"),
          f"• Cuesta cerca de {F.pct(_costo(ctx.cfg), 1)} entre salir de una y entrar a la otra.",
          "• El movimiento esperado es para arriba o para abajo: no dice hacia dónde.",
          f"• Sólo cambiaría {F.esc(t)} si te está costando venderla en trii o si ya vas por encima del corte y quieres cuidar lo ganado.",
          "• Si cambias: vende por partes con orden límite y compra la otra a medida que te ejecuten.",
          "", _consejo_rank(est)]
    return "\n".join(L)


# ------------------------------------------------------------------ /ganar
def resp_ganar(ctx: S.Contexto) -> str:
    est = S.leer_estado(ctx)
    base = est.activo
    filas = _filas(ctx, base)
    yo = next((x for x in filas if x["es_base"]), None)
    if yo is None:
        return "🏆 No pude hacer el análisis ahora (faltan datos de tu acción principal). Intenta de nuevo en unos minutos."
    liquidas = [x for x in filas if not x["es_base"] and x["grupo"] == "local" and x["nivel"] == LQ.BUENA]
    mejor = liquidas[0] if liquidas else None
    L = [f"🏆 {F.b('¿Con cuál tengo más opción de ganar?')}",
         F.it("Ningún filtro predice hacia dónde va el precio. Esto es lo que dicen todos juntos:"), ""]
    # 1) movimiento
    mov = f"📏 {F.b('Movimiento hasta el final')} ({F.fecha(_fin(ctx.cfg))}): {F.esc(base)} ±{F.n(yo['mee'] * 100, 1)}%."
    if mejor:
        mov += f" La mejor de las que se negocian bien, {F.esc(mejor['ticker'])}, ±{F.n(mejor['mee'] * 100, 1)}% ({F.n(mejor['vs_base'], 2)} veces)."
    L.append(mov)
    # 2) semáforo
    try:
        res, _, _ = S.evaluar_activo(ctx.f, base, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor)
        cuando = "hoy" if (res.fecha is None or res.fecha >= ctx.ahora.date()) else f"en la última sesión ({F.fecha(res.fecha)})"
        L.append(f"🚦 {F.b('Semáforo')}: {F.esc(base)} en {F.EMOJI.get(res.color, '')} {F.esc(res.color)}; {cuando} {F.pct(res.r_hoy, 1, True)}.")
    except Exception:                                                                  # noqa: BLE001 — un filtro sin datos se dice, no tumba el análisis
        L.append(f"🚦 {F.b('Semáforo')}: no pude revisarlo ahora.")
    # 3) macro
    try:
        tab, imp = S.panorama_macro(ctx)
        mios = [(next(f["nombre"] for f in tab if f["clave"] == k), x["efecto"]) for k, i in imp.items() for x in i["mias"] if x["ticker"] == base]
        if mios:
            L.append(f"🌍 {F.b('Macro hoy')}: " + "; ".join(f"{F.esc(nom)} le suma {F.pct(e, 1, True)} estimado" if e > 0 else f"{F.esc(nom)} le resta {F.pct(e, 1, True)} estimado"
                                                     for nom, e in mios) + ".")
        else:
            L.append(f"🌍 {F.b('Macro hoy')}: nada que esté empujando a {F.esc(base)} más de lo normal.")
    except Exception:                                                                  # noqa: BLE001
        L.append(f"🌍 {F.b('Macro hoy')}: no pude revisarla ahora.")
    # 4) noticias y costo
    n = len([x for x in S.senales_recientes(24, ctx.ahora) if x.get("ticker") in est.tenidos()])
    cuantas = "ninguna noticia fuerte" if n == 0 else ("1 noticia fuerte" if n == 1 else f"{n} noticias fuertes")
    L.append(f"📰 {F.b('Noticias')}: {cuantas} de tus acciones en 24 horas. "
             f"Comprar por la noticia del día no paga el costo (lo medí con casi 1.900 anuncios).")
    L.append(f"💸 {F.b('Costo de cambiar')}: cerca de {F.pct(_costo(ctx.cfg), 1)} cada vez.")
    # 5) liquidez
    valores = _valores(ctx, est)
    lq = LQ.medir(ctx.f, base, ctx.cfg)
    L.append(f"💧 {F.b('Liquidez en trii')}: {F.esc(base)} — {NIVEL_TXT[lq['nivel']]}{_parte_del_dia(valores.get(base), dict(liquidez_mm=lq.get('mediana_mm')))}.")
    # conclusión
    L.append("")
    ventaja = ctx.cfg["seleccion_base"]["ventaja_minima"]
    if mejor is None or mejor["mee"] < yo["mee"]:
        L.append(f"👉 {F.b('La que más opción te da de quedar arriba es la que ya tienes: ' + base)}. Es la que más puede moverse.")
        if lq["nivel"] != LQ.BUENA:
            L.append("Su punto débil es la liquidez: no compres más, y si sales hazlo con orden límite y por partes. Cambiarla por una más fácil de vender "
                     f"te deja con cerca del {F.pct(mejor['vs_base'], 0)} del movimiento." if mejor else
                     "Su punto débil es la liquidez: no compres más, y si sales hazlo con orden límite y por partes.")
    elif mejor["mee"] >= yo["mee"] * (1 + ventaja):
        L.append(f"👉 {F.b(mejor['ticker'] + ' puede moverse bastante más que ' + base)} ({F.n(mejor['vs_base'], 2)} veces) y se negocia bien en trii: vale la pena "
                 f"revisar el cambio. Mira {F.b('/reemplazo ' + base.lower())}.")
    else:
        L.append(f"👉 {F.b('Mantén ' + base)}: {F.esc(mejor['ticker'])} se mueve un poco más, pero la diferencia no paga el costo de cambiar.")
    otras = []
    for t in sorted(est.tenidos() - {base}):
        l = LQ.medir(ctx.f, t, ctx.cfg)
        if l["nivel"] != LQ.BUENA:
            otras.append(f"{F.esc(t)} ({NIVEL_TXT[l['nivel']].split(' ', 1)[1]})")
    if otras:
        L.append("• También tienes " + ", ".join(otras) + ": no hay afán de venderlas, pero cuando salgas que sea sin prisa y con orden límite.")
    if mejor:
        L.append(f"• Si tienes efectivo libre: la que más puede moverse entre las que se negocian bien es {F.b(mejor['ticker'])}. Antes de comprarla, "
                 f"pídeme {F.b('/revisar ' + mejor['ticker'].lower())}.")
    L += ["", _consejo_rank(est), "", F.it("Movimiento esperado = cuánto puede subir o bajar. Más movimiento es más oportunidad y más riesgo; no garantiza ganar.")]
    return "\n".join(L)


# ------------------------------------------------------------------ /revisar
def resp_revisar(ctx: S.Contexto, args: list[str]) -> str:
    est = S.leer_estado(ctx)
    if not args:
        return f"{F.b('Cómo usar /revisar')}\nEscribe la acción que estás pensando comprar: {F.b('/revisar ecopetrol')}. Te digo qué tiene hoy a favor y en contra."
    t = S.resolver(ctx, " ".join(args), est.tenidos())
    if t not in S.universo_permitido(ctx.cfg) and t not in est.tenidos():
        return f"❌ {F.esc(t)} no está permitida en el concurso (o está en la lista negra), o no reconocí el nombre."
    pros: list[str] = []
    contras: list[str] = []
    L = [f"🔍 {F.b('¿Compro ' + t + ' hoy?')}  {F.it(F.fecha(ctx.ahora.date()) + ' ' + ctx.ahora.strftime('%H:%M'))}", ""]
    # precio y semáforo
    try:
        res, _, _ = S.evaluar_activo(ctx.f, t, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor)
        de_hoy = res.fecha is None or res.fecha >= ctx.ahora.date()
        cuando = "hoy" if de_hoy else f"en la última sesión ({F.fecha(res.fecha)})"
        verbo = ("sube" if res.r_hoy >= 0 else "baja") if de_hoy else ("subió" if res.r_hoy >= 0 else "bajó")
        L.append(f"📈 {F.b('Precio')}: {verbo} {F.pct(abs(res.r_hoy), 1)} {cuando}. Semáforo {F.EMOJI.get(res.color, '')} {F.esc(res.color)}.")
        if res.color in ("ROJO", "NEGRO"):
            contras.append(f"el semáforo está en {res.color}")
        if res.z >= 2:
            contras.append("viene de una subida fuerte, y en la BVC las que saltan así suelen devolver parte en los días siguientes")
        try:
            mee, _, _ = S.mee_de(ctx.f, t, res.sigma20, ctx.ahora, ctx.cfg)
            corte = C.corte_vigente(ctx.ahora, ctx.cfg)
            if corte and mee:
                L.append(f"📏 {F.b('Cuánto puede moverse')}: cerca de ±{F.n(mee * 100, 1)}% hasta el corte del {F.fecha(corte['fecha'])} (para arriba o para abajo).")
        except Exception:                                                              # noqa: BLE001
            pass
    except Exception:                                                                  # noqa: BLE001
        L.append(f"📈 {F.b('Precio')}: no pude leerlo ahora.")
    # macro
    try:
        from . import macro as M
        tab = [x for x in M.tablero(ctx.f, ctx.cfg, ctx.ahora) if x["cuando"] == "hoy"]
        sens = M.sensibilidades(ctx.f, ctx.cfg, list(dict.fromkeys(S._tickers_macro(ctx, est) + [t])))
        efectos = [(x["nombre"], x["cambio"], sens.get(x["clave"], {}).get(t, {}).get("beta", 0.0) * x["cambio"]) for x in tab]
        efectos = sorted([e for e in efectos if abs(e[2]) >= ctx.cfg["macro_vivo"]["efecto_minimo"]], key=lambda e: -abs(e[2]))
        if efectos:
            L.append(f"🌍 {F.b('Macro hoy')}: " + "; ".join(f"{F.esc(nom)} {'sube' if c >= 0 else 'cae'} {F.pct(abs(c), 1)} → a {F.esc(t)} suele {'sumarle' if e > 0 else 'restarle'} "
                                                     f"{F.pct(abs(e), 1)}" for nom, c, e in efectos[:3]) + ".")
            total = sum(e for _, _, e in efectos)
            if total <= -0.005:
                contras.append("la macro de hoy la empuja hacia abajo (" + ", ".join(nom for nom, _, e in efectos if e < 0) + ")")
            elif total >= 0.005:
                pros.append("la macro de hoy la favorece (" + ", ".join(nom for nom, _, e in efectos if e > 0) + ")")
        else:
            L.append(f"🌍 {F.b('Macro hoy')}: nada que le pegue a {F.esc(t)} más de lo normal.")
    except Exception:                                                                  # noqa: BLE001
        L.append(f"🌍 {F.b('Macro hoy')}: no pude revisarla ahora.")
    # noticias
    try:
        filas, _ = S.buscar_noticias(ctx, horas=24)
        mias = [x for x in filas if x["ticker"] == t][:5]
        if mias:
            L.append(f"📰 {F.b('Noticias de las últimas 24 horas')}:")
            for x in mias:
                marca = "🟢" if x["sentido"] > 0 else ("🔴" if x["sentido"] < 0 else "⚪")
                L.append(f"  {marca} {F.esc(x['titulo'][:150])} {F.it('(' + F.esc(('oficial, ' if x['oficial'] else '') + x['fuente']) + ')')}")
            malas, buenas = sum(x["sentido"] < 0 for x in mias), sum(x["sentido"] > 0 for x in mias)
            if malas > buenas:
                contras.append("las noticias del día pintan más negativas que positivas")
            elif buenas > malas:
                pros.append("las noticias del día pintan más positivas que negativas")
            L.append(F.it("  Leo el titular, no la nota completa: ⚪ = el titular no dice si es bueno o malo."))
        else:
            L.append(f"📰 {F.b('Noticias')}: no encontré noticias de {F.esc(t)} en las últimas 24 horas (ni anuncios oficiales en la Superfinanciera).")
    except Exception:                                                                  # noqa: BLE001
        L.append(f"📰 {F.b('Noticias')}: no pude consultarlas ahora.")
    # liquidez
    lq = LQ.medir(ctx.f, t, ctx.cfg)
    L.append(F.esc(LQ.frase(lq, None, ctx.cfg)))
    es_bvc = t in ctx.cfg["universe"]["local"]
    apta = lq["nivel"] == LQ.BUENA and es_bvc
    if apta:
        pros.append("se compra y se vende fácil en trii")
    # veredicto
    L.append("")
    if pros:
        L.append("✅ A favor: " + F.esc("; ".join(pros)) + ".")
    if contras:
        L.append("⚠️ En contra: " + F.esc("; ".join(contras)) + ".")
    if not apta:
        L.append(f"👉 {F.b('No la compraría')}: " + ("no pasa el filtro de liquidez en trii." if es_bvc else "no es de la BVC y en trii se negocia poco."))
    elif len(contras) > len(pros) - 1:                                                 # la liquidez sola no alcanza: hace falta algo más a favor que en contra
        L.append(f"👉 {F.b('Hoy tiene más en contra que a favor')}. Si no tienes afán, espera; nada obliga a entrar hoy.")
    else:
        L.append(f"👉 {F.b('Hoy no tiene nada serio en contra')}. Eso no es una señal de que vaya a subir: sólo que no hay motivo para evitarla.")
    L.append(F.it(f"No sé si va a subir o bajar. Entrar y salir cuesta cerca de {F.pct(_costo(ctx.cfg), 1)}: no la compres para venderla en uno o dos días. Siempre con orden límite."))
    return "\n".join(L)
