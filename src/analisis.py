"""Los tres análisis que juntan TODOS los filtros en una sola respuesta (antes había que pedir cada filtro por separado y armar la conclusión a mano):

  · `resp_ganar`      (/ganar)            — ¿con cuál tengo más opción de ganar? Movimiento esperado, semáforo, macro, noticias, costo y liquidez de tu base.
  · `resp_reemplazo`  (/reemplazo NUCO)   — ¿cuál puede reemplazar a esa acción? Las de la BVC que más se le acercan en movimiento, con su liquidez en trii.
  · `resp_revisar`    (/revisar ECOPETROL)— ¿la compro hoy? Precio, macro, noticias del día y liquidez de ESA acción, con lo que tiene a favor y en contra.
  · `resp_variaciones`(/variaciones)      — rentabilidad de cada acción que se analiza: hoy, en el concurso y en el último mes.

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
    """Todas las acciones analizadas con UNA sola medida de movimiento: la misma del banco de relevo (movimiento esperado hasta el próximo corte, `mee_corte`),
    llevada también hasta el final del concurso (`mee`). Así /comprar, /banco, /ganar y /reemplazo ordenan igual y nombran a la misma acción.
    Cada fila: ticker, grupo, mee_corte, mee, vs_base, color, nivel de liquidez en trii (con `a_ratos` y `continuidad`), liquidez_mm, es_base."""
    import math

    from .construir import candidatos_banco, construir_candidato
    cands = {c.ticker: c for c in candidatos_banco(ctx.f, ctx.cfg, ctx.ahora, base, None, ctx.puntuador, ctx.motor)}
    if base not in cands:
        cb = construir_candidato(ctx.f, base, ctx.ahora, ctx.cfg, None, ctx.puntuador, ctx.motor)
        if cb is not None:
            cands[base] = cb
    yo = cands.get(base)
    corte = C.corte_vigente(ctx.ahora, ctx.cfg)
    ses_corte = max(C.sesiones_restantes(ctx.ahora, ctx.cfg), 1)
    ses_fin = ses_corte + (len([d for d in C.sesiones(ctx.cfg) if d > corte["fecha"]]) if corte else 0)
    factor = math.sqrt(ses_fin / ses_corte)
    filas = []
    for t, c in cands.items():
        if not c.mee or c.mee <= 0:
            continue
        lq = LQ.medir(ctx.f, t, ctx.cfg)
        filas.append(dict(ticker=t, grupo=c.grupo, mee_corte=c.mee, mee=c.mee * factor, vs_base=(c.mee / yo.mee) if (yo and yo.mee) else None, color=c.color,
                          nivel=lq["nivel"], liquida=lq["nivel"] == LQ.BUENA, a_ratos=bool(lq.get("a_ratos")), continuidad=lq.get("continuidad"),
                          liquidez_mm=lq.get("mediana_mm") or 0.0, es_base=t == base, supera=False))
    return sorted(filas, key=lambda x: -x["mee"])


def _mov(x: dict[str, Any], cfg: dict[str, Any], ahora: Any = None) -> str:
    """'±3,2% hasta el corte del vie 09/10 (±9,0% hasta el final)': el mismo número que muestra /comprar, y el del final al lado."""
    fin = f"±{F.n(x['mee'] * 100, 1)}%"
    if not x.get("mee_corte"):
        return fin
    return f"±{F.n(x['mee_corte'] * 100, 1)}% hasta el corte ({fin} hasta el final)"


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


def _consejo_rank(est: Any, ctx: S.Contexto | None = None) -> str:
    mia, corte = est.rent.get("mia"), est.rent.get("umbral")
    if mia is None or corte is None:
        return (f"🏁 No sé cómo vas en el ranking. Escríbeme {F.b('voy 6,5 y el corte está en 11')}: si vas por encima conviene cuidar lo ganado; "
                "si vas por debajo, mantener lo que más se mueve.")
    nota = S.nota_ranking(ctx, est) if ctx is not None else ""
    if mia >= corte:
        t = f"🏁 Vas {F.n(mia, 1, True)}% y el corte está en {F.n(corte, 1, True)}%: vas por encima. Cuida lo ganado; no cambies por cambiar."
    else:
        t = (f"🏁 Vas {F.n(mia, 1, True)}% y el corte está en {F.n(corte, 1, True)}%: vas por debajo. Para alcanzarlo te conviene mantener lo que más se mueve, "
             "no pasarte a algo más quieto.")
    return t + ("\n" + nota if nota else "")


# ------------------------------------------------------------------ el chequeo ÚNICO de "¿la compro hoy?"
def _comun(ctx: S.Contexto, est: Any, tickers: list[str]) -> dict[str, Any]:
    """Lo que se consulta UNA vez y sirve para revisar varias acciones: factores macro de hoy, sensibilidades medidas y noticias de 24 horas."""
    from . import macro as M
    out: dict[str, Any] = dict(tab=None, sens={}, noticias=None)
    try:
        out["tab"] = [x for x in M.tablero(ctx.f, ctx.cfg, ctx.ahora) if x["cuando"] == "hoy"]
        out["sens"] = M.sensibilidades(ctx.f, ctx.cfg, list(dict.fromkeys(S._tickers_macro(ctx, est) + list(tickers))))
    except Exception:                                                                  # noqa: BLE001 — un filtro sin datos se dice, no tumba el análisis
        out["tab"] = None
    try:
        out["noticias"] = S.buscar_noticias(ctx, horas=24)[0]
    except Exception:                                                                  # noqa: BLE001
        out["noticias"] = None
    return out


def chequeo(ctx: S.Contexto, t: str, est: Any, comun: dict[str, Any] | None = None) -> dict[str, Any]:
    """LA evaluación de "¿la compro hoy?" de una acción. La usan /revisar, /comprar, los cambios sugeridos y la sugerencia de los avisos: como todos preguntan
    aquí, el bot no puede recomendar en una respuesta lo que en otra desaconseja.
    Devuelve {ticker, lineas, pros, contras, apta, veredicto}: veredicto = 'sin_liquidez' | 'en_contra' | 'sin_contras'."""
    comun = comun or _comun(ctx, est, [t])
    pros: list[str] = []
    contras: list[str] = []
    L: list[str] = []
    hoy_accion: float | None = None
    try:
        res, _, _ = S.evaluar_activo(ctx.f, t, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor)
        de_hoy = res.fecha is None or res.fecha >= ctx.ahora.date()
        cuando = "hoy" if de_hoy else f"en la última sesión ({F.fecha(res.fecha)})"
        verbo = ("sube" if res.r_hoy >= 0 else "baja") if de_hoy else ("subió" if res.r_hoy >= 0 else "bajó")
        hoy_accion = res.r_hoy if de_hoy else None
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
    # macro: sólo cuenta a favor o en contra si la acción HOY va en esa misma dirección (si no, se dice que no lo está siguiendo)
    tab = comun.get("tab")
    if tab is None:
        L.append(f"🌍 {F.b('Macro hoy')}: no pude revisarla ahora.")
    else:
        sens = comun.get("sens") or {}
        efectos = [(x["nombre"], x["cambio"], sens.get(x["clave"], {}).get(t, {}).get("beta", 0.0) * x["cambio"]) for x in tab]
        efectos = sorted([e for e in efectos if abs(e[2]) >= ctx.cfg["macro_vivo"]["efecto_minimo"]], key=lambda e: -abs(e[2]))
        if efectos:
            L.append(f"🌍 {F.b('Macro')} {F.it('(a las ' + ctx.ahora.strftime('%H:%M') + ', frente al cierre de ayer)')}: "
                     + "; ".join(f"{F.esc(nom)} {'sube' if c >= 0 else 'cae'} {F.pct(abs(c), 1)} → a {F.esc(t)} suele {'sumarle' if e > 0 else 'restarle'} {F.pct(abs(e), 1)}"
                                 for nom, c, e in efectos[:3]) + ".")
            total = sum(e for _, _, e in efectos)
            sigue = hoy_accion is not None and (hoy_accion > 0) == (total > 0)
            if abs(total) >= 0.005 and hoy_accion is None:                             # aún no hay precio de HOY de la acción: no se puede decir que lo sigue
                L.append(F.it(f"  Todavía no tengo el precio de hoy de {F.esc(t)} para confirmar que lo esté siguiendo: no lo cuento ni a favor ni en contra."))
            elif abs(total) >= 0.005 and not sigue:
                L.append(F.it(f"  Pero hoy {F.esc(t)} no lo está siguiendo ({'sube' if hoy_accion >= 0 else 'baja'} {F.pct(abs(hoy_accion), 1)}): ese efecto es lo que SUELE pasar, "
                              "no lo que está pasando. No lo cuento ni a favor ni en contra."))
            elif total <= -0.005:
                contras.append("la macro de hoy la empuja hacia abajo (" + ", ".join(nom for nom, _, e in efectos if e < 0) + ")")
            elif total >= 0.005:
                pros.append("la macro de hoy la favorece (" + ", ".join(nom for nom, _, e in efectos if e > 0) + ") y la acción lo está siguiendo")
        else:
            L.append(f"🌍 {F.b('Macro hoy')}: nada que le pegue a {F.esc(t)} más de lo normal.")
    # noticias propias y de rebote
    filas = comun.get("noticias")
    if filas is None:
        L.append(f"📰 {F.b('Noticias')}: no pude consultarlas ahora.")
    else:
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
        ajenas = [(x, a) for x in filas if x["ticker"] != t for a in x.get("afectadas") or [] if a["ticker"] == t and not a["propia"]][:4]
        if ajenas:
            L.append(f"↪️ {F.b('Noticias de otras empresas que le pegan de rebote')}:")
            for x, a in ajenas:
                cifra = ("±" + F.pct(abs(a["efecto"]), 1)) if a["sentido"] == 0 else F.pct(a["efecto"], 1, True)
                L.append(f"  {'🟢' if a['sentido'] > 0 else ('🔴' if a['sentido'] < 0 else '⚪')} {F.esc(x['ticker'])}: {F.esc(x['titulo'][:120])} → a {F.esc(t)} {cifra} estimado "
                         f"{F.it('(' + F.esc(a['motivo']) + ')')}")
            neto = sum(a["efecto"] for _, a in ajenas if a["sentido"])
            if neto <= -0.003:
                contras.append("le pegan de rebote noticias negativas de empresas relacionadas")
            elif neto >= 0.003:
                pros.append("le ayudan de rebote noticias positivas de empresas relacionadas")
    lq = LQ.medir(ctx.f, t, ctx.cfg)
    L.append(F.esc(LQ.frase(lq, None, ctx.cfg)))
    es_bvc = t in ctx.cfg["universe"]["local"]
    apta = lq["nivel"] == LQ.BUENA                                                     # también una extranjera, si de verdad se negocia bien en Colombia (NUCO)
    if apta:
        pros.append("se compra y se vende fácil en trii")
    veredicto = "sin_liquidez" if not apta else ("en_contra" if len(contras) > len(pros) - 1 else "sin_contras")     # la liquidez sola no basta: hace falta algo más a favor que en contra
    return dict(ticker=t, lineas=L, pros=pros, contras=contras, apta=apta, es_bvc=es_bvc, veredicto=veredicto)


def banco_revisado(ctx: S.Contexto, banco: list[Any] | None = None, n: int = 5) -> tuple[list[Any], dict[str, dict[str, Any]]]:
    """(candidatas del banco que HOY no tienen más en contra que a favor, en el mismo orden; {ticker: su chequeo}). La primera es LA acción que el bot
    nombra cuando sugiere comprar o cambiar; si la lista queda vacía, no sugiere ninguna."""
    est = S.leer_estado(ctx)
    banco = S.banco_bvc(ctx) if banco is None else banco
    tope = list(banco[:n])
    comun = _comun(ctx, est, [e.c.ticker for e in tope])
    ch = {e.c.ticker: chequeo(ctx, e.c.ticker, est, comun) for e in tope}
    return [e for e in tope if ch[e.c.ticker]["veredicto"] == "sin_contras"], ch


def resp_comprar(ctx: S.Contexto) -> str:
    """/comprar: la lista de candidatas y UNA elegida, que es la primera que pasa el mismo chequeo de /revisar. Si ninguna lo pasa, se dice."""
    est = S.leer_estado(ctx)
    r = S.ejecutar_motor(ctx.f, est, ctx.cfg, ctx.ahora, ctx.puntuador, ctx.motor)
    hora = C.hora_orden_manana(C.proxima_sesion(ctx.ahora, ctx.cfg) or ctx.ahora.date(), ctx.cfg)
    from . import noticias_bvc as N
    buenas, ch = banco_revisado(ctx, r.banco) if r.banco else ([], {})
    eleccion = dict(ticker=buenas[0].c.ticker if buenas else None, chequeos=ch) if r.banco else None
    liq = S.avisos_liquidez(ctx, [buenas[0].c.ticker], {buenas[0].c.ticker: float(ctx.cfg["capital"]) * 0.25}, todos=True) if buenas else []
    nota = S.nota_ranking(ctx, est)
    return F.msg_que_comprar(r.banco, r.activo, r.mee, r.decision, r.corte, C.sesiones_restantes(ctx.ahora, ctx.cfg), S.senales_recientes(24, ctx.ahora), hora, ctx.cfg,
                             n_estudio=N.cargar_estudio().get("n_total"), liquidez_txt=liq[0] if liq else None, eleccion=eleccion) + ("\n" + nota if nota else "")


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
    cands = [x for x in filas if not x["es_base"] and x["grupo"] == "local" and x.get("color") not in ("ROJO", "NEGRO") and (x["nivel"] == LQ.BUENA or _casi(x, ctx.cfg))][:5]
    if not cands:
        return f"🔁 Ahora mismo ninguna acción de la BVC pasa el filtro de liquidez para reemplazar a {F.esc(t)}."
    valor = _valores(ctx, est).get(t)
    L = [f"🔁 {F.b('¿Cuál puede reemplazar a ' + t + '?')}",
         F.it(f"Las de la BVC que más pueden moverse, de mayor a menor. {t} se espera que se mueva {_mov(yo, ctx.cfg)}."), ""]
    for i, x in enumerate(cands, 1):
        L.append(f"{i}. {F.b(x['ticker'])}: {_mov(x, ctx.cfg)} · {F.n(x['vs_base'], 2)} veces {F.esc(t)} · {_liq_txt(x)}{_parte_del_dia(valor, x)}")
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
          f"• Antes de comprar la otra, pídeme {F.b('/revisar ' + (segura or cerca)['ticker'].lower())}: ahí miro si HOY tiene algo en contra.",
          "• Si cambias: vende por partes con orden límite y compra la otra a medida que te ejecuten.",
          "", _consejo_rank(est, ctx)]
    return "\n".join(L)


# ------------------------------------------------------------------ /ganar
def resp_ganar(ctx: S.Contexto) -> str:
    est = S.leer_estado(ctx)
    base = est.activo
    filas = _filas(ctx, base)
    yo = next((x for x in filas if x["es_base"]), None)
    if yo is None:
        return "🏆 No pude hacer el análisis ahora (faltan datos de tu acción principal). Intenta de nuevo en unos minutos."
    liquidas = [x for x in filas if not x["es_base"] and x["grupo"] == "local" and x["nivel"] == LQ.BUENA and x.get("color") not in ("ROJO", "NEGRO") and x["ticker"] not in est.tenidos()]
    mejor = liquidas[0] if liquidas else None
    L = [f"🏆 {F.b('¿Con cuál tengo más opción de ganar?')}",
         F.it("Ningún filtro predice hacia dónde va el precio. Esto es lo que dicen todos juntos:"), ""]
    mov = f"📏 {F.b('Movimiento esperado')}: {F.esc(base)} {_mov(yo, ctx.cfg)}."
    if mejor:
        mov += f" La que más se mueve de las que se negocian bien, {F.esc(mejor['ticker'])}: {_mov(mejor, ctx.cfg)}, o sea {F.n(mejor['vs_base'], 2)} veces lo de {F.esc(base)}."
    L.append(mov)
    es_eeuu = base in ctx.cfg["universe"].get("mgc", [])
    try:
        res, _, _ = S.evaluar_activo(ctx.f, base, ctx.ahora, ctx.cfg, est, ctx.puntuador, ctx.motor)
        cuando = "hoy" if (res.fecha is None or res.fecha >= ctx.ahora.date()) else f"en la última sesión ({F.fecha(res.fecha)})"
        donde = " en Nueva York (en trii, en pesos, puede variar distinto)" if es_eeuu else ""
        L.append(f"🚦 {F.b('Semáforo')}: {F.esc(base)} en {F.EMOJI.get(res.color, '')} {F.esc(res.color)}; {cuando} {F.pct(res.r_hoy, 1, True)}{donde}.")
    except Exception:                                                                  # noqa: BLE001 — un filtro sin datos se dice, no tumba el análisis
        L.append(f"🚦 {F.b('Semáforo')}: no pude revisarlo ahora.")
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
    n = len([x for x in S.senales_recientes(24, ctx.ahora) if x.get("ticker") in est.tenidos()])
    cuantas = "ninguna noticia fuerte" if n == 0 else ("1 noticia fuerte" if n == 1 else f"{n} noticias fuertes")
    L.append(f"📰 {F.b('Noticias')}: {cuantas} de tus acciones en 24 horas. Comprar por la noticia del día no paga el costo (lo medí con casi 1.900 anuncios).")
    L.append(f"💸 {F.b('Costo de cambiar')}: cerca de {F.pct(_costo(ctx.cfg), 1)} cada vez.")
    valores = _valores(ctx, est)
    lq = LQ.medir(ctx.f, base, ctx.cfg)
    L.append(f"💧 {F.b('Liquidez en trii')}: {F.esc(base)} — {NIVEL_TXT[lq['nivel']]}{_parte_del_dia(valores.get(base), dict(liquidez_mm=lq.get('mediana_mm')))}.")
    L.append("")
    ventaja = ctx.cfg["seleccion_base"]["ventaja_minima"]
    if mejor is None or mejor["mee"] < yo["mee"]:
        L.append(f"👉 {F.b('La que más opción te da de quedar arriba es la que ya tienes: ' + base)}. Ninguna de las que se negocian bien en trii se mueve más.")
        mas = next((x for x in filas if not x["es_base"] and x["grupo"] == "local" and x["mee"] > yo["mee"] and _casi(x, ctx.cfg)), None)
        if mas:                                                                        # /reemplazo la va a nombrar: aquí se dice lo mismo, para no contradecirse
            L.append(f"({F.esc(mas['ticker'])} se mueve un poco más —{F.n(mas['vs_base'], 2)} veces—, pero se negocia a ratos y queda fuera del filtro de liquidez por poco.)")
        if lq["nivel"] != LQ.BUENA:
            L.append("Su punto débil es la liquidez: no compres más, y si sales hazlo con orden límite y por partes."
                     + (f" Cambiarla por una más fácil de vender te deja con cerca del {F.pct(mejor['vs_base'], 0)} del movimiento." if mejor else ""))
    elif mejor["mee"] >= yo["mee"] * (1 + ventaja):
        L.append(f"👉 {F.b(mejor['ticker'] + ' puede moverse bastante más que ' + base)} ({F.n(mejor['vs_base'], 2)} veces) y se negocia bien en trii: vale la pena "
                 f"revisar el cambio. Mira {F.b('/reemplazo ' + base.lower())}.")
    else:
        L.append(f"👉 {F.b('Mantén ' + base)}: {F.esc(mejor['ticker'])} se mueve un poco más, pero la diferencia no paga el costo de cambiar.")
    malas, justas = [], []
    for t in sorted(est.tenidos() - {base}):
        nivel = LQ.medir(ctx.f, t, ctx.cfg)["nivel"]
        (malas if nivel in (LQ.MALA, LQ.SIN_DATO) else justas if nivel == LQ.JUSTA else []).append(t)
    if malas:
        L.append(f"• {F.esc(', '.join(malas))}: casi no se negocia{'n' if len(malas) > 1 else ''} en trii. No hay afán, pero si vas a mover algo, empieza por ahí"
                 + (f" (hacia {F.esc(mejor['ticker'])})" if mejor else "") + ", con orden límite y sin prisa.")
    if justas:
        L.append(f"• {F.esc(', '.join(justas))}: se negocia{'n' if len(justas) > 1 else ''} poco o a ratos. Puedes mantener; no compres más.")
    if mejor:
        L.append(f"• Si tienes efectivo libre: la que más puede moverse entre las que se negocian bien es {F.b(mejor['ticker'])}. Antes de comprarla, "
                 f"pídeme {F.b('/revisar ' + mejor['ticker'].lower())} (o {F.b('/comprar')}): ahí miro si HOY tiene algo en contra.")
    L += ["", _consejo_rank(est, ctx), "", F.it("Movimiento esperado = cuánto puede subir o bajar. Más movimiento es más oportunidad y más riesgo; no garantiza ganar.")]
    return "\n".join(L)


# ------------------------------------------------------------------ /revisar
def resp_revisar(ctx: S.Contexto, args: list[str]) -> str:
    est = S.leer_estado(ctx)
    if not args:
        return f"{F.b('Cómo usar /revisar')}\nEscribe la acción que estás pensando comprar: {F.b('/revisar ecopetrol')}. Te digo qué tiene hoy a favor y en contra."
    t = S.resolver(ctx, " ".join(args), est.tenidos())
    if t not in S.universo_permitido(ctx.cfg) and t not in est.tenidos():
        return f"❌ {F.esc(t)} no está permitida en el concurso (o está en la lista negra), o no reconocí el nombre."
    ch = chequeo(ctx, t, est)
    L = [f"🔍 {F.b('¿Compro ' + t + ' hoy?')}  {F.it(F.fecha(ctx.ahora.date()) + ' ' + ctx.ahora.strftime('%H:%M'))}", "", *ch["lineas"], ""]
    if ch["pros"]:
        L.append("✅ A favor: " + F.esc("; ".join(ch["pros"])) + ".")
    if ch["contras"]:
        L.append("⚠️ En contra: " + F.esc("; ".join(ch["contras"])) + ".")
    if ch["veredicto"] == "sin_liquidez":
        L.append(f"👉 {F.b('No la compraría')}: " + ("no pasa el filtro de liquidez en trii." if ch["es_bvc"] else "no es de la BVC y en trii se negocia poco."))
    elif ch["veredicto"] == "en_contra":
        L.append(f"👉 {F.b('Hoy tiene más en contra que a favor')}. Si no tienes afán, espera; nada obliga a entrar hoy.")
    else:
        L.append(f"👉 {F.b('Hoy no tiene nada serio en contra')}. Eso no es una señal de que vaya a subir: sólo que no hay motivo para evitarla.")
    L.append(F.it(f"No sé si va a subir o bajar. Entrar y salir cuesta cerca de {F.pct(_costo(ctx.cfg), 1)}: no la compres para venderla en uno o dos días. Siempre con orden límite."))
    return "\n".join(L)


# ------------------------------------------------------------------ /variaciones
def _cierres(ctx: S.Contexto, sym: str) -> pd.Series | None:
    """Cierres diarios de 3 meses de un símbolo (caché de 2 minutos). None si no hay."""
    def bajar() -> pd.DataFrame:
        d = ctx.f.yf.download(sym, period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if d is None or d.empty:
            raise RuntimeError("Yahoo devolvió vacío")
        d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
        return d[["Close"]].dropna()
    try:
        d = ctx.f._con_cache(f"rentabilidad|{sym}", 2, bajar, f"rentabilidad {sym}")
        return d["Close"] if d is not None and len(d) >= 2 else None
    except Exception:                                                                  # noqa: BLE001
        return None


def _desde(c: pd.Series, ctx: S.Contexto) -> tuple[float | None, float | None]:
    """(rentabilidad desde antes del primer día del concurso, rentabilidad de las últimas 21 sesiones) de una serie de cierres."""
    ult = float(c.iloc[-1])
    antes = c[c.index < pd.Timestamp(ctx.cfg["concurso"]["inicio"])]
    return ((ult / float(antes.iloc[-1]) - 1) if len(antes) and len(antes) < len(c) else None, (ult / float(c.iloc[-22]) - 1) if len(c) >= 22 else None)


def _rentabilidad(ctx: S.Contexto, t: str) -> dict[str, Any] | None:
    """Rentabilidad de la acción con su precio en pesos (el que se ve en trii): {hoy, es_hoy, fecha, concurso, mes, aprox}.
      · hoy      = último precio frente al cierre anterior (con el día al que corresponde);
      · concurso = desde el cierre anterior al primer día del concurso hasta ahora;
      · mes      = últimas 21 sesiones.
    Para las acciones de EE. UU. cuyo historial en Colombia no es fiable (NUCO: la fuente repite el mismo precio días enteros), "hoy" sale de lo que reporta
    la Bolsa de Colombia y "concurso" y "mes" se aproximan con el precio de Nueva York pasado a pesos (`aprox`). Lo que no se pueda calcular queda en None."""
    sym = LQ.simbolo_trii(t, ctx.cfg)
    fiable = sym is not None and (t in ctx.cfg["universe"]["local"] or LQ._medir_historia(ctx.f, t, ctx.cfg)["nivel"] != LQ.SIN_DATO)
    c = _cierres(ctx, sym) if fiable else None
    if c is not None and float(c.iloc[-2]) > 0 and abs(float(c.iloc[-1]) / float(c.iloc[-2]) - 1) < 0.5:
        fecha = pd.Timestamp(c.index[-1]).date()
        concurso, mes = _desde(c, ctx)
        return dict(hoy=float(c.iloc[-1]) / float(c.iloc[-2]) - 1, es_hoy=fecha >= ctx.ahora.date(), fecha=fecha, concurso=concurso, mes=mes, aprox=False)
    fl = LQ.fila_libro(ctx.f, t, ctx.cfg)
    hoy = fl["cambio"] if fl and fl.get("cambio") is not None else None
    concurso = mes = None
    fecha, es_hoy = None, True
    if sym is not None and t not in ctx.cfg["universe"]["local"]:                      # global sin historial fiable en Colombia: Nueva York × dólar
        from .data_loader import yahoo_symbol
        ny, dolar = _cierres(ctx, yahoo_symbol(t, ctx.cfg)), _cierres(ctx, ctx.cfg["macro_vivo"]["factores"]["dolar"]["simbolo"])
        if ny is not None and dolar is not None:
            pesos = (ny * dolar.reindex(ny.index, method="ffill")).dropna()
            if len(pesos) >= 2:
                concurso, mes = _desde(pesos, ctx)
                if hoy is None:
                    hoy, fecha = float(pesos.iloc[-1]) / float(pesos.iloc[-2]) - 1, pd.Timestamp(pesos.index[-1]).date()
                    es_hoy = fecha >= ctx.ahora.date()
    if hoy is None:
        return None
    return dict(hoy=hoy, es_hoy=es_hoy, fecha=fecha, concurso=concurso, mes=mes, aprox=concurso is not None or mes is not None)


def resp_variaciones(ctx: S.Contexto) -> str:
    from concurrent.futures import ThreadPoolExecutor

    from . import construir as CO
    est = S.leer_estado(ctx)
    tenidos = est.tenidos()
    tickers = list(dict.fromkeys(sorted(tenidos) + sorted(CO.universo_candidatos(ctx.cfg))))

    def una(t: str) -> dict[str, Any] | None:
        try:
            v = _rentabilidad(ctx, t)
            return dict(v, ticker=t) if v else None
        except Exception:                                                              # noqa: BLE001 — una acción sin dato se omite y se cuenta abajo
            return None
    with ThreadPoolExecutor(max_workers=ctx.cfg["banco"]["max_hilos"]) as ex:
        filas = [x for x in ex.map(una, tickers) if x]
    if not filas:
        return "📊 No pude leer las rentabilidades ahora. Intenta de nuevo en unos minutos."
    sin = [t for t in tickers if t not in {x["ticker"] for x in filas}]
    o = lambda v, aprox=False: "s/d" if v is None else ("≈ " if aprox else "") + F.pct(v, 1, True)      # noqa: E731

    def linea(x: dict[str, Any]) -> str:
        c = x["hoy"]
        marca = "🟢" if c > 0.0005 else ("🔴" if c < -0.0005 else "⚪")
        viejo = "" if x["es_hoy"] else " 🕘"
        mia = " 💼" if x["ticker"] in tenidos else ""
        return f"{marca} {F.b(x['ticker'])}{mia}: hoy {F.b(F.pct(c, 1, True))}{viejo} · concurso {o(x['concurso'], x.get('aprox'))} · mes {o(x['mes'], x.get('aprox'))}"
    filas.sort(key=lambda x: -x["hoy"])
    inicio = pd.Timestamp(ctx.cfg["concurso"]["inicio"]).date()
    L = [f"📊 {F.b('Rentabilidad de las acciones')}  {F.it(F.fecha(ctx.ahora.date()) + ' ' + ctx.ahora.strftime('%H:%M'))}",
         F.it(f"Cuánto ha subido o bajado cada acción: hoy, desde que empezó el concurso ({F.fecha(inicio)}) y en el último mes. De la que más sube hoy a la que más baja."), ""]
    L += [linea(x) for x in filas]
    L.append("")
    de_hoy = [x for x in filas if x["es_hoy"]]
    if de_hoy:
        suben, bajan = sum(x["hoy"] > 0.0005 for x in de_hoy), sum(x["hoy"] < -0.0005 for x in de_hoy)
        L.append(f"👉 {F.b('Hoy')}: {suben} suben, {bajan} bajan y {len(de_hoy) - suben - bajan} no se mueven. La que más sube: {F.esc(de_hoy[0]['ticker'])} "
                 f"({F.pct(de_hoy[0]['hoy'], 1, True)}); la que más baja: {F.esc(de_hoy[-1]['ticker'])} ({F.pct(de_hoy[-1]['hoy'], 1, True)}).")
    else:
        L.append(f"👉 {F.b('Hoy todavía no ha abierto la bolsa')}: la columna de hoy es la última sesión.")
    con = sorted([x for x in filas if x["concurso"] is not None], key=lambda x: -x["concurso"])
    if con:
        L.append(f"🏁 {F.b('En el concurso')}: las que más han rendido son " + ", ".join(f"{F.esc(x['ticker'])} ({F.pct(x['concurso'], 1, True)})" for x in con[:3])
                 + "; las que menos, " + ", ".join(f"{F.esc(x['ticker'])} ({F.pct(x['concurso'], 1, True)})" for x in con[-3:][::-1]) + ".")
    mias = [x for x in filas if x["ticker"] in tenidos]
    if mias:
        L.append(f"💼 {F.b('Las tuyas')}: " + ", ".join(f"{F.esc(x['ticker'])} hoy {F.pct(x['hoy'], 1, True)} (concurso {o(x['concurso'])})" for x in mias) + ".")
    notas = ["Precio en pesos en la Bolsa de Colombia, el que ves en trii. Es la rentabilidad de la ACCIÓN, no la tuya: la tuya depende de a qué precio compraste (/cartera)."]
    if any(x.get("aprox") for x in filas):
        notas.append("≈ = acción de EE. UU. sin historial fiable en Colombia: lo del concurso y el mes es su precio de Nueva York pasado a pesos; en trii puede diferir.")
    if any(not x["es_hoy"] for x in filas):
        notas.append("🕘 = todavía no ha negociado hoy: es la variación de su última sesión.")
    if sin:
        notas.append("Sin dato ahora: " + ", ".join(sin) + ".")
    notas.append("Que una acción haya subido no dice que vaya a seguir subiendo. Antes de comprar una, pídeme /revisar y su nombre (ahí miro también si se negocia bien en trii).")
    return "\n".join(L + [F.it(x) for x in notas])
