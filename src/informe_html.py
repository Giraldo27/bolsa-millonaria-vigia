"""Informe HTML fundamentado (un solo archivo, sin internet): se genera SOLO cuando alguien lo pide en el bot (/informe o la palabra `html`).

Cada número va con su explicación en palabras sencillas; incluye los pasos de la Regla Maestra, el banco completo (también por qué se descartó cada
candidato), noticias con su tono, fechas que vienen, lo que dice la validación histórica y los límites. Función pura: recibe resultados ya calculados."""
from __future__ import annotations

import datetime as dt
import html as _html
import json
from pathlib import Path
from typing import Any, Callable

from . import concurso as C
from . import formato as F
from .regla_maestra import multiplo_requerido
from .relevo import liquidez_ok
from .semaforo import AMARILLO, AZUL, EMOJI, NEGRO, ROJO, VERDE

COLOR_CSS = {VERDE: "#0ca30c", AZUL: "#2a78d6", AMARILLO: "#eda100", ROJO: "#d03b3b", NEGRO: "#3b3b3b"}
CSS = """
:root{--bg:#f9f9f7;--sf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;--line:#e1e0d9;--warn:#8a5a00;--warnbg:#fff4d6;--acc:#2a78d6}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--sf:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--mut:#898781;--line:#2c2c2a;--warn:#fab219;--warnbg:#3a2e08;--acc:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:18px 14px 60px}h1{font-size:clamp(24px,5vw,34px);margin:0 0 4px}h2{font-size:21px;margin:34px 0 10px;padding-top:6px;border-top:2px solid var(--line)}
h3{font-size:17px;margin:18px 0 6px}p{margin:.5em 0}.mut{color:var(--mut)}.sm{font-size:14px}
.card{background:var(--sf);border:1px solid var(--line);border-radius:12px;padding:14px 16px;margin:10px 0}
.big{display:flex;gap:14px;align-items:center}.dot{width:54px;height:54px;border-radius:50%;flex:none}
.warn{background:var(--warnbg);border:1px solid var(--warn);border-radius:10px;padding:10px 14px;margin:12px 0}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--sf);margin:8px 0}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child,td.l,th.l{text-align:left}td.l{white-space:normal}th{color:var(--ink2);font-size:13px}
.pos{color:#0a7a0a;font-weight:600}.neg{color:#b32525;font-weight:600}.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:700;color:#fff}
.bar{height:10px;border-radius:5px;background:var(--acc);min-width:2px}svg{width:100%;height:auto}
ul{padding-left:20px}li{margin:5px 0}dl dt{font-weight:700;margin-top:10px}dl dd{margin:2px 0 0;color:var(--ink2)}footer{color:var(--mut);font-size:13px;margin-top:30px}
"""


def e(x: Any) -> str:
    return _html.escape(str(x), quote=True)


def _tabla(cab: list[str], filas: list[list[str]], izq: int = 1) -> str:
    """Tabla HTML con TODAS las columnas alineadas a la izquierda (casi todo es texto: a la derecha se leía descuadrado)."""
    izq = len(cab)
    th = "".join(f"<th class='{'l' if i < izq else ''}'>{c}</th>" for i, c in enumerate(cab))
    tr = "".join("<tr>" + "".join(f"<td class='{'l' if i < izq else ''}'>{c}</td>" for i, c in enumerate(f)) + "</tr>" for f in filas)
    return f"<div class='tw'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


def _pill(color: str) -> str:
    return f"<span class='pill' style='background:{COLOR_CSS[color]}'>{e(F.NOMBRE_COLOR[color])}</span>"


def _signo(x: float | None, txt: str) -> str:
    return "<span class='mut'>n. d.</span>" if x is None or x != x else f"<span class='{'pos' if x > 0 else 'neg' if x < 0 else ''}'>{e(txt)}</span>"


def sparkline(cierres: Any, precio: float | None) -> str:
    """Precio de los últimos 60 cierres (+ el de hoy) como línea SVG."""
    v = [float(x) for x in list(cierres.tail(60))] + ([float(precio)] if precio else [])
    if len(v) < 3:
        return ""
    lo, hi = min(v), max(v)
    w, h, pad = 600, 140, 8
    pts = " ".join(f"{pad + i * (w - 2 * pad) / (len(v) - 1):.1f},{h - pad - (x - lo) / ((hi - lo) or 1) * (h - 2 * pad):.1f}" for i, x in enumerate(v))
    return (f"<svg viewBox='0 0 {w} {h}' role='img' aria-label='Precio de los últimos {len(v)} cierres'><polyline fill='none' stroke='var(--acc)' stroke-width='2.2' points='{pts}'/>"
            f"<text x='{pad}' y='14' fill='var(--mut)' font-size='12'>máx {hi:,.2f}</text><text x='{pad}' y='{h - 2}' fill='var(--mut)' font-size='12'>mín {lo:,.2f}</text></svg>")


def porque_descartado(c: Any, actual: str, cfg: dict[str, Any], permitidos: set[str], excluidos: set[str]) -> str:
    """Motivo (en palabras) por el que un candidato calculado NO está en el banco."""
    if c.ticker == actual:
        return "es tu acción actual"
    if c.ticker not in permitidos:
        return "no está permitida (lista negra o fuera del universo)"
    if c.ticker in excluidos:
        return "semana final sin ir de primero: TSLA y NVDA no se recomiendan como nuevas"
    if not c.mee or c.mee <= 0:
        return "no pude calcular cuánto se espera que se mueva"
    if c.color in (ROJO, NEGRO):
        return f"está en {F.NOMBRE_COLOR[c.color]}: cae por su cuenta con mala noticia"
    if not liquidez_ok(c, cfg):
        return "se negocia poco (riesgo de no poder entrar o salir al precio esperado)"
    if cfg["banco"].get("exigir_noticias") and c.sin_noticias:
        return "sin datos de noticias (no se le puede detectar una mala noticia)"
    return "quedó fuera del top " + str(cfg["banco"]["top_n"]) + " por tener menor movimiento esperado"


def _validacion(cfg: dict[str, Any]) -> str:
    """Resumen de la Fase 3 (si existe outputs/validacion/tabla_eventos.csv); si no, texto general."""
    from . import validacion as V
    f = Path(cfg["paths"]["outputs_dir"]) / "validacion" / "tabla_eventos.csv"
    base = ("<p>Entre 2015 y 2026 probé qué hace una acción <b>después</b> de cada color y simulé el concurso con las reglas oficiales. "
            "Informe completo: <code>reports/validacion_semaforo.html</code>.</p>")
    try:
        import pandas as pd
        t = pd.read_csv(f)
        c = V.criterio_negro(t, cfg)
        d = c["detalle"]
        txt = (f"<li><b>NEGRO no anticipó más caídas</b> en acciones grandes: retorno posterior a 5 sesiones {F.pct(d[5]['media'], 2, True)} "
               f"(intervalo {F.pct(d[5]['ic_lo'], 2, True)} a {F.pct(d[5]['ic_hi'], 2, True)}) y a 10 sesiones {F.pct(d[10]['media'], 2, True)}. Por eso no se vende solo por el color.</li>"
               if c["datos_suficientes"] and 5 in d and 10 in d else "")
        return base + f"<ul>{txt}<li>Mantener una sola acción volátil pasa los cortes con una probabilidad modesta y con mucha variación entre años; ninguna regla garantiza ganar.</li></ul>"
    except Exception:                                                                                    # noqa: BLE001 — sin la validación, el resto del informe sigue
        return base


def armar_informe(est: Any, r: Any, cats: list[dict[str, Any]], noticias: list[dict[str, Any]] | None, cfg: dict[str, Any], ahora: dt.datetime,
                  traductor: Callable[[str], str | None] | None = None, universo: set[str] | None = None, base_rank: list[dict[str, Any]] | None = None) -> str:
    """HTML completo. `r` es un `construir.ResultadoMotor`; `noticias` son titulares puntuados (`semaforo.titulares_puntuados`) o None si no hubo datos."""
    res, dec, s = r.res, r.decision, cfg["semaforo"]
    p, rent = est.d["posicion"], est.rent
    permitidos = universo or {c.ticker for c in r.candidatos}
    hora = C.hora_orden_manana(C.proxima_sesion(ahora, cfg) or ahora.date(), cfg)
    corte = r.corte
    sem = C.semana_concurso(ahora.date(), cfg)

    # ---------- 1. resumen
    rend_entrada = (r.ent.precio / p["precio_entrada"] - 1) if p.get("precio_entrada") else None
    resumen = (f"<div class='card big'><div class='dot' style='background:{COLOR_CSS[res.color]}'></div><div><h3 style='margin:0'>{e(res.ticker)}: {e(F.NOMBRE_COLOR[res.color])}</h3>"
               f"<p>{F.que_paso(res)}</p><p><b>¿Qué significa?</b> {e(F.EXPLICA_COLOR[res.color])}</p><p><b>👉 Qué hacer:</b> {e(F.ACCION_COLOR[res.color])}</p></div></div>"
               f"<div class='card'><h3 style='margin-top:0'>⚖️ ¿Cambio de acción?</h3><p>{'</p><p>'.join(x for x in F.texto_decision(dec, res.ticker, hora))}</p></div>")

    # ---------- 2. posición y ranking
    falta = (rent["umbral"] - rent["mia"]) if rent.get("mia") is not None and rent.get("umbral") is not None else None
    a = cfg["concurso"]["actividad"]
    filas_pos = [
        ["Acción que tienes", f"{e(p['activo'])} ({F.cop(p['monto_cop'])})", "Tu posición registrada en el bot."],
        ["Precio de compra", e(F.n(p["precio_entrada"], 2)) if p.get("precio_entrada") else "aún sin comprar", "Lo registras con /pos."],
        ["Rendimiento desde la compra", _signo(rend_entrada, F.pct(rend_entrada, 1, True)) if rend_entrada is not None else "n. d.", "Precio actual frente a tu precio de compra."],
        ["Tu rentabilidad en el ranking", e(F.n(rent["mia"], 1, True) + "%") if rent.get("mia") is not None else "sin dato", "La que me dices con /rank."],
        ["Objetivo del próximo corte", e(F.n(rent["umbral"], 1, True) + "%") if rent.get("umbral") is not None else "sin dato", "Rentabilidad que marca hoy el corte (o la del #1 en la semana 5)."],
        ["Te falta", e(F.puntos(falta)) if falta is not None and falta > 0 else ("vas por encima" if falta is not None else "n. d."), "Diferencia en puntos porcentuales."],
        ["Cambios de acción usados", f"{est.cambios_usados()} de {cfg['estado']['max_cambios']}", "Máximo 4 en total y 1 por día."],
        ["Operaciones de actividad", f"{est.ops_semana(ahora.date())} de {a['ops_por_semana']} esta semana · {est.ops_total()} de {a['minimo_total']} en total", "Requisito para optar a premios (no da puntos)."],
    ]
    if corte:
        filas_pos.append(["Próximo corte", f"{e(F.fecha(corte['fecha']))} · " + (f"top {corte['top_pct']}%" if corte.get("top_pct") else "final: #1"),
                          f"Faltan {C.sesiones_restantes(ahora, cfg)} días de bolsa" + (f"; semana {sem} de 5." if sem else ".")])

    # ---------- 3. semáforo explicado
    vr = res.vol_rel
    filas_met = [
        ["Movimiento de hoy", _signo(res.r_hoy, F.pct(res.r_hoy, 2, True)), "—", "Cambio del precio frente al cierre anterior."],
        ["Fuerza del movimiento (z)", e(F.n(res.z, 2, True)), f"rara si ≤ {s['z_caida']:g}", "Cuántas \"variaciones diarias típicas\" (σ de 20 días) se movió hoy. 0 = normal."],
        ["Parte explicada por el mercado", _signo(res.r_mercado, F.pct(res.r_mercado, 2, True)), "—", "Beta × movimiento del Nasdaq-100: lo que se movió solo por arrastre del mercado."],
        ["Parte propia de la acción", _signo(res.resid_hoy, F.pct(res.resid_hoy, 2, True)), "—", "Movimiento menos la parte del mercado."],
        ["Fuerza de la parte propia (z_res)", e(F.n(res.z_res, 2, True)), f"caída propia si ≤ {s['z_res_caida']:g}; mercado si > {s['z_res_azul']:g}", "Lo mismo que z pero solo con la parte propia."],
        ["Beta", e(F.n(res.beta, 2)), "—", "Sensibilidad al Nasdaq-100 (1 = se mueve igual)."],
        ["Variación diaria típica (σ 20 días)", e(F.pct(res.sigma20, 2)), "—", "Qué tanto se mueve normalmente en un día."],
        ["Volumen frente al normal", e(F.n(vr, 2) + "×") if vr is not None else "n. d.", f"alerta roja exige ≥ {s['vol_rel_min']:g}×", "Volumen del día entre el promedio de 20 días."],
        ["Suma de lo propio en 5 días", _signo(res.resid5, F.pct(res.resid5, 2, True)), f"alerta si ≤ {F.pct(res.umbral5, 2)} con mala noticia", "Detecta caídas lentas que no se ven en un solo día."],
        ["Noticias en 24 h", e(res.n_24h), "—", "Titulares encontrados en el último día."],
        ["Tono de las noticias", e(F.n(res.sentimiento, 2, True)) if res.sentimiento is not None else "sin dato", f"negativo si < {s['sentimiento_negativo']:g}", f"Calculado con {e(res.motor_sentimiento.upper())}: de −1 (muy malo) a +1 (muy bueno)."],
        ["¿Mala noticia?", "sí" if res.noticia_negativa else "no", "—", "Tono negativo o palabra clave de riesgo en un titular que no sea positivo."],
    ]
    adv = "".join(f"<li>{e(x)}</li>" for x in res.advertencias)
    sec_sem = (f"<p><b>Por qué este color:</b> {e(res.motivo)}</p>{_tabla(['Medida', 'Valor', 'Umbral', 'Qué significa'], filas_met, 1)}"
               + (f"<div class='warn'><b>Avisos:</b><ul>{adv}</ul></div>" if adv else "")
               + f"<h3>Precio reciente</h3><div class='card'>{sparkline(r.ent.cierres, r.ent.precio)}<p class='sm mut'>Últimos 60 cierres y el precio actual.</p></div>")

    # ---------- 4. noticias
    if noticias is None:
        sec_not = "<p>No pude consultar noticias (fuentes caídas o sin cobertura). Mira la ficha de la acción en Yahoo Finance.</p>"
    elif not noticias:
        sec_not = "<p>No hay noticias recientes.</p>"
    else:
        filas = []
        for k in noticias[:15]:
            es = k.get("idioma", "en") == "es"
            trad = traductor(k["titulo"]) if (traductor and not es and len(filas) < 8) else None
            tono = F.tono(k["puntaje"], k.get("palabras"))
            filas.append([tono, e(trad or k["titulo"]) + ("" if trad or es else " <span class='mut sm'>(en inglés)</span>"),
                          e(trad and k["titulo"] or "") if trad else "", e(F.n(k["puntaje"], 2, True)), e(k.get("proveedor", "")), e(str(k["ts"])[:16].replace("T", " "))])
        sec_not = ("<p class='sm mut'>😟 mala · 😐 neutral · 🙂 buena. El tono lo calcula un programa que lee el titular en inglés: no entiende contexto financiero, úsalo como pista, no como verdad.</p>"
                   + _tabla(["", "Titular", "Original", "Tono", "Fuente", "Fecha (UTC)"], filas, 2))

    # ---------- 5. Regla Maestra paso a paso
    g = dec.g
    pasos = []
    if rent.get("mia") is None or rent.get("umbral") is None:
        pasos.append("Falta tu ranking: escribe <code>/rank 6,5 11</code> (tu rentabilidad y la del corte, en %) y la regla se puede calcular.")
    else:
        pasos.append(f"<b>1. Cuánto te falta (g).</b> g = objetivo ({F.n(rent['umbral'], 1)}%) − tu rentabilidad ({F.n(rent['mia'], 1)}%) + {cfg['regla_maestra']['margen_g_pp']:g} de margen = <b>{F.n(g, 2) if g is not None else 'n. d.'} puntos</b>. "
                     + ("Como es 0 o menos, ya vas por encima del objetivo y lo mejor es mantener." if g is not None and g <= 0 else ""))
        if dec.multiplo:
            pasos.append(f"<b>2. Qué tanto más debe moverse el nuevo.</b> Cambiar cuesta ≈ {cfg['regla_maestra']['costo_cambio_pp']:g}%, así que el candidato debe moverse al menos <b>{F.n(dec.multiplo, 2)} veces</b> lo que se mueve tu acción "
                         f"(fórmula: (g + {cfg['regla_maestra']['costo_cambio_pp']:g}) / g" + (f", o con d = {cfg['regla_maestra']['d_negro']:g} si tu acción está en NEGRO y esa opción está activa" if cfg["regla_maestra"]["cambio_por_negro"] else "") + ").")
        pasos.append(f"<b>3. Cuánto se espera que se mueva tu acción.</b> ±{F.pct(r.mee, 2) if r.mee else 'n. d.'} hasta el corte (método: {e(r.metodo_mee)}"
                     + (f"; vencimiento de opciones {e(r.iv['vencimiento'])}" if r.iv and r.iv.get("vencimiento") else "") + ").")
    evalu = "".join(f"<tr><td class='l'>{e(x['ticker'])}</td><td>±{F.pct(x['mee'], 2)}</td><td>{F.n(x['ratio'], 2)}×</td><td>{'✅ cumple' if x['cumple'] else 'no alcanza'}</td></tr>" for x in dec.evaluados)
    sec_regla = ("<ol>" + "".join(f"<li>{x}</li>" for x in pasos) + "</ol>"
                 + (f"<h3>Candidatos revisados (en el orden del banco)</h3><div class='tw'><table><thead><tr><th class='l'>Candidato</th><th>Se espera que se mueva</th><th>Veces tu acción</th><th>¿Alcanza?</th></tr></thead><tbody>{evalu}</tbody></table></div>" if evalu else "")
                 + f"<div class='card'><b>Conclusión:</b> {'<br>'.join(x for x in F.texto_decision(dec, res.ticker, hora))}</div>")

    # ---------- 6. banco
    en_banco = {x.c.ticker for x in r.banco}
    maxm = max([x.c.mee for x in r.banco] + [r.mee or 0.0001])
    filas_b = []
    for i, x in enumerate(r.banco, 1):
        c = x.c
        parecido = F._parecido(c.corr, res.ticker) or "n. d."
        filas_b.append([str(i), f"<b>{e(c.ticker)}</b> <span class='mut sm'>({e(c.grupo)})</span>", _pill(c.color),
                        f"±{F.pct(c.mee, 1)}<div class='bar' style='width:{c.mee / maxm * 100:.0f}%'></div>", e(F.n(c.mee / r.mee, 2) + "×") if r.mee else "n. d.", e(parecido),
                        e(f"{c.valor_negociado_mm:,.0f}") if c.valor_negociado_mm else "n. d.",
                        e("; ".join(m.replace("correlación", "se parece a tu acción en").replace("sentimiento", "tono de noticias") for m in x.razones if not m.startswith("MEE")))])
    sec_banco = (_tabla(["#", "Candidato", "Semáforo", "Movimiento esperado", "Veces tu acción", "Parecido a tu acción", "Valor negociado/día (millones)", "Motivos"], filas_b, 2) if filas_b
                 else "<p>Hoy ningún candidato cumple los filtros.</p>")
    resto = [c for c in sorted(r.candidatos, key=lambda c: -(c.mee or 0)) if c.ticker not in en_banco]
    filas_d = [[e(c.ticker), e(c.grupo), f"±{F.pct(c.mee, 1)}" if c.mee else "n. d.", e(porque_descartado(c, res.ticker, cfg, permitidos, set(r.excluidos)))] for c in resto[:30]]
    sec_banco += ("<h3>Los demás evaluados y por qué no están en el top</h3>" + _tabla(["Acción", "Grupo", "Movimiento esperado", "Por qué no"], filas_d, 1)) if filas_d else ""
    sec_banco += (f"<p class='sm mut'>Se revisan todas las acciones del concurso con datos: las {len([c for c in r.candidatos if c.grupo == 'mgc'])} de EE. UU. (MGC) y las {len([c for c in r.candidatos if c.grupo == 'local'])} de la BVC. "
                  "Orden: más movimiento esperado primero; si hay empate (menos de 10 % de diferencia), gana la que menos se parezca a tu acción, luego la que tenga una fecha importante antes del corte.</p>")

    # ---------- 6b. ¿hay una base mejor para TODO el concurso?
    if base_rank:
        from .seleccion import veredicto
        v, mejores = veredicto(base_rank, cfg)
        maxa = max(x["mee"] for x in base_rank)
        filas_r = [[str(i), f"<b>{e(x['ticker'])}</b> <span class='mut sm'>({'BVC' if x['grupo'] == 'local' else 'EE. UU.'})</span>" + (" ← tu base" if x["es_base"] else ""),
                    f"±{F.pct(x['mee'], 1)}<div class='bar' style='width:{x['mee'] / maxa * 100:.0f}%'></div>", e(F.n(x["vs_base"], 2) + "×"),
                    e(F.pct(x["real"], 0)), e(F.pct(x["iv"], 0)) if x["iv"] else "<span class='mut'>n. d.</span>", e(f"{x['liquidez_mm']:,.0f}"), "⚠️ supera el umbral" if x["supera"] else ""]
                   for i, x in enumerate(base_rank[:15], 1)]
        veredicto_txt = (f"<p><b>⚠️ Revisa la base:</b> {e(mejores[0]['ticker'])} supera a {e(res.ticker)} por más del {F.pct(cfg['seleccion_base']['ventaja_minima'], 0)}.</p>" if mejores else
                         f"<p><b>✅ Mantén {e(res.ticker)}:</b> ninguna otra acción líquida (EE. UU. o BVC) supera su movimiento esperado por el {F.pct(cfg['seleccion_base']['ventaja_minima'], 0)} necesario para justificar un cambio.</p>")
        sec_base = (veredicto_txt + _tabla(["#", "Acción", "Movimiento esperado hasta el final", "Veces la base", "Volatilidad anual (60 días)", "Volatilidad de opciones", "Valor negociado/día (millones)", ""], filas_r, 2)
                    + "<p class='sm mut'>Movimiento esperado = volatilidad anual × √(días hasta el 6-nov / 365); si hay opciones se mezcla 50 % opciones y 50 % últimos 60 días. "
                      "Es oportunidad y riesgo por igual; no es una promesa de ganancia. Revisa las acciones de la BVC y las de EE. UU. por igual.</p>")
    else:
        sec_base = "<p>No pude calcular la comparación de base en esta consulta.</p>"

    # ---------- 7. catalizadores
    icono = {"corte": "🏁", "macro": "🏦", "reporte": "📊"}
    filas_c = [[e(F.fecha(x["fecha"])) + (f" {e(x['hora'])}" if x.get("hora") else ""), icono.get(x["tipo"], "•") + " " + e(x["detalle"]) + (" <span class='mut sm'>(fecha por confirmar)</span>" if x.get("verificada") is False else "")] for x in cats]
    sec_cat = _tabla(["Fecha", "Qué pasa"], filas_c, 2) if filas_c else f"<p>No hay fechas importantes conocidas en los próximos {cfg['radar']['dias_catalizadores']} días hábiles.</p>"

    # ---------- 8. salud / supuestos
    salud = est.d.get("salud", {})
    ok_txt = ", ".join(f"{k}: {'✅' if v.get('fallos', 0) == 0 else '⚠️'}" for k, v in salud.items()) or "sin lecturas guardadas todavía"
    avis = "".join(f"<li>{e(x)}</li>" for x in r.avisos[:12])
    limites = [
        "Es una ayuda para decidir, no una orden: el bot <b>nunca compra ni vende</b>. Las órdenes las das tú en trii.",
        f"El movimiento esperado usa {'las opciones (volatilidad implícita)' if r.metodo_mee == 'IV' else 'la variación diaria de los últimos 20 días'}; en la BVC siempre se usa la variación diaria.",
        "Las noticias vienen de Finnhub, Yahoo y Google News. Si todas fallan, el sistema sospecha de una mala noticia cuando hay mucho volumen y un hueco de apertura (se avisa en el mensaje).",
        "La traducción de titulares es automática y literal; puede sonar torpe.",
        "Los resultados históricos comparan reglas con datos del pasado; no garantizan nada para este concurso.",
    ]
    sec_sal = (f"<p><b>Fuentes de datos:</b> {e(ok_txt)}.</p>" + (f"<div class='warn'><b>Avisos de esta consulta:</b><ul>{avis}</ul></div>" if avis else "")
               + "<ul>" + "".join(f"<li>{x}</li>" for x in limites) + "</ul>")

    gl = [("Semáforo", "Resumen del estado de tu acción: verde normal, azul cae el mercado, amarillo cae sola, rojo cae sola con mala noticia, negro no se recupera."),
          ("z", "Cuántas variaciones diarias típicas se movió el precio hoy."), ("Residual", "Lo que se mueve la acción después de quitar el arrastre del mercado."),
          ("Movimiento esperado", "Rango típico de movimiento hasta el próximo corte (±)."), ("Corte", "Viernes en que solo pasan los mejores del ranking."),
          ("Banco de relevo", "Lista de acciones candidatas para reemplazar la tuya si la Regla Maestra lo recomienda.")]
    dl = "".join(f"<dt>{e(a)}</dt><dd>{e(b)}</dd>" for a, b in gl)

    nav = f"<p class='mut sm'>Generado el {e(F.fecha(ahora))} a las {ahora:%H:%M} (hora de Bogotá), a pedido en el bot. Datos de Yahoo Finance y Finnhub.</p>"
    cuerpo = (f"<h1>Informe de {e(res.ticker)} y de tus alternativas</h1>{nav}{resumen}"
              f"<h2>1 · Tu posición y el ranking</h2>{_tabla(['Concepto', 'Dato', 'Qué es'], filas_pos, 3)}"
              f"<h2>2 · El semáforo, número por número</h2>{sec_sem}<h2>3 · Noticias</h2>{sec_not}"
              f"<h2>4 · La Regla Maestra, paso a paso</h2>{sec_regla}<h2>5 · Banco de relevo: ¿hay algo mejor que {e(res.ticker)}?</h2>{sec_banco}"
              f"<h2>6 · ¿Hay una base mejor para todo el concurso?</h2>{sec_base}"
              f"<h2>7 · Qué viene</h2>{sec_cat}<h2>8 · Qué dice la validación histórica</h2>{_validacion(cfg)}"
              f"<h2>9 · Fuentes y límites</h2>{sec_sal}<h2>Glosario</h2><dl>{dl}</dl>"
              "<footer>Informe generado por el bot a pedido. Sin garantías: el bot nunca envía órdenes.</footer>")
    return (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Informe {e(res.ticker)} · {ahora:%d/%m/%Y %H:%M}</title><style>{CSS}</style></head><body><main>{cuerpo}</main></body></html>")


__all__ = ["armar_informe", "json", "EMOJI", "AMARILLO", "AZUL"]
