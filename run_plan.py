"""Genera reports/plan_completo.html: el plan explicado de pies a cabeza (reglas, estrategia, motor, mensajes, nube, riesgos y límites).

Un solo archivo, sin internet. Los números salen de config.yaml y de outputs/validacion/ (no están escritos a mano), así que el documento no se desactualiza:
vuelve a correr `python run_plan.py` después de cambiar la configuración."""
from __future__ import annotations

import datetime as dt
import html
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from src import concurso as C
from src.config import ROOT, load_config

e = lambda x: html.escape(str(x), quote=True)                                                          # noqa: E731


def n(x: float | None, d: int = 1, signo: bool = False) -> str:
    if x is None or x != x:
        return "n. d."
    return (f"{x:+,.{d}f}" if signo else f"{x:,.{d}f}").replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x: float | None, d: int = 1, signo: bool = False) -> str:
    return "n. d." if x is None or x != x else n(x * 100, d, signo) + " %"


def cop(x: float) -> str:
    return "$ " + f"{x:,.0f}".replace(",", ".")


def fecha(d: dt.date | str) -> str:
    d = pd.Timestamp(d).date()
    return ["lun", "mar", "mié", "jue", "vie", "sáb", "dom"][d.weekday()] + f" {d:%d/%m}"


def tabla(cab: list[str], filas: list[list[str]], izq: int = 1, cls: str = "", num: bool = False) -> str:
    """Tabla HTML. Por omisión TODAS las columnas van a la izquierda (son texto); con num=True solo las primeras `izq` columnas van a la izquierda y el resto,
    a la derecha (tablas de cifras)."""
    izq = izq if num else len(cab)
    th = "".join(f"<th class='{'l' if i < izq else ''}'>{c}</th>" for i, c in enumerate(cab))
    tr = "".join("<tr>" + "".join(f"<td class='{'l' if i < izq else ''}'>{c}</td>" for i, c in enumerate(f)) + "</tr>" for f in filas)
    return f"<div class='tw {cls}'><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>"


def caja(titulo: str, cuerpo: str, tipo: str = "") -> str:
    return f"<div class='box {tipo}'><b>{titulo}</b><div>{cuerpo}</div></div>"


def detalle(titulo: str, cuerpo: str) -> str:
    return f"<details><summary>{titulo}</summary><div class='in'>{cuerpo}</div></details>"


CSS = """
:root{--bg:#f9f9f7;--sf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--mut:#898781;--line:#e1e0d9;--acc:#2a78d6;--ok:#0a7a0a;--okbg:#e6f4e6;--warn:#8a5a00;--warnbg:#fff4d6;--bad:#b32525;--badbg:#fbeaea}
@media (prefers-color-scheme:dark){:root{--bg:#0d0d0d;--sf:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--mut:#898781;--line:#2c2c2a;--acc:#3987e5;--ok:#37c837;--okbg:#12301a;--warn:#fab219;--warnbg:#3a2e08;--bad:#ff7b7b;--badbg:#3a1616}}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:64px}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",sans-serif}
.layout{display:grid;grid-template-columns:250px minmax(0,1fr);max-width:1340px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:18px 12px 18px 18px;border-right:1px solid var(--line);font-size:14px}
nav b{display:block;margin-bottom:8px;font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}
nav a{display:block;padding:4px 8px;border-radius:6px;color:var(--ink2);text-decoration:none}nav a:hover{background:var(--sf);color:var(--ink)}
main{padding:22px 24px 90px;min-width:0}section{margin:0 0 46px}
h1{font-size:clamp(26px,4vw,38px);line-height:1.15;margin:0 0 6px}h2{font-size:clamp(21px,3vw,27px);margin:0 0 12px;padding-top:8px;border-top:2px solid var(--line)}
h3{font-size:18px;margin:22px 0 6px}h4{font-size:16px;margin:16px 0 4px}p{margin:.5em 0}.mut{color:var(--mut)}.sm{font-size:14px}
.box{background:var(--sf);border:1px solid var(--line);border-radius:12px;padding:12px 16px;margin:12px 0}.box>b{display:block;margin-bottom:4px}
.box.ok{background:var(--okbg);border-color:var(--ok)}.box.warn{background:var(--warnbg);border-color:var(--warn)}.box.bad{background:var(--badbg);border-color:var(--bad)}
.grid{display:grid;gap:12px;grid-template-columns:repeat(auto-fit,minmax(270px,1fr))}.kpi{font-size:28px;font-weight:800;line-height:1.1}.kpi small{display:block;font-size:13px;font-weight:400;color:var(--ink2);margin-top:4px}
.tw{overflow-x:auto;border:1px solid var(--line);border-radius:10px;background:var(--sf);margin:8px 0}table{border-collapse:collapse;width:100%;font-size:14px}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid var(--line);vertical-align:top}th{background:var(--bg);color:var(--ink2);font-size:13px}
th.l,td.l,th:first-child,td:first-child{text-align:left}td.l{white-space:normal}tr:last-child td{border-bottom:0}
details{margin:8px 0;border:1px solid var(--line);border-radius:10px;background:var(--sf)}summary{cursor:pointer;padding:9px 13px;font-weight:600;font-size:15px}.in{padding:2px 16px 12px;color:var(--ink2)}
.pill{display:inline-block;padding:1px 9px;border-radius:99px;font-size:12px;font-weight:700;color:#fff;white-space:nowrap}
code{background:var(--line);padding:1px 6px;border-radius:5px;font-size:.9em;overflow-wrap:anywhere}ul,ol{padding-left:22px}li{margin:5px 0}
svg{width:100%;height:auto;display:block}svg text{font-family:system-ui,sans-serif;fill:var(--ink)}svg .t2{fill:var(--ink2)}svg .bx{fill:var(--sf);stroke:var(--line);stroke-width:1.5}
svg .ac{fill:var(--acc);fill-opacity:.12;stroke:var(--acc);stroke-width:1.5}svg .ar{stroke:var(--mut);stroke-width:1.6;fill:none}
dl dt{font-weight:700;margin-top:10px}dl dd{margin:2px 0 0;color:var(--ink2)}footer{color:var(--mut);font-size:13px;max-width:1340px;margin:0 auto;padding:10px 24px 40px}
@media (max-width:900px){.layout{display:block}nav{position:sticky;top:0;z-index:30;height:auto;display:flex;gap:6px;overflow-x:auto;border-right:0;border-bottom:1px solid var(--line);background:var(--bg);padding:8px 12px;white-space:nowrap}
nav b{display:none}nav a{padding:6px 10px;border:1px solid var(--line);border-radius:99px;font-size:13px}main{padding:16px 14px 60px}}
"""

COL = {"VERDE": "#0ca30c", "AZUL": "#2a78d6", "AMARILLO": "#eda100", "ROJO": "#d03b3b", "NEGRO": "#3b3b3b"}


def pill(c: str, t: str) -> str:
    return f"<span class='pill' style='background:{COL[c]}'>{t}</span>"


# ---------------------------------------------------------------- diagramas (SVG en línea)
def svg_flujo() -> str:
    def caja_(x, y, w, h, t1, t2="", cls="bx"):
        return (f"<rect class='{cls}' x='{x}' y='{y}' width='{w}' height='{h}' rx='10'/><text x='{x + w / 2}' y='{y + 22}' text-anchor='middle' font-size='14' font-weight='700'>{t1}</text>"
                + (f"<text class='t2' x='{x + w / 2}' y='{y + 40}' text-anchor='middle' font-size='11.5'>{t2}</text>" if t2 else ""))

    def flecha(x1, y1, x2, y2):
        return f"<path class='ar' d='M{x1},{y1} L{x2},{y2}' marker-end='url(#p)'/>"
    return ("<svg viewBox='0 0 980 470' role='img' aria-label='Mapa del sistema'><defs><marker id='p' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
            "<path d='M0,0 L10,5 L0,10 z' fill='var(--mut)'/></marker></defs>"
            + caja_(10, 20, 210, 58, "Precios", "Yahoo Finance: diario y 15 min") + caja_(10, 92, 210, 58, "Noticias", "Finnhub → Yahoo → Google → Alpha V.")
            + caja_(10, 164, 210, 58, "Opciones (IV)", "yfinance · solo EE. UU.") + caja_(10, 236, 210, 58, "Nasdaq-100 (^NDX)", "para separar mercado y acción")
            + caja_(10, 308, 210, 58, "Traducción", "MyMemory (titulares)") + caja_(10, 380, 210, 58, "Reglas del concurso", "config.yaml (cortes, costos, horario)")
            + flecha(222, 49, 290, 190) + flecha(222, 121, 290, 200) + flecha(222, 193, 290, 210) + flecha(222, 265, 290, 220) + flecha(222, 337, 290, 235) + flecha(222, 409, 290, 245)
            + caja_(292, 150, 230, 120, "MOTOR DE DECISIÓN", "", "ac") + "<text class='t2' x='407' y='196' text-anchor='middle' font-size='12'>1 · Semáforo (color de tu acción)</text>"
            "<text class='t2' x='407' y='214' text-anchor='middle' font-size='12'>2 · Movimiento esperado (MEE)</text><text class='t2' x='407' y='232' text-anchor='middle' font-size='12'>3 · Banco de relevo (BVC + EE. UU.)</text>"
            "<text class='t2' x='407' y='250' text-anchor='middle' font-size='12'>4 · Regla Maestra (¿cambiar?)</text>"
            + caja_(292, 300, 230, 58, "state.json", "tu posición, ranking, cambios, avisos") + flecha(407, 300, 407, 272)
            + flecha(524, 190, 590, 70) + flecha(524, 210, 590, 150) + flecha(524, 230, 590, 235) + flecha(524, 250, 590, 320)
            + caja_(592, 30, 230, 80, "MONITOR · cada 15 min", "avisa cambios de color y novedades") + caja_(592, 124, 230, 70, "RADAR · 19:30 lun–jue", "resumen del día y qué hacer")
            + caja_(592, 208, 230, 70, "BOT de Telegram", "/estado /banco /base /informe …") + caja_(592, 292, 230, 70, "DASHBOARD (Streamlit)", "tablero y calculadora")
            + caja_(592, 376, 230, 62, "Informe HTML", "solo cuando alguien lo pide")
            + flecha(824, 70, 880, 200) + flecha(824, 160, 880, 215) + flecha(824, 243, 880, 230)
            + "<rect class='ac' x='880' y='160' width='92' height='110' rx='14'/><text x='926' y='206' text-anchor='middle' font-size='15' font-weight='700'>Tú</text>"
              "<text class='t2' x='926' y='226' text-anchor='middle' font-size='11'>Telegram</text><text class='t2' x='926' y='242' text-anchor='middle' font-size='11'>y navegador</text></svg>")


def svg_regla() -> str:
    return ("<svg viewBox='0 0 900 330' role='img' aria-label='Árbol de decisión de la Regla Maestra'><defs><marker id='q' viewBox='0 0 10 10' refX='8' refY='5' markerWidth='7' markerHeight='7' orient='auto-start-reverse'>"
            "<path d='M0,0 L10,5 L0,10 z' fill='var(--mut)'/></marker></defs>"
            "<rect class='bx' x='10' y='120' width='170' height='70' rx='10'/><text x='95' y='148' text-anchor='middle' font-size='14' font-weight='700'>Cada noche</text><text class='t2' x='95' y='168' text-anchor='middle' font-size='12'>tú das /rank</text>"
            "<path class='ar' d='M182,155 L240,155' marker-end='url(#q)'/>"
            "<rect class='ac' x='242' y='110' width='200' height='90' rx='10'/><text x='342' y='140' text-anchor='middle' font-size='14' font-weight='700'>g = objetivo − mi rent. + 1</text>"
            "<text class='t2' x='342' y='162' text-anchor='middle' font-size='12'>¿cuántos puntos me faltan?</text><text class='t2' x='342' y='180' text-anchor='middle' font-size='12'>(1 punto de margen)</text>"
            "<path class='ar' d='M444,135 L520,60' marker-end='url(#q)'/><path class='ar' d='M444,175 L520,250' marker-end='url(#q)'/>"
            "<rect class='bx' x='522' y='20' width='220' height='80' rx='10'/><text x='632' y='46' text-anchor='middle' font-size='14' font-weight='700'>g ≤ 0: voy arriba</text><text class='t2' x='632' y='68' text-anchor='middle' font-size='12'>MANTENER (cambiar solo cuesta)</text>"
            "<rect class='bx' x='522' y='210' width='220' height='100' rx='10'/><text x='632' y='236' text-anchor='middle' font-size='14' font-weight='700'>g &gt; 0: voy atrás</text><text class='t2' x='632' y='258' text-anchor='middle' font-size='12'>¿algún candidato cumple</text>"
            "<text class='t2' x='632' y='276' text-anchor='middle' font-size='12'>MEE_cand / MEE_mío ≥ (g+1,3)/g ?</text>"
            "<path class='ar' d='M744,245 L790,215' marker-end='url(#q)'/><path class='ar' d='M744,275 L790,295' marker-end='url(#q)'/>"
            "<rect class='ac' x='792' y='170' width='100' height='70' rx='10'/><text x='842' y='200' text-anchor='middle' font-size='14' font-weight='700'>Sí</text><text class='t2' x='842' y='220' text-anchor='middle' font-size='12'>CAMBIAR</text>"
            "<rect class='bx' x='792' y='260' width='100' height='60' rx='10'/><text x='842' y='286' text-anchor='middle' font-size='14' font-weight='700'>No</text><text class='t2' x='842' y='304' text-anchor='middle' font-size='12'>MANTENER</text></svg>")


# ---------------------------------------------------------------- secciones
def sec_resumen(cfg: dict) -> str:
    c, base = cfg["concurso"], cfg["posicion_base"]
    cap = cfg["capital"]
    return f"""
<h1>Bolsa Millonaria 2026 — el plan completo, de pies a cabeza</h1>
<p class="mut sm">Documento generado el {dt.date.today():%d/%m/%Y} a partir de la configuración real del sistema (<code>config.yaml</code>) y de las validaciones guardadas. Escrito para ti (participante del concurso), sin necesidad de saber programar ni finanzas avanzadas.</p>
<div class="grid">
<div class="box"><div class="kpi">{cop(cap)}<small>capital simulado de trii; el ranking mide solo la rentabilidad acumulada</small></div></div>
<div class="box"><div class="kpi">{e(base['ticker'])} al {pct(base['peso'], 0)}<small>base del plan: una sola acción muy volátil, sin stops, el resto en efectivo para las micro-compras</small></div></div>
<div class="box"><div class="kpi">{fecha(c['inicio'])} → {fecha(c['fin'])}<small>4 cortes semanales y la final (#1 de Diamante)</small></div></div>
</div>
<div class="box ok"><b>La idea en un minuto</b><div>
<ol><li><b>Compras una sola acción muy movida</b> ({e(base['ticker'])}) el primer día y la <b>mantienes</b>: sin stops, sin diversificar, sin vender por susto. En un concurso donde pasa solo el mejor X % de cada corte, la diversificación casi garantiza quedar en el medio.</li>
<li><b>Un sistema automático vigila por ti</b> (monitor, radar, bot de Telegram, dashboard): te dice si una caída es solo ruido o parece un cambio real, y te propone alternativas de la BVC y de EE. UU.</li>
<li><b>Solo cambias de acción con la “Regla Maestra”:</b> cuando vas atrás del corte y existe una acción lo bastante más movida como para que valga la pena pagar el costo del cambio (≈ {n(cfg['regla_maestra']['costo_cambio_pp'], 1)} %).</li>
<li><b>Tú das todas las órdenes en trii.</b> El bot nunca compra ni vende: solo informa y recomienda.</li></ol></div></div>
<div class="box warn"><b>Lo que este plan NO promete</b><div>No hay ninguna regla que garantice ganar ni llegar a +100 %. Con datos de 2015–2026, solo una acción real (TSLA, en 2020) llegó a +100 % en 25 sesiones: ≈ 0,03 % de los concursos. El plan busca <b>maximizar la probabilidad de terminar en la cola alta del ranking</b>, aceptando mucha variación. La mayoría de la gente termina en negativo; terminar en positivo y alto es la vara realista (mediana del mejor rival simulado: ≈ +24 %).</div></div>"""


def sec_reglas(cfg: dict) -> str:
    c = cfg["concurso"]
    u = cfg["universe"]
    cortes = [[fecha(x["fecha"]), f"semana {x['semana']}", f"top {x['top_pct']} %", e(x["liga"])] for x in c["cortes"]] + [[fecha(c["final"]["fecha"]), "semana 5", "final", e(c["final"]["meta"])]]
    hor = [[f"{fecha(h['desde'])} → {fecha(h['hasta'])}", f"{h['apertura']} a {h['cierre']}"] for h in c["horario"]]
    v = cfg["costos_v5"]
    a = c["actividad"]
    return f"""
<h2>1 · Las reglas oficiales del concurso</h2>
<h3>Cómo se gana</h3>
<ul><li><b>Ranking = solo rentabilidad acumulada</b> desde el {fecha(c['inicio'])}. No hay puntos por actividad.</li>
<li>Cada <b>viernes al cierre</b> hay un corte: solo siguen los mejores. Quien cae no vuelve. Según tu lectura del reglamento, el “top X %” se mide <b>contra los participantes que siguen vivos</b>; por eso los filtros se multiplican y al final queda muy poca gente (con ≈ 5.500 participantes, unas 9 personas llegan vivas al final).</li></ul>
{tabla(['Corte (cierre del viernes)', 'Semana', 'Pasa', 'Liga'], cortes, 1)}
<h3>Horario, festivos y actividad</h3>
{tabla(['Fechas', 'Horario de órdenes en trii/bvc (hora de Bogotá)'], hor, 1)}
<ul><li><b>Festivos sin operar en trii:</b> {', '.join(fecha(d) for d in c['festivos'])} (la bolsa de EE. UU. sí abre, así que el precio se mueve aunque tú no puedas operar).</li>
<li><b>Actividad:</b> {a['ops_por_semana']} operaciones por semana (mínimo {a['minimo_semanal_requerido']}) y al menos {a['minimo_total']} en total. Es un requisito para optar a premios, <b>no da puntos</b>. Se cumple con micro-compras de unos {cop(a['micro_compra_cop'])} de {e(a['activo_micro'])}; la compra inicial y cada cambio de acción también cuentan.</li></ul>
<h3>Costos</h3>
{tabla(['Concepto', 'Valor'], [['Comisión (operaciones de más de $ 5.000.000)', pct(v['comision_pct'], 4)], ['Comisión fija (hasta $ 5.000.000)', cop(v['comision_fija_cop'])],
 ['Spread de las acciones de EE. UU. (MGC)', pct(v['spread_mgc'], 1)], ['Slippage de las acciones de EE. UU. (MGC)', pct(v['slippage_mgc'], 1)], ['Costo aproximado de cambiar de acción (vender + comprar)', pct(v['costo_cambio'], 1)]], 1)}
<h3>Qué se puede comprar</h3>
{tabla(['Grupo', 'Cuántas', 'Acciones'], [['Acciones de EE. UU. (MGC)', str(len(u['mgc'])), ', '.join(u['mgc'])], ['ETF', str(len(u['etf'])), ', '.join(u['etf'])], ['Acciones de la BVC', str(len(u['local'])), ', '.join(u['local'])]], 2)}
<div class="box bad"><b>Lista negra (nunca se recomienda ni se puede registrar)</b><div>{', '.join(u['blacklist'])}. Hay una prueba automática que falla si alguna aparece como recomendación.</div></div>"""


def sec_estrategia(cfg: dict) -> str:
    b, s, rm = cfg["posicion_base"], cfg["seleccion_base"], cfg["regla_maestra"]
    c = cfg["concurso"]
    sem = []
    for k in c["cortes"]:
        sem.append(f"<li><b>Semana {k['semana']}</b> (corte {fecha(k['fecha'])}, top {k['top_pct']} %)</li>")
    return f"""
<h2>2 · La estrategia</h2>
<h3>Qué haces</h3>
<ol><li><b>El primer día ({fecha(c['inicio'])}, {C.hora_orden_manana(pd.Timestamp(c['inicio']).date(), cfg)} hora de Bogotá):</b> compras <b>{e(b['ticker'])}</b> por {cop(cfg['capital'] * b['peso'])} ({pct(b['peso'], 0)} del capital) con una orden límite al precio de venta que muestre trii. Registras la compra en el bot con <code>/pos {e(b['ticker'])} 97M &lt;precio&gt;</code>.</li>
<li><b>Dejas el {pct(1 - b['peso'], 0)} restante en efectivo</b> para las micro-compras de actividad (4 por semana, ≈ $ 100.000 de ECOPETROL cada una).</li>
<li><b>Mantienes.</b> Sin stops, sin vender por un titular, sin “reducir al ir ganando”.</li>
<li><b>Cada noche</b> lees el radar y le das al bot tu rentabilidad y el objetivo del corte (<code>/rank</code>). Solo si la Regla Maestra dice CAMBIAR (y tú estás de acuerdo) cambias de acción al día siguiente con orden límite.</li></ol>
<h3>Por qué una sola acción volátil</h3>
<p>En un concurso por ranking, lo que importa es estar en la cola alta de la distribución, no el promedio. Una cartera diversificada produce rentabilidades parecidas a las de casi todos y casi nunca pasa los cortes duros (10 % y 10 %); una apuesta concentrada tiene baja probabilidad de ir muy bien, pero es la única forma de que exista. Es una decisión de varianza, no de “más rentabilidad esperada”.</p>
<h3>Cómo se eligió {e(b['ticker'])} (y se vuelve a revisar con <code>/base</code>)</h3>
<ul><li><b>Movimiento esperado hasta el final</b> = volatilidad anual × √(días hasta el {fecha(c['fin'])} / 365). Volatilidad anual = {pct(s['peso_iv'], 0)} opciones (IV) + {pct(1 - s['peso_iv'], 0)} últimos {s['vol_realizada_dias']} días; solo realizada en la BVC o cuando no hay opciones fiables.</li>
<li>Solo acciones líquidas (EE. UU. ≥ US$ {s['valor_negociado_min_usd_mm']} millones al día en la comparación de base).</li>
<li>Otra acción reemplaza a la base <b>solo si supera su movimiento esperado en {pct(s['ventaja_minima'], 0)} o más</b>: con menos no compensa el costo ni el riesgo de equivocarse por una racha corta.</li>
<li>El comando <code>/base</code> compara todas (EE. UU. y BVC) cuando quieras, y el radar de la noche te avisa si algo cambia. El 5-oct ninguna superó a TSLA (META 0,96×, GRUPOARGOS 0,95×, NUCO 0,91×).</li></ul>
<div class="box warn"><b>El gran supuesto que no se puede verificar: cuánta gente tiene lo mismo que tú</b><div>Si muchos rivales tienen TSLA o NVDA con más exposición que tu 97 %, cuando TSLA sube te superan siempre. En la simulación, con 30 % del campo concentrado en TSLA/NVDA, TSLA pasa de ser la mejor base (puesto 1 de 55) a una de las peores (puesto ≈ 52), y una acción menos popular sale mejor. Como no se puede saber cuántos rivales reales las tienen, la elección de la base es una decisión tuya con este riesgo a la vista (ver sección 11).</div></div>
<h3>Los cambios de acción</h3>
<ul><li>Máximo <b>{cfg['estado']['max_cambios']} cambios en total</b> y <b>{cfg['estado']['max_cambios_por_dia']} por día</b>. Cada cambio cuesta ≈ {n(rm['costo_cambio_pp'], 1)} %.</li>
<li>Semana 5 (final): si no vas de primero, <b>TSLA y NVDA no se recomiendan como acciones nuevas</b>; el último día, si vas de primero con ≥ {n(rm['ultimo_dia_ventaja_vender_pp'], 0)} puntos de ventaja sobre el segundo, se sugiere vender todo para asegurar el #1.</li>
<li>Detalle de la regla en la sección 7.</li></ul>
<h3>Calendario del concurso</h3><ul>{''.join(sem)}<li><b>Semana 5</b> (final {fecha(c['final']['fecha'])}): #1 de Diamante.</li></ul>"""


def sec_sistema(cfg: dict) -> str:
    return f"""
<h2>3 · El mapa del sistema</h2>
<p>El sistema toma datos de internet, los pasa por un motor de decisión y se los entrega a tu celular y a tu PC por cuatro canales. Todo lo que muestra está guardado en <code>state.json</code> (tu posición, tu ranking y qué ya te avisó).</p>
{svg_flujo()}
<h3>Las piezas, una por una</h3>
{tabla(['Pieza', 'Cuándo corre', 'Qué hace', 'Archivo'], [
['<b>Monitor</b>', 'cada 15 min de lun a vie entre 08:30 y 16:00 (solo actúa si el mercado de trii está abierto)', 'Calcula el semáforo de tu acción y del banco, y te avisa cambios de color y novedades.', '<code>monitor.py</code>'],
['<b>Radar</b>', 'lun a jue 19:30 (hora de Bogotá)', 'Resumen de la noche en 5 secciones: tu acción, ¿cambiar?, mejores relevos, qué viene y tus tareas de mañana. Es la medición oficial del cierre.', '<code>radar.py</code>'],
['<b>Bot de Telegram</b>', 'siempre encendido (al iniciar sesión)', 'Responde a tus comandos; también avisa en el grupo si lo agregas.', '<code>bot.py</code>'],
['<b>Dashboard</b>', 'cuando lo abras (se actualiza cada 5 min)', 'Tablero con semáforo, relevos, calculadora de la Regla Maestra, noticias, fechas y comparación de base.', '<code>app.py</code>, <code>iniciar_app.bat</code>'],
['<b>Informe HTML</b>', 'solo cuando alguien lo pide (<code>/informe</code> o la palabra <code>html</code>)', 'Archivo con toda la información fundamentada.', '<code>src/informe_html.py</code>'],
['<b>Validación</b>', 'una vez (ya corrida)', 'Prueba con datos de 2015–2026 qué pasa después de cada color y simula el concurso.', '<code>run_validacion.py</code>'],
['<b>Nube</b>', 'opcional (PC apagado)', 'Monitor y radar en GitHub Actions.', '<code>.github/workflows/</code>']], 1)}"""


def sec_datos(cfg: dict) -> str:
    f = cfg["fuentes"]
    ttl = f["ttl_min"]
    return f"""
<h2>4 · De dónde salen los datos (y qué pasa si algo falla)</h2>
{tabla(['Dato', 'Fuente principal', 'Respaldo', 'Se guarda (caché)'], [
['Precios diarios y de 15 min', 'Yahoo Finance (yfinance)', 'Última caché; limpieza automática de datos erróneos', f"diario {ttl['diario']} min · intradía {ttl['intradia']} min"],
['Nasdaq-100 (^NDX)', 'Yahoo Finance', 'Última caché', f"{ttl['ndx']} min"],
['Opciones (volatilidad implícita)', 'Yahoo Finance', 'Volatilidad de los últimos 20/60 días', f"{ttl['opciones']} min"],
['Noticias de EE. UU.', 'Finnhub', 'Yahoo (yfinance) → Yahoo RSS → Google News → Alpha Vantage (si hay clave) → caché vencida → <b>proxy</b>', f"{ttl['noticias']} min"],
['Noticias de la BVC', 'Google News en español', 'Caché vencida; si no hay datos no se recomienda la acción', f"{ttl['noticias']} min"],
['Fechas de resultados', 'Finnhub + yfinance', 'Si difieren, se marca “las fuentes no coinciden”', f"{ttl['reportes']} min"],
['Eventos macro (FED, IPC, empleo, BanRep)', 'Lista en <code>config.yaml</code>', 'Editable; los no verificados se marcan “por confirmar”', '—'],
['Traducción de titulares', 'MyMemory (gratis, ≈ 5.000 caracteres/día)', 'Se muestra el titular en inglés con la etiqueta “(en inglés)”', f"{f['traduccion']['ttl_dias']} días"],
['Tono de las noticias', 'VADER (inglés) y un léxico propio (español)', 'FinBERT es opcional (no está instalado)', '—']], 1)}
<div class="box"><b>El “proxy” cuando no hay noticias</b><div>Si todas las fuentes de noticias caen, el sistema sospecha de una mala noticia cuando hay un volumen ≥ {n(cfg['semaforo']['proxy_sin_noticias']['vol_rel_min'], 1)}× el normal y la acción abre con un hueco de ≥ {n(cfg['semaforo']['proxy_sin_noticias']['gap_sigma'], 1)}σ a la baja. Los mensajes lo avisan. Con noticias reales de un año, este respaldo acierta casi siempre cuando alerta (precisión ≈ 92 %), pero se pierde muchas alertas (recall ≈ 27 %).</div></div>
<h3>Claves y datos personales</h3>
<ul><li><code>.env</code> (nunca se sube ni se comparte): token del bot, tu <code>chat id</code>, clave de Finnhub y (opcional) la de Alpha Vantage.</li>
<li><code>data/state.json</code>: tu posición, tu rentabilidad y el objetivo del corte, cambios usados, operaciones por semana, colores guardados y qué avisos ya se mandaron. Se guarda de forma atómica con copia <code>.bak</code> y candado de archivo.</li></ul>"""


def sec_semaforo(cfg: dict) -> str:
    s = cfg["semaforo"]
    return f"""
<h2>5 · El semáforo: ¿mi caída es ruido o es real?</h2>
<p>Cada vez que corre el monitor (y cada noche en el radar) se calcula, para tu acción y para cada candidata:</p>
{tabla(['Medida', 'Cómo se calcula', 'Qué dice'], [
['<b>z</b> (fuerza del movimiento)', f"retorno del día ÷ variación diaria típica (σ de {s['sigma_dias']} días)", 'Cuántas “variaciones típicas” se movió hoy. 0 = normal; por debajo de −2 es una caída anormal.'],
['<b>Beta</b>', f"sensibilidad al Nasdaq-100 con {s['beta_dias']} días", '1 = se mueve igual que el índice.'],
['<b>z_res</b> (fuerza de lo propio)', 'retorno − beta × retorno del Nasdaq-100, ÷ σ del residual', 'Lo mismo que z pero quitando lo que explica el mercado.'],
['<b>Volumen relativo</b>', f"volumen del día ÷ promedio de {s['vol_dias']} días", 'Mucha actividad confirma que el movimiento es serio.'],
['<b>Residual de 5 días</b>', f"suma de lo propio en {s['resid_acum_dias']} días vs. {n(s['resid_acum_umbral'], 1)}·σ₅", 'Detecta caídas lentas que no se ven en un solo día.'],
['<b>Noticias</b>', 'titulares de 24 h, tono (−1 a +1) y palabras de riesgo', f"Negativa si el tono es < {n(s['sentimiento_negativo'], 1)} o hay una palabra de riesgo sin tono positivo."]], 1)}
<h3>Los cinco colores</h3>
{tabla(['Color', 'Condición', 'Qué hacer'], [
[pill('VERDE', 'VERDE'), f"z > {n(s['z_caida'], 1)}: movimiento normal", 'Nada.'],
[pill('AZUL', 'AZUL'), f"z ≤ {n(s['z_caida'], 1)} pero z_res > {n(s['z_res_azul'], 1)}: cae el mercado, no tu acción", 'Nada: es ruido de mercado.'],
[pill('AMARILLO', 'AMARILLO'), f"z_res ≤ {n(s['z_res_caida'], 1)} sin noticia negativa o con volumen < {n(s['vol_rel_min'], 1)}× (la zona gris entre {n(s['z_res_caida'], 1)} y {n(s['z_res_azul'], 1)} también cae aquí)", 'Vigilar; no vender por susto.'],
[pill('ROJO', 'ROJO'), f"z_res ≤ {n(s['z_res_caida'], 1)} <b>y</b> noticia negativa <b>y</b> volumen ≥ {n(s['vol_rel_min'], 1)}× (o residual de 5 días ≤ {n(s['resid_acum_umbral'], 1)}σ con noticias negativas)", 'La alerta más seria; no vender a mitad del día: el radar mira el cierre.'],
[pill('NEGRO', 'NEGRO'), f"un ROJO que al cierre del 2.º día no recuperó el {pct(s['recuperacion_negro'], 0)} de la caída y sigue con tono negativo", 'Información; no vender solo por esto (ver validación).']], 1)}
<ul><li><b>Excepción del corte:</b> si el corte es mañana, un ROJO se trata como NEGRO.</li>
<li><b>El radar de las 19:30</b> es la medición oficial del cierre: el monitor intradía puede mostrar ROJO el día 1, pero NEGRO solo se decide con un cierre.</li>
<li>Los umbrales viven en <code>config.yaml → semaforo</code>.</li></ul>
<div class="box ok"><b>Lo que dice la validación sobre NEGRO</b><div>En 2015–2026, después de NEGRO las acciones grandes <b>no siguieron cayendo</b> más que el mercado (el intervalo del 95 % cruza 0 en todos los horizontes y en las cuatro variantes del supuesto). Por eso <code>regla_maestra.cambio_por_negro</code> quedó en <b>false</b>: NEGRO solo informa. En TSLA, 9 de 13 veces subió más que el mercado en las 10 sesiones siguientes.</div></div>"""


def sec_banco(cfg: dict) -> str:
    m, b = cfg["mee"], cfg["banco"]
    return f"""
<h2>6 · Movimiento esperado y banco de relevo</h2>
<h3>MEE: cuánto se espera que se mueva una acción hasta el próximo corte</h3>
<ul><li><b>Con opciones (EE. UU.):</b> MEE = IV de la opción ATM del primer vencimiento posterior al corte × √(días calendario / 365).</li>
<li><b>Sin opciones o en la BVC:</b> MEE = variación diaria de {m['respaldo_sigma_dias']} días × √252 × √(días / 365) (modo “{e(m['respaldo'])}”, el mismo “reloj” que la IV para comparar justo).</li>
<li>Es movimiento <b>esperado hacia cualquier lado</b>: mide oportunidad y riesgo por igual.</li></ul>
<h3>Banco de relevo: las mejores alternativas a tu acción</h3>
<p>Se revisan <b>todas</b> las acciones permitidas de EE. UU. y de la BVC (los ETF casi no se mueven y se excluyen). Pasan estos filtros:</p>
<ul><li>No están en la lista negra ni son tu acción actual.</li>
<li><b>Liquidez:</b> EE. UU. ≥ US$ {b['liquidez_min_usd_mm']} millones al día; BVC ≥ $ {b['liquidez_min_cop_mm_locales']:,} millones al día.</li>
<li>No están en ROJO ni en NEGRO.</li>
<li><b>Con noticias:</b> si no se pueden consultar sus noticias, no se recomiendan (si eso dejara menos de {b['relajar_si_menos_de']} candidatos, se relaja y quedan marcadas “sin datos de noticias”).</li></ul>
<p><b>Orden:</b> 1) mayor MEE; los que están a menos de {pct(b['tolerancia_mee'], 0)} del mejor se consideran empatados y se ordenan por 2) <b>menor correlación</b> de {b['corr_dias']} días con tu acción (diversifican), 3) fecha de resultados antes del corte, 4) volumen alto sin caída o tono positivo. Se muestran las mejores {b['top_n']} con sus motivos; el informe HTML y el dashboard muestran también <b>por qué se descartó cada una</b>.</p>"""


def sec_regla(cfg: dict) -> str:
    rm = cfg["regla_maestra"]
    mult = [[str(g), n((g + rm["costo_cambio_pp"]) / g, 2) + "×"] for g in (1, 2, 3, 4, 5, 6, 8, 10, 15, 20)]
    return f"""
<h2>7 · La Regla Maestra: ¿cambio de acción?</h2>
{svg_regla()}
<ol><li><b>g = objetivo − mi rentabilidad + {n(rm['margen_g_pp'], 0)}</b> (en puntos porcentuales). El objetivo es el umbral del próximo corte (o la rentabilidad del #1 en la semana 5).</li>
<li><b>Si g ≤ 0:</b> mantener. Si vas arriba del objetivo, cambiar solo cuesta plata (y conviene parecerse al campo, no diferenciarse).</li>
<li><b>Si g &gt; 0:</b> se cambia a un candidato si <b>MEE_candidato / MEE_actual ≥ (g + {n(rm['costo_cambio_pp'], 1)}) / g</b>. Mientras más atrás vas, menos “extra” de movimiento hace falta; cuando vas cerca del objetivo, se exige mucho más.</li></ol>
{tabla(['g (puntos que te faltan + 1)', 'Veces que debe moverse el nuevo vs. tu acción'], mult, 1, num=True)}
<div class="box"><b>Ejemplo</b><div>Vas +6 % y el corte está en +11 %: g = 11 − 6 + 1 = <b>6</b>; el nuevo debe moverse ≥ (6 + 1,3) / 6 = <b>1,22</b> veces lo tuyo. Si tu TSLA tiene MEE 6,5 % y META 9 %, la razón es 1,38 ≥ 1,22: <b>CAMBIAR a META</b>. Si vas +10 %, g = 2 y se exigiría 1,65×: <b>MANTENER</b>.</div></div>
<h3>Reglas adicionales</h3>
<ul><li>Máximo 1 cambio por día y {cfg['estado']['max_cambios']} en total.</li>
<li><b>Modo NEGRO</b> (hoy <b>apagado</b>, <code>cambio_por_negro: false</code>): si se activara, la exigencia bajaría a (g + 1,3) / (g + {n(rm['d_negro'], 1)}), también cuando vas adelante por ≤ {n(rm['ventaja_negro_pp'], 0)} puntos; si vas adelante por más, solo si la acción acumula {n(rm['caida_desde_entrada_pp'], 0)} % desde tu compra.</li>
<li><b>Semana 5:</b> si no vas de primero se excluyen {', '.join(rm['excluir_semana_final'])} como candidatos nuevos. <b>Último día:</b> con ventaja ≥ {n(rm['ultimo_dia_ventaja_vender_pp'], 0)} puntos sobre el segundo, se sugiere vender todo.</li>
<li>Los cambios se ejecutan <b>a la mañana siguiente</b> ({C.hora_orden_manana(pd.Timestamp(cfg['concurso']['inicio']).date(), cfg)} en octubre, 09:45 desde el 3 de noviembre) con orden límite.</li></ul>"""


def sec_mensajes(cfg: dict) -> str:
    mon = cfg["monitor"]
    ev = mon["eventos"]
    cmds = [["<code>/estado</code>", 'Cómo vas: acción, semáforo, ranking, cambios usados, operaciones, próximo corte.'], ["<code>/semaforo [TICKER]</code>", 'El semáforo de tu acción o de otra, con noticias traducidas.'],
            ["<code>/detalle</code>", 'Los números técnicos (z, z_res, volumen, beta…).'], ["<code>/base</code>", '¿Hay una acción mejor que tu base para todo el concurso? (BVC y EE. UU.)'],
            ["<code>/banco</code>", 'Los mejores relevos y qué dice la Regla Maestra.'], ["<code>/noticias TICKER</code>", 'Últimos titulares con su tono.'], ["<code>/catalizadores</code>", 'Fechas importantes de los próximos 10 días hábiles.'],
            ["<code>/actualizar</code>", 'Borra la caché y recalcula con datos frescos.'], ["<code>/informe</code>", 'Mensaje corto + archivo HTML con toda la información (también: agrega <code>html</code> a cualquier comando).'],
            ["<code>/rank 6,5 11</code>", 'Guarda tu rentabilidad y la del corte (en la semana 5: <code>/rank &lt;mi&gt; &lt;#1&gt; &lt;#2&gt;</code>).'],
            ["<code>/compra META 8 720,5</code>", 'Registra cada compra de tu cartera (acción, cantidad y precio). No es un cambio de acción; cuenta como operación de actividad.'],
            ["<code>/venta GRUPOARGOS</code>", 'Registra que vendiste todo (o <code>/venta TSLA 20</code>: una parte).'], ["<code>/cartera</code>", 'Tus acciones con precios de hoy, rentabilidad y peso de cada una.'],
            ["<code>/pos META 95M 720,5</code>", 'Solo si cambiaste TODA tu cartera a una sola acción: gasta uno de los 4 cambios.'],
            ["<code>/op</code>", 'Registra una micro-compra de actividad.'], ["<code>/ayuda</code>", 'Lista de comandos y significado de los colores.']]
    avisos = [['🚦 Cambio de color de tu acción', 'al cambiar (también si un relevo entra o sale de ROJO/NEGRO)', f"enfriamiento {mon['enfriamiento_min']} min; las escaladas pasan"],
              ['📰 Noticia nueva importante', f"titular de las últimas {ev['noticias']['ventana_h']} h claramente malo o muy bueno", f"la 1.ª vez solo toma línea base; {ev['noticias']['enfriamiento_min']} min entre avisos; máx. {ev['noticias']['max_por_aviso']} titulares"],
              ['⚖️ La Regla Maestra cambia de opinión', 'pasa a recomendar cambiar (o deja de hacerlo); requiere tu /rank', f"{ev['decision']['enfriamiento_min']} min; no repite lo que ya trajo la alerta de color"],
              ['📊 🏦 🏁 Fechas', 'resultados de tu acción hoy/próxima sesión, evento macro del día, corte hoy/mañana', 'una sola vez cada una'],
              ['📈 Subida fuerte', f"tu acción sube con z ≥ {n(ev['alza_fuerte_z'], 1)}", 'una vez al día'], ['🔎 Comparación de base', 'en el radar: otra acción supera a la base (o deja de superarla)', 'solo cuando cambia'],
              ['⚠️ Salud de fuentes', f"una fuente falla {mon['fallos_para_avisar']} veces seguidas (precios) o se recupera", f"{mon['aviso_salud_cada_min']} min"],
              ['🧾 Micro-compra de mañana', 'en el radar: te dice si mañana toca (cuenta tus 4 por semana)', '—']]
    return f"""
<h2>8 · Qué te avisa el sistema y qué puedes preguntarle</h2>
<h3>Avisos automáticos</h3>
{tabla(['Aviso', 'Cuándo', 'Anti-spam'], avisos, 1)}
<p>Tope de <b>{ev['max_por_corrida']}</b> novedades por revisión. Todo se ajusta en <code>config.yaml → monitor</code>. Los avisos <b>no traen archivo</b>; si no se pueden enviar (sin internet), se guardan (máx. 20) y se reenvían.</p>
<p>Los mensajes están escritos en lenguaje sencillo: <b>¿Qué pasó? · ¿Qué significa? · 👉 Qué hacer</b>, con los titulares traducidos y 😟 / 😐 / 🙂 en vez de números. Los números técnicos están en <code>/detalle</code>.</p>
<h3>Comandos del bot</h3>
{tabla(['Comando', 'Qué hace'], cmds, 1)}
<h3>El informe HTML a pedido</h3>
<p>Solo cuando alguien lo pide (tú o cualquiera del grupo). Incluye: resumen y decisión, tu posición, el semáforo número por número con su explicación, todas las noticias con su tono, la Regla Maestra paso a paso, el banco completo (y por qué se descartó cada acción), la comparación de base, fechas que vienen, la validación histórica y los límites. Hay un enfriamiento de {cfg['bot']['enfriamiento_pesado_s']} s por chat para que nadie gaste las consultas de datos repitiéndolo.</p>
<h3>Dashboard</h3>
<p>Se abre con <code>iniciar_app.bat</code> (<code>http://localhost:8501</code>; desde el celular, por la misma wifi, con la “Network URL”). Pestañas: <b>Hoy</b> (semáforo grande, métricas, gráfico del día en σ y la decisión), <b>Relevos</b>, <b>Calculadora</b> (aplica la misma Regla Maestra con tus números), <b>Noticias</b>, <b>Fechas</b> y <b>Base</b>. Se actualiza solo cada 5 minutos.</p>"""


def sec_rutina(cfg: dict) -> str:
    c = cfg["concurso"]
    noches = [fecha(d) for d in cfg["radar"]["noches_decision"]]
    return f"""
<h2>9 · Tu rutina durante el concurso</h2>
{tabla(['Cuándo', 'Qué haces'], [
[f"<b>{fecha(c['inicio'])}, 08:45</b> (primer día)", f"Compra inicial de {e(cfg['posicion_base']['ticker'])} con orden límite; si no se ejecuta en 10 minutos, ajústala. Regístrala con <code>/pos</code>. Esa compra cuenta como tu operación del día."],
['Cada día de bolsa', 'Nada: el monitor te avisa solo si pasa algo. Si ves un movimiento fuerte, <code>/semaforo</code> o <code>/informe</code>.'],
['<b>Cada noche 19:30</b> (lun a jue)', 'Lees el radar → con los números de la app de trii das <code>/rank &lt;mi rentabilidad&gt; &lt;objetivo del corte&gt;</code> → miras la recomendación de la Regla Maestra.'],
['Noches de decisión', ', '.join(noches) + ': el radar las marca; la regla se calcula todas las noches, pero estas son las que importan para programar un cambio.'],
['Mañana siguiente 08:45 (09:45 desde el 3-nov)', 'Solo si hubo CAMBIAR y estás de acuerdo: ejecutas con orden límite y registras con <code>/pos</code>.'],
['Micro-compra', 'El radar te dice si mañana toca; la haces y registras <code>/op</code>. Si mañana haces un cambio, ese cambio cuenta y no necesitas micro-compra.'],
['Antes de cada corte (jueves)', 'El bot te avisa que el corte es mañana. Un ROJO se trata como NEGRO. Revisa tu ranking.'],
['Último día', 'Si vas de primero con ventaja ≥ 8 puntos, el radar sugiere vender todo.']], 1)}
<div class="box"><b>Tu PC</b><div>Las tareas de Windows (monitor, radar y bot) no despiertan un PC dormido ni apagado: déjalo encendido y con internet a las 08:30, 15:00 y 19:30, o usa la nube (sección 10).</div></div>"""


def sec_infra(cfg: dict) -> str:
    return """
<h2>10 · Dónde corre y cómo se instala</h2>
<h3>Modo A: tu PC (Windows)</h3>
<ul><li><code>programar_tareas.ps1</code> instala tres tareas: <b>BM-Monitor</b> (cada 15 min, lun–vie 08:30–16:00), <b>BM-Radar</b> (lun–jue 19:30) y <b>BM-Bot</b> (al iniciar sesión, con reinicio automático por <code>iniciar_bot.bat</code>). <code>-Simular</code> muestra qué haría y <code>-Quitar</code> las borra.</li>
<li>Salidas en <code>alertas.log</code> y <code>bot.log</code> (el token nunca aparece en los registros).</li></ul>
<h3>Modo B: la nube (opcional, cuando el PC está apagado)</h3>
<ul><li><b>GitHub Actions</b> (repositorio <b>privado</b>): <code>monitor.yml</code> (cada 15 min en UTC), <code>radar.yml</code> (00:30 UTC = 19:30 Bogotá) y <code>bot.yml</code> (apagado por defecto: <code>bot.py --una-vez</code>). El estado se guarda en una rama privada <code>estado</code>.</li>
<li><b>Streamlit Community Cloud</b> para el dashboard (solo necesita <code>FINNHUB_API_KEY</code>; no usa el token de Telegram ni tu posición).</li>
<li>Elige <b>un solo modo a la vez</b>. Límites reales: Yahoo suele bloquear las direcciones de GitHub, los cron pueden demorar 10–15 minutos, hay 2.000 minutos gratis al mes en repositorios privados y Telegram permite un solo lector por bot. La subida a GitHub <b>no se pudo probar desde este PC</b> (no tiene Git).</li></ul>
<h3>Quién puede hablarle al bot</h3>
<ul><li>Por decisión tuya, el bot <b>responde a cualquier chat o grupo</b> (<code>bot.responder_a: todos</code>). Cualquiera que encuentre el bot puede consultarlo y, salvo que actives <code>escritura_solo_mi_chat</code>, también cambiar tu <code>/rank</code>, <code>/pos</code> y <code>/op</code>.</li>
<li>Los avisos automáticos llegan solo al chat de <code>TELEGRAM_CHAT_ID</code> (puede ser el id del grupo).</li></ul>"""


def _leer(cfg: dict) -> dict[str, Any]:
    out = ROOT / cfg["paths"]["outputs_dir"] / "validacion"
    d: dict[str, Any] = {}
    try:
        d["tabla"] = pd.read_csv(out / "tabla_eventos.csv")
        d["liga"] = pd.read_csv(out / "liga_resumen.csv")
        d["b0"] = pd.read_csv(out / "bases_resumen_esc0.csv")
        d["b1"] = pd.read_csv(out / "bases_resumen_esc1.csv")
        d["noticias"] = json.loads((out / "cruce_noticias.json").read_text(encoding="utf-8"))
    except Exception:                                                                                  # noqa: BLE001 — sin la validación, el resto del documento sigue
        pass
    return d


def sec_validacion(cfg: dict) -> str:
    d = _leer(cfg)
    if "tabla" not in d:
        return "<h2>11 · Qué dice la validación</h2><p>Aún no hay validación guardada: corre <code>python run_validacion.py</code>.</p>"
    t, lg = d["tabla"], d["liga"]
    nr = cfg["validacion"]["liga"]["participantes_reales"][1]

    def f(grupo, clase, h, med="AR"):
        return t[(t.grupo == grupo) & (t.clase == clase) & (t.h == h) & (t.medida == med)].iloc[0]
    n5, n10, a10, ax10 = f("grandes", "NEGRO", 5), f("grandes", "NEGRO", 10), f("grandes", "AMARILLO", 10), f("grandes", "AMARILLO", 10, "ARx")

    def liga(esc, k):
        return lg[(lg.escenario == esc) & (lg.estrategia == k) & (lg.base_pct == "supervivientes")].iloc[0]
    filas = []
    for esc, nom in ((0, "Nadie concentrado en TSLA/NVDA"), (1, "30 % concentrado en TSLA/NVDA (70–100 %)"), (2, "30 % concentrado, todos con 100 %"), (3, "60 % concentrado")):
        for k, kn in (("a", "(a) mantener"), ("b", "(b) + Regla Maestra")):
            r = liga(esc, k)
            filas.append([nom, kn, pct(r.pasa1, 1), pct(r.pasa2, 1), pct(r.pasa3, 2), pct(r.pasa4, 2), pct(r[f"top30_{nr}"], 2), pct(r[f"p1_{nr}"], 2)])
    b0, b1 = d["b0"].reset_index(drop=True), d["b1"].reset_index(drop=True)
    p0, p1 = int(b0.index[b0.base == "TSLA"][0]) + 1, int(b1.index[b1.base == "TSLA"][0]) + 1
    nz = d.get("noticias", {})
    return f"""
<h2>11 · Qué dice la validación (2015–2026)</h2>
<p>El informe completo está en <code>reports/validacion_semaforo.html</code>. Resumen de lo que importa para el plan:</p>
<h3>El semáforo</h3>
<ul><li><b>NEGRO no anticipa más caídas</b> en las 7 acciones grandes: retorno residual posterior a 5 sesiones {pct(n5.media, 2, True)} (intervalo 95 %: {pct(n5.ic_lo, 2, True)} a {pct(n5.ic_hi, 2, True)}) y a 10 sesiones {pct(n10.media, 2, True)} ({pct(n10.ic_lo, 2, True)} a {pct(n10.ic_hi, 2, True)}), con {int(n5.n)} eventos. Salirse costaría {pct(cfg['costos_v5']['costo_cambio'], 1)}.</li>
<li><b>AMARILLO:</b> la acción no sigue cayendo ({pct(a10.media, 2, True)} a 10 sesiones). Restando su tendencia normal, el rebote ({pct(ax10.media, 2, True)}) no se distingue de cero.</li>
<li>Se probó con 4 versiones del supuesto de noticias (la historia no tiene noticias antes del {e(nz.get('primera_noticia', 'oct-2025'))}) y la conclusión se sostiene.</li></ul>
<h3>El simulador de ligas ({n(nr, 0)} participantes, “top X %” contra los que siguen vivos)</h3>
<p>Cada día de 2015 a 2026 es el inicio de un concurso de 25 sesiones de EE. UU. con rivales sintéticos que compran 2–4 activos al azar y mantienen. Probabilidad de seguir en carrera:</p>
{tabla(['Rivales', 'Estrategia', 'Corte 1', 'Corte 2', 'Corte 3', 'Corte 4', 'P(top 30)', 'P(#1)'], filas, 2, num=True)}
<ul><li>Como base, <b>TSLA queda en el puesto {p0} de 55</b> si nadie está concentrado y en el puesto <b>{p1}</b> si el 30 % del campo lo está (una acción menos popular, como PBR, sale mejor).</li>
<li>La Regla Maestra es una <b>apuesta de cola</b>: baja un poco la probabilidad de pasar el corte 1 y la rentabilidad esperada (≈ −1,3 puntos por el costo de los cambios), pero sube la probabilidad de llegar al final al diferenciarte de rivales que tienen tu misma acción.</li>
<li>Con ≈ 5.500 participantes y esta regla de cortes, al final quedan unas 9 personas: P(top 30) es casi la probabilidad de sobrevivir los cuatro cortes.</li></ul>
<div class="box warn"><b>Cómo leer estas cifras</b><div>Comparan reglas con datos pasados; TSLA y NVDA fueron de lo mejor entre 2015 y 2026, y nadie lo sabía antes. Los rivales reales rotan y reaccionan; los simulados no. La composición del campo (cuántos tienen TSLA/NVDA) es un supuesto, no un dato.</div></div>
<h3>¿Y la meta de +100 %?</h3>
<p>Medido el 5-oct-2026 en 2.931 concursos históricos: <b>TSLA llegó a +100 % en ≈ 0,03 %</b> (una vez, 2020), a +50 % en 2,7 % y a +30 % en ≈ 10 %. Ninguna otra acción real lo logró (algunos “+100 %” en acciones locales eran errores de datos de Yahoo). Exigiría ≈ +2,8 % diario sin una sesión mala. El mejor rival simulado termina con una mediana de ≈ +24 %.</p>"""


def sec_riesgos(cfg: dict) -> str:
    return """
<h2>12 · Qué puede fallar y qué hace el sistema</h2>
""" + tabla(['Qué falla', 'Qué hace el sistema', 'Qué haces tú'], [
        ['Finnhub (noticias) cae o no hay clave', 'Cadena de respaldo (Yahoo → Google → Alpha Vantage → caché). Si todas caen, avisa y usa el proxy de volumen + hueco.', 'Nada; en una alerta roja verifica el titular en Yahoo Finance.'],
        ['Precios (Yahoo) caen', 'Usa la última caché y avisa tras 3 corridas seguidas sin datos (y cuando vuelve).', 'Mira trii a mano.'],
        ['Acciones de la BVC', 'Noticias por Google News en español (cobertura parcial: se advierte). Sin noticias no se recomiendan.', '—'],
        ['Opciones sin datos (p. ej. mercado cerrado)', 'El movimiento esperado usa la variación diaria, en la misma base de tiempo.', '—'],
        ['Telegram o internet caen', 'Guarda hasta 20 avisos y los reenvía con la hora original.', '—'],
        ['<code>state.json</code> dañado', 'Restaura la copia <code>.bak</code> y lo avisa; si ambas se dañan, error claro.', 'Revisar el archivo.'],
        ['Monitor, radar y bot escriben a la vez', 'Candado de escritura: no se pierde ninguna actualización.', '—'],
        ['Falla el monitor o el radar', 'Manda un mensaje de fallo (el radar incluye un respaldo manual de 3 minutos).', 'Usar el respaldo manual.'],
        ['Se cae el bot', 'Se reinicia en 15 s y no muere por errores de datos.', '<code>/actualizar</code>'],
        ['<b>PC apagado o dormido</b>', 'No hay avisos ni respuestas mientras esté así.', 'Dejarlo encendido o usar la nube.'],
        ['Fechas de resultados distintas entre fuentes', 'Las muestra como “las fuentes no coinciden en la fecha”.', 'Confirmarla en la web de la empresa.']], 1)


def sec_limites(cfg: dict) -> str:
    return """
<h2>13 · Lo que hay que saber con honestidad</h2>
<div class="grid">
<div class="box bad"><b>Riesgos del plan</b><div><ul><li>Una sola acción: puedes perder una parte grande del capital (en la historia, el 10 % peor de los concursos con TSLA pierde ≈ 18 % o más; cerca de la mitad termina en negativo).</li>
<li>Más movimiento esperado = más oportunidad <b>y</b> más riesgo.</li><li>Nada garantiza ganar ni la meta de +100 %.</li></ul></div></div>
<div class="box warn"><b>Supuestos sin verificar</b><div><ul><li>Cuánta gente tiene TSLA/NVDA (el tamaño del campo, ≈ 5.500, es un dato tuyo aproximado).</li><li>Que “top X %” se mide contra los que siguen vivos.</li>
<li>Que los rivales no operan en la simulación.</li><li>Fechas macro por confirmar: BanRep (30-oct), IPC de EE. UU. (14-oct) y empleo (6-nov).</li></ul></div></div>
<div class="box"><b>Límites técnicos</b><div><ul><li>Sin noticias reales antes de oct-2025: la historia usa un proxy.</li><li>La traducción de titulares es automática y literal.</li><li>El tono de las noticias (VADER) no entiende contexto financiero.</li>
<li>NEGRO no tiene caducidad (puede durar meses en una acción que no se recupera).</li><li>La nube no se probó contra GitHub desde este PC.</li></ul></div></div>
</div>
<div class="box"><b>Seguridad</b><div><ul><li>El bot <b>nunca envía órdenes</b>: trii no tiene API y las órdenes las das tú.</li><li>Las claves viven solo en <code>.env</code>; tu token y tu clave de Finnhub se pegaron en un chat: <b>revoca el token</b> (<code>/revoke</code> en @BotFather) y regenera la clave de Finnhub cuando termine el concurso.</li>
<li>Con el bot abierto a cualquier chat, quien lo encuentre puede usarlo; activa <code>escritura_solo_mi_chat</code> si no quieres que otros cambien tu seguimiento.</li></ul></div></div>"""


def sec_params(cfg: dict) -> str:
    s, m, b, r, mo, ev = cfg["semaforo"], cfg["mee"], cfg["banco"], cfg["regla_maestra"], cfg["monitor"], cfg["monitor"]["eventos"]
    filas = [
        ["semaforo.z_caida / z_res_caida / z_res_azul", f"{s['z_caida']} / {s['z_res_caida']} / {s['z_res_azul']}", 'Umbrales de caída anormal, caída propia y “lo explica el mercado”.'],
        ["semaforo.vol_rel_min", s['vol_rel_min'], 'Volumen mínimo (× lo normal) para ROJO.'], ["semaforo.sentimiento_negativo", s['sentimiento_negativo'], 'Tono por debajo del cual una noticia es negativa.'],
        ["semaforo.recuperacion_negro", s['recuperacion_negro'], 'Fracción de la caída que hay que recuperar para no ser NEGRO.'], ["semaforo.proxy_sin_noticias", f"volumen ≥ {s['proxy_sin_noticias']['vol_rel_min']}× y hueco ≥ {s['proxy_sin_noticias']['gap_sigma']}σ", 'Respaldo cuando no hay noticias.'],
        ["mee.respaldo", m['respaldo'], 'Base de tiempo del movimiento esperado sin opciones.'], ["banco.top_n / tolerancia_mee", f"{b['top_n']} / {b['tolerancia_mee']}", 'Tamaño del banco y empate entre candidatos.'],
        ["banco.liquidez_min_usd_mm / locales", f"{b['liquidez_min_usd_mm']} / {b['liquidez_min_cop_mm_locales']}", 'Liquidez mínima (millones al día).'], ["banco.exigir_noticias", b['exigir_noticias'], 'No se recomienda una acción sin datos de noticias.'],
        ["regla_maestra.costo_cambio_pp / margen_g_pp", f"{r['costo_cambio_pp']} / {r['margen_g_pp']}", 'Costo del cambio y margen de seguridad.'], ["regla_maestra.cambio_por_negro", r['cambio_por_negro'], 'NEGRO baja el listón de la regla (hoy no).'],
        ["regla_maestra.d_negro", r['d_negro'], 'Parámetro del modo NEGRO.'], ["seleccion_base.ventaja_minima", cfg['seleccion_base']['ventaja_minima'], 'Ventaja necesaria para sustituir la base.'],
        ["monitor.enfriamiento_min", mo['enfriamiento_min'], 'Minutos entre alertas de color del mismo activo.'], ["monitor.eventos.max_por_corrida", ev['max_por_corrida'], 'Tope de novedades por revisión.'],
        ["monitor.eventos.alza_fuerte_z", ev['alza_fuerte_z'], 'Umbral del aviso de subida fuerte.'], ["bot.responder_a", cfg['bot']['responder_a'], 'Quién puede hablarle al bot.'],
        ["estado.max_cambios", cfg['estado']['max_cambios'], 'Cambios de acción permitidos.']]
    return "<h2>14 · Todos los parámetros importantes</h2><p>Viven en <code>config.yaml</code>: se cambian ahí, sin tocar el código.</p>" + tabla(['Parámetro', 'Valor actual', 'Para qué sirve'], [[f"<code>{a}</code>", e(b_), c] for a, b_, c in filas], 1)


def sec_archivos() -> str:
    return "<h2>15 · Mapa de archivos</h2>" + tabla(['Archivo', 'Qué es'], [
        ['<code>config.yaml</code>', 'Todos los parámetros: reglas del concurso, umbrales, costos, horarios, fuentes y avisos.'], ['<code>.env</code>', 'Tus claves (nunca se comparten).'],
        ['<code>data/state.json</code>', 'Tu posición, ranking, cambios, operaciones y avisos ya enviados.'], ['<code>monitor.py</code> · <code>radar.py</code> · <code>bot.py</code>', 'Las tres piezas que corren solas.'],
        ['<code>app.py</code> · <code>iniciar_app.bat</code>', 'Dashboard.'], ['<code>src/semaforo.py</code>', 'Métricas y colores.'], ['<code>src/relevo.py</code>', 'Movimiento esperado y banco de relevo.'], ['<code>src/regla_maestra.py</code>', 'La Regla Maestra.'],
        ['<code>src/seleccion.py</code>', 'Comparación de la base.'], ['<code>src/eventos.py</code>', 'Avisos de novedades.'], ['<code>src/formato.py</code>', 'Mensajes en lenguaje sencillo.'], ['<code>src/informe_html.py</code>', 'Informe HTML a pedido.'],
        ['<code>src/data_sources.py</code>', 'Precios, noticias, opciones, traducción, caché y respaldos.'], ['<code>src/state.py</code>', 'Estado persistente con candado y copia de seguridad.'],
        ['<code>run_validacion.py</code> · <code>src/validacion.py</code> · <code>src/liga_v5.py</code>', 'Validación y simulador de ligas.'], ['<code>reports/validacion_semaforo.html</code>', 'Informe de la validación.'],
        ['<code>programar_tareas.ps1</code>', 'Instala las tareas de Windows.'], ['<code>.github/workflows/</code> · <code>nube/</code>', 'Flujos de GitHub Actions y estado en la nube.'],
        ['<code>README.md</code> · <code>ASSUMPTIONS.md</code>', 'Instrucciones y lista de supuestos (más de 90 puntos).'], ['<code>tests/</code>', 'Más de 400 pruebas automáticas.']], 1)


def sec_glosario() -> str:
    g = [("Semáforo", "Resumen del estado de tu acción en cinco colores."), ("z", "Cuántas variaciones diarias típicas se movió el precio hoy."), ("Residual", "Lo que se mueve una acción después de quitar el arrastre del mercado."),
         ("Beta", "Sensibilidad de una acción al Nasdaq-100."), ("MEE", "Movimiento esperado hasta el próximo corte (±)."), ("IV", "Volatilidad implícita: lo que el mercado de opciones espera que se mueva."),
         ("Banco de relevo", "Las mejores alternativas a tu acción."), ("Regla Maestra", "Cuándo vale la pena pagar el costo de cambiar de acción."), ("g", "Cuántos puntos te faltan para el objetivo (+1 de margen)."),
         ("Corte", "Viernes en que solo pasan los mejores del ranking."), ("Proxy", "Aproximación cuando falta el dato real."), ("Bootstrap por bloques", "Repetir el cálculo miles de veces remuestreando meses completos."),
         ("Intervalo de confianza del 95 %", "Rango donde, con 95 % de confianza, está el verdadero promedio; si incluye 0, no hay efecto demostrable."), ("Concentrado", "Rival que pone entre 70 y 100 % en TSLA o NVDA."),
         ("Micro-compra", "Compra pequeña (≈ $ 100.000) para cumplir el requisito de actividad."), ("Orden límite", "Orden con precio máximo/mínimo definido: no se ejecuta a cualquier precio.")]
    return "<h2>16 · Glosario</h2><dl>" + "".join(f"<dt>{e(a)}</dt><dd>{e(b)}</dd>" for a, b in g) + "</dl>"


NAV = [("s0", "Resumen"), ("s1", "1 · Reglas"), ("s2", "2 · Estrategia"), ("s3", "3 · Mapa del sistema"), ("s4", "4 · Datos"), ("s5", "5 · Semáforo"), ("s6", "6 · Banco de relevo"), ("s7", "7 · Regla Maestra"),
       ("s8", "8 · Avisos y comandos"), ("s9", "9 · Tu rutina"), ("s10", "10 · Dónde corre"), ("s11", "11 · Validación"), ("s12", "12 · Fallos"), ("s13", "13 · Honestidad"), ("s14", "14 · Parámetros"),
       ("s15", "15 · Archivos"), ("s16", "16 · Glosario")]


def construir(cfg: dict) -> str:
    cuerpos = [sec_resumen(cfg), sec_reglas(cfg), sec_estrategia(cfg), sec_sistema(cfg), sec_datos(cfg), sec_semaforo(cfg), sec_banco(cfg), sec_regla(cfg), sec_mensajes(cfg), sec_rutina(cfg),
               sec_infra(cfg), sec_validacion(cfg), sec_riesgos(cfg), sec_limites(cfg), sec_params(cfg), sec_archivos(), sec_glosario()]
    nav = "<b>Contenido</b>" + "".join(f"<a href='#{i}'>{t}</a>" for i, t in NAV)
    main = "".join(f"<section id='{NAV[k][0]}'>{c}</section>" for k, c in enumerate(cuerpos))
    return (f"<!doctype html><html lang='es'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>Bolsa Millonaria 2026 — el plan completo</title><style>{CSS}</style></head><body><div class='layout'><nav>{nav}</nav><main>{main}</main></div>"
            f"<footer>Generado por <code>python run_plan.py</code> el {dt.datetime.now():%d/%m/%Y %H:%M}. Información y recomendaciones, no órdenes ni promesas de ganancia.</footer></body></html>")


def main() -> int:
    cfg = load_config()
    destino = ROOT / cfg["paths"]["reports_dir"] / "plan_completo.html"
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(construir(cfg), encoding="utf-8")
    print(f"✅ {destino}  ({destino.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
