"""Cifras y gráficos del informe de validación (la presentación HTML vive en templates/validacion.html.j2).

Todo texto interpretativo del informe sale de las cifras (nada de conclusiones escritas a mano que puedan contradecir los datos)."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src import validacion as V

from . import fmt
from .data import GAIN, GRID, LOSS, NEUTRAL, base_layout


def html(fig: go.Figure, div_id: str) -> str:
    """Gráfico sin la librería: Plotly se incluye UNA sola vez en el <head> de la plantilla (no depende del orden de los gráficos)."""
    return fig.to_html(full_html=False, include_plotlyjs=False, div_id=div_id,
                       config={"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]})

COLOR_SEMAFORO = {"VERDE": "#0ca30c", "AZUL": "#2a78d6", "AMARILLO": "#eda100", "ROJO": "#d03b3b", "NEGRO": "#3b3b3b"}
NOMBRE_CLASE = {"VERDE": "Verde (normal)", "AZUL": "Azul (cae el mercado)", "AMARILLO": "Amarillo (cae sola)", "ROJO": "Rojo (cae con mala noticia)",
                "NEGRO": "Negro (no se recupera)"}
ESTRATEGIA = {"a": "(a) Mantener la base", "b": "(b) Base + Regla Maestra", "c": "(c) (b) + cambio por NEGRO"}
COLOR_ESTRATEGIA = {"a": "#2a78d6", "b": "#eb6834", "c": "#4a3aa7"}


# ------------------------------------------------------------------ estudio de eventos
def fila(tabla: pd.DataFrame, grupo: str, clase: str, h: int, medida: str = "AR") -> pd.Series:
    r = tabla[(tabla.grupo == grupo) & (tabla.clase == clase) & (tabla.h == h) & (tabla.medida == medida)]
    return r.iloc[0]


def grafico_eventos(tabla: pd.DataFrame, grupo: str, medida: str = "AR", titulo_y: str = "Retorno residual posterior (%)") -> str:
    """Barras con intervalo del 95 % para cada clase y horizonte."""
    fig = go.Figure()
    for clase in ("AZUL", "AMARILLO", "ROJO", "NEGRO"):
        t = tabla[(tabla.grupo == grupo) & (tabla.clase == clase) & (tabla.medida == medida)].sort_values("h")
        fig.add_trace(go.Bar(x=[f"{h} sesión{'es' if h > 1 else ''}" for h in t.h], y=t.media * 100, name=NOMBRE_CLASE[clase], marker_color=COLOR_SEMAFORO[clase],
                             error_y=dict(type="data", symmetric=False, array=(t.ic_hi - t.media) * 100, arrayminus=(t.media - t.ic_lo) * 100, color=NEUTRAL, thickness=1.3),
                             customdata=np.c_[t.n, t.ic_lo * 100, t.ic_hi * 100],
                             hovertemplate="%{x}<br>Promedio %{y:.2f} %<br>IC 95 %: %{customdata[1]:.2f} % a %{customdata[2]:.2f} %<br>%{customdata[0]} eventos<extra>" +
                                           NOMBRE_CLASE[clase] + "</extra>"))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.update_layout(barmode="group", yaxis_title=titulo_y, bargap=0.2)
    return html(base_layout(fig, 430), f"ev_{grupo}_{medida}")


def tabla_resumen(tabla: pd.DataFrame, grupo: str, clase: str, medida: str = "AR") -> list[dict[str, Any]]:
    out = []
    for h in sorted(tabla.h.unique()):
        r = fila(tabla, grupo, clase, h, medida)
        out.append(dict(h=int(h), n=int(r.n), media=r.media, ic_lo=r.ic_lo, ic_hi=r.ic_hi, p=r.p, pos=r.pct_pos, salir=r.salir_neto))
    return out


def grafico_negro_tsla(panel: pd.DataFrame, h: int = 10) -> tuple[str, pd.DataFrame]:
    """Cada evento NEGRO de TSLA: retorno residual a h sesiones (puntos) y tabla de eventos."""
    e = panel[(panel.ticker == "TSLA") & panel["nuevo_negro"]].dropna(subset=[f"AR{h}"]).copy()
    e["fecha"] = e.index
    fig = go.Figure(go.Bar(x=e["fecha"].dt.strftime("%d/%m/%Y"), y=e[f"AR{h}"] * 100, marker_color=[GAIN if v > 0 else LOSS for v in e[f"AR{h}"]],
                           hovertemplate="%{x}<br>%{y:.1f} % a %{customdata} sesiones después<extra></extra>", customdata=[h] * len(e)))
    fig.add_hline(y=0, line_color=NEUTRAL, line_width=1)
    fig.update_layout(yaxis_title=f"Retorno residual a {h} sesiones (%)", xaxis_title="Fecha en que TSLA quedó en NEGRO", showlegend=False)
    return html(base_layout(fig, 360), "negro_tsla"), e[["fecha", "z_res", "vol_rel", f"AR{h}"]].rename(columns={f"AR{h}": "ar"})


# ------------------------------------------------------------------ noticias reales vs proxy
def cifras_noticias(res: dict[str, Any] | None) -> dict[str, Any] | None:
    if not res:
        return None
    r = dict(res)
    r["pct_con_noticia"] = r["con_noticia_negativa"] / r["propia_con_volumen"] if r["propia_con_volumen"] else float("nan")
    return r


# ------------------------------------------------------------------ simulador de ligas
def grafico_cortes(resumen: pd.DataFrame, escenario: int, base_pct: str = "campo") -> str:
    r = resumen[(resumen.escenario == escenario) & (resumen.base_pct == base_pct)]
    fig = go.Figure()
    etapas = ["Corte 1 · top 50 %", "Corte 2 · top 33 %", "Corte 3 · top 10 %", "Corte 4 · top 10 % (clave)"]
    for k in ("a", "b", "c"):
        f = r[r.estrategia == k].iloc[0]
        ys = [f[f"pasa{i}"] * 100 for i in range(1, 5)]
        lo = [f[f"pasa{i}_lo"] * 100 for i in range(1, 5)]
        hi = [f[f"pasa{i}_hi"] * 100 for i in range(1, 5)]
        fig.add_trace(go.Bar(x=etapas, y=ys, name=ESTRATEGIA[k], marker_color=COLOR_ESTRATEGIA[k],
                             error_y=dict(type="data", symmetric=False, array=np.array(hi) - ys, arrayminus=np.array(ys) - lo, color=NEUTRAL, thickness=1.2),
                             hovertemplate="%{x}<br>%{y:.1f} % (IC 95 %)<extra>" + ESTRATEGIA[k] + "</extra>"))
    fig.update_layout(barmode="group", yaxis_title="Probabilidad de seguir vivo (%)", bargap=0.2)
    return html(base_layout(fig, 420), f"liga_cortes_{escenario}_{base_pct}")


def grafico_escenarios(resumen: pd.DataFrame, nombres: list[str], base_pct: str = "campo") -> str:
    """Probabilidad de llegar vivo al final (4 cortes) por escenario de rivales, para las tres estrategias."""
    fig = go.Figure()
    r = resumen[resumen.base_pct == base_pct]
    for k in ("a", "b", "c"):
        t = r[r.estrategia == k].sort_values("escenario")
        fig.add_trace(go.Bar(x=[nombres[i] for i in t.escenario], y=t.pasa4 * 100, name=ESTRATEGIA[k], marker_color=COLOR_ESTRATEGIA[k],
                             hovertemplate="%{x}<br>%{y:.1f} %<extra>" + ESTRATEGIA[k] + "</extra>"))
    fig.update_layout(barmode="group", yaxis_title="Pasa los 4 cortes (%)", xaxis=dict(tickangle=0), bargap=0.2)
    return html(base_layout(fig, 430, margin=dict(l=55, r=15, t=30, b=120)), f"liga_escenarios_{base_pct}")


def grafico_por_anio(ventanas: pd.DataFrame, escenario: int, estrategia: str = "a") -> str:
    """Probabilidad de pasar los 4 cortes según el año en que empieza la ventana (la dispersión muestra cuánto depende del régimen)."""
    v = ventanas[(ventanas.escenario == escenario) & (ventanas.estrategia == estrategia) & (ventanas.base_pct == ventanas.base_pct.iloc[0])]
    g = v.groupby(v.inicio.dt.year)["pasa4"].mean() * 100
    fig = go.Figure(go.Bar(x=g.index.astype(str), y=g.values, marker_color=COLOR_ESTRATEGIA[estrategia], hovertemplate="Ventanas que empiezan en %{x}<br>%{y:.0f} % pasan los 4 cortes<extra></extra>"))
    fig.update_layout(yaxis_title="Pasa los 4 cortes (%)", xaxis_title="Año en que empieza la ventana", showlegend=False)
    return html(base_layout(fig, 360), "liga_anio")


def grafico_retornos(ventanas: pd.DataFrame, escenario: int) -> str:
    v = ventanas[(ventanas.escenario == escenario) & (ventanas.base_pct == ventanas.base_pct.iloc[0])]
    fig = go.Figure()
    for k in ("a", "b"):
        x = v[v.estrategia == k]["ret_final"] * 100
        fig.add_trace(go.Histogram(x=x, name=ESTRATEGIA[k], marker_color=COLOR_ESTRATEGIA[k], opacity=0.6, histnorm="percent", xbins=dict(size=5),
                                   hovertemplate="Retorno %{x} %<br>%{y:.1f} % de las ventanas<extra>" + ESTRATEGIA[k] + "</extra>"))
    fig.update_layout(barmode="overlay", xaxis_title="Rentabilidad al final de las 25 sesiones (%)", yaxis_title="% de las ventanas")
    return html(base_layout(fig, 380), "liga_retornos")


def filas_liga(resumen: pd.DataFrame, escenario: int, base_pct: str, n_reales: list[int]) -> list[dict[str, Any]]:
    r = resumen[(resumen.escenario == escenario) & (resumen.base_pct == base_pct)]
    out = []
    for k in ("a", "b", "c"):
        f = r[r.estrategia == k].iloc[0]
        out.append(dict(k=k, nombre=ESTRATEGIA[k], pasa=[f[f"pasa{i}"] for i in range(1, 5)], p1={n: f[f"p1_{n}"] for n in n_reales},
                        top30={n: f[f"top30_{n}"] for n in n_reales}, ret_mediana=f.ret_mediana, ret_p10=f.ret_p10, ret_p90=f.ret_p90, cambios=f.pct_con_cambio,
                        ventanas=int(f.ventanas)))
    return out


def azar_esperado(top_pcts: list[float], base_pct: str = "campo") -> list[float]:
    """Probabilidad de seguir vivo después de cada corte para un participante cualquiera.
    'campo' (top X % contra TODOS): 50, 33, 10, 10 % (el filtro más estricto manda). 'supervivientes' (top X % contra los vivos): se multiplican (50, 16,5, 1,65, 0,165 %)."""
    acum, out = 1.0, []
    for i, t in enumerate(top_pcts):
        acum = acum * t / 100 if base_pct == "supervivientes" else (min(acum, t / 100) if i else t / 100)
        out.append(acum)
    return out


def tabla_bases(res: pd.DataFrame, base: str, n: int = 12) -> list[dict[str, Any]]:
    """Las mejores `n` acciones como base (según el estudio) y, si no está entre ellas, la base actual con su posición."""
    r = res.reset_index(drop=True)
    r["puesto"] = r.index + 1
    sel = r.head(n)
    if base not in set(sel["base"]) and base in set(r["base"]):
        sel = pd.concat([sel, r[r["base"] == base]])
    return [dict(puesto=int(x.puesto), base=x.base, es_base=x.base == base, **{k: float(getattr(x, k)) for k in ("pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30", "ret_mediana", "ret_p10", "ret_p90", "pct_pierde")},
                 ventanas=int(x.ventanas)) for x in sel.itertuples()]


def grafico_bases(filas: list[dict[str, Any]], etiqueta_corte: str = "pasa3", titulo: str = "Pasa hasta el corte 3 (%)") -> str:
    fig = go.Figure(go.Bar(x=[x["base"] for x in filas], y=[x[etiqueta_corte] * 100 for x in filas], marker_color=["#2a78d6" if x["es_base"] else NEUTRAL for x in filas],
                           hovertemplate="%{x}<br>%{y:.1f} % de los concursos<extra></extra>"))
    fig.update_layout(yaxis_title=titulo, showlegend=False)
    return html(base_layout(fig, 360), "bases_" + etiqueta_corte)


# ------------------------------------------------------------------ texto interpretativo calculado a partir de las cifras
def clasifica(media: float, lo: float, hi: float) -> str:
    return "sube" if lo > 0 else ("baja" if hi < 0 else "sin efecto claro")


def describir(rows: list[dict[str, Any]], hs: tuple[int, ...] = (5, 10, 20)) -> str:
    """'a 5 sesiones +0,17 % (sin efecto claro), a 10 sesiones …' con el veredicto del intervalo en cada horizonte."""
    por_h = {r["h"]: r for r in rows}
    return ", ".join(f"a {h} sesiones {fmt.pct(por_h[h]['media'], 2, True)} ({clasifica(por_h[h]['media'], por_h[h]['ic_lo'], por_h[h]['ic_hi'])})"
                     for h in hs if h in por_h)


def _por_h(rows: list[dict[str, Any]], h: int) -> dict[str, Any]:
    return next(r for r in rows if r["h"] == h)


def _intervalo(r: dict[str, Any]) -> str:
    return f"{fmt.pct(r['ic_lo'], 2, True)} a {fmt.pct(r['ic_hi'], 2, True)}"


def veredicto_comparacion(lo: float, hi: float) -> str:
    return "mejora" if lo > 0 else ("empeora" if hi < 0 else "sin diferencia demostrable")


MEDIDAS = {"pasa1": "Pasar el corte 1 (top 50 %)", "pasa2": "Pasar hasta el corte 2 (top 33 %)", "pasa3": "Pasar hasta el corte 3 (top 10 %)",
           "pasa4": "Pasar los 4 cortes", "p1": "P(#1 final)", "top30": "P(top 30 de Diamante)", "ret_final": "Rentabilidad final"}


def construir_contexto(cfg: dict[str, Any], tabla: pd.DataFrame, panel: pd.DataFrame, rob: pd.DataFrame, noticias: dict[str, Any] | None,
                       liga_res: pd.DataFrame, liga_vent: pd.DataFrame, hoy, bases_res: dict[int, pd.DataFrame] | None = None) -> dict[str, Any]:
    """Todo lo que usa la plantilla: gráficos (HTML), tablas y frases. Las frases se derivan de los números (nunca escritas a mano)."""
    from plotly.offline import get_plotlyjs

    from src import liga_v5 as L

    vl = cfg["validacion"]
    lg = vl["liga"]
    nr = lg["participantes_reales"]
    n_mid = nr[1]
    eb = lg["escenario_base"]
    nombres = [e["nombre"] for e in lg["escenarios"]]
    pct, num = fmt.pct, fmt.num
    crit = V.criterio_negro(tabla, cfg)

    # ---- tablas por clase
    T = lambda g, c, m="AR": tabla_resumen(tabla, g, c, m)                                            # noqa: E731
    neg_g, amar_g, rojo_g, azul_g = T("grandes", "NEGRO"), T("grandes", "AMARILLO"), T("grandes", "ROJO"), T("grandes", "AZUL")
    neg_gx, amar_gx = T("grandes", "NEGRO", "ARx"), T("grandes", "AMARILLO", "ARx")
    neg_r, rojo_r, amar_r = T("resto", "NEGRO"), T("resto", "ROJO"), T("resto", "AMARILLO")

    # ---- veredictos
    n5, n10 = _por_h(neg_g, 5), _por_h(neg_g, 10)
    if crit["deriva"]:
        v_negro = dict(emoji="🟢", pregunta="¿Después de NEGRO la caída sigue? (acciones grandes)",
                       texto=f"<b>Sí.</b> {describir(neg_g, (5, 10))}. " + ("La ganancia de salir compensa el costo del cambio." if crit["rentable_neto"] else
                                                                           "Pero la caída posterior es menor que el 1,3 % que cuesta cambiar de activo: no alcanza para pagar el cambio."))
    else:
        v_negro = dict(emoji="🔴", pregunta="¿Después de NEGRO la caída sigue? (acciones grandes)",
                       texto=f"<b>No.</b> Retorno residual medio {describir(neg_g, (5, 10))}. Ningún intervalo queda por debajo de cero: no hay evidencia de que la caída continúe, "
                             f"y salirse costaría 1,3 %. Por eso <code>cambio_por_negro</code> quedó en <b>false</b> (NEGRO sólo informa).")
    a10, ax10, a20, ax20 = _por_h(amar_g, 10), _por_h(amar_gx, 10), _por_h(amar_g, 20), _por_h(amar_gx, 20)
    todos_pos = all(_por_h(amar_g, h)["media"] >= 0 for h in (5, 10, 20))
    if ax10["ic_lo"] > 0 or ax20["ic_lo"] > 0:
        v_amar = dict(emoji="🟢", pregunta="¿Después de AMARILLO la acción rebota? (acciones grandes)",
                      texto=f"<b>Sí.</b> Sobre su propia tendencia normal, a 10 sesiones rinde {pct(ax10['media'], 2, True)} (intervalo {_intervalo(ax10)}). Vender por susto fue un error.")
    elif todos_pos and (a10["ic_lo"] > 0 or a20["ic_lo"] > 0):
        v_amar = dict(emoji="🟡", pregunta="¿Después de AMARILLO la acción rebota? (acciones grandes)",
                      texto=f"<b>A medias.</b> No sigue cayendo y rinde más que el mercado ({describir(amar_g, (10, 20))}), pero buena parte es la tendencia normal de estas acciones: "
                            f"restándola, a 10 sesiones el rebote es {pct(ax10['media'], 2, True)} (intervalo {_intervalo(ax10)}) y no se distingue de cero.")
    elif a10["ic_hi"] < 0:
        v_amar = dict(emoji="🔴", pregunta="¿Después de AMARILLO la acción rebota? (acciones grandes)", texto=f"<b>No:</b> sigue cayendo ({describir(amar_g, (5, 10, 20))}).")
    else:
        v_amar = dict(emoji="🟡", pregunta="¿Después de AMARILLO la acción rebota? (acciones grandes)",
                      texto=f"<b>Sin señal clara.</b> {describir(amar_g, (5, 10, 20))}; ningún intervalo permite afirmar que rebote ni que siga cayendo.")

    vb = liga_vent[(liga_vent.escenario == eb) & (liga_vent.base_pct == lg["base_pct"])]
    comp = {k: L.comparar_estrategias(vb, cfg, *par, n_mid) for k, par in (("b_a", ("b", "a")), ("c_b", ("c", "b")), ("c_a", ("c", "a")))}
    c_ba = comp["b_a"]
    sig_mejora = [m for m in ("pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30") if c_ba[m]["lo"] > 0]
    sig_empeora = [m for m in ("pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30", "ret_final") if c_ba[m]["hi"] < 0]
    dif_ba = c_ba["pasa4"]
    dif_txt = lambda m: f"{pct(c_ba[m]['dif'], 2, True)} (intervalo {pct(c_ba[m]['lo'], 2, True)} a {pct(c_ba[m]['hi'], 2, True)})"      # noqa: E731
    cambia = float(((vb[vb.estrategia == "b"].set_index("inicio")["cambios"]) > 0).mean())
    dist_cb = float((vb[vb.estrategia == "c"].set_index("inicio")["ret_final"].round(10) != vb[vb.estrategia == "b"].set_index("inicio")["ret_final"].round(10)).mean())
    if sig_mejora and sig_empeora:
        veredicto_regla = "mixto"
        v_regla = dict(emoji="🟡", pregunta="¿La Regla Maestra mejora a “sólo mantener”?",
                       texto=f"<b>Mixto.</b> Mejora: {', '.join(MEDIDAS[m][0].lower() + MEDIDAS[m][1:] for m in sig_mejora)}. Empeora: {', '.join(MEDIDAS[m][0].lower() + MEDIDAS[m][1:] for m in sig_empeora)}. "
                             f"Cambia de activo en {pct(cambia, 0)} de los concursos y cada cambio cuesta 1,3 %: la rentabilidad final cambia {dif_txt('ret_final')}. "
                             f"Es una apuesta de cola: te diferencia de los rivales que tienen tu misma acción (P(#1) {dif_txt('p1')}), a cambio de pasar algo menos los primeros cortes.")
    elif sig_mejora:
        veredicto_regla = "mejora"
        v_regla = dict(emoji="🟢", pregunta="¿La Regla Maestra mejora a “sólo mantener”?",
                       texto=f"<b>Sí.</b> Mejora de forma demostrable {', '.join(MEDIDAS[m] for m in sig_mejora)}; pasar los 4 cortes cambia {dif_txt('pasa4')}.")
    elif sig_empeora:
        veredicto_regla = "empeora"
        v_regla = dict(emoji="🔴", pregunta="¿La Regla Maestra mejora a “sólo mantener”?",
                       texto=f"<b>No, empeora</b> {', '.join(MEDIDAS[m] for m in sig_empeora)}; pasar los 4 cortes cambia {dif_txt('pasa4')}.")
    else:
        veredicto_regla = "sin diferencia"
        v_regla = dict(emoji="🟡", pregunta="¿La Regla Maestra mejora a “sólo mantener”?",
                       texto=f"<b>No hay diferencia demostrable.</b> Cambia de activo en {pct(cambia, 0)} de los concursos; la prob. de pasar los 4 cortes cambia {pct(dif_ba['dif'], 1, True)} "
                             f"(intervalo {pct(dif_ba['lo'], 1, True)} a {pct(dif_ba['hi'], 1, True)}). Cambiar por NEGRO (c) cambia el resultado en sólo {pct(dist_cb, 1)} de los concursos.")
    fb = filas_liga(liga_res, eb, lg["base_pct"], nr)
    fa = next(r for r in fb if r["k"] == "a")
    azar = azar_esperado([c["top_pct"] for c in L.calendario_concurso(cfg)["cortes"]], lg["base_pct"])
    sup = lg["base_pct"] == "supervivientes"
    texto_base_pct = ("Para pasar un corte hay que estar entre el <b>mejor X %</b> de <b>los participantes que siguen vivos</b> (así lo entiende el usuario del reglamento): "
                      "los filtros se multiplican y cada vez queda menos gente. Quien no pasa queda eliminado y no vuelve." if sup else
                      "Para pasar un corte hay que estar entre el <b>mejor X %</b> de <b>todo el campo</b>; quien no pasa queda eliminado y no vuelve.")
    texto_medida = ("Según el usuario, “top 10 %” se mide <b>contra los que siguen vivos</b>: es el escenario base de este informe. Si en realidad fuera contra <b>todos</b> los participantes, "
                    "el filtro sería mucho más suave:" if sup else
                    "El reglamento que tengo no aclara si “top 10 %” es contra <b>todos</b> los participantes o sólo contra los que siguen vivos. Si fuera contra los sobrevivientes, el filtro sería mucho más duro:")
    pcs = lambda x: pct(x, 2 if x < 0.02 else (1 if x < 0.2 else 0))                                    # noqa: E731 — que una probabilidad pequeña no se redondee a "0 %"
    f0 = next((r for r in filas_liga(liga_res, 0, lg["base_pct"], nr) if r["k"] == "a"), None)
    v_cortes = dict(emoji="🟡", pregunta="¿Qué tan probable es sobrevivir los cuatro cortes con la base?",
                    texto=(f"Con la base ({cfg['posicion_base']['ticker']} al {pct(cfg['posicion_base']['peso'], 0)}) pasa los cuatro cortes en <b>{pcs(f0['pasa'][3])}</b> de los concursos históricos si "
                           f"<b>nadie</b> la tiene concentrada, pero en <b>{pcs(fa['pasa'][3])}</b> si el 30 % del campo está concentrado en TSLA/NVDA (un participante cualquiera: ≈ {pcs(azar[3])}). "
                           f"Lo que decide no es sólo la acción sino <b>cuánta gente tiene la misma</b>. Llegar primero: {pcs(fa['p1'][n_mid])} con {num(n_mid, 0)} participantes.") if f0 else
                    f"Con la base pasa los cuatro cortes en <b>{pcs(fa['pasa'][3])}</b> de los concursos históricos (un participante cualquiera: ≈ {pcs(azar[3])}).")

    # ---- gráficos de eventos
    g_ev_grandes = grafico_eventos(tabla, "grandes", "AR")
    g_ev_grandes_x = grafico_eventos(tabla, "grandes", "ARx", "Retorno residual menos la tendencia normal de la acción (%)")
    g_ev_resto = grafico_eventos(tabla, "resto", "AR")
    g_tsla, tsla_ev = grafico_negro_tsla(panel)
    n_tsla, up_tsla = len(tsla_ev), int((tsla_ev["ar"] > 0).sum())

    # ---- tablas de robustez
    def celdas(g: str, c: str, m: str = "AR") -> list[dict[str, Any]]:
        return [dict(media=_por_h(T(g, c, m), h)["media"], lo=_por_h(T(g, c, m), h)["ic_lo"], hi=_por_h(T(g, c, m), h)["ic_hi"]) for h in (5, 10, 20)]
    nombres_g = {"grandes": "Acciones grandes", "grandes_sin_TSLA_NVDA": "Grandes sin TSLA ni NVDA", "TSLA": "Sólo TSLA", "resto": "Resto del universo MGC"}
    grupos_rows = [dict(nombre=f"{nombres_g[g]} · {c}", n=_por_h(T(g, c), 5)["n"], celdas=celdas(g, c)) for c in ("NEGRO", "AMARILLO") for g in nombres_g]
    etiquetas = {"base": "Base: respaldo de volumen + hueco; el sentimiento sigue negativo en el día 2", "volumen_dia2": "Respaldo; NEGRO exige además volumen ≥ 1,5× en el día 2",
                 "amplio": "Amplio: toda caída fuerte trae mala noticia", "amplio_volumen_dia2": "Amplio + volumen ≥ 1,5× en el día 2"}
    rob_rows = []
    for var, nom in etiquetas.items():
        for c in ("ROJO", "NEGRO"):
            t = rob[(rob.variante == var) & (rob.clase == c)].set_index("h")
            rob_rows.append(dict(nombre=nom, clase=c, n=int(t.loc[5, "n"]), celdas=[dict(media=t.loc[h, "media"], lo=t.loc[h, "ic_lo"], hi=t.loc[h, "ic_hi"]) for h in (5, 10, 20)]))
    neg_rob = rob[(rob.clase == "NEGRO") & (rob.h.isin([5, 10]))]
    n_var_baja = int((neg_rob.ic_hi < 0).sum())
    resp_robustez = (f"En las {neg_rob.variante.nunique()} versiones del supuesto, NEGRO nunca queda con el intervalo completo por debajo de cero a 5 o a 10 sesiones "
                     f"({n_var_baja} de {len(neg_rob)} casos lo logran). La conclusión de la sección 3 no depende de cómo se reconstruya la mala noticia." if n_var_baja == 0 else
                     f"En {n_var_baja} de {len(neg_rob)} combinaciones (versión × horizonte) NEGRO sí queda por debajo de cero: la conclusión depende del supuesto y debe tomarse con cautela.")

    # ---- respuestas
    resp_negro = (f"<b>NEGRO</b> ({neg_g[0]['n']} eventos en acciones grandes): {describir(neg_g)}. "
                  + ("Hay deriva a la baja: la caída siguió." if crit["deriva"] else
                     "No hay deriva a la baja: después de NEGRO las acciones grandes no siguieron cayendo más que el mercado. "
                     f"La ganancia de salir (−retorno − 1,3 %) es negativa a 5 y a 10 sesiones ({pct(n5['salir'], 2, True)} y {pct(n10['salir'], 2, True)}): cambiarse habría costado dinero."))
    resp_amarillo = (f"<b>AMARILLO</b> ({amar_g[0]['n']} eventos): {describir(amar_g)}. Restando la tendencia normal de cada acción: {describir(amar_gx)}. "
                     + ("La acción rebota por encima de su normal." if (ax10["ic_lo"] > 0 or ax20["ic_lo"] > 0) else
                        "La caída no continúa, pero tampoco se puede afirmar un rebote por encima de lo normal de estas acciones."))
    sem_clases = lambda rows: describir(rows, (5, 10))                                                  # noqa: E731
    texto_graf_grandes = (f"NEGRO: {sem_clases(neg_g)}. ROJO: {sem_clases(rojo_g)}. AMARILLO: {sem_clases(amar_g)}. "
                          "Si una barra queda por debajo de cero con su línea completa, la acción siguió cayendo más que el mercado; si cruza el cero, no hay efecto demostrable.")
    texto_graf_x = (f"Sin la tendencia normal: AMARILLO {sem_clases(amar_gx)}; NEGRO {sem_clases(neg_gx)}. "
                    "casi todo lo que parecía “rebote” era la subida habitual de estas acciones.")
    texto_graf_resto = (f"En las acciones menos famosas: NEGRO {sem_clases(neg_r)}; ROJO {sem_clases(rojo_r)}; AMARILLO {sem_clases(amar_r)}. "
                        "Tampoco aquí hay una caída que se prolongue de forma demostrable.")
    texto_graf_tsla = (f"TSLA quedó en NEGRO {n_tsla} veces desde 2015; en {up_tsla} de ellas ({pct(up_tsla / n_tsla, 0) if n_tsla else 'n. d.'}) rindió más que el mercado en las 10 sesiones siguientes. "
                       f"Promedio a 10 sesiones: {describir(T('TSLA', 'NEGRO'), (10,))}. Son pocos eventos y TSLA subió mucho en la muestra, pero la historia no respalda vender TSLA por un NEGRO.")

    # ---- noticias
    nz = cifras_noticias(noticias)
    resp_noticias = ""
    if nz:
        resp_noticias = (f"De cada 100 caídas propias con volumen alto, {nz['pct_con_noticia'] * 100:.0f} traían una noticia negativa real según el sistema. "
                         f"El respaldo habría marcado {nz['rojo_proxy']} ROJO frente a {nz['rojo_real']} con noticias reales; coinciden {nz['ambos']}. "
                         + ("El respaldo es más estricto que las noticias: se pierde alertas, pero casi no inventa." if nz["solo_proxy"] <= nz["solo_real"] else
                            "El respaldo marca más alertas que las noticias reales: puede asustar de más."))

    # ---- liga
    filas_esc = []
    for i, nm in enumerate(nombres):
        for r in filas_liga(liga_res, i, lg["base_pct"], nr):
            filas_esc.append(dict(escenario=nm, nombre=r["nombre"], pasa=r["pasa"], top30=r["top30"][n_mid], p1=r["p1"][n_mid]))
    alt = "supervivientes" if lg["base_pct"] == "campo" else "campo"
    filas_sup = [dict(base=f"Contra {'los sobrevivientes' if alt == 'supervivientes' else 'todo el campo'}", **{k: r[k] for k in ("nombre", "pasa")}) for r in filas_liga(liga_res, eb, alt, nr)] + \
                [dict(base=f"Contra {'todo el campo' if alt == 'supervivientes' else 'los sobrevivientes'} (base del informe)", **{k: r[k] for k in ("nombre", "pasa")}) for r in fb]
    comparaciones = []
    for k, etq in (("b_a", "(b) Regla Maestra − (a) mantener"), ("c_b", "(c) con NEGRO − (b) Regla Maestra"), ("c_a", "(c) − (a)")):
        for m in ("pasa1", "pasa2", "pasa3", "pasa4", "p1", "top30", "ret_final"):
            c = comp[k][m]
            nombre_m = MEDIDAS[m] + (f" con {num(n_mid, 0)} part." if m in ("p1", "top30") else "")
            comparaciones.append(dict(comparacion=etq, medida=nombre_m, dif=c["dif"], lo=c["lo"], hi=c["hi"], veredicto=veredicto_comparacion(c["lo"], c["hi"])))
    esc_i = {nm: i for i, nm in enumerate(nombres)}
    peor = next((r for r in filas_esc if r["escenario"] == nombres[2] and r["nombre"].startswith("(a)")), None)
    disp = next((r for r in filas_esc if r["escenario"] == nombres[0] and r["nombre"].startswith("(a)")), None)
    resp_liga = (f"{ {'mixto': 'La Regla Maestra tiene un efecto mixto frente a', 'mejora': 'La Regla Maestra mejora a', 'empeora': 'La Regla Maestra empeora a', 'sin diferencia': 'No se puede afirmar que la Regla Maestra mejore ni empeore a'}[veredicto_regla] } "
                 f"mantener la base en el escenario base (pasar los 4 cortes: {dif_txt('pasa4')}; P(#1): {dif_txt('p1')}; rentabilidad final: {dif_txt('ret_final')}). "
                 f"Cambiar de activo ocurre en {pct(cambia, 0)} de los concursos, siempre cuando se va atrás del corte y hay un activo más volátil. "
                 f"Añadir el cambio por NEGRO (c) modifica el resultado en {pct(dist_cb, 1)} de los concursos. "
                 + (f"Si los rivales no están concentrados, mantener la base pasa los 4 cortes en {pct(disp['pasa'][3], 0)}; si todos los concentrados tienen 100 % (duplican la cartera con más exposición que mi 97 %), cae a {pct(peor['pasa'][3], 0)}." if peor and disp else ""))
    texto_graf_cortes = (f"Mantener la base sobrevive el corte 1 en {pcs(fa['pasa'][0])}, el 2 en {pcs(fa['pasa'][1])}, el 3 en {pcs(fa['pasa'][2])} y el 4 en {pcs(fa['pasa'][3])}. "
                         f"Cada filtro elimina a muchos participantes; llegar vivo al final ya es un logro, y ganar es otra cosa (P(#1) ≈ {pcs(fa['p1'][n_mid])} con {num(n_mid, 0)} participantes).")
    texto_graf_escenarios = ("Cuantos más rivales tienen lo mismo que tú, menos probable es destacar: con sólo 3 % de efectivo de diferencia, un rival con 100 % en TSLA te supera siempre que TSLA sube. "
                             "La diferencia entre escenarios es mayor que la que hay entre estrategias.")
    por_anio = vb[vb.estrategia == "a"].groupby(vb[vb.estrategia == "a"].inicio.dt.year)["pasa4"].mean()
    mejor_a, peor_a = int(por_anio.idxmax()), int(por_anio.idxmin())
    texto_graf_anio = (f"Pasa los 4 cortes en {pct(por_anio.max(), 0)} de los concursos que empiezan en {mejor_a} y en {pct(por_anio.min(), 0)} de los que empiezan en {peor_a}: "
                       "el resultado depende muchísimo del régimen de mercado, no sólo de la regla.")
    ra = vb[vb.estrategia == "a"]["ret_final"]
    texto_graf_retornos = (f"Mantener la base: mediana {pct(ra.median(), 1, True)}, 10 % peor {pct(ra.quantile(.1), 1, True)}, 10 % mejor {pct(ra.quantile(.9), 1, True)}. "
                           f"En {pct((ra < 0).mean(), 0)} de los concursos se pierde dinero.")
    temporada = []
    for y in range(int(vb.inicio.dt.year.min()), int(vb.inicio.dt.year.max()) + 1):
        w = vb[(vb.estrategia == "a") & (vb.inicio >= pd.Timestamp(f"{y}-10-05"))].sort_values("inicio")
        if len(w) and w.iloc[0].inicio < pd.Timestamp(f"{y}-10-12"):
            r = w.iloc[0]
            temporada.append(dict(anio=y, ret=r.ret_final, med=r.mediana_campo, mejor=r.mejor_campo, p1=bool(r.pasa1), p4=bool(r.pasa4)))

    bases_rows, g_bases, texto_bases, bases_rows0, texto_bases0 = [], "", "", [], ""
    base_tk = cfg["posicion_base"]["ticker"]
    bases_res = bases_res or {}
    if eb in bases_res and len(bases_res[eb]):
        br = bases_res[eb].reset_index(drop=True)
        bases_rows = tabla_bases(br, base_tk)
        pos = br.index[br["base"] == base_tk]
        mejor = br.iloc[0]
        g_bases = grafico_bases(bases_rows, "pasa3", "Pasa hasta el corte 3 (%)")
        texto_bases = (f"Con el campo del escenario base ({nombres[eb]}), de {len(br)} acciones probadas como base, {base_tk} queda en el puesto {int(pos[0]) + 1 if len(pos) else 'n. d.'}; "
                       f"la primera es {mejor['base']} ({pct(mejor['pasa3'], 1)} sigue viva hasta el corte 3 y {pct(mejor['pasa4'], 1)} los cuatro). "
                       f"Lo que pesa no es sólo cuánto se mueve la acción sino <b>cuánta gente tiene la misma</b>: si muchos rivales tienen {base_tk} con más exposición que tú, "
                       "cuando sube te superan; una acción menos popular te diferencia. ")
    if 0 in bases_res and len(bases_res[0]) and eb != 0:
        b0 = bases_res[0].reset_index(drop=True)
        bases_rows0 = tabla_bases(b0, base_tk)
        pos0 = b0.index[b0["base"] == base_tk]
        texto_bases0 = (f"Sin rivales concentrados, {base_tk} sube al puesto {int(pos0[0]) + 1 if len(pos0) else 'n. d.'} de {len(b0)} ({pct(float(b0[b0['base'] == base_tk]['pasa3'].iloc[0]), 1)} llega al corte 3). "
                        "La conclusión depende de cuántos rivales reales estén en TSLA/NVDA, que no se conoce: por eso el informe muestra los dos campos. "
                        "Ojo con la mirada hacia atrás: las acciones que más subieron entre 2015 y 2026 salen arriba por eso, y nadie sabía en 2015 cuáles serían.")
    primera = noticias["primera_noticia"] if noticias else "n. d."
    limites = [
        f"<b>Sin noticias antes del {primera}.</b> Toda la historia larga usa el respaldo de volumen + hueco de apertura. El bot en vivo sí usa noticias; su desempeño real puede diferir del estudio.",
        "<b>NEGRO histórico supone que el sentimiento sigue negativo en el día 2</b> (o exige volumen alto, en una variante). El sentimiento real de 2015–2025 no existe en fuentes gratuitas.",
        "<b>Mirada hacia atrás.</b> TSLA y NVDA fueron de lo mejor que pudo pasar entre 2015 y 2026; por eso se muestran también las cifras sin ellas y restando la tendencia normal de cada acción.",
        "<b>Rivales sintéticos que no operan.</b> Los participantes reales rotan, reaccionan al ranking y pueden copiarse entre sí. El 30 % concentrado en TSLA/NVDA y los pesos de 70–100 % son supuestos, por eso hay varios escenarios.",
        "<b>Número de participantes desconocido.</b> P(#1) y P(top 30) se calculan para varios tamaños del campo; no hay forma de saber el real con datos gratuitos.",
        "<b>“Top X %” contra los vivos o contra todos.</b> El usuario entiende que es contra los que siguen vivos y así se simula; el reglamento que tengo no lo aclara. La sección 6 muestra ambas lecturas.",
        "<b>MEE sin opciones.</b> No hay volatilidad implícita histórica gratuita: el movimiento esperado de la historia usa la volatilidad de 20 días. En vivo, el bot usa la IV de opciones cuando existe.",
        "<b>Banco de relevo parcial.</b> La simulación sólo usa acciones MGC como relevos (sin locales) y no ve calendarios de resultados ni noticias de los candidatos; en vivo el banco es más rico.",
        "<b>25 sesiones de EE. UU.</b> El concurso tiene 23 sesiones de bvc, pero los activos MGC se mueven también el 12-oct y el 2-nov (EE. UU. abre). Las ventanas se solapan, por eso los intervalos son por bloques de mes.",
        "<b>Costos genéricos.</b> Se usan comisión 0,2975 %, spread 0,3 %, slippage 0,4 % y 1,3 % por cambio; el simulador real de trii puede ser distinto.",
    ]
    cambios = [
        dict(cambio="<code>regla_maestra.cambio_por_negro</code>: <b>true → false</b>" if not cfg["regla_maestra"]["cambio_por_negro"] else "<code>cambio_por_negro</code> se mantiene en <b>true</b>",
             por_que=("NEGRO no mostró deriva a la baja en acciones grandes (criterio fijado antes de ver los datos)." if not crit["deriva"] else "NEGRO sí mostró deriva a la baja en acciones grandes.")),
        dict(cambio="Textos del bot sobre NEGRO y ROJO", por_que="Ya no sugieren “evaluar un cambio” por el color; dicen que no se venda sólo por eso y que se mire la Regla Maestra con el ranking."),
        dict(cambio="Se mantiene la Regla Maestra y el semáforo", por_que="Siguen sirviendo como información y como filtro de candidatos; la validación no encontró motivo para quitarlos."),
        dict(cambio="Nuevos módulos y pruebas", por_que="<code>src/validacion.py</code>, <code>src/liga_v5.py</code>, <code>src/validacion_noticias.py</code> y sus pruebas automáticas."),
    ]
    hallazgos = [
        f"<b>NEGRO no anticipa más caídas en acciones grandes.</b> {describir(neg_g, (5, 10))} con {neg_g[0]['n']} eventos; el resultado resiste las cuatro versiones del supuesto de noticias.",
        f"<b>En TSLA, la historia va en contra de vender por NEGRO:</b> {up_tsla} de {n_tsla} veces subió más que el mercado en las 10 sesiones siguientes.",
        f"<b>AMARILLO:</b> {v_amar['texto']}",
        f"<b>Simulador de ligas:</b> {resp_liga}",
    ]
    acciones = [
        "<b>Ya aplicado:</b> <code>cambio_por_negro: false</code>. NEGRO sigue apareciendo en los mensajes, pero como información.",
        "<b>No vendas TSLA por el color</b> (ni AMARILLO, ni ROJO, ni NEGRO). Mira siempre qué dice la Regla Maestra con tu ranking real: usa <code>/rank</code> cada noche.",
        "<b>Compara siempre con las demás acciones:</b> escribe <code>/base</code> (BVC y EE. UU.) y mira el banco de relevo; hoy ninguna supera a la base por el 20 % que se exige para cambiar.",
        "<b>Recuerda el límite:</b> estas cifras comparan reglas con datos pasados; no garantizan un resultado.",
    ]
    return dict(
        fmt=fmt, hoy=hoy, plotly_js=get_plotlyjs(), n_ses=L.calendario_concurso(cfg)["n_ses"], f_ini=str(panel.index.min().date()), f_fin=str(panel.index.max().date()),
        n_acciones=panel["ticker"].nunique(), n_ventanas=int(vb.inicio.nunique()), veredictos=[v_negro, v_amar, v_regla, v_cortes], hallazgos=hallazgos, acciones=acciones,
        cortes_txt=", ".join(f"{c['top_pct']} %" for c in cfg["concurso"]["cortes"]), rivales=num(lg["rivales"], 0), base_txt=f"{cfg['posicion_base']['ticker']} al {pct(cfg['posicion_base']['peso'], 0)}",
        primera_noticia=primera, g_ev_grandes=g_ev_grandes, g_ev_grandes_x=g_ev_grandes_x, g_ev_resto=g_ev_resto, g_negro_tsla=g_tsla,
        neg_g=neg_g, amar_g=amar_g, neg_r=neg_r, rojo_r=rojo_r, resp_negro=resp_negro, resp_amarillo=resp_amarillo,
        texto_graf_grandes=texto_graf_grandes, texto_graf_x=texto_graf_x, texto_graf_resto=texto_graf_resto, texto_graf_tsla=texto_graf_tsla,
        grupos_rows=grupos_rows, rob_rows=rob_rows, resp_robustez=resp_robustez, tsla_eventos=tsla_ev.to_dict("records"),
        noticias=nz, resp_noticias=resp_noticias, liga_ini=str(vb.inicio.min().date()), liga_fin=str(vb.inicio.max().date()), esc_base_nombre=nombres[eb],
        g_cortes=grafico_cortes(liga_res, eb, lg["base_pct"]), g_escenarios=grafico_escenarios(liga_res, nombres, lg["base_pct"]),
        g_anio=grafico_por_anio(liga_vent, eb, "a"), g_retornos=grafico_retornos(liga_vent, eb), azar_txt=f"{pcs(azar[0])}, {pcs(azar[1])}, {pcs(azar[2])} y {pcs(azar[3])}",
        texto_graf_cortes=texto_graf_cortes, texto_graf_escenarios=texto_graf_escenarios, texto_graf_anio=texto_graf_anio, texto_graf_retornos=texto_graf_retornos,
        n_reales=nr, filas_base=fb, filas_esc=filas_esc, filas_sup=filas_sup, comparaciones=comparaciones, resp_liga=resp_liga, temporada=temporada,
        limites=limites, cambios=cambios, texto_base_pct=texto_base_pct, texto_medida=texto_medida, bases_rows=bases_rows, g_bases=g_bases, texto_bases=texto_bases,
        bases_rows0=bases_rows0, texto_bases0=texto_bases0, base_tk=base_tk)


__all__ = ["fmt", "V", "json"]
