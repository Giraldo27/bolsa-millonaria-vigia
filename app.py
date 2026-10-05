"""Dashboard de Bolsa Millonaria (Streamlit): se lee bien en el celular. Sólo muestra información y recomendaciones: NUNCA envía órdenes.

Uso local:   python -m streamlit run app.py
Nube:        Streamlit Community Cloud (ver README, sección 8). Sólo necesita FINNHUB_API_KEY como secreto; NO necesita el token de Telegram ni tu posición."""
from __future__ import annotations

import os

import plotly.graph_objects as go
import streamlit as st

from src import concurso as C
from src import formato as F
from src import servicio as S
from src import tablero as T
from src.informe_html import armar_informe
from src.semaforo import EMOJI

for _k in ("FINNHUB_API_KEY", "ALPHAVANTAGE_API_KEY"):                       # en la nube las claves vienen de los "secrets" de Streamlit, no de archivos
    try:
        if _k in st.secrets and not os.environ.get(_k):
            os.environ[_k] = str(st.secrets[_k])
    except Exception:                                                        # noqa: BLE001 — sin archivo de secretos (uso local con .env)
        pass

st.set_page_config(page_title="Bolsa Millonaria", page_icon="🚦", layout="centered", initial_sidebar_state="collapsed")
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:760px}
.sem{display:flex;gap:16px;align-items:center;border:1px solid rgba(137,135,129,.35);border-radius:16px;padding:14px 18px;margin:6px 0 12px}
.sem .dot{width:72px;height:72px;border-radius:50%;flex:none}.sem h2{margin:0;font-size:1.5rem}.sem p{margin:2px 0 0;opacity:.8}
</style>""", unsafe_allow_html=True)

COLOR = {"VERDE": "#0ca30c", "AZUL": "#2a78d6", "AMARILLO": "#eda100", "ROJO": "#d03b3b", "NEGRO": "#3b3b3b"}


@st.cache_data(ttl=300, show_spinner="Consultando precios, noticias y relevos (≈ 1 minuto la primera vez)…")
def datos(_clave: int = 0) -> T.DatosTablero:
    return T.cargar(S.crear_contexto())


def traductor():
    return S.crear_contexto().f.traducir


@st.fragment(run_every="5m")                                                 # se recalcula solo cada 5 minutos (los datos se guardan 5 min)
def tablero() -> None:
    ctx = S.crear_contexto()
    cfg = ctx.cfg
    try:
        d = datos(st.session_state.get("clave", 0))
    except Exception as e:                                                   # noqa: BLE001 — sin datos no hay tablero, pero se explica y se puede reintentar
        st.error("No pude cargar los datos ahora (alguna fuente no responde). Prueba otra vez en un minuto o mira trii a mano.")
        st.caption(f"Detalle técnico: {type(e).__name__}")
        if st.button("Reintentar"):
            st.cache_data.clear()
            st.rerun(scope="app")
        return
    r, res = d.r, d.r.res
    c1, c2 = st.columns([3, 2])
    c1.markdown(f"### 🚦 {d.activo}")
    c1.caption(f"{F.fecha(d.ahora)} {d.ahora:%H:%M} (hora de Bogotá) · se actualiza solo cada 5 min")
    if c2.button("🔄 Actualizar ahora", width="stretch"):
        S.purgar_cache(ctx)
        st.cache_data.clear()
        st.session_state["clave"] = st.session_state.get("clave", 0) + 1
        st.rerun(scope="app")
    for a in d.avisos:
        st.warning(a)

    t_hoy, t_rel, t_calc, t_not, t_fec, t_base = st.tabs(["🚦 Hoy", "🔁 Relevos", "🧮 Calculadora", "📰 Noticias", "📅 Fechas", "🔎 Base"])

    with t_hoy:
        st.markdown(f"<div class='sem'><div class='dot' style='background:{COLOR[res.color]}'></div><div><h2>{F.esc(res.ticker)}: {F.esc(F.NOMBRE_COLOR[res.color])}</h2>"
                    f"<p>{F.esc(F.EXPLICA_COLOR[res.color])}</p></div></div>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Hoy", F.pct(res.r_hoy, 1, True))
        m2.metric("Fuerza (σ)", F.n(res.z, 1, True), help="Cuántas variaciones diarias típicas se movió hoy. Por debajo de −2 es una caída anormal.")
        m3.metric("Volumen", (F.n(res.vol_rel, 1) + "×") if res.vol_rel is not None else "n. d.", help="Frente al volumen normal de la acción.")
        st.markdown(f"**¿Qué pasó?** {F.plano(F.que_paso(res))}")
        st.markdown(f"**👉 Qué hacer:** {F.ACCION_COLOR[res.color]}")
        hora = C.hora_orden_manana(C.proxima_sesion(d.ahora, cfg) or d.ahora.date(), cfg)
        st.info("⚖️ **¿Cambio de acción?**\n\n" + "\n\n".join(F.plano(x) for x in F.texto_decision(r.decision, res.ticker, hora)))
        if d.intradia is not None:
            fig = go.Figure(go.Scatter(x=d.intradia["hora"], y=d.intradia["sigma"], mode="lines+markers", line=dict(color="#2a78d6"), name=res.ticker,
                                       hovertemplate="%{x}<br>%{y:.2f} σ<extra></extra>"))
            fig.add_hrect(y0=-10, y1=-2, fillcolor="rgba(208,59,59,.12)", line_width=0)
            for y in (-2, -1, 0, 1, 2):
                fig.add_hline(y=y, line_dash="dot" if y else "solid", line_color="rgba(137,135,129,.6)", line_width=1)
            fig.update_layout(height=300, margin=dict(l=40, r=10, t=10, b=30), yaxis_title="Movimiento del día (σ)", showlegend=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", yaxis=dict(range=[min(-3, d.intradia["sigma"].min() - .3), max(3, d.intradia["sigma"].max() + .3)]))
            st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
            st.caption("Cada punto es una barra de 15 minutos. Por debajo de la franja roja (−2σ) la caída ya es anormal para esta acción.")
        g = d.estado
        if g.get("mia") is not None and g.get("umbral") is not None:
            st.caption(F.plano(F.linea_ranking({"mia": g["mia"], "umbral": g["umbral"]})))
        st.caption(f"🔁 Cambios de acción usados: {g.get('cambios_usados', 0)} de {g.get('cambios_max', cfg['estado']['max_cambios'])}")

    with t_rel:
        st.markdown(f"**Mejores relevos para {d.activo}** (de mayor a menor movimiento esperado hasta el próximo corte)")
        tb = T.df_banco(r)
        st.dataframe(tb, hide_index=True, width="stretch") if len(tb) else st.write("Hoy ningún candidato cumple los filtros.")
        with st.expander("Los demás evaluados y por qué no están"):
            st.dataframe(T.df_descartados(r, cfg, d.permitidos), hide_index=True, width="stretch")
        st.caption("Se revisan las acciones de EE. UU. y de la BVC. Más movimiento esperado = más oportunidad y más riesgo; no es una promesa de ganancia.")

    with t_calc:
        st.markdown("**¿Me conviene cambiar?** Escribe tus números (en %, por ejemplo 6,5) y aplico la Regla Maestra con los datos de hoy.")
        if g.get("mia") is None or g.get("umbral") is None:
            st.caption("Aún no hay un ranking guardado: escribe tu rentabilidad y la del corte para que el cálculo tenga sentido.")
        a1, a2 = st.columns(2)
        mia = a1.number_input("Mi rentabilidad (%)", value=float(g.get("mia") or 0.0), step=0.1, format="%.1f")
        obj = a2.number_input("Objetivo del corte (%)", value=float(g.get("umbral") or 0.0), step=0.1, format="%.1f", help="Rentabilidad que marca hoy el corte; en la semana 5, la del #1.")
        b1, b2 = st.columns(2)
        usados = b1.number_input("Cambios ya usados", min_value=0, max_value=cfg["estado"]["max_cambios"], value=int(g.get("cambios_usados", 0)))
        hoy_cambio = b2.checkbox("Ya hice un cambio hoy", value=bool(g.get("cambio_hoy", False)))
        dec, banco = T.calc_regla(d, cfg, mia, obj, int(usados), hoy_cambio)
        if dec.g is not None:
            k1, k2 = st.columns(2)
            k1.metric("Te faltan (g)", F.n(dec.g, 1) + " pts", help="objetivo − tu rentabilidad + 1 de margen de seguridad")
            k2.metric("El nuevo debe moverse", (F.n(dec.multiplo, 2) + "× lo tuyo") if dec.multiplo else "—", help="(g + 1,3) / g: el 1,3 es el costo de cambiar")
        st.success("\n\n".join(F.plano(x) for x in F.texto_decision(dec, d.activo, C.hora_orden_manana(C.proxima_sesion(d.ahora, cfg) or d.ahora.date(), cfg))))
        ev = T.evaluados(dec)
        if len(ev):
            st.dataframe(ev, hide_index=True, width="stretch")

    with t_not:
        nt = T.df_noticias(d.noticias, d.ahora, traductor())
        if d.noticias is None:
            st.write("No pude consultar noticias ahora. Mira la ficha en Yahoo Finance.")
        elif nt.empty:
            st.write("No hay noticias recientes.")
        else:
            st.caption("😟 mala · 😐 neutral · 🙂 buena. El tono lo calcula un programa: úsalo como pista, no como verdad.")
            st.dataframe(nt, hide_index=True, width="stretch")

    with t_fec:
        tc = T.df_catalizadores(d.cats)
        st.dataframe(tc, hide_index=True, width="stretch") if len(tc) else st.write("No hay fechas importantes conocidas en los próximos días hábiles.")

    with t_base:
        if d.base_rank:
            from src.seleccion import veredicto
            v, mejores = veredicto(d.base_rank, cfg)
            (st.warning if v == "revisar" else st.success)(F.plano(F.msg_base(d.base_rank, v, mejores, d.activo, cfg, top=3)).split("\n\n")[1])
            st.dataframe(T.df_base(d.base_rank), hide_index=True, width="stretch")
        else:
            st.write("No pude calcular la comparación de base ahora.")

    st.divider()
    if st.button("📎 Preparar informe completo (HTML)", width="stretch"):
        est = S.leer_estado(ctx)
        html = armar_informe(est, r, d.cats, d.noticias, cfg, d.ahora, traductor(), d.permitidos, d.base_rank)
        st.download_button("⬇️ Descargar informe", html.encode("utf-8"), file_name=f"informe_{d.ahora:%Y%m%d_%H%M}.html", mime="text/html", width="stretch")
    st.caption("Esto es información y recomendaciones, no órdenes ni promesas. Tú das las órdenes en trii. " + EMOJI["VERDE"] + EMOJI["AZUL"] + EMOJI["AMARILLO"] + EMOJI["ROJO"] + EMOJI["NEGRO"])


st.title("Bolsa Millonaria 2026")
tablero()
