"""Cálculo de cifras y gráficos del informe (sin HTML de presentación: eso vive en templates/informe.html.j2)."""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_loader import load_prices
from src.universe import atr_pct

from . import fmt

# ---------- paleta ----------
GAIN, LOSS, NEUTRAL = "#0ca30c", "#d03b3b", "#898781"
ASSET = {"TSLA": "#2a78d6", "NVDA": "#eb6834", "ECOPETROL": "#4a3aa7"}
MODE = {"blindaje": "#7f8c8d", "mantener": "#eda100", "ataque": "#e87ba4", "allin": "#1baf7a"}
MODE_NAME = {"blindaje": "Blindaje", "mantener": "Mantener", "ataque": "Ataque", "allin": "All-In"}
MODES = list(MODE_NAME)
GRID = "rgba(137,135,129,0.25)"
REASON = {"SL": "stop loss", "TP": "take profit", "tiempo": "stop de tiempo", "reporte": "venta antes del reporte",
          "catalizador": "fin del setup B", "brent_cae": "caída del Brent", "SL_trailing": "stop de trailing",
          "SL_usdcop": "stop por dólar", "fin_datos": "fin de datos"}


def base_layout(fig: go.Figure, height: int = 420, **kw) -> go.Figure:
    opts = dict(height=height, margin=dict(l=55, r=15, t=30, b=50), separators=",.", paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)", font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", size=13, color=NEUTRAL),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0), hoverlabel=dict(font_size=13))
    opts.update(kw)
    fig.update_layout(**opts)
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


_first = [True]


def html(fig: go.Figure, div_id: str) -> str:
    """Un solo bloque con Plotly inline (el primero); el resto reutiliza la misma librería."""
    mode = "inline" if _first[0] else False
    _first[0] = False
    return fig.to_html(full_html=False, include_plotlyjs=mode, div_id=div_id, config={"displaylogo": False, "responsive": True,
                                                                                       "modeBarButtonsToRemove": ["lasso2d", "select2d"]})


def wstats(x: np.ndarray | pd.Series) -> dict[str, float]:
    x = np.asarray(x, float)
    return dict(n=len(x), media=x.mean(), mediana=np.median(x), p10=np.percentile(x, 10), p90=np.percentile(x, 90),
                gt0=(x > 0).mean(), gt3=(x > 0.03).mean(), gt10=(x > 0.10).mean())


def build_context(cfg: dict[str, Any]) -> dict[str, Any]:
    out = cfg["paths"]["outputs_dir"]
    f3, f4, f4k, f4i = out / "fase3", out / "fase4", out / "fase4k", out / "fase4i"
    ctx: dict[str, Any] = {"fmt": fmt}

    # ---------- datos base ----------
    met = pd.read_csv(f3 / "metricas_continuo.csv", index_col="modo")
    fix = pd.read_csv(f3 / "metricas_tamano_fijo.csv", index_col="modo")
    boot = pd.read_csv(f3 / "bootstrap.csv", index_col="modo")
    mc = pd.read_csv(f3 / "montecarlo.csv", index_col="modo")
    oos = pd.read_csv(f3 / "dentro_fuera_muestra.csv")
    brk = pd.read_csv(f3 / "desglose_setup_activo.csv")
    bench_m = pd.read_csv(f3 / "benchmarks_metricas.csv", index_col=0)
    avd = pd.read_csv(f4 / "antes_vs_despues.csv")
    bavd = pd.read_csv(f4 / "bootstrap_antes_despues.csv")
    mavd = pd.read_csv(f4 / "montecarlo_antes_despues.csv")
    dec = pd.read_csv(f4 / "decisiones.csv")
    liga = pd.read_csv(f4 / "liga_resumen.csv", index_col="estrategia")
    vol = pd.read_csv(f4k / "volatilidad_ventana.csv")
    conc = pd.read_csv(f4i / "concentrada.csv")
    rank = pd.read_csv(out / "fase1_ranking.csv", index_col=0)
    paths = np.load(f4k / "paths_plan.npy")
    paths_tsla = np.load(f4k / "paths_tsla.npy")
    wdates = pd.to_datetime(np.load(f4k / "dates_plan.npy"))
    win = {m: pd.read_parquet(f3 / f"ventanas_{m}.parquet") for m in MODES}
    winm = {m: pd.read_parquet(f4 / f"ventanas_mejorada_{m}.parquet") for m in MODES}
    wb = pd.read_parquet(f3 / "ventanas_benchmarks.parquet")
    plan_final = paths[:, -1]
    ps = wstats(plan_final)
    py = pd.Series(plan_final, index=wdates).groupby(wdates.year).mean()
    rng = np.random.default_rng(5)
    boot_years = np.array([rng.choice(py.values, len(py)).mean() for _ in range(5000)])
    plan_ci = (np.percentile(boot_years, 2.5), np.percentile(boot_years, 97.5))

    def vrow(sel: str, campo: str, tramo: str = "todo") -> pd.Series:
        return vol[(vol.seleccion == sel) & (vol.campo == campo) & (vol.tramo == tramo)].iloc[0]

    PLAN = "más volátil 250d (mgc)"
    plan_a, plan_m = vrow(PLAN, "amplio"), vrow(PLAN, "mgc")
    plan_a_te, plan_m_te = vrow(PLAN, "amplio", "test"), vrow(PLAN, "mgc", "test")
    tsla_a = vrow("TSLA fijo", "amplio")
    tsla_with_matrix = conc[(conc.seleccion == "TSLA fijo (ref. hindsight)") & (conc.politica == "matriz") & (conc.campo == "amplio") & (conc.tramo == "todo")].iloc[0]
    tsla_no_matrix = conc[(conc.seleccion == "TSLA fijo (ref. hindsight)") & (conc.politica == "ninguna") & (conc.campo == "amplio") & (conc.tramo == "todo")].iloc[0]
    ctx.update(plan_stats=ps, plan_ci=plan_ci, plan_years_pos=int((py > 0).sum()), plan_years=len(py),
               plan_a=plan_a, plan_m=plan_m, plan_a_te=plan_a_te, plan_m_te=plan_m_te, tsla_a=tsla_a,
               p_gt20=float((plan_final > 0.20).mean()), p_lt10=float((plan_final < -0.10).mean()),
               tsla_matrix=tsla_with_matrix, tsla_nomatrix=tsla_no_matrix, plan_pcts={q: float(np.percentile(plan_final, q)) for q in (5, 25, 50, 75, 90, 95)})

    # ---------- 1. veredicto ----------
    o = lambda m, c: met.loc[m, c]                                                       # noqa: E731
    ctx["verdict"] = dict(
        rent_min=min(met.rent_neta), rent_max=max(met.rent_neta), win_min=met.win_rate.min(), win_max=met.win_rate.max(),
        exp_orig=met.loc["blindaje", "expectancy_pct"], exp_mej=avd[(avd.version == "mejorada") & (avd.modo == "blindaje") & (avd.periodo == "completo")].iloc[0].expectancy_pct,
        costo_trade=float((brk[brk.modo == "blindaje"].costo_pct * brk[brk.modo == "blindaje"].n).sum() / brk[brk.modo == "blindaje"].n.sum()))

    # ---------- 2. tarjetas ----------
    def vals(col: str, f, source=met) -> list[dict]:
        return [dict(modo=MODE_NAME[m], v=f(source.loc[m, col]), c=fmt.cls(source.loc[m, col])) for m in MODES]
    cards = [
        dict(t="Rentabilidad neta total", tip="Cuánto creció (o se redujo) tu dinero entre 2015 y 2026 después de todos los costos. Ejemplo: si empiezas con $ 100.000.000 y terminas con $ 25.000.000, la rentabilidad neta es −75 %.",
             vals=vals("rent_neta", lambda x: fmt.pct(x, 0, True)),
             mean=f"Con interés compuesto, el sistema original terminó con entre {fmt.pct(1 + met.rent_neta.max(), 0)} y {fmt.pct(1 + met.rent_neta.min(), 0)} del capital inicial."),
        dict(t="Retorno anualizado (CAGR)", tip="El retorno promedio por año si el dinero hubiera crecido (o caído) al mismo ritmo cada año. Un −11 % significa perder 11 de cada 100 pesos, cada año, en promedio.",
             vals=vals("cagr", lambda x: fmt.pct(x, 1, True)), mean="Un CAGR negativo significa que la estrategia destruye capital año tras año."),
        dict(t="Máximo drawdown", tip="La peor caída desde un máximo hasta un mínimo posterior. Si tu patrimonio llegó a $ 100.000.000 y luego cayó a $ 40.000.000 antes de recuperarse, el drawdown es −60 %.",
             vals=vals("max_dd", lambda x: fmt.pct(x, 0, True)), mean="Es el dolor máximo que habrías tenido que aguantar."),
        dict(t="Sharpe / Sortino", tip="Retorno por unidad de riesgo. Sharpe usa toda la volatilidad; Sortino sólo la de las caídas. Por encima de 1 es bueno; negativo significa que el riesgo no se pagó.",
             vals=[dict(modo=MODE_NAME[m], v=f"{fmt.num(met.loc[m, 'sharpe'], 2)} / {fmt.num(met.loc[m, 'sortino'], 2)}", c=fmt.cls(met.loc[m, 'sharpe'])) for m in MODES],
             mean="Todos negativos: el riesgo asumido no se compensó con retorno."),
        dict(t="Tasa de acierto real vs. mínima teórica", tip="Porcentaje de operaciones ganadoras. La 'mínima teórica' (44 % en ECOPETROL y NVDA, 43 % en TSLA) es el acierto que necesitas para no perder dado el TP/SL y los costos.",
             vals=[dict(modo=MODE_NAME[m], v=f"{fmt.pct(met.loc[m, 'win_rate'], 0)} (mín. {fmt.pct(met.loc[m, 'win_rate_min'], 0)})", c="neg" if met.loc[m, "win_rate"] < met.loc[m, "win_rate_min"] else "pos") for m in MODES],
             mean="En todos los modos el acierto real quedó muy por debajo del mínimo necesario."),
        dict(t="Payoff y profit factor", tip="Payoff = ganancia media ÷ pérdida media. Profit factor = suma de ganancias ÷ suma de pérdidas. Por encima de 1 la estrategia gana más de lo que pierde.",
             vals=[dict(modo=MODE_NAME[m], v=f"{fmt.num(met.loc[m, 'payoff'], 2)} / {fmt.num(met.loc[m, 'profit_factor'], 2)}", c="pos" if met.loc[m, "profit_factor"] > 1 else "neg") for m in MODES],
             mean="Con profit factor menor que 1, por cada $ 1 que se pierde sólo se gana entre $ 0,46 y $ 0,63."),
        dict(t="Expectancy por trade", tip="Lo que esperas ganar (o perder) en promedio en cada operación. Ejemplo: −1,2 % sobre $ 40.000.000 son −$ 467.000 por operación.",
             vals=[dict(modo=MODE_NAME[m], v=f"{fmt.pct(met.loc[m, 'expectancy_pct'], 2, True)} · {fmt.cop(met.loc[m, 'expectancy_cop_40m'])}", c=fmt.cls(met.loc[m, "expectancy_pct"])) for m in MODES],
             mean="Cada operación del sistema original perdía, en promedio, más de un punto porcentual."),
        dict(t="Número de trades y días en posición", tip="Cuántas operaciones se hicieron en 11,8 años y cuántas sesiones se mantuvo cada una en promedio.",
             vals=[dict(modo=MODE_NAME[m], v=f"{int(met.loc[m, 'n_trades'])} · {fmt.num(met.loc[m, 'dias_prom'], 1)} d", c="neu") for m in MODES],
             mean="Operaciones muy cortas (≈ 2 días) pagan costos casi sin tiempo para que el precio se mueva."),
        dict(t="Exposición media", tip="Qué fracción del patrimonio estuvo invertida en promedio. El resto estuvo en efectivo.",
             vals=[dict(modo=MODE_NAME[m], v=fmt.pct(met.loc[m, "exposicion_media"], 0), c="neu") for m in MODES],
             mean="Aun con poca exposición, los costos por operación pesan."),
        dict(t="Costos pagados", tip="Comisiones, spread de entrada y slippage de salida. 'Como % del P&L bruto' compara los costos con la ganancia antes de costos; si el P&L bruto es negativo no tiene sentido (n. a.).",
             vals=[dict(modo=MODE_NAME[m], v=f"{fmt.cop_m(met.loc[m, 'costos_cop'], 0)} · bruto {fmt.cop_m(met.loc[m, 'pnl_bruto_cop'], 0)}", c="neg") for m in MODES],
             mean="Antes de costos el sistema empata (P&L bruto ≈ 0); los costos convierten el empate en pérdida."),
    ]
    for c_ in cards:
        c_["vals"] = [dict(v) for v in c_["vals"]]
    ctx["cards"] = cards

    # ---------- 3. simulación del concurso ----------
    series = {"Plan final (acción más volátil)": plan_final, "Sistema original · Mantener": win["mantener"].ret.values,
              "Sistema mejorado · Mantener": winm["mantener"].ret.values, "Comprar y mantener TSLA": paths_tsla[:, -1]}
    cols = {"Plan final (acción más volátil)": "#2a78d6", "Sistema original · Mantener": MODE["mantener"],
            "Sistema mejorado · Mantener": "#e87ba4", "Comprar y mantener TSLA": "#4a3aa7"}
    fig = go.Figure()
    for k, v in series.items():
        fig.add_trace(go.Histogram(x=v * 100, name=k, xbins=dict(start=-40, end=60, size=2), marker_color=cols[k], opacity=0.6,
                                   histnorm="percent", visible=True if "Plan" in k or "original" in k else "legendonly",
                                   hovertemplate="Retorno %{x} %<br>%{y:.1f} % de las ventanas<extra>" + k + "</extra>"))
    fig.update_layout(barmode="overlay", xaxis_title="Retorno neto en 23 sesiones (%)", yaxis_title="% de las ventanas")
    ctx["hist_html"] = html(base_layout(fig, 420), "hist")
    rows = []
    for k, v in series.items():
        s = wstats(v)
        rows.append(dict(nombre=k, **s))
    for m in ("blindaje", "ataque", "allin"):
        rows.append(dict(nombre=f"Sistema original · {MODE_NAME[m]}", **wstats(win[m].ret.values)))
    ctx["win_rows"] = rows

    # liga
    names = [("Plan final (acción más volátil)", None), ("TSLA comprar y mantener (referencia)", None)]
    lg = pd.DataFrame({
        "Plan final": [plan_a[f"pasa_{c}"] for c in (5, 10, 15, 20)] + [plan_a.campeon],
        "Sistema mejorado (matriz dinámica)": [liga.loc["mejor_dinamica", f"pasa_corte_{c}"] for c in (5, 10, 15, 20)] + [liga.loc["mejor_dinamica", "es_primero"]],
        "Sistema original (matriz dinámica)": [liga.loc["orig_dinamica", f"pasa_corte_{c}"] for c in (5, 10, 15, 20)] + [liga.loc["orig_dinamica", "es_primero"]]},
        index=["Pasa corte 1", "Pasa cortes 1-2", "Pasa cortes 1-3", "Pasa cortes 1-4", "Pasa todo y termina #1 (aprox.)"])
    fig = go.Figure()
    for k, colr in zip(lg.columns, ("#2a78d6", "#e87ba4", "#eda100")):
        fig.add_trace(go.Bar(x=lg.index, y=lg[k] * 100, name=k, marker_color=colr, hovertemplate="%{x}<br>%{y:.1f} %<extra>" + k + "</extra>"))
    fig.update_layout(barmode="group", yaxis_title="Probabilidad (%)", bargap=0.25)
    ctx["liga_html"] = html(base_layout(fig, 400), "liga")
    ctx["liga"] = lg

    # ---------- 4. robustez ----------
    fb = go.Figure()
    labels, mids, los, his, colr = [], [], [], [], []
    for v in ("original", "mejorada"):
        for m in ("blindaje", "mantener", "ataque", "allin"):
            r = bavd[(bavd.version == v) & (bavd.modo == m)].iloc[0]
            labels.append(f"{'Original' if v == 'original' else 'Mejorado'} · {MODE_NAME[m]}")
            mids.append(r.media * 100); los.append(r.ic_lo * 100); his.append(r.ic_hi * 100)
            colr.append(LOSS if r.ic_hi < 0 else NEUTRAL)
    fb.add_trace(go.Scatter(x=mids, y=labels, mode="markers", marker=dict(size=10, color=colr),
                            error_x=dict(type="data", symmetric=False, array=np.array(his) - np.array(mids), arrayminus=np.array(mids) - np.array(los), color=NEUTRAL, thickness=2),
                            hovertemplate="%{y}<br>Expectancy %{x:.2f} %<extra></extra>", name="Expectancy por trade (IC 95 %)"))
    fb.add_vline(x=0, line_color=NEUTRAL, line_dash="dash")
    fb.update_layout(xaxis_title="Expectancy por trade (%) · IC 95 %", showlegend=False)
    fb.update_yaxes(autorange="reversed")
    ctx["boot_html"] = html(base_layout(fb, 400, margin=dict(l=120, r=15, t=20, b=55)), "boot")
    mcm = np.load(f3 / "mc_blindaje_mean.npy") * 100
    fm = go.Figure()
    fm.add_trace(go.Histogram(x=mcm, nbinsx=40, marker_color=NEUTRAL, opacity=0.7, name="1.000 estrategias aleatorias",
                              hovertemplate="%{x:.2f} %<br>%{y} corridas<extra></extra>"))
    fm.add_vline(x=mc.loc["blindaje", "strat_mean_ret"] * 100, line_color=LOSS, line_width=3,
                 annotation_text="Sistema original", annotation_position="top")
    fm.update_layout(xaxis_title="Retorno neto medio por trade (%)", yaxis_title="Corridas", showlegend=False)
    ctx["mc_html"] = html(base_layout(fm, 340), "mc")
    ctx["robust"] = dict(boot=boot, mc=mc, oos=oos, avd=avd, bavd=bavd, mavd=mavd)
    ctx["oos_rows"] = [dict(modo=MODE_NAME[r.modo], periodo=r.periodo, n=int(r.n_trades), win=r.win_rate, exp=r.expectancy_pct, pnl=r.pnl_neto_cop)
                       for r in oos[oos.modo.isin(["blindaje", "mantener"])].itertuples()]

    # ---------- 6. activos ----------
    bl = []
    for t in cfg["universe"]["blacklist"]:
        try:
            d = load_prices(t, cfg)
        except Exception:
            continue
        if d.empty:
            continue
        rec = d[d.index > d.index[-1] - pd.DateOffset(years=cfg["data"]["recent_years"])]
        bl.append(dict(ticker=t, atr_pct=atr_pct(d).reindex(rec.index).mean(), valor_cop_mm=(rec.Close * rec.Volume).mean() / 1e6,
                       pct_vol0=(rec.Volume.fillna(0) == 0).mean() * 100))
    bl = pd.DataFrame(bl)
    ctx["blacklist"] = bl
    fs = go.Figure()
    others = rank[~rank.index.isin(cfg["universe"]["system_assets"])]
    # eje en miles de millones de COP por día (valor_cop_mm está en millones)
    fs.add_trace(go.Scatter(x=(others.valor_cop_mm / 1000).clip(lower=1e-3), y=others.atr_pct, mode="markers", name="Universo permitido", text=others.index,
                            marker=dict(size=9, color="#9aa0a6", line=dict(width=1, color="rgba(255,255,255,0.8)")),
                            hovertemplate="%{text}<br>Valor negociado: %{x:,.2f} mil millones COP/día<br>Volatilidad (ATR): %{y:.2f} %<extra></extra>"))
    if len(bl):
        fs.add_trace(go.Scatter(x=(bl.valor_cop_mm / 1000).clip(lower=1e-3), y=bl.atr_pct, mode="markers+text", name="Lista negra (no operar)", text=bl.ticker,
                                textposition="top center", textfont=dict(size=10), marker=dict(size=10, color="rgba(0,0,0,0)", symbol="x", line=dict(width=2, color="#4d4d4d")),
                                hovertemplate="%{text} (LISTA NEGRA)<br>Valor negociado: %{x:,.3f} mil millones COP/día<br>ATR: %{y:.2f} %<extra></extra>"))
    for t, cl_ in ASSET.items():
        r = rank.loc[t]
        fs.add_trace(go.Scatter(x=[max(r.valor_cop_mm / 1000, 1e-3)], y=[r.atr_pct], mode="markers+text", name=t, text=[t], textposition="top center",
                                marker=dict(size=16, color=cl_, line=dict(width=2, color="white")),
                                hovertemplate=t + f"<br>Puesto {int(r['rank'])} de {len(rank)}<br>Valor negociado: %{{x:,.1f}} mil millones COP/día<br>ATR: %{{y:.2f}} %<extra></extra>"))
    ticks = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4, 1e5]
    fs.update_xaxes(type="log", title="Valor negociado/día (mil millones COP, log)", tickvals=ticks,
                    ticktext=[fmt.num(v, 3 if v < 0.01 else (2 if v < 0.1 else (1 if v < 1 else 0))) for v in ticks])
    fs.update_yaxes(title="Volatilidad ATR 14 d (%)")
    ctx["scatter_html"] = html(base_layout(fs, 480), "scatter")
    top = rank.head(12)
    ctx["rank_rows"] = [dict(pos=int(r["rank"]), t=t, g={"mgc": "MGC", "etf": "ETF", "local": "Local"}[r.grupo], atr=r.atr_pct, val=r.valor_cop_mm, v0=r.pct_vol0,
                             ev=r.eventos_ventana_ano, pre=r.pre_media_pct, score=r.score, sel=t in cfg["universe"]["system_assets"]) for t, r in top.iterrows()]
    ctx["rank_sys"] = [dict(pos=int(rank.loc[t, "rank"]), t=t, atr=rank.loc[t].atr_pct, val=rank.loc[t].valor_cop_mm, v0=rank.loc[t].pct_vol0,
                            ev=rank.loc[t].eventos_ventana_ano, pre=rank.loc[t].pre_media_pct, score=rank.loc[t].score) for t in cfg["universe"]["system_assets"]]
    ctx["n_universe"] = len(rank)

    # ---------- 7. una operación de principio a fin ----------
    tr = pd.read_csv(f3 / "trades_blindaje.csv", parse_dates=["entry_date", "exit_date"])
    cand = tr[(tr.asset == "TSLA") & (tr.exit_reason == "SL_trailing") & (tr.sessions >= 3)]
    if cand.empty:
        cand = tr[(tr.asset == "TSLA") & (tr.exit_reason == "TP")]
    t = cand.sort_values("net_pnl", ascending=False).iloc[0]
    px = load_prices("TSLA", cfg)
    w = px.loc[t.entry_date - pd.Timedelta(days=8): t.exit_date + pd.Timedelta(days=5)]
    p = cfg["assets"]["TSLA"]
    ent = t.entry_px
    fc = go.Figure(go.Candlestick(x=w.index, open=w.Open, high=w.High, low=w.Low, close=w.Close, name="TSLA (USD)",
                                  increasing_line_color=GAIN, decreasing_line_color=LOSS, showlegend=False))
    for y, lab, col, dash in ((ent, "Entrada", NEUTRAL, "solid"), (ent * (1 + p["tp"]), "Take profit +8 %", GAIN, "dash"),
                              (ent * (1 - p["sl"]), "Stop loss −4 %", LOSS, "dash"),
                              (ent * (1 + p["trail_trigger"]), "Gatillo del trailing +5,5 %", "#eda100", "dot"),
                              (ent * (1 + p["trail_sl"]), "Stop sube a +1,5 %", "#e87ba4", "dot")):
        fc.add_hline(y=y, line_color=col, line_dash=dash, line_width=2, annotation_text=f"{lab}: {fmt.num(y, 2)}", annotation_position="top left", annotation_bgcolor="rgba(128,128,128,0.18)",
                     annotation_font_size=11)
    fc.add_vrect(x0=t.entry_date, x1=t.exit_date, fillcolor="rgba(42,120,214,0.10)", line_width=0)
    fc.update_layout(xaxis_rangeslider_visible=False, yaxis_title="Precio (USD)")
    fc.update_xaxes(tickformat="%d/%m", rangebreaks=[dict(bounds=["sat", "mon"])])
    ctx["trade_html"] = html(base_layout(fc, 460, showlegend=False), "trade")
    ctx["trade"] = dict(t=t, ent=ent, fx_ent=t.fx_entry, fx_exit=t.fx_exit, exit_reason=REASON.get(t.exit_reason, t.exit_reason), tp=ent * 1.08, sl=ent * 0.96)

    # ---------- 8. resultados generales ----------
    eq = {m: pd.read_parquet(f3 / f"equity_{m}.parquet") for m in MODES}
    bench = pd.read_parquet(f3 / "benchmarks_continuo.parquet")
    cap = cfg["capital"]
    fe = go.Figure()
    for m in MODES:
        fe.add_trace(go.Scatter(x=eq[m].index, y=eq[m]["equity"] / cap * 100, name=f"Sistema · {MODE_NAME[m]}", line=dict(color=MODE[m], width=2),
                                hovertemplate="%{x|%d/%m/%Y}<br>$ %{y:,.1f} por cada $ 100<extra>" + MODE_NAME[m] + "</extra>"))
    for c_, lab, col, dash in (("bh_TSLA", "Comprar y mantener TSLA", ASSET["TSLA"], "dash"), ("bh_NVDA", "Comprar y mantener NVDA", ASSET["NVDA"], "dash"),
                               ("bh_ECOPETROL", "Comprar y mantener ECOPETROL", ASSET["ECOPETROL"], "dash"), ("bh_tercios", "1/3 en cada activo", "#52514e", "dot"),
                               ("efectivo", "Efectivo", "#c3c2b7", "solid")):
        fe.add_trace(go.Scatter(x=bench.index, y=bench[c_] / cap * 100, name=lab, line=dict(color=col, width=2, dash=dash),
                                hovertemplate="%{x|%d/%m/%Y}<br>$ %{y:,.1f} por cada $ 100<extra>" + lab + "</extra>"))
    fe.update_yaxes(type="log", title="Valor de $ 100 invertidos (log)")
    fe.update_xaxes(tickformat="%Y")
    ctx["equity_html"] = html(base_layout(fe, 460), "equity")
    fd = go.Figure()
    for m in MODES:
        e = eq[m]["equity"]
        fd.add_trace(go.Scatter(x=e.index, y=(e / e.cummax() - 1) * 100, name=MODE_NAME[m], line=dict(color=MODE[m], width=2),
                                hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} %<extra>" + MODE_NAME[m] + "</extra>"))
    b3 = bench["bh_tercios"]
    fd.add_trace(go.Scatter(x=b3.index, y=(b3 / b3.cummax() - 1) * 100, name="1/3 en cada activo", line=dict(color="#52514e", width=2, dash="dot")))
    fd.update_yaxes(title="Caída desde el máximo (%)")
    fd.update_xaxes(tickformat="%Y")
    ctx["dd_html"] = html(base_layout(fd, 360), "dd")
    ctx["bench_m"] = bench_m

    # ---------- 9. por setup y activo ----------
    b = brk[brk.modo == "blindaje"].copy()
    b["etq"] = "Setup " + b.setup + " · " + b.asset
    b = b.sort_values("pnl_neto_mm")
    fbar = go.Figure(go.Bar(y=b.etq, x=b.pnl_neto_mm, orientation="h", marker_color=[GAIN if v > 0 else LOSS for v in b.pnl_neto_mm],
                            hovertemplate="%{y}<br>P&L neto: $ %{x:,.1f} M<extra></extra>"))
    fbar.update_xaxes(title="P&L neto (millones de COP)")
    ctx["setup_html"] = html(base_layout(fbar, 380, margin=dict(l=105, r=15, t=20, b=55)), "setup")
    ctx["setup_rows"] = [dict(setup=r.setup, asset=r.asset, n=int(r.n), pnl=r.pnl_neto_mm, win=r.acierto, bruto=r.bruto_pct, costo=r.costo_pct, neto=r.neto_pct)
                         for r in brk[brk.modo == "blindaje"].itertuples()]

    # ---------- 10. mejoras ----------
    fh = make_subplots(rows=2, cols=3, subplot_titles=[f"{a} · {tr_}" for tr_ in ("entrenam.", "validación") for a in ("ECOPETROL", "NVDA", "TSLA")],
                       horizontal_spacing=0.07, vertical_spacing=0.14)
    allv = []
    for j, a in enumerate(("ECOPETROL", "NVDA", "TSLA"), start=1):
        e1 = pd.read_csv(f4 / f"e1_tp_sl_{a}.csv")
        for i, col in enumerate(("exp_is", "exp_oos"), start=1):
            piv = e1.pivot(index="tp", columns="sl", values=col) * 100
            allv.append(piv.values)
            fh.add_trace(go.Heatmap(z=piv.values, x=[f"{c * 100:g} %" for c in piv.columns], y=[f"{r * 100:g} %" for r in piv.index],
                                    zmid=0, zmin=-2, zmax=2, colorscale=[[0, LOSS], [0.5, "#f0efec"], [1, GAIN]], showscale=(i == 1 and j == 3),
                                    colorbar=dict(title="Expectancy<br>por trade (%)", len=0.9), texttemplate="%{z:.1f}", textfont=dict(size=9),
                                    hovertemplate=a + "<br>TP %{y} · SL %{x}<br>Expectancy neta %{z:.2f} %<extra></extra>"), row=i, col=j)
    fh.update_xaxes(title_text="Stop loss", row=2); fh.update_yaxes(title_text="Take profit", col=1)
    fh.update_layout(height=620, margin=dict(l=60, r=20, t=50, b=50), separators=",.", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     font=dict(color=NEUTRAL, size=12))
    ctx["heat_html"] = html(fh, "heat")
    ctx["pos_cells_oos"] = {a: float((pd.read_csv(f4 / f"e1_tp_sl_{a}.csv").exp_oos > 0).mean()) for a in ("ECOPETROL", "NVDA", "TSLA")}
    cmp_ = avd[avd.periodo == "completo"].copy()
    ctx["cmp_rows"] = [dict(modo=MODE_NAME[m], o=cmp_[(cmp_.version == "original") & (cmp_.modo == m)].iloc[0], n=cmp_[(cmp_.version == "mejorada") & (cmp_.modo == m)].iloc[0],
                            ooso=avd[(avd.version == "original") & (avd.modo == m) & (avd.periodo == "fuera (>=2022)")].iloc[0],
                            oosn=avd[(avd.version == "mejorada") & (avd.modo == m) & (avd.periodo == "fuera (>=2022)")].iloc[0]) for m in MODES]
    ctx["accepted"] = dec[dec.aceptada == True]                                                            # noqa: E712
    ctx["rejected_n"] = int((dec.aceptada == False).sum())                                                  # noqa: E712
    ctx["wstats_mej"] = pd.read_csv(f4 / "ventanas_estadisticas_mejorada.csv").set_index("modo")
    ctx["wstats_orig"] = pd.read_csv(f3 / "ventanas_estadisticas.csv").set_index("modo")
    # rediseño: tabla resumen
    redes = []
    for name, lab in ((PLAN, "Plan final: la acción MGC más volátil (250 sesiones)"), ("TSLA fijo", "Comprar y mantener TSLA (con hindsight)"),
                      ("más volátil 20d (mgc)", "La más volátil (20 sesiones)")):
        r = vrow(name, "amplio", "test")
        redes.append(dict(lab=lab, ret=r.ret_medio, pctpos=r.pct_pos, p=r.p_final, top10=r.top10, camp=vrow(name, "amplio", "todo").campeon))
    nv = {tr_: conc[(conc.seleccion == "NVDA fijo (ref. hindsight)") & (conc.politica == "ninguna") & (conc.campo == "amplio") & (conc.tramo == tr_)].iloc[0]
          for tr_ in ("test", "todo")}
    redes.insert(2, dict(lab="Comprar y mantener NVDA (con hindsight)", ret=nv["test"].ret_medio, pctpos=nv["test"].pct_pos, p=nv["test"].p_final,
                         top10=nv["test"].top10, camp=nv["todo"].campeon))
    ctx["redes"] = redes
    ctx["matrix_effect"] = dict(con=tsla_with_matrix.campeon, sin=tsla_no_matrix.campeon)

    # ---------- 14. bitácora ----------
    recs = []
    for m in MODES:
        t_ = pd.read_csv(f3 / f"trades_{m}.csv", parse_dates=["entry_date", "exit_date"])
        for r in t_.itertuples():
            recs.append([MODE_NAME[m], r.asset, r.setup, r.entry_date.strftime("%Y-%m-%d"), r.exit_date.strftime("%Y-%m-%d"), REASON.get(r.exit_reason, r.exit_reason), int(r.sessions),
                         round(r.invested_cop), round(r.net_pnl), round(r.ret_net * 100, 2), r.entry_date.year])
    ctx["trades_json"] = json.dumps(recs, separators=(",", ":"))
    ctx["n_trades_total"] = len(recs)

    # ---------- varios ----------
    ctx["met"], ctx["fix"], ctx["o"] = met, fix, o
    ctx["contest_win"] = pd.read_csv(f3 / "ventanas_concurso_por_anio.csv")
    ctx["hoy"] = pd.Timestamp.today()
    return ctx
