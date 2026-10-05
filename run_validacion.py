"""Fase 3 · Validación del semáforo y de la Regla Maestra.

Uso:  python run_validacion.py estudio     # estudio de eventos 2015–2026 → outputs/validacion/*.csv
      python run_validacion.py noticias    # cruce con la historia de noticias de Finnhub (≈ 1 año) → ¿qué tan bueno es el proxy?
      python run_validacion.py liga        # simulador de ligas con las reglas oficiales
      python run_validacion.py informe     # arma reports/validacion_semaforo.html con lo anterior
      python run_validacion.py todo"""
from __future__ import annotations

import sys
import warnings

import pandas as pd

from src import validacion as V
from src.config import load_config
from src.data_loader import load_macro, load_prices

warnings.filterwarnings("ignore")


def panel_estudio(cfg: dict) -> pd.DataFrame:
    ndx = load_macro("ndx", cfg)
    v = cfg["validacion"]
    partes = []
    for t in cfg["universe"]["mgc"]:
        px = load_prices(t, cfg)
        partes.append(V.estudio_activo(t, px, ndx, cfg, v["inicio"], v["negro_sentimiento"]))
    return pd.concat(partes)


def grupos(cfg: dict) -> dict[str, list[str]]:
    g = cfg["validacion"]["grandes"]
    return {"grandes": g, "resto": [t for t in cfg["universe"]["mgc"] if t not in g], "todas": list(cfg["universe"]["mgc"]),
            "grandes_sin_TSLA_NVDA": [t for t in g if t not in ("TSLA", "NVDA")], "TSLA": ["TSLA"]}


# Variantes del supuesto con el que se reconstruye la "noticia negativa" y el sentimiento del día 2 (la historia no tiene noticias): la conclusión
# sobre NEGRO debe sostenerse en todas, no sólo en la base.
VARIANTES_NEGRO = {"base": ("siempre", "proxy"), "volumen_dia2": ("volumen_dia2", "proxy"), "amplio": ("siempre", "amplio"),
                   "amplio_volumen_dia2": ("volumen_dia2", "amplio")}


def robustez_negro(cfg: dict) -> pd.DataFrame:
    """ROJO y NEGRO de las acciones grandes bajo cada variante del supuesto de noticias/sentimiento (tabla larga)."""
    ndx = load_macro("ndx", cfg)
    v = cfg["validacion"]
    filas = []
    for nombre, (negro_s, modo) in VARIANTES_NEGRO.items():
        panel = pd.concat([V.estudio_activo(t, load_prices(t, cfg), ndx, cfg, v["inicio"], negro_s, modo) for t in v["grandes"]])
        t = V.tabla_eventos(panel, cfg, {"grandes": v["grandes"]})
        t = t[(t.clase.isin(["ROJO", "NEGRO"])) & (t.medida == "AR") & (t.h.isin([5, 10, 20]))]
        filas.append(t.assign(variante=nombre))
    return pd.concat(filas)


def estudio(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = cfg["paths"]["outputs_dir"] / "validacion"
    out.mkdir(parents=True, exist_ok=True)
    panel = panel_estudio(cfg)
    tabla = V.tabla_eventos(panel, cfg, grupos(cfg))
    panel.to_parquet(out / "panel_eventos.parquet")
    tabla.to_csv(out / "tabla_eventos.csv", index=False)
    robustez_negro(cfg).to_csv(out / "robustez_negro.csv", index=False)
    return panel, tabla


def mostrar(tabla: pd.DataFrame, cfg: dict) -> None:
    pd.set_option("display.width", 220, "display.max_columns", 30)
    for g in ("grandes", "resto"):
        t = tabla[(tabla.grupo == g) & (tabla.medida == "AR")].copy()
        t["media%"] = (t.media * 100).round(2)
        t["IC95%"] = [f"[{a * 100:+.2f}, {b * 100:+.2f}]" for a, b in zip(t.ic_lo, t.ic_hi)]
        t["salir_neto%"] = (t.salir_neto * 100).round(2)
        t["pos%"] = (t.pct_pos * 100).round(0)
        print(f"\n===== {g.upper()} · retorno residual posterior (AR) =====")
        print(t[["clase", "h", "n", "n_acciones", "media%", "IC95%", "p", "pos%", "salir_neto%"]].to_string(index=False))
    print("\nCriterio NEGRO (grandes):", V.criterio_negro(tabla, cfg))


def noticias(cfg: dict) -> dict:
    """Descarga (reanudable, con caché) la historia de noticias de las acciones grandes y la cruza con el proxy."""
    import datetime as dt
    import os

    from dotenv import load_dotenv

    from src import validacion_noticias as N
    from src.config import ROOT
    from src.data_sources import FuentesDatos
    from src.sentimiento import crear_puntuador

    load_dotenv(ROOT / ".env")
    if not os.getenv("FINNHUB_API_KEY"):
        raise SystemExit("Falta FINNHUB_API_KEY en .env: sin ella no hay historia de noticias.")
    v = cfg["validacion"]
    out = cfg["paths"]["outputs_dir"] / "validacion"
    out.mkdir(parents=True, exist_ok=True)
    f = FuentesDatos(cfg)
    puntuador, motor = crear_puntuador(cfg)
    ndx = load_macro("ndx", cfg)
    hasta = pd.Timestamp(load_prices("TSLA", cfg).index[-1]).date()
    desde = dt.date.fromisoformat(v["noticias"]["desde"])
    carpeta = ROOT / v["noticias"]["carpeta"]
    filas, resumen, primera = [], {}, {}
    for t in v["grandes"]:
        hist = N.descargar_historia(f, t, desde, hasta, carpeta)
        primera[t] = str(hist["ts"].min().date()) if len(hist) else None
        px = load_prices(t, cfg)
        m = V.metricas_vectorizadas(px, ndx, cfg)
        ini = (hist["ts"].min().normalize().tz_localize(None) + pd.Timedelta(days=v["noticias"]["margen_dias"])) if len(hist) else None
        if ini is None:
            continue
        fechas = m.loc[ini:].index
        d = N.analisis_diario(hist, fechas, cfg, puntuador)
        c = N.cruce(m, d, cfg)
        c["ticker"] = t
        filas.append(c)
        print(f"  {t}: {len(hist):,} titulares · primera noticia {primera[t]} · {len(fechas)} sesiones", flush=True)
    todo = pd.concat(filas)
    resumen = N.resumen_cruce(todo)
    resumen.update(motor=motor, primera_noticia=min(x for x in primera.values() if x), ultima_sesion=str(hasta), tickers=v["grandes"])
    todo.to_parquet(out / "cruce_noticias.parquet")
    import json
    (out / "cruce_noticias.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(resumen, indent=1, ensure_ascii=False))
    return resumen


def panel_liga(cfg: dict):
    """Panel de precios en COP de todo el universo permitido + semáforo de las acciones MGC, alineado al calendario de EE. UU. desde mediados de 2014
    (con margen para σ20 y la beta de 60 días)."""
    from src.league import build_cop_prices
    from src.liga_v5 import construir_panel

    v = cfg["validacion"]
    ndx = load_macro("ndx", cfg)
    fechas = ndx.index[ndx.index >= pd.Timestamp("2014-06-02")]
    O, C = build_cop_prices(cfg, fechas)
    precios = {t: load_prices(t, cfg) for t in cfg["universe"]["mgc"]}
    estudios = {t: V.estudio_activo(t, precios[t], ndx, cfg, "2014-06-02", v["negro_sentimiento"]) for t in precios}
    return construir_panel(cfg, fechas, O, C, estudios, precios)


def liga(cfg: dict, rapido: bool = False) -> pd.DataFrame:
    """Corre todos los escenarios de rivales (con 'campo' como base del top X %) y el escenario base también con 'supervivientes'. Guarda las ventanas y el
    resumen en outputs/validacion/."""
    import time

    from src import liga_v5 as L

    v = cfg["validacion"]["liga"]
    out = cfg["paths"]["outputs_dir"] / "validacion"
    out.mkdir(parents=True, exist_ok=True)
    p = panel_liga(cfg)
    paso = v["paso_rapido"] if rapido else 1
    nr, t0 = v["participantes_reales"], time.time()
    ventanas, resumenes = [], []
    corridas = [(i, e, v["base_pct"]) for i, e in enumerate(v["escenarios"])]
    alt = "supervivientes" if v["base_pct"] == "campo" else "campo"
    corridas.append((v["escenario_base"], v["escenarios"][v["escenario_base"]], alt))                   # sensibilidad a cómo se mide "top X %"
    for i, e, base_pct in corridas:
        df = L.correr_liga(p, cfg, v["desde"], "TSLA", v["rivales"], e["concentrados"], v["semilla"], base_pct, paso, peso_conc_min=e["peso_min"])
        df = df.assign(escenario=i, nombre=e["nombre"], base_pct=base_pct)
        ventanas.append(df)
        resumenes.append(L.resumen_liga(df, cfg, nr).assign(escenario=i, nombre=e["nombre"], base_pct=base_pct))
        print(f"  [{i}] {e['nombre']} · top X % contra {base_pct}: {df.inicio.nunique()} ventanas · {time.time() - t0:.0f} s", flush=True)
    suf = "_rapido" if rapido else ""
    todas, resumen = pd.concat(ventanas), pd.concat(resumenes)
    todas.to_parquet(out / f"liga_ventanas{suf}.parquet")
    resumen.to_csv(out / f"liga_resumen{suf}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 48)
    cols = ["escenario", "base_pct", "estrategia", "ventanas", "pasa1", "pasa2", "pasa3", "pasa4", f"p1_{nr[1]}", f"top30_{nr[1]}", "ret_mediana", "pct_con_cambio"]
    print(resumen[cols].round(4).to_string(index=False))
    return todas


def bases(cfg: dict, escenarios: tuple[int, ...] | None = None) -> None:
    """¿Y si la base fuera OTRA acción (cualquiera de EE. UU., la BVC o ETF)? Mantener cada una contra los mismos rivales de cada escenario
    (por omisión el escenario base y el campo disperso, para ver el efecto de que TSLA sea la acción 'llena de gente')."""
    from src import liga_v5 as L

    lg = cfg["validacion"]["liga"]
    p = panel_liga(cfg)
    out = cfg["paths"]["outputs_dir"] / "validacion"
    for i in escenarios if escenarios is not None else sorted({lg["escenario_base"], 0}):
        esc = lg["escenarios"][i]
        df = L.correr_bases(p, cfg, lg["desde"], list(p.tickers), lg["rivales"], esc["concentrados"], lg["semilla"], lg["base_pct"], 1, esc["peso_min"])
        r = L.resumen_bases(df, lg["participantes_reales"][1])
        r.to_csv(out / f"bases_resumen_esc{i}.csv", index=False)
        print(f"[escenario {i}] {esc['nombre']}")
        pd.set_option("display.width", 220)
        print(r.head(10).round(4).to_string(index=False), flush=True)


def informe(cfg: dict) -> None:
    """Arma reports/validacion_semaforo.html con lo que hay en outputs/validacion/ (corre antes los pasos estudio, noticias y liga)."""
    import datetime as dt
    import json

    from jinja2 import Environment, FileSystemLoader, StrictUndefined

    from report.validacion import construir_contexto
    from src.config import ROOT

    out = cfg["paths"]["outputs_dir"] / "validacion"
    tabla = pd.read_csv(out / "tabla_eventos.csv")
    panel = pd.read_parquet(out / "panel_eventos.parquet")
    rob = pd.read_csv(out / "robustez_negro.csv")
    nf = out / "cruce_noticias.json"
    noticias_res = json.loads(nf.read_text(encoding="utf-8")) if nf.exists() else None
    liga_res = pd.read_csv(out / "liga_resumen.csv")
    liga_vent = pd.read_parquet(out / "liga_ventanas.parquet")
    bases_res = {i: pd.read_csv(f) for i in range(len(cfg["validacion"]["liga"]["escenarios"])) if (f := out / f"bases_resumen_esc{i}.csv").exists()}
    ctx = construir_contexto(cfg, tabla, panel, rob, noticias_res, liga_res, liga_vent, dt.date.today(), bases_res)
    env = Environment(loader=FileSystemLoader(ROOT / "report" / "templates"), undefined=StrictUndefined, autoescape=False)
    html = env.get_template("validacion.html.j2").render(**ctx)
    dest = cfg["paths"]["reports_dir"] / "validacion_semaforo.html"
    dest.write_text(html, encoding="utf-8")
    print(f"✅ {dest}  ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    cfg = load_config()
    paso = sys.argv[1] if len(sys.argv) > 1 else "estudio"
    if paso in ("bases", "todo"):
        bases(cfg)
    if paso in ("informe", "todo"):
        informe(cfg)
    if paso in ("noticias", "todo"):
        noticias(cfg)
    if paso in ("liga", "todo"):
        liga(cfg, rapido="--rapido" in sys.argv)
    if paso in ("estudio", "todo"):
        panel, tabla = estudio(cfg)
        print(f"panel: {len(panel):,} filas · {panel.ticker.nunique()} acciones · {panel.index.min().date()} → {panel.index.max().date()}")
        mostrar(tabla, cfg)
