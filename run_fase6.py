"""Fase 6: genera reports/informe_backtest.html (un solo archivo autocontenido, abre sin internet).
Uso: python run_fase6.py"""
from __future__ import annotations

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from report.data import MODE_NAME, build_context
from src.config import ROOT, load_config
from src.data_loader import load_prices
from src.plan import build_message, contest_status, rank_by_volatility


def main() -> None:
    cfg = load_config()
    ctx = build_context(cfg)
    out = cfg["paths"]["outputs_dir"]

    # ---- acción elegida hoy, con los datos en caché (cierres ya publicados) ----
    closes = pd.DataFrame({t: load_prices(t, cfg)["Close"] for t in cfg["universe"]["mgc"]}).dropna(how="all")
    ranking = rank_by_volatility(closes, cfg["plan"]["vol_window"], cfg["plan"]["exclude_from_ranking"])
    selected = ranking.index[0]
    empty = pd.DataFrame(columns=["ticker", "fecha_compra", "precio_compra_usd"])
    ctx["msg_before"] = build_message(status=contest_status(cfg, pd.Timestamp("2026-10-04")), ranking=ranking, selected=selected, positions=empty,
                                      prices={}, next_earnings={}, capital=cfg["capital"], percentil=None, cfg=cfg, bought_today=False)
    pos = pd.DataFrame([[selected, "2026-10-05", 400.0]], columns=empty.columns)
    ctx["msg_during"] = build_message(status=contest_status(cfg, pd.Timestamp("2026-10-19")), ranking=ranking, selected=selected, positions=pos,
                                      prices={selected: 368.0}, next_earnings={selected: pd.Timestamp("2026-10-21")}, capital=cfg["capital"],
                                      percentil=22, cfg=cfg, bought_today=False) + "\n(Ejemplo con precios ficticios: compra a 400 USD, hoy 368 USD.)"

    # ---- alias y cifras que usa la plantilla ----
    cells = [pd.read_csv(out / "fase4" / f"e1_tp_sl_{a}.csv") for a in ("ECOPETROL", "NVDA", "TSLA")]
    ctx.update(
        capital=cfg["capital"], v=ctx["verdict"], matrix=ctx["matrix_effect"], selected=selected, tr=ctx["trade"],
        n_cells=sum(len(c) for c in cells), cells_pos_oos=int(sum((c.exp_oos > 0).sum() for c in cells)),
        n_tested=sum(len(pd.read_csv(f)) for f in (out / "fase4").glob("e*.csv")),
        blacklist_names=", ".join(cfg["universe"]["blacklist"]),
        rob=dict(bavd_o=ctx["robust"]["bavd"].query("version=='original' and modo=='blindaje'").iloc[0],
                 bavd_m=ctx["robust"]["bavd"].query("version=='mejorada' and modo=='blindaje'").iloc[0],
                 mavd_o=ctx["robust"]["mavd"].query("version=='original' and modo=='blindaje'").iloc[0],
                 mavd_m=ctx["robust"]["mavd"].query("version=='mejorada' and modo=='blindaje'").iloc[0]),
        mode_name=MODE_NAME)
    env = Environment(loader=FileSystemLoader(ROOT / "report" / "templates"), undefined=StrictUndefined, autoescape=False)
    env.filters["safe"] = lambda x: x
    html = env.get_template("informe.html.j2").render(**ctx)
    dest = cfg["paths"]["reports_dir"] / "informe_backtest.html"
    dest.write_text(html, encoding="utf-8")
    print(f"✅ {dest}  ({dest.stat().st_size / 1e6:.1f} MB)  · acción elegida hoy: {selected}")


if __name__ == "__main__":
    main()
