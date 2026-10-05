"""Estudio de eventos de las noticias de la BVC (ver src/estudio_noticias.py). Descarga la "Información relevante" histórica de cada emisor desde la
Superfinanciera y los precios diarios, mide la reacción y lo que vino después, y guarda la tabla que el bot usa en sus mensajes.

Uso:  python run_estudio_noticias.py            # usa lo ya descargado (data/cache/sfc_hist) y descarga lo que falte
      python run_estudio_noticias.py --refrescar  # vuelve a descargar todo
Salida: data/estudio_noticias_bvc.json (tabla) y outputs/estudio_noticias_bvc.csv (un renglón por anuncio, para revisar a mano)."""
from __future__ import annotations

import datetime as dt
import json
import sys
import time

import pandas as pd

from src import estudio_noticias as E
from src import noticias_bvc as N
from src.config import ROOT, load_config
from src.data_sources import FuentesDatos


def bajar_historia(cfg: dict, emisor: dict, desde: dt.date, hasta: dt.date, refrescar: bool) -> list[dict]:
    carpeta = ROOT / "data" / "cache" / "sfc_hist"
    carpeta.mkdir(parents=True, exist_ok=True)
    tipo, cod = emisor["sfc_id"]
    arch = carpeta / f"{tipo}_{cod}_{desde}_{hasta}.json"
    if arch.exists() and not refrescar:
        return json.loads(arch.read_text(encoding="utf-8"))
    import requests
    s = cfg["noticias_bvc"]["fuentes"]["sfc"]
    url = s["url"].replace("/general/pagina", "/entidad/pagina")
    filas, pag = [], 0
    while True:
        r = requests.get(url, params=dict(tipoEntidad=tipo, codigoEntidad=cod, page=pag, size=200, fechaDesde=f"{desde:%Y/%m/%d}", fechaHasta=f"{hasta:%Y/%m/%d}"),
                         headers={"User-Agent": cfg["noticias_bvc"]["agente"], "api-key": s["api_key"], "Accept": "application/json"}, timeout=90)
        r.raise_for_status()
        j = r.json()
        filas += j.get("content", [])
        pag += 1
        if pag >= j.get("totalPages", 1):
            break
        time.sleep(0.4)
    arch.write_text(json.dumps(filas, ensure_ascii=False), encoding="utf-8")
    return filas


def main() -> int:
    cfg = load_config()
    e = cfg["noticias_bvc"]["estudio"]
    desde, hasta = dt.date.fromisoformat(e["desde"]), dt.date.today()
    f = FuentesDatos(cfg)
    dias = int((hasta - desde).days * 0.72) + 60
    import yfinance as yf
    d = yf.download(cfg["macro"]["icolcap"], start=str(desde - dt.timedelta(days=60)), progress=False, auto_adjust=False)      # el mercado: ETF del índice COLCAP
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.index = pd.DatetimeIndex(d.index).tz_localize(None).normalize()
    ref = d["Close"].dropna()
    medidos, sin_precio = [], []
    for em in cfg["noticias_bvc"]["emisores"]:
        if not em.get("sfc_id"):
            continue
        try:
            crudo = bajar_historia(cfg, em, desde, hasta, "--refrescar" in sys.argv)
        except Exception as ex:                                                        # noqa: BLE001
            print(f"  {em['nombre']}: no se pudo descargar ({type(ex).__name__})")
            continue
        items = N.items_de_sfc(crudo)
        px, usado = None, None
        for t in em["tickers"]:                                                        # la más líquida con datos
            d = f.diario(t, dias)
            if d is not None and len(d) > 200 and (px is None or (d["Close"] * d["Volume"]).tail(60).mean() > (px["Close"] * px["Volume"]).tail(60).mean()):
                px, usado = d, t
        if px is None or (px["Volume"].tail(250) == 0).mean() > 0.3:
            sin_precio.append(em["nombre"])
            continue
        evs = E.eventos_de(items, em["nombre"], cfg)
        ok = [m for m in (E.medir({**ev, "ticker": usado}, px, ref, e["horizontes"]) for ev in evs) if m]
        medidos += ok
        print(f"  {em['nombre']:<32} {usado:<11} {len(items):4d} anuncios · {len(ok):4d} medidos · {sum(1 for m in ok if m['cat'] != 'otros'):4d} con tipo")
    dep = E.depurar(medidos, cfg["noticias_bvc"]["pesos"])
    t = E.tabla(dep, cfg)
    (ROOT / N.ESTUDIO).write_text(json.dumps(t, ensure_ascii=False, indent=1), encoding="utf-8")
    cfg["paths"]["outputs_dir"].mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dep).drop(columns=["ts"]).assign(ts=[m["ts"].isoformat() for m in dep]).to_csv(cfg["paths"]["outputs_dir"] / "estudio_noticias_bvc.csv", index=False, encoding="utf-8-sig")
    print(f"\n{len(dep)} anuncios (uno por emisor y día) desde {desde}. Sin precios fiables: {', '.join(sin_precio) or 'ninguno'}")
    hs = e["horizontes"]

    def fila(nombre: str, r: dict) -> str:
        if not r.get("n"):
            return f"{nombre:<26} n=0"
        partes = [f"{nombre:<26} n={r['n']:4d} |z| mediana {r['z_abs_mediana']:.2f} · fuerte(≥2) {r['pct_fuerte'] * 100:4.0f}%"]
        for h in hs:
            x = r.get(f"h{h}")
            if x:
                partes.append(f"h{h}: {x['media'] * 100:+.2f}% [{x['ic_lo'] * 100:+.2f},{x['ic_hi'] * 100:+.2f}] sube {x['pct_pos'] * 100:.0f}%")
        return " | ".join(partes)
    print("\nA) Comprando al CIERRE del día de reacción, según cómo reaccionó el precio ese día (retorno frente al mercado; IC95 entre corchetes)")
    for k, r in t["global"].items():
        print(fila("GLOBAL " + k, r))
    for k, r in t["categorias"].items():
        print(fila(k, r))
    print("\nB) Anuncios FUERA de horario, comprando en la APERTURA siguiente (h1 = hasta ese mismo cierre)")
    for k, r in t["apertura"].items():
        print(fila("APERTURA " + k, r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
