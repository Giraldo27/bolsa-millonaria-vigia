"""¿Sirve cambiarse a "la que va mejor"? Con las acciones líquidas de la BVC (3 años): la que más subió en k días frente al mercado, ¿sigue ganando los k siguientes?

Uso:  python run_estudio_relativo.py
Salida: data/estudio_relativo_bvc.json (lo citan los avisos de cambio). Ventanas sin solape; diferencia = tercio ganador − tercio perdedor; IC95 por remuestreo."""
from __future__ import annotations

import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

from src.cambios import ESTUDIO_REL
from src.config import ROOT, load_config
from src.construir import universo_candidatos
from src.data_loader import yahoo_symbol


def main() -> int:
    import yfinance as yf
    cfg = load_config()
    ref = cfg["macro"]["icolcap"]
    simbolos = {t: yahoo_symbol(t, cfg) for t in universo_candidatos(cfg)}
    d = yf.download(sorted(set(simbolos.values())) + [ref], period="3y", auto_adjust=True, progress=False)
    cierre, volumen = d["Close"], d["Volume"]
    liquidas = [s for s in set(simbolos.values()) if s in cierre and (cierre[s] * volumen[s]).tail(60).median() / 1e6 >= cfg["liquidez"]["buena_cop_mm"]]
    c = cierre[liquidas + [ref]]
    c = c[c.pct_change().abs() < 0.25].ffill(limit=2)                                  # ticks erróneos de Yahoo fuera
    r = np.log(c).diff()
    rel = r[liquidas].sub(r[ref], axis=0)
    rng = np.random.default_rng(2026)
    filas = []
    for k in (1, 3, 5, 10):
        pas, fut = rel.rolling(k).sum(), rel.rolling(k).sum().shift(-k)
        x = []
        for i in range(k + 20, len(c) - k, k):
            p, f = pas.iloc[i].dropna(), fut.iloc[i].dropna()
            com = p.index.intersection(f.index)
            if len(com) < 9:
                continue
            o = p[com].sort_values()
            n = max(len(com) // 3, 3)
            x.append((f[o.index[-n:]].mean(), f[o.index[:n]].mean()))
        x = np.array(x)
        dif = x[:, 0] - x[:, 1]
        b = dif[rng.integers(0, len(dif), (4000, len(dif)))].mean(1)
        filas.append(dict(k=k, n=int(len(dif)), dif=float(dif.mean()), ic_lo=float(np.percentile(b, 2.5)), ic_hi=float(np.percentile(b, 97.5)),
                          ganadoras=float(x[:, 0].mean()), rezagadas=float(x[:, 1].mean())))
        print(f"k={k:2d} n={len(dif):3d} | ganadoras − rezagadas en los {k} días siguientes: {dif.mean() * 100:+.2f}% [{filas[-1]['ic_lo'] * 100:+.2f}, {filas[-1]['ic_hi'] * 100:+.2f}] "
              f"| ganadoras {x[:, 0].mean() * 100:+.2f}% · rezagadas {x[:, 1].mean() * 100:+.2f}% (frente al mercado)")
    (ROOT / ESTUDIO_REL).write_text(json.dumps(dict(hecho=dt.date.today().isoformat(), acciones=len(liquidas), filas=filas), ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(liquidas)} acciones líquidas · guardado en {ESTUDIO_REL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
