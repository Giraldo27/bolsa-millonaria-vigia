"""Verificación de cobertura de datos (Fase 0). Solo lectura: no escribe en data/."""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

MGC = "AAPL AMZN BAC C DIS F GE GOOGL JNJ JPM KO MA META MSFT NUCO NVDA PBR PFE TSLA UBER V WMT".split()
ETF = "CSPX EQQQ IB01 ICLN SUWS".split()
LOCAL = ("AVAL BCOLOMBIA BHI BOGOTA BVC CELSIA CEMARGOS CNEC CORFICOLCF ECOPETROL ENKA EXITO GEB GRUPOARGOS "
         "GRUPOSURA HCOLSEL ISA MINEROS NUTRESA PEI PFAVAL PFBCOLOM PFCEMARGOS PFCORFICOL PFGRUPOARG PFSURA "
         "PROMIGAS TERPEL").split()
NEGRA = "CONCONCRET ELCONDOR ETB FABRICATO OCCIDENTE POPULAR VILLAS".split()

# candidatos de símbolo Yahoo por ticker (se prueba en orden y se queda el de mayor historia)
EXTRA = {
    "NUCO": ["NUCO", "NU", "NUE"],
    "CSPX": ["CSPX.L", "CSPX.AS", "SXR8.DE"],
    "EQQQ": ["EQQQ.L", "EQQQ.DE", "EQQQ.MI"],
    "IB01": ["IB01.L", "IB01.AS"],
    "ICLN": ["ICLN", "INRG.L"],
    "SUWS": ["SUWS.L", "SUWS.SW"],
    "PFGRUPOARG": ["PFGRUPOARG.CL"],
    "PFSURA": ["PFGRUPSURA.CL", "PFSURA.CL"],
    "GRUPOSURA": ["GRUPOSURA.CL"],
    "ECOPETROL": ["ECOPETROL.CL", "EC"],
}
MACRO = {"BRENT": "BZ=F", "USDCOP": "COP=X", "NDX": "^NDX", "ICOLCAP": "ICOLCAP.CL", "COLCAP": "^COLCAP",
         "ADR_EC": "EC"}


def candidatos(t: str) -> list[str]:
    if t in EXTRA:
        return EXTRA[t]
    if t in LOCAL or t in NEGRA:
        return [f"{t}.CL"]
    return [t]


def bajar(sym: str) -> pd.DataFrame:
    try:
        df = yf.download(sym, start="2005-01-01", auto_adjust=False, progress=False, threads=False)
    except Exception:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df.dropna(how="all")


def resumen(grupo: str, t: str, sym: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return dict(grupo=grupo, ticker=t, simbolo=sym, ok=False)
    c = df["Close"].dropna()
    idx = c.index
    gaps = idx.to_series().diff().dt.days
    vol0 = (df["Volume"].fillna(0) == 0).mean() * 100
    desde2015 = c[c.index >= "2015-01-01"]
    return dict(grupo=grupo, ticker=t, simbolo=sym, ok=True, desde=idx[0].date(), hasta=idx[-1].date(),
                filas=len(c), filas_2015=len(desde2015), gap_max_dias=int(gaps.max()),
                gaps_gt7=int((gaps > 7).sum()), pct_vol0=round(vol0, 1))


def main() -> None:
    filas, cache = [], {}
    items = ([("MGC", t) for t in MGC] + [("ETF", t) for t in ETF] + [("LOCAL", t) for t in LOCAL]
             + [("NEGRA", t) for t in NEGRA])
    for g, t in items:
        mejor = None
        for sym in candidatos(t):
            df = bajar(sym)
            r = resumen(g, t, sym, df)
            if r["ok"] and (mejor is None or r["filas"] > mejor[0]["filas"]):
                mejor = (r, df)
            if r["ok"] and r["filas"] > 1500:
                break
        if mejor is None:
            filas.append(dict(grupo=g, ticker=t, simbolo=",".join(candidatos(t)), ok=False))
        else:
            filas.append(mejor[0]); cache[t] = mejor[1]
        print(filas[-1], flush=True)
    for k, sym in MACRO.items():
        df = bajar(sym)
        r = resumen("MACRO", k, sym, df)
        filas.append(r); cache[k] = df
        print(r, flush=True)
    out = pd.DataFrame(filas)
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(out.to_string(index=False))
    out.to_csv(Path(__file__).parent / "data" / "cobertura_inicial.csv", index=False)
    pd.to_pickle(cache, Path(__file__).parent / "data" / "_tmp_cache.pkl")


if __name__ == "__main__":
    main()
