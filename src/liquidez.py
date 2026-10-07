"""Liquidez REAL en trii: cuánto se negocia cada acción en la Bolsa de Colombia, que es donde trii ejecuta tus órdenes.

El error que corrige (5-oct-2026): las acciones de EE. UU. que muestra trii (TSLACO, NVDACO, METACO…) son las del Mercado Global Colombiano y se negocian
MUY poco aquí (TSLACO movió 42 acciones en todo un día), aunque en Nueva York muevan miles de millones. Medir la liquidez con el volumen de Nueva York
decía "muy líquida" de algo que en trii casi no tiene compradores: cada entrada y salida cuesta caro. Aquí se mide con el libro local (`<TICKER>CO.CL`).

Niveles: buena (se puede entrar y salir sin mover el precio) · justa (sólo con orden límite y paciencia) · mala (evítala: salir puede costar mucho) ·
sin_dato (Yahoo no trae un historial fiable de esa acción en Colombia: se dice, no se inventa)."""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd

from .config import group_of
from .data_loader import yahoo_symbol

BUENA, JUSTA, MALA, SIN_DATO = "buena", "justa", "mala", "sin_dato"


def simbolo_trii(ticker: str, cfg: dict[str, Any]) -> str | None:
    """Símbolo de Yahoo del libro donde trii negocia la acción: `<T>.CL` para las locales y `<T>CO.CL` para las globales (NUCO ya trae el CO)."""
    try:
        g = group_of(ticker, cfg)
    except KeyError:
        return None
    if g == "local":
        return yahoo_symbol(ticker, cfg)
    if g == "mgc":
        return (ticker if ticker.endswith("CO") else ticker + "CO") + ".CL"
    return None


def clasificar(valor_diario: pd.Series, volumen: pd.Series, cfg: dict[str, Any]) -> dict[str, Any]:
    """Nivel de liquidez a partir del valor negociado diario (pesos) de las últimas sesiones. Usa la MEDIANA (un día de bloque grande no engaña) y cuenta los
    días sin negociación. Con historial lleno de ceros no se concluye "mala": Yahoo no lo trae completo (pasa con NUCO) y se reporta sin_dato."""
    c = cfg["liquidez"]
    v, vol = valor_diario.tail(c["dias"]), volumen.tail(c["dias"])
    if len(v) < c["dias"] // 2:
        return dict(nivel=SIN_DATO, mediana_mm=None, hoy_mm=None, acciones_hoy=None, dias_sin_negociar=None, dias=len(v), motivos=["muy pocos días de historia"])
    sin = int((vol <= 0).sum())
    mediana, hoy = float(v.median()) / 1e6, float(v.iloc[-1]) / 1e6
    v60 = valor_diario.tail(60)
    mediana60 = float(v60.median()) / 1e6 if len(v60) >= 40 else None
    flojo = float(v.quantile(0.25)) / 1e6
    base = dict(mediana_mm=mediana, mediana60_mm=mediana60, dia_flojo_mm=flojo, hoy_mm=hoy, acciones_hoy=float(vol.iloc[-1]), dias_sin_negociar=sin, dias=len(v))
    if sin > len(v) // 2:
        # Más de la mitad de los días "sin volumen": o no se negocia, o Yahoo no trae el volumen de esta acción (le pasa con NUCO, que es de las más negociadas).
        # Con estos datos no se puede saber cuál de las dos: se dice "sin dato" (y sin dato NUNCA se recomienda), no "mala".
        return dict(base, nivel=SIN_DATO, mediana_mm=None, motivos=["Yahoo no trae un volumen fiable de esta acción en Colombia"])
    motivos = []
    if mediana < c["buena_cop_mm"]:
        motivos.append(f"negocia unos $ {mediana:,.0f} millones al día (pido {c['buena_cop_mm']:,})".replace(",", "."))
    if mediana60 is not None and mediana60 < c["buena_60_cop_mm"]:
        motivos.append((f"en 3 meses su día normal es de $ {mediana60:,.0f} millones (pido {c['buena_60_cop_mm']:,})".replace(",", "."))
                       + (": la liquidez de ahora puede no durar" if mediana >= c["buena_cop_mm"] else ""))
    if flojo < c["buena_dia_flojo_cop_mm"]:
        motivos.append(f"en sus días flojos baja a $ {flojo:,.0f} millones (pido {c['buena_dia_flojo_cop_mm']:,})".replace(",", "."))
    if sin > c["max_dias_sin_negociar"]:
        motivos.append(f"{sin} de los últimos {len(v)} días no se negoció")
    if not motivos:
        return dict(base, nivel=BUENA, motivos=[])
    return dict(base, nivel=JUSTA if (mediana >= c["justa_cop_mm"] and sin <= 1) else MALA, motivos=motivos)      # "justa" tolera un día sin negociar; "buena", ninguno


def fila_libro(f: Any, ticker: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """La fila de esta acción en el libro de la Bolsa de Colombia (`f.libro_bvc()`), o None. Las globales aparecen con o sin el "CO" final (NU, TSLACO, META)."""
    sym = simbolo_trii(ticker, cfg)
    leer = getattr(f, "libro_bvc", None)
    if sym is None or leer is None:
        return None
    try:
        tabla = leer()
    except Exception:                                                                  # noqa: BLE001
        return None
    if not tabla:
        return None
    s = sym[:-3] if sym.endswith(".CL") else sym
    for cand in dict.fromkeys([s, ticker, s[:-2] if s.endswith("CO") else s, ticker[:-2] if ticker.endswith("CO") else ticker]):
        if cand in tabla:
            return tabla[cand]
    return None


def en_sesion(ahora: dt.datetime | None = None) -> bool:
    """¿Está abierta la bolsa? La de Colombia sigue el horario de Nueva York (lun–vie, 9:30–16:00 de allá: 8:30–15:00 o 9:30–16:00 de Bogotá según la época).
    Mientras lo está, "lo negociado hoy" va por la mitad y no se compara con un día completo."""
    a = pd.Timestamp(ahora or dt.datetime.now(dt.timezone.utc)).tz_convert("America/New_York")
    return a.weekday() < 5 and 570 <= a.hour * 60 + a.minute < 965


def cruzar(l: dict[str, Any], fila: dict[str, Any] | None, cfg: dict[str, Any], ahora: dt.datetime | None = None) -> dict[str, Any]:
    """Cruza la medición con el libro de la Bolsa de Colombia. Dos usos:
      · si la otra fuente no trae volumen (NUCO), se mide con el libro; como sólo trae PROMEDIOS (un día de bloque grande los infla), se exige que el menor
        promedio pase el mínimo, que el volumen no esté concentrado en pocos días y que la última sesión no haya sido floja;
      · si la otra fuente dice "buena" y el libro dice claramente menos, gana la más prudente.
    Nunca sube de nivel algo que ya se midió como justo o malo."""
    if not fila:
        return l
    c = cfg["liquidez"]
    proms = [fila[k] for k in ("prom10_mm", "prom30_mm", "prom60_mm") if fila.get(k)]
    if not proms:
        return l
    menor = min(proms)
    mil = lambda v: f"{v:,.0f}".replace(",", ".")                                       # noqa: E731
    l = dict(l, libro=fila)
    if l["nivel"] == SIN_DATO:
        motivos = []
        if menor < c["buena_cop_mm"]:
            motivos.append(f"en promedio negocia unos $ {mil(menor)} millones al día (pido {mil(c['buena_cop_mm'])})")
        p10, p60 = fila.get("prom10_mm"), fila.get("prom60_mm")
        # "Volumen de pocos días" sólo descalifica si el nivel de 3 meses NO alcanza por sí solo. Si en 3 meses ya promedia más de lo que se pide (NUCO:
        # $ 7.400 millones al día) que además negocie el triple estas semanas es MÁS liquidez, no menos (corregido el 7-oct-2026: el filtro la rechazaba).
        if p10 and p60 and p60 < c["buena_cop_mm"] and p10 > c["libro_bvc"]["concentrado_veces"] * p60:
            veces = f"{p10 / p60:.1f}".replace(".", ",")
            motivos.append(f"en las últimas 2 semanas negoció {veces} veces lo de los últimos 3 meses: es un volumen de pocos días que puede no durar")
        if not en_sesion(ahora) and fila["valor_ult_mm"] < c["buena_dia_flojo_cop_mm"]:
            motivos.append(f"en su última sesión negoció sólo $ {mil(fila['valor_ult_mm'])} millones (pido {mil(c['buena_dia_flojo_cop_mm'])})")
        nivel = BUENA if not motivos else (JUSTA if menor >= c["justa_cop_mm"] else MALA)
        return dict(l, nivel=nivel, mediana_mm=menor, mediana60_mm=fila.get("prom60_mm"), hoy_mm=fila["valor_ult_mm"], acciones_hoy=fila["acciones_ult"],
                    motivos=motivos, fuente="bvc")
    if l["nivel"] == BUENA and menor < c["buena_60_cop_mm"]:
        return dict(l, nivel=JUSTA, motivos=[f"la Bolsa de Colombia reporta en promedio sólo $ {mil(menor)} millones al día (pido {mil(c['buena_60_cop_mm'])}): "
                                             "mis dos fuentes no coinciden y me quedo con la más prudente"])
    return l


def resumen_continuidad(velas: pd.DataFrame, hoy: dt.date | None = None) -> dict[str, Any] | None:
    """De las velas de 5 minutos (sólo existen o traen volumen cuando hubo operaciones): en qué parte del día se negocia la acción.
    {mediana, flojo (percentil 25), pausa (minutos, mediana de la pausa más larga del día), dias}. La sesión tiene 78 tramos de 5 minutos."""
    if velas is None or len(velas) == 0:
        return None
    idx = pd.DatetimeIndex(velas.index)
    idx = idx.tz_localize("UTC") if idx.tz is None else idx
    idx = idx.tz_convert("America/New_York")
    minuto = idx.hour * 60 + idx.minute
    ok = (minuto >= 570) & (minuto < 960) & (velas["Volume"].fillna(0).to_numpy() > 0)
    s = pd.Series(minuto[ok], index=idx[ok].date)
    fraccion, pausas = [], []
    for dia, m in s.groupby(level=0):
        if hoy is not None and dia >= hoy:                                             # el día en curso va por la mitad: no cuenta
            continue
        puntos = [570, *sorted(m.to_numpy()), 960]
        fraccion.append(len(m) / 78)
        pausas.append(max(b - a for a, b in zip(puntos, puntos[1:])))
    if not fraccion:
        return None
    fr = pd.Series(fraccion)
    return dict(mediana=float(fr.median()), flojo=float(fr.quantile(0.25)), pausa=float(pd.Series(pausas).median()), dias=len(fr))


def continuidad(f: Any, ticker: str, cfg: dict[str, Any]) -> dict[str, Any] | None:
    """¿La acción se negocia todo el día o a ratos? (velas de 5 minutos del último mes, caché de 4 horas). None si no se puede medir."""
    c = (cfg.get("liquidez") or {}).get("continuidad") or {}
    sym = simbolo_trii(ticker, cfg)
    if not c.get("activo") or sym is None:
        return None

    def bajar() -> pd.DataFrame:
        h = f.yf.Ticker(sym).history(period="1mo", interval="5m")
        if h is None or h.empty:
            raise RuntimeError("Yahoo devolvió vacío")
        return h[["Volume"]]
    try:
        h = f._con_cache(f"continuidad|{sym}", c.get("cache_min", 240), bajar, f"continuidad {ticker}")
        r = resumen_continuidad(h, dt.datetime.now(dt.timezone(dt.timedelta(hours=-5))).date())
        return r if r and r["dias"] >= c.get("dias_min", 10) else None
    except Exception:                                                                  # noqa: BLE001 — sin dato no se castiga ni se inventa
        return None


def exigir_continuidad(l: dict[str, Any], cont: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """Una acción puede mover mucha plata al día y aun así negociarse A RATOS (pausas largas sin ninguna operación): en trii eso se ve como "no hay con
    quién negociar". "Buena" exige además operaciones en buena parte del día; si no, baja a "justa". Sin dato de continuidad no se castiga."""
    if not cont:
        return l
    l = dict(l, continuidad=cont)
    c = cfg["liquidez"]["continuidad"]
    if l["nivel"] != BUENA or (cont["mediana"] >= c["min"] and cont["flojo"] >= c["min_dia_flojo"]):
        return l
    if cont["mediana"] < c["min"]:
        motivo = f"se negocia a ratos: sólo hay operaciones en el {cont['mediana']:.0%} del día (pido {c['min']:.0%})"
    else:
        motivo = f"en sus días flojos sólo hay operaciones en el {cont['flojo']:.0%} del día (pido {c['min_dia_flojo']:.0%})"
    return dict(l, nivel=JUSTA, a_ratos=True, motivos=[motivo + f", con pausas de unos {cont['pausa']:.0f} minutos sin que nadie compre ni venda"])


def afinar(f: Any, ticker: str, l: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """Las dos comprobaciones que se suman a la medición diaria: lo negociado en la Bolsa de Colombia y la continuidad dentro del día."""
    l = cruzar(dict(l, ticker=ticker), fila_libro(f, ticker, cfg), cfg)
    return exigir_continuidad(l, continuidad(f, ticker, cfg) if l["nivel"] == BUENA else None, cfg)


def medir(f: Any, ticker: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Liquidez de la acción en trii: medición diaria + lo negociado en la Bolsa de Colombia + continuidad dentro del día. Nunca lanza: sin datos, sin_dato."""
    return afinar(f, ticker, _medir_historia(f, ticker, cfg), cfg)


def _medir_historia(f: Any, ticker: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """La medición día por día (mediana, días flojos, días sin negociar), con caché de 1 hora."""
    sym = simbolo_trii(ticker, cfg)
    vacio = dict(ticker=ticker, simbolo=sym, nivel=SIN_DATO, mediana_mm=None, hoy_mm=None, acciones_hoy=None, dias_sin_negociar=None, dias=0)
    if sym is None:
        return vacio

    def bajar() -> pd.DataFrame:
        d = f.yf.download(sym, period="3mo", interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = d.columns.get_level_values(0)
        if d is None or d.empty:
            raise RuntimeError("Yahoo devolvió vacío")
        return d[["Close", "Volume"]].dropna(subset=["Close"])
    try:
        d = f._con_cache(f"liquidez|{sym}", 60, bajar, f"liquidez {ticker}")
        if d is None or len(d) == 0:
            return vacio
        return dict(vacio, **clasificar(d["Close"] * d["Volume"].fillna(0), d["Volume"].fillna(0), cfg))
    except Exception:                                                                  # noqa: BLE001 — sin datos no se inventa
        return vacio


def frase(l: dict[str, Any], orden_cop: float | None = None, cfg: dict[str, Any] | None = None) -> str:
    """La liquidez en palabras sencillas (texto plano; quien llama le pone el formato)."""
    t = l["ticker"]
    if l["nivel"] == SIN_DATO:
        return (f"💧 Liquidez de {t} en trii: NO LA PUEDO MEDIR (mi fuente no trae un volumen fiable de esta acción en Colombia). No significa que sea mala, "
                "pero tampoco puedo decir que sea buena: por eso no la recomiendo. Antes de operar mira en trii cuántas acciones se han negociado hoy, y usa orden límite.")
    med = f"$ {l['mediana_mm']:,.0f} millones".replace(",", ".")
    acc = f"{l['acciones_hoy']:,.0f}".replace(",", ".")
    parte = ""
    if orden_cop and l["mediana_mm"]:
        pct = orden_cop / (l["mediana_mm"] * 1e6) * 100
        cuanto = "menos del 1%" if pct < 1 else f"el {pct:,.0f}%"
        tope = ((cfg or {}).get("liquidez") or {}).get("orden_max_pct", 0.02) * 100
        grande = " Es una posición grande para esta acción." if pct > tope else ""
        parte = f" $ {orden_cop / 1e6:,.0f} millones serían {cuanto} de lo que se negocia en un día.{grande}".replace(",", ".")
    if l["nivel"] == BUENA and l.get("fuente") == "bvc":
        return f"💧 Liquidez de {t} en trii: buena (se negocian en promedio unos {med} al día, según la Bolsa de Colombia).{parte}"
    if l["nivel"] == BUENA:
        cont = l.get("continuidad")
        seguido = f" Hay operaciones en el {cont['mediana']:.0%} del día." if cont else ""
        return f"💧 Liquidez de {t} en trii: buena (se negocian unos {med} al día, también en sus días flojos).{seguido}{parte}"
    if l.get("a_ratos"):
        return (f"💧 Liquidez de {t} en trii: JUSTA. Mueve unos {med} al día, pero {por_ratos(l)}.{parte} No la recomiendo: si la tienes o decides entrar, "
                "hazlo sólo con orden límite y con paciencia; tu orden puede tardar un buen rato en ejecutarse.")
    por = ("; ".join(l.get("motivos") or [])) or f"unos {med} al día"
    if l.get("fuente") == "bvc":                                                       # medida con el libro de la bolsa: se dice cuántas acciones fueron, que es lo que se ve en trii
        por += f"; última sesión: {acc} acciones, $ " + f"{l['hoy_mm']:,.0f}".replace(",", ".") + " millones"
        if l["nivel"] == JUSTA:
            return (f"💧 Liquidez de {t} en trii: JUSTA ({por}).{parte} No la recomiendo para entrar. Si la tienes, sal sólo con orden límite y por partes: "
                    "una orden a mercado te vende barato.")
        return (f"💧 Liquidez de {t} en trii: MALA ({por}).{parte} Entrar es fácil, salir no: puedes quedarte sin comprador o vender muy por debajo. "
                "No la compres; si la tienes, sal con orden límite y sin prisa.")
    if l["nivel"] == JUSTA:
        return (f"💧 Liquidez de {t} en trii: JUSTA ({por}; hoy {acc} acciones).{parte} No la recomiendo. Si la tienes, sal sólo con orden límite y sin apuro: "
                "una orden a mercado te vende barato.")
    return (f"💧 Liquidez de {t} en trii: MALA ({por}; hoy {acc} acciones).{parte} Entrar es fácil, salir no: "
            "puedes quedarte sin comprador o vender muy por debajo. No la compres; si la tienes, sal con orden límite y sin prisa.")


def por_ratos(l: dict[str, Any]) -> str:
    return "; ".join(l.get("motivos") or [])


def veredicto(l: dict[str, Any], es_bvc: bool) -> tuple[bool, str]:
    """(¿apta para comprar según el filtro?, frase corta). Sólo es apta una acción de la BVC con liquidez BUENA; sin dato cuenta como NO apta."""
    if l["nivel"] == BUENA and es_bvc:
        return True, "✅ APTA: pasa el filtro de liquidez (y es de la BVC)."
    if l["nivel"] == BUENA:
        return True, "✅ APTA: se negocia bien en trii (es una acción extranjera que también se negocia en la Bolsa de Colombia)."
    if l["nivel"] == SIN_DATO:
        return False, "⛔ NO APTA para comprar: no puedo comprobar su liquidez, y lo que no puedo comprobar no lo apruebo."
    if l.get("a_ratos"):
        return False, "⛔ NO APTA para comprar: mueve buena plata al día, pero se negocia a ratos."
    return False, "⛔ NO APTA para comprar: no pasa el filtro de liquidez en trii."
