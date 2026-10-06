"""Pruebas de la macro en vivo (factores, sensibilidades medidas, titulares macro), de los cambios sugeridos y del resumen de la mañana."""
import datetime as dt
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src import cambios as CB
from src import formato as F
from src import liquidez as LQ
from src import macro as M
from src import noticias_bvc as N
from src import servicio as S
from src.relevo import Candidato, EntradaBanco
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg
from tests.test_fase4 import bog, sin_jerga

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)
UTC = dt.timezone.utc
IDX = pd.bdate_range(end="2026-10-05", periods=300)
RNG = np.random.default_rng(7)
R_BRENT = pd.Series(RNG.normal(0, 0.02, 300), index=IDX)
R_BRASIL = pd.Series(RNG.normal(0, 0.015, 300), index=IDX)
R_OTRO = pd.Series(RNG.normal(0, 0.01, 300), index=IDX)
SIMB = {k: v["simbolo"] for k, v in CFG["macro_vivo"]["factores"].items()}


def precios(r):
    return 100 * np.cumprod(1 + r)


class Fuentes:
    """Historia: ECOPETROL sigue al petróleo (0,5 por 1), NUCO a Brasil (1 por 1), el resto no sigue a nada. `hoy` = cambio de hoy de cada factor."""
    def __init__(self, hoy=None, dia=None, sin_sesion_hoy=()):
        self.hoy = hoy or {}
        self.dia = pd.Timestamp(dia or AHORA.date())                                   # fecha de la barra "de hoy"
        self.sin_sesion_hoy = set(sin_sesion_hoy)                                      # factores que hoy aún no han negociado (su último dato es de ayer)
        ruido = lambda: pd.Series(RNG.normal(0, 0.004, 300), index=IDX)                # noqa: E731
        self.acciones = {"ECOPETROL": 0.5 * R_BRENT + ruido(), "NUCO": 1.0 * R_BRASIL + ruido()}
        self.fact = {SIMB["petroleo"]: R_BRENT, SIMB["brasil"]: R_BRASIL}
        self.avisos = []
        self.yf = SimpleNamespace(download=self._download, Ticker=self._ticker)

    def _con_cache(self, clave, ttl, fn, nombre):
        return fn()

    def _download(self, sym, **k):
        r = self.fact.get(sym, R_OTRO if sym in SIMB.values() else None)
        if r is None:
            return pd.DataFrame()
        clave = next((k2 for k2, v in SIMB.items() if v == sym), None)
        base = np.asarray(precios(r), dtype=float)
        if clave in self.sin_sesion_hoy:                                               # la última barra es la de la sesión anterior, con el movimiento pedido
            base = np.r_[base[:-1], base[-2] * (1 + self.hoy.get(clave, 0.0))]
            return pd.DataFrame({"Close": base}, index=IDX)
        return pd.DataFrame({"Close": np.r_[base, base[-1] * (1 + self.hoy.get(clave, 0.0))]}, index=IDX.append(pd.DatetimeIndex([self.dia])))

    def _ticker(self, sym):
        clave = next((k for k, v in SIMB.items() if v == sym), None)
        return SimpleNamespace(fast_info={"last_price": 100.0 * (1 + self.hoy.get(clave, 0.0)), "previous_close": 100.0})

    def diario(self, t, dias=None):
        r = self.acciones.get(t, pd.Series(RNG.normal(0, 0.012, 300), index=IDX))
        return pd.DataFrame({"Close": precios(r), "Volume": 1e6}, index=IDX)

    def cotizacion(self, t):
        return {"precio": 100.0, "cierre_previo": 100.0}


def ctx(tmp_path, f, ahora=AHORA, dry=False):
    return S.Contexto(CFG, f, puntuar_vader, "vader", ahora, dry, tmp_path / "s.json", salida=lambda s: None)


# =============================== factores y sensibilidades ===============================
def test_el_tablero_dice_cuanto_se_movio_cada_factor_frente_a_lo_normal():
    tab = {x["clave"]: x for x in M.tablero(Fuentes({"petroleo": -0.06, "brasil": 0.004}), CFG, AHORA)}
    assert set(tab) == set(SIMB) and tab["petroleo"]["fuerte"] and tab["petroleo"]["z"] < -2 and tab["petroleo"]["cambio"] == pytest.approx(-0.06)
    assert not tab["brasil"]["fuerte"] and not tab["oro"]["notable"]
    roto = M.tablero(Fuentes({"petroleo": 0.9}), CFG, AHORA)                                               # +90 % en un día: tick erróneo, se descarta
    assert "petroleo" not in {x["clave"] for x in roto}


def test_las_sensibilidades_se_miden_y_solo_quedan_las_firmes():
    s = M.sensibilidades(Fuentes(), CFG, ["ECOPETROL", "NUCO", "ISA", "GEB"])
    assert s["petroleo"]["ECOPETROL"]["beta"] == pytest.approx(0.5, abs=0.05) and abs(s["petroleo"]["ECOPETROL"]["t"]) > 10
    assert s["brasil"]["NUCO"]["beta"] == pytest.approx(1.0, abs=0.05)
    assert "ISA" not in s["petroleo"] and "GEB" not in s["brasil"] and "NUCO" not in s["petroleo"]   # sin relación firme no aparece


def test_el_impacto_separa_beneficiadas_y_afectadas_y_marca_lo_tuyo():
    f = Fuentes({"petroleo": -0.06})
    tab = {x["clave"]: x for x in M.tablero(f, CFG, AHORA)}
    sens = M.sensibilidades(f, CFG, ["ECOPETROL", "NUCO", "ISA"])
    imp = M.impacto(tab["petroleo"], sens["petroleo"], CFG, {"ECOPETROL"})
    assert [x["ticker"] for x in imp["afectadas"]] == ["ECOPETROL"] and imp["beneficiadas"] == [] and imp["mias"][0]["tengo"]
    assert imp["afectadas"][0]["efecto"] == pytest.approx(-0.03, abs=0.004)                          # 0,5 × −6 %
    sube = M.impacto(dict(tab["petroleo"], cambio=0.04), sens["petroleo"], CFG, set())
    assert [x["ticker"] for x in sube["beneficiadas"]] == ["ECOPETROL"]


def test_el_dolar_afecta_directo_a_las_acciones_globales_que_tienes():
    dolar = dict(clave="dolar", nombre="el dólar en Colombia", cambio=-0.018, z=-1.6, en_puntos=False)
    imp = M.impacto(dolar, {}, CFG, {"NUCO", "META", "GRUPOARGOS"})
    assert {x["ticker"] for x in imp["afectadas"]} == {"NUCO", "META"} and all(x["directo"] and x["efecto"] == -0.018 for x in imp["afectadas"])


def test_un_movimiento_fuerte_se_avisa_una_vez_por_dia_y_sentido_salvo_que_crezca_mucho():
    mem = {}
    fuerte = [dict(clave="petroleo", z=-2.4, fuerte=True), dict(clave="oro", z=0.3, fuerte=False)]
    assert [x["clave"] for x in M.movimientos_nuevos(fuerte, mem, AHORA)] == ["petroleo"]
    assert M.movimientos_nuevos(fuerte, mem, AHORA) == []
    assert M.movimientos_nuevos([dict(clave="petroleo", z=-3.0, fuerte=True)], mem, AHORA) == []      # un poco más: no se repite
    assert len(M.movimientos_nuevos([dict(clave="petroleo", z=-4.2, fuerte=True)], mem, AHORA)) == 1  # mucho más: sí
    assert len(M.movimientos_nuevos([dict(clave="petroleo", z=2.5, fuerte=True)], mem, AHORA)) == 1   # el otro sentido es otra noticia
    assert len(M.movimientos_nuevos(fuerte, mem, bog(2026, 10, 7, 10))) == 1                          # otro día


# =============================== titulares macro ===============================
def item(titulo, fuente, minutos=5):
    return N.Item(AHORA.astimezone(UTC) - dt.timedelta(minutes=minutos), titulo, "", fuente, "", "rss")


@pytest.mark.parametrize("titulo, tema", [
    ("Banco de la República sube su tasa de interés a 9,75 %", "banrep"), ("Inflación en Colombia de septiembre fue 5,1 %, según el Dane", "inflacion_co"),
    ("La FED mantiene las tasas y Powell enfría expectativas", "fed"), ("Fitch rebaja la calificación de Colombia", "riesgo_pais"),
    ("Gobierno radica reforma tributaria por $16 billones", "riesgo_pais"), ("OPEP+ acuerda recortar producción", "petroleo"),
    ("Segunda vuelta en Brasil: así llegan los candidatos", "brasil"), ("Precio del dólar hoy en casas de cambio", None),
    ("Ecopetrol anuncia dividendo extraordinario", None), ("Millonarios gana el clásico", None),
])
def test_los_titulares_macro_se_reconocen_por_tema(titulo, tema):
    assert M.tema_de(titulo) == tema


def test_un_tema_macro_se_avisa_si_lo_traen_dos_medios_y_no_se_repite():
    mem = {}
    assert M.titulares_macro([item("Banco de la República sube su tasa de interés", "Portafolio")], mem, AHORA, CFG) == []      # primera vez: línea base
    uno = [item("Banco de la República sorprende y baja la tasa de interés", "Portafolio")]
    assert M.titulares_macro(uno, mem, AHORA, CFG) == []                                              # un solo medio: aún no
    dos = [item("La FED recorta tasas medio punto", "Portafolio"), item("Reserva Federal baja sus tasas más de lo esperado", "La República"),
           item("FED: Powell explica el recorte", "Semana", minutos=600)]
    (g,) = M.titulares_macro(dos, mem, AHORA, CFG)
    assert g["tema"] == "fed" and g["fuentes"] == ["Portafolio", "La República"] and len(g["items"]) == 2 and "nasdaq" in g["factores"]
    mas = [item("La FED y lo que viene para los mercados", "El Tiempo"), item("Tras la FED, el dólar reacciona en Colombia", "Valora Analitik")]
    assert M.titulares_macro(mas, mem, bog(2026, 10, 6, 10, 30), CFG) == []                           # mismo tema antes de 3 horas: no se repite


# =============================== mensajes ===============================
def test_el_aviso_de_movimiento_macro_dice_quien_gana_y_quien_pierde():
    f = Fuentes({"petroleo": -0.06})
    tab = {x["clave"]: x for x in M.tablero(f, CFG, AHORA)}
    imp = M.impacto(tab["petroleo"], M.sensibilidades(f, CFG, ["ECOPETROL", "NUCO", "ISA"])["petroleo"], CFG, {"ECOPETROL"})
    m = F.msg_macro_movimiento(tab["petroleo"], imp, AHORA)
    t = F.plano(m)
    for esperado in ("Movimiento macro fuerte", "el petróleo (Brent) cae 6,0%", "veces lo normal", "Afectadas: ECOPETROL (la tienes) -3,", "En tu cartera: ECOPETROL",
                     "Qué hacer: nada por reflejo", "/macro"):
        assert esperado in t, esperado
    assert "Beneficiadas" not in t and m.count("<b>") == m.count("</b>")
    sin_jerga(m)


def test_el_tablero_macro_y_el_aviso_de_titular():
    f = Fuentes({"petroleo": -0.06, "brasil": 0.03})
    c = S.Contexto(CFG, f, puntuar_vader, "vader", AHORA, True, None, salida=lambda s: None)
    tab = M.tablero(f, CFG, AHORA)
    sens = M.sensibilidades(f, CFG, ["ECOPETROL", "NUCO"])
    imp = {x["clave"]: M.impacto(x, sens.get(x["clave"], {}), CFG, {"NUCO"}) for x in tab if x["notable"]}
    t = F.plano(F.msg_macro_tablero(tab, imp, AHORA))
    assert "🔥 El petróleo (Brent) cae 6,0%" in t and "La bolsa de Brasil sube 3,0%" in t and "— normal" in t and "Beneficiadas: NUCO (la tienes) +3," in t and "Afectadas: ECOPETROL" in t
    g = dict(tema="petroleo", nombre="petróleo", items=[item("OPEP+ sorprende y aumenta la producción", "Portafolio")], fuentes=["Portafolio", "Semana"], factores=["petroleo"])
    n = F.plano(F.msg_macro_titular(g, tab, imp, AHORA))
    assert "Noticia macro: petróleo" in n and "OPEP+ sorprende" in n and "Cómo reacciona el mercado ahora: el petróleo (Brent) cae 6,0%" in n and "Afectadas: ECOPETROL" in n
    quieto = F.plano(F.msg_macro_titular(dict(g, factores=["oro"]), tab, imp, AHORA))
    assert "el oro se mueve dentro de lo normal" in quieto and "todavía no está moviendo los precios" in quieto
    assert "no pude consultar" in F.plano(F.msg_macro_tablero([], {}, AHORA)) and c.dry


def test_la_revision_macro_avisa_sola_el_movimiento_y_el_titular_con_botones(tmp_path, monkeypatch):
    enviados = []
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, botones=None, **k: enviados.append((texto, botones)) or True)
    mem = N.Memoria(tmp_path / "n.json")
    Estado(CFG, tmp_path / "s.json").registrar_compra("ECOPETROL", 1000, 2775, AHORA)
    assert S.tick_macro(ctx(tmp_path, Fuentes()), mem, []) == []                                      # día tranquilo: silencio
    out = S.tick_macro(ctx(tmp_path, Fuentes({"petroleo": -0.06})), mem, [])
    assert len(out) == 1 and "Movimiento macro fuerte" in out[0] and "ECOPETROL (la tienes)" in F.plano(out[0]) and enviados[0][1] == S.BOTONES_MACRO
    assert S.tick_macro(ctx(tmp_path, Fuentes({"petroleo": -0.06})), mem, []) == []                   # ya avisado hoy
    dos = [item("La FED recorta tasas medio punto", "Portafolio"), item("Reserva Federal baja sus tasas", "La República")]
    out = S.tick_macro(ctx(tmp_path, Fuentes({"petroleo": -0.06})), mem, dos)
    assert len(out) == 1 and "Noticia macro: Reserva Federal" in F.plano(out[0]) and len(enviados) == 2
    assert "macro_mov" in N.Memoria(tmp_path / "n.json").d                                           # la memoria quedó guardada


def test_el_comando_macro_responde_aunque_no_haya_datos(tmp_path):
    class SinRed(Fuentes):
        def _download(self, sym, **k):
            return pd.DataFrame()
    assert "no pude consultar" in F.plano(S.resp_macro(ctx(tmp_path, SinRed())))
    assert "Nada se mueve HOY más de lo normal" in F.plano(S.resp_macro(ctx(tmp_path, Fuentes())))


# =============================== cambios sugeridos ===============================
def liq(t, nivel, mediana=184.0, acciones=42.0):
    return dict(ticker=t, nivel=nivel, mediana_mm=mediana, hoy_mm=mediana, acciones_hoy=acciones, dias_sin_negociar=0, dias=20)


BANCO = [EntradaBanco(Candidato("PFSURA", "local", 0.042, "σ20", valor_negociado_mm=11300)), EntradaBanco(Candidato("ISA", "local", 0.035, "σ20", valor_negociado_mm=8400))]


def test_se_sugiere_salir_de_lo_que_casi_no_se_negocia_en_trii_hacia_la_mejor_de_la_bvc():
    liqs = {"TSLA": liq("TSLA", LQ.MALA), "META": liq("META", LQ.JUSTA, 534, 98), "NUCO": liq("NUCO", LQ.SIN_DATO, None), "GRUPOARGOS": liq("GRUPOARGOS", LQ.BUENA, 4982)}
    (s,) = CB.sugerencias(["GRUPOARGOS", "META", "NUCO", "TSLA"], liqs, BANCO, {"TSLA": 8.4e6}, CFG)
    assert (s["de"], s["a"], s["tipo"]) == ("TSLA", "PFSURA", "liquidez")
    assert "casi no se negocia en trii" in s["motivo"] and "$ 184 millones al día" in s["motivo"] and "hoy 42 acciones" in s["motivo"] and "es el 5 %" in s["motivo"]
    assert "±4,2 %" in s["por_que_a"] and "$ 11.300 millones al día" in s["por_que_a"]
    assert CB.sugerencias(["TSLA"], liqs, [], {}, CFG) == []                                         # sin una buena candidata no se manda a nadie a ninguna parte
    assert CB.sugerencias(["GRUPOARGOS", "NUCO"], liqs, BANCO, {}, CFG) == []                        # lo bueno y lo que no se puede medir no generan cambio


def test_el_mismo_cambio_no_se_repite_antes_de_24_horas_y_se_olvida_cuando_ya_no_aplica():
    mem = {}
    s = [dict(de="TSLA", a="PFSURA", tipo="liquidez")]
    assert CB.nuevos(s, mem, AHORA, CFG) == s and CB.nuevos(s, mem, bog(2026, 10, 6, 15), CFG) == []
    assert CB.nuevos(s, mem, bog(2026, 10, 7, 10, 1), CFG) == s                                      # al día siguiente lo recuerda una vez
    assert CB.nuevos([], mem, bog(2026, 10, 7, 11), CFG) == [] and mem["cambios_avisados"] == {}     # ya vendiste: se olvida


def test_el_mensaje_de_cambio_explica_el_motivo_el_destino_y_como_hacerlo():
    c = dict(de="TSLA", a="PFSURA", tipo="liquidez", motivo="TSLA casi no se negocia en trii (unos $ 184 millones al día; hoy 42 acciones).",
             por_que_a="es la acción de la BVC con liquidez buena que más puede moverse hasta el próximo corte (±4,2 %).", liquidez_a="💧 Liquidez de PFSURA en trii: buena.")
    m = F.msg_cambio(c, dict(k=3, n=226, dif=-0.0094))
    t = F.plano(m)
    for esperado in ("Cambio sugerido: TSLA → PFSURA", "¿Por qué? TSLA casi no se negocia", "¿Por qué PFSURA?", "Liquidez de PFSURA en trii: buena", "orden límite",
                     "vendí tsla", "rindieron 0,9% menos que las rezagadas en los 3 siguientes (medido en 226 períodos)", "no una garantía"):
        assert esperado in t, esperado
    assert m.count("<b>") == m.count("</b>") and "rezagadas" not in F.plano(F.msg_cambio(c))


def test_el_estudio_relativo_guardado_dice_que_perseguir_a_la_ganadora_pierde():
    r = CB.estudio_relativo(3)
    if r is None:
        pytest.skip("aún no se ha corrido run_estudio_relativo.py")
    assert r["dif"] < 0 and r["ic_hi"] < 0 and r["n"] >= 100                                         # si esto cambia, hay que repensar la regla (y este aviso lo dirá)


def test_el_aviso_de_cambio_sale_solo_una_vez_y_no_reescribe_el_estado_si_no_hay_novedad(tmp_path, monkeypatch):
    enviados = []
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, botones=None, **k: enviados.append(texto) or True)
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: liq(t, LQ.MALA if t == "TSLA" else LQ.BUENA, 184 if t == "TSLA" else 9000))
    e = Estado(CFG, tmp_path / "s.json")
    e.registrar_compra("TSLA", 8, 329.0, AHORA, 8.4e6)
    e.registrar_compra("GRUPOARGOS", 300, 21500, AHORA)
    c = ctx(tmp_path, Fuentes())
    out = S.tick_cambios(c, BANCO)
    assert len(out) == 1 and "Cambio sugerido: TSLA → PFSURA" in F.plano(out[0]) and len(enviados) == 1
    antes = (tmp_path / "s.json").stat().st_mtime_ns
    assert S.tick_cambios(c, BANCO) == [] and (tmp_path / "s.json").stat().st_mtime_ns == antes      # sin novedad: ni mensaje ni escritura


# =============================== resumen de la mañana ===============================
def test_el_resumen_de_la_manana_junta_macro_calendario_liquidez_y_cambios(tmp_path, monkeypatch):
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: liq(t, LQ.MALA if t == "TSLA" else LQ.BUENA, 184 if t == "TSLA" else 9000))
    e = Estado(CFG, tmp_path / "s.json")
    e.registrar_compra("TSLA", 8, 329.0, AHORA, 8.4e6)
    e.registrar_compra("NUCO", 1200, 15.2, AHORA, 58e6)
    t = F.plano(S.resumen_manana(ctx(tmp_path, Fuentes({"brasil": 0.05}, dia="2026-10-28"), ahora=bog(2026, 10, 28, 8, 10)), BANCO))
    for esperado in ("Antes de abrir (la bolsa abre a las 08:30)", "Qué pasó mientras dormías", "La bolsa de Brasil sube 5,0%", "Beneficiadas: NUCO (la tienes)",
                     "Hoy en el calendario", "Decisión de la FED", "Ojo con lo que se negocia poco", "Liquidez de TSLA en trii: MALA", "Cambios sugeridos", "TSLA → PFSURA",
                     "Órdenes límite, nunca a mercado", "Aún no sé cómo vas"):
        assert esperado in t, esperado
    e.actualizar_rank(1.0, 3.0, AHORA)
    assert "Aún no sé cómo vas" not in F.plano(S.resumen_manana(ctx(tmp_path, Fuentes(), ahora=bog(2026, 10, 7, 8, 10)), []))


# =============================== ¿de cuándo es el movimiento? y lo de fuera de horario ===============================
def test_lo_de_ayer_no_se_presenta_como_de_hoy_ni_vuelve_a_disparar_avisos(tmp_path, monkeypatch):
    """Caso real (6-oct-2026, 6:30 a. m.): el bot decía "el Colcap sube 2,9 %" y "NUCO sube 13 % hoy" cuando eso había sido el día anterior."""
    f = Fuentes({"colombia": 0.029, "petroleo": -0.06}, sin_sesion_hoy=["colombia"])
    tab = {x["clave"]: x for x in M.tablero(f, CFG, AHORA)}
    col = tab["colombia"]
    assert col["cuando"] == "ayer" and col["fecha"] == IDX[-1].date() and not col["fuerte"] and not col["notable"] and abs(col["z"]) > 2
    assert tab["petroleo"]["cuando"] == "hoy" and tab["petroleo"]["fuerte"]
    t = F.plano(F.msg_macro_tablero(list(tab.values()), {}, AHORA))
    assert "🕘 La bolsa de Colombia (Colcap) subió 2,9% en la última sesión (lun 05/10); hoy aún no abre" in t and "El petróleo (Brent) cae 6,0%" in t
    enviados = []
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, botones=None, **k: enviados.append(texto) or True)
    out = S.tick_macro(ctx(tmp_path, f), N.Memoria(tmp_path / "n.json"), [])
    assert len(out) == 1 and "petróleo" in out[0] and "Colcap" not in out[0].split("¿A quién le pega?")[0]     # sólo se avisa lo que pasa HOY


def test_lo_normal_se_mide_sin_contar_la_sesion_que_se_esta_juzgando():
    f = Fuentes({"petroleo": -0.06})
    x = next(x for x in M.tablero(f, CFG, AHORA) if x["clave"] == "petroleo")
    assert x["sigma"] == pytest.approx(float(R_BRENT.tail(60).std(ddof=1)), rel=0.02) and x["z"] == pytest.approx(-0.06 / x["sigma"])


class FuentesExt:
    """Barras de 5 minutos con pre-mercado y horario normal en Nueva York."""
    def __init__(self, barras):
        self.barras = barras
        self.yf = SimpleNamespace(Ticker=lambda s: SimpleNamespace(history=lambda **k: self._hist()))

    def _con_cache(self, clave, ttl, fn, nombre):
        return fn()

    def _hist(self):
        if not self.barras:
            return pd.DataFrame()
        idx = pd.DatetimeIndex([pd.Timestamp(t, tz="America/New_York") for t, _ in self.barras])
        return pd.DataFrame({"Close": [v for _, v in self.barras]}, index=idx)


def dia_normal(fecha, cierre):
    return [(f"{fecha} {h:02d}:{m:02d}", cierre) for h in range(10, 16) for m in (0, 30)]


def test_el_movimiento_antes_de_abrir_y_despues_de_cerrar_se_mide_contra_el_ultimo_cierre():
    from src import sesion as SE
    pre = FuentesExt(dia_normal("2026-10-05", 15.18) + [("2026-10-06 07:00", 16.0), ("2026-10-06 07:25", 16.70)])
    x = SE.fuera_de_horario(pre, "NU", dt.datetime(2026, 10, 6, 11, 30, tzinfo=UTC))                 # 07:30 en Nueva York
    assert x["estado"] == "pre" and x["cierre"] == 15.18 and x["cambio"] == pytest.approx(16.70 / 15.18 - 1) and x["hora"] == "06:25"     # hora de Bogotá
    post = FuentesExt(dia_normal("2026-10-05", 15.18) + [("2026-10-05 17:30", 14.42)])
    y = SE.fuera_de_horario(post, "NU", dt.datetime(2026, 10, 5, 22, 0, tzinfo=UTC))
    assert y["estado"] == "post" and y["cambio"] == pytest.approx(14.42 / 15.18 - 1)
    abierto = FuentesExt(dia_normal("2026-10-05", 15.18) + [("2026-10-06 08:00", 16.0), ("2026-10-06 10:00", 16.2)])
    assert SE.fuera_de_horario(abierto, "NU", dt.datetime(2026, 10, 6, 14, 5, tzinfo=UTC)) is None      # mercado abierto: no hay "fuera de horario"
    viejo = FuentesExt(dia_normal("2026-10-02", 15.18) + [("2026-10-02 17:30", 15.5)])
    assert SE.fuera_de_horario(viejo, "NU", dt.datetime(2026, 10, 4, 15, 0, tzinfo=UTC)) is None        # domingo: lo del viernes ya no es noticia
    assert SE.fuera_de_horario(FuentesExt([]), "NU") is None and F.linea_fuera(None) == ""
    assert F.linea_fuera(x) == "antes de abrir hoy va +10,0% (06:25)"


def test_el_semaforo_dice_de_que_dia_es_la_sesion_y_una_subida_fuerte_no_es_un_dia_normal():
    from tests.test_fase4 import mk_res
    from src.semaforo import VERDE
    r = mk_res(VERDE, ticker="NUCO")
    r.r_hoy, r.z, r.fecha = 0.132, 4.3, dt.date(2026, 10, 5)
    t = F.plano(F.msg_semaforo(r, bog(2026, 10, 6, 6, 30)))
    assert "En la última sesión (lun 05/10) NUCO subió 13,2%. Hoy todavía no ha abierto la bolsa." in t and "sube 13,2% hoy" not in t
    assert "fue una subida fuerte" in t and "no es un día normal" in t and "dentro de lo habitual" not in t
    mismo_dia = F.plano(F.msg_semaforo(r, bog(2026, 10, 5, 14, 0)))
    assert "NUCO sube 13,2% hoy." in mismo_dia
    r.fecha = None                                                                                   # resultados antiguos sin fecha: como antes
    assert "NUCO sube 13,2% hoy." in F.plano(F.msg_semaforo(r, bog(2026, 10, 6, 6, 30)))


def test_la_cartera_y_el_semaforo_muestran_el_movimiento_fuera_de_horario_de_tus_acciones_de_ee_uu(tmp_path, monkeypatch):
    from src import sesion as SE
    monkeypatch.setattr(SE, "fuera_de_horario", lambda f, sym, ahora=None: dict(estado="pre", precio=16.70, cierre=15.18, cambio=0.10, hora="06:25") if sym == "NU" else None)
    c = ctx(tmp_path, Fuentes())
    lineas = S.lineas_fuera_de_horario(c, ["NUCO", "META", "GRUPOARGOS"])
    assert len(lineas) == 1 and "NUCO antes de abrir (pre-mercado): +10,0% frente al último cierre, a las 06:25" in F.plano(lineas[0])
