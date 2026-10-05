"""Pruebas de la Fase 5: lógica del tablero (sin pantalla) y la app de Streamlit con datos simulados (sin red)."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import servicio as S
from src import tablero as T
from src.regla_maestra import CAMBIAR, MANTENER, ContextoRegla, Decision, regla_maestra
from src.relevo import Candidato, construir_banco
from src.semaforo import AMARILLO, ROJO, VERDE
from src.sentimiento import puntuar_vader
from tests.helpers import make_cfg
from tests.test_fase4 import CLAVE, ent_falsa, estado_tmp, mk_res, motor_falso

CFG = make_cfg()
RAIZ = Path(__file__).resolve().parents[1]
AHORA = pd.Timestamp("2026-10-27 19:30", tz="America/Bogota").to_pydatetime()


def intradia_falso():
    idx = pd.date_range("2026-10-26 09:30", periods=26, freq="15min", tz="America/New_York")
    return pd.DataFrame({"Close": np.linspace(100, 93, len(idx))}, index=idx)


def datos(color=AMARILLO, ranking=(6.0, 11.0), con_base=True, noticias=True, **kw):
    cands = [Candidato("META", "mgc", 0.09, "σ20", corr=0.26, valor_negociado_mm=9999), Candidato("GRUPOARGOS", "local", 0.08, "σ20", corr=0.1, valor_negociado_mm=9000),
             Candidato("ETB", "local", 0.30, "σ20", valor_negociado_mm=9000), Candidato("AMZN", "mgc", 0.07, "σ20", color=ROJO, valor_negociado_mm=9999)]
    r = motor_falso(mk_res(color, **kw), construir_banco(cands, "TSLA", CFG), Decision(CAMBIAR, g=6.0, multiplo=1.22, candidato="META", ratio=1.38, codigo="cambiar"))
    r.candidatos = cands
    brank = [dict(ticker="TSLA", grupo="mgc", es_base=True, mee=0.14, vs_base=1.0, iv=None, real=0.5, liquidez_mm=1e4, supera=False, liquida=True),
             dict(ticker="GRUPOARGOS", grupo="local", es_base=False, mee=0.15, vs_base=1.07, iv=0.4, real=0.5, liquidez_mm=1e4, supera=False, liquida=True)] if con_base else None
    return T.DatosTablero(AHORA, "TSLA", r, [dict(fecha=AHORA.date(), hora="", detalle="Corte del concurso: pasa el top 10% (Oro → Diamante)", tipo="corte")],
                          [CLAVE | {"idioma": "en"}] if noticias else None, brank, T.serie_sigma(intradia_falso(), 100.0, 0.02),
                          dict(mia=ranking[0], umbral=ranking[1], cambios_usados=1, cambios_max=4, cambio_hoy=False), {"TSLA", "META", "GRUPOARGOS", "AMZN"})


# =============================== lógica del tablero ===============================
def test_serie_sigma_expresa_el_movimiento_del_dia_en_sigmas():
    s = T.serie_sigma(intradia_falso(), 100.0, 0.02)
    assert len(s) == 26 and s["sigma"].iloc[0] == pytest.approx(0.0, abs=0.2) and s["sigma"].iloc[-1] == pytest.approx((93 / 100 - 1) / 0.02)
    assert list(s.columns) == ["hora", "precio", "sigma"] and s["hora"].iloc[0] == "09:30"


def test_serie_sigma_no_inventa_cuando_faltan_datos():
    assert T.serie_sigma(None, 100, 0.02) is None and T.serie_sigma(pd.DataFrame(), 100, 0.02) is None
    assert T.serie_sigma(intradia_falso(), 100, 0.0) is None and T.serie_sigma(intradia_falso(), 0, 0.02) is None


def test_tabla_del_banco_en_palabras_sencillas():
    d = datos()
    t = T.df_banco(d.r)
    assert list(t["Acción"]) == ["META", "GRUPOARGOS"] and list(t["Mercado"]) == ["EE. UU.", "BVC"] and t["Semáforo"].iloc[0].endswith("VERDE")
    assert t["Veces tu acción"].iloc[0] == pytest.approx(0.09 / 0.065, abs=0.01) and "se mueve distinto a TSLA" in t["Parecido a tu acción"].iloc[0]
    assert not any("MEE" in x for x in t["Motivos"])


def test_tabla_de_descartados_dice_el_motivo():
    t = T.df_descartados(datos().r, CFG, {"TSLA", "META", "GRUPOARGOS", "AMZN"}).set_index("Acción")
    assert "no está permitida" in t.loc["ETB", "Por qué no"] and "ROJO" in t.loc["AMZN", "Por qué no"] and "META" not in t.index


def test_tablas_de_fechas_noticias_y_base():
    d = datos()
    assert "Corte del concurso" in T.df_catalizadores(d.cats)["Qué pasa"].iloc[0]
    n = T.df_noticias(d.noticias, AHORA, lambda t: "Tesla retira vehículos")
    assert n["Tono"].iloc[0] == "😟" and n["Titular"].iloc[0] == "Tesla retira vehículos" and T.df_noticias(None, AHORA).empty
    assert "(en inglés)" in T.df_noticias(d.noticias, AHORA, lambda t: None)["Titular"].iloc[0]
    b = T.df_base(d.base_rank)
    assert b["Acción"].iloc[0].endswith("← tu base") and b["Mercado"].iloc[1] == "BVC" and T.df_base(None).empty


def test_la_calculadora_aplica_la_misma_regla_maestra_del_bot():
    d = datos()
    dec, banco = T.calc_regla(d, CFG, 6.0, 11.0, cambios_usados=0)
    directo, _ = regla_maestra(ContextoRegla(mia=6.0, objetivo=11.0, activo="TSLA", color_actual=AMARILLO, mee_actual=d.r.mee, cambios_restantes=4, cambio_hoy=False), d.r.candidatos, CFG)
    assert (dec.accion, dec.candidato, dec.g, dec.multiplo) == (directo.accion, directo.candidato, directo.g, directo.multiplo)
    assert dec.g == pytest.approx(6.0) and dec.multiplo == pytest.approx(1.2167, abs=1e-3) and dec.accion == CAMBIAR and dec.candidato == "META"
    assert "ETB" not in [x.c.ticker for x in banco]                                                   # la lista negra nunca es recomendación
    ev = T.evaluados(dec)
    assert ev["¿Alcanza?"].iloc[0] == "✅ sí"


def test_la_calculadora_respeta_ir_arriba_los_limites_y_los_cambios_usados():
    d = datos()
    assert T.calc_regla(d, CFG, 12.0, 11.0)[0].accion == MANTENER
    assert T.calc_regla(d, CFG, 6.0, 11.0, cambios_usados=4)[0].codigo == "limite_total"
    assert T.calc_regla(d, CFG, 6.0, 11.0, cambios_usados=0, cambio_hoy=True)[0].codigo == "limite_dia"


# =============================== la app ===============================
@pytest.fixture
def app(monkeypatch, tmp_path):
    st.cache_data.clear()                                                                              # la caché de Streamlit vive en el proceso: cada prueba parte limpia
    ctx = S.Contexto(CFG, SimpleNamespace(traducir=lambda t: "Tesla retira vehículos"), puntuar_vader, "vader", AHORA, True, tmp_path / "s.json", salida=lambda s: None)
    estado_tmp(tmp_path).guardar()
    monkeypatch.setattr(S, "crear_contexto", lambda *a, **k: ctx)
    monkeypatch.setattr(S, "purgar_cache", lambda c: 0)
    estado = {"n": 0}
    monkeypatch.setattr(T, "cargar", lambda c: (estado.__setitem__("n", estado["n"] + 1), datos())[1])
    at = AppTest.from_file(str(RAIZ / "app.py"), default_timeout=60)
    at.estado = estado
    return at


def textos(at):
    out = [m.value for m in at.markdown] + [i.value for i in at.info] + [s.value for s in at.success] + [c.value for c in at.caption] + [w.value for w in at.warning]
    return "\n".join(out)


def test_la_app_arranca_sin_errores_y_muestra_el_semaforo_en_palabras_sencillas(app):
    app.run()
    assert not app.exception
    t = textos(app)
    assert "TSLA: AMARILLO (vigilar)" in t and "¿Cambio de acción?" in t and "cambia a META" in t.replace("**", "")
    assert [tb.label for tb in app.tabs] == ["🚦 Hoy", "🔁 Relevos", "🧮 Calculadora", "📰 Noticias", "📅 Fechas", "🔎 Base"]
    for jerga in ("z_res", "MEE", "σ20"):
        assert jerga not in t


def test_la_app_muestra_las_metricas_del_dia(app):
    app.run()
    m = {x.label: x.value for x in app.metric}
    assert m["Hoy"] == "-1,2%" and m["Volumen"] == "1,1×" and "Te faltan (g)" in m and m["Te faltan (g)"] == "6,0 pts"


def test_la_calculadora_se_recalcula_con_los_numeros_del_usuario(app):
    app.run()
    entradas = {n.label: n for n in app.number_input}
    assert entradas["Mi rentabilidad (%)"].value == 6.0 and entradas["Objetivo del corte (%)"].value == 11.0                    # parte de lo guardado
    entradas["Mi rentabilidad (%)"].set_value(12.0).run()
    assert not app.exception
    assert "lo mejor es mantener" in "\n".join(s.value for s in app.success).lower() or "por encima del objetivo" in "\n".join(s.value for s in app.success)
    app.number_input[0].set_value(6.0).run()
    assert "cambia a META" in "\n".join(s.value for s in app.success).replace("**", "")


def test_el_boton_actualizar_ahora_vuelve_a_consultar(app):
    app.run()
    antes = app.estado["n"]
    boton = next(b for b in app.button if "Actualizar ahora" in b.label)
    boton.click().run()
    assert not app.exception and app.estado["n"] == antes + 1


def test_si_fallan_los_datos_la_app_explica_y_no_se_cae(monkeypatch, app):
    def roto(c):
        raise RuntimeError("Yahoo caído")
    monkeypatch.setattr(T, "cargar", roto)
    app.run()
    assert not app.exception and any("No pude cargar los datos" in e.value for e in app.error)
    assert any(b.label == "Reintentar" for b in app.button)


def test_sin_noticias_ni_base_la_app_sigue_funcionando(monkeypatch, app):
    monkeypatch.setattr(T, "cargar", lambda c: datos(con_base=False, noticias=False))
    app.run()
    assert not app.exception
    assert "No pude consultar noticias" in textos(app) and "No pude calcular la comparación de base" in "\n".join(m.value for m in app.markdown)


def test_sin_ranking_guardado_la_calculadora_arranca_en_cero(monkeypatch, app):
    monkeypatch.setattr(T, "cargar", lambda c: datos(ranking=(None, None)))
    app.run()
    assert not app.exception and app.number_input[0].value == 0.0 and app.number_input[1].value == 0.0


def test_la_app_no_pide_ni_expone_el_token_de_telegram_ni_la_posicion():
    codigo = (RAIZ / "app.py").read_text(encoding="utf-8") + (RAIZ / "src" / "tablero.py").read_text(encoding="utf-8")
    assert "TELEGRAM" not in codigo
    assert "FINNHUB_API_KEY" in codigo                                                                 # única clave que necesita
