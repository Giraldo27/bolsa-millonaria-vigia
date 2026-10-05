"""Pruebas de los textos calculados del informe de validación (las frases deben salir de las cifras, no de conclusiones escritas a mano)."""
import pytest

from report import validacion as RV


def fila(h, media, lo, hi):
    return dict(h=h, media=media, ic_lo=lo, ic_hi=hi, n=50, p=0.5, pos=0.5, salir=-media - 0.013)


def test_clasifica_segun_el_intervalo_no_segun_el_signo_de_la_media():
    assert RV.clasifica(0.01, 0.002, 0.02) == "sube"
    assert RV.clasifica(-0.01, -0.02, -0.002) == "baja"
    assert RV.clasifica(0.01, -0.005, 0.03) == "sin efecto claro"                          # media positiva pero el intervalo cruza 0: no es efecto demostrable


def test_describir_resume_cada_horizonte_con_su_veredicto():
    t = RV.describir([fila(5, 0.0017, -0.008, 0.011), fila(10, 0.0103, 0.003, 0.017)], (5, 10))
    assert "a 5 sesiones +0,17 % (sin efecto claro)" in t and "a 10 sesiones +1,03 % (sube)" in t
    assert RV.describir([fila(5, 0.0, -0.1, 0.1)], (5, 10, 20)).count("sesiones") == 1       # sólo describe los horizontes que existen


def test_veredicto_de_comparacion_exige_que_el_intervalo_no_cruce_cero():
    assert RV.veredicto_comparacion(0.001, 0.02) == "mejora"
    assert RV.veredicto_comparacion(-0.02, -0.001) == "empeora"
    assert RV.veredicto_comparacion(-0.01, 0.02) == "sin diferencia demostrable"


def test_azar_esperado_anida_los_cortes_con_campo_total():
    assert RV.azar_esperado([50, 33, 10, 10]) == pytest.approx([0.5, 0.33, 0.10, 0.10])


def test_cifras_noticias_calcula_la_fraccion_con_noticia_negativa():
    r = RV.cifras_noticias(dict(propia_con_volumen=24, con_noticia_negativa=20))
    assert r["pct_con_noticia"] == pytest.approx(20 / 24) and RV.cifras_noticias(None) is None
