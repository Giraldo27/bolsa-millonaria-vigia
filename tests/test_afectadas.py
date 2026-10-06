"""Pruebas de "a qué acciones de la BVC afecta y en qué porcentaje": noticias de empresas (la propia, su otra serie y su grupo) y noticias macro."""
import datetime as dt

import pytest

from src import formato as F
from src import macro as M
from src import noticias_bvc as N
from tests.helpers import make_cfg
from tests.test_fase4 import bog
from tests.test_macro import Fuentes, item

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)


def test_solo_cuentan_como_relacionadas_la_otra_serie_y_las_empresas_del_mismo_grupo():
    r = M.relacionadas(CFG)
    assert r["GRUPOARGOS"]["PFGRUPOARG"] == "misma" and r["PFBCOLOM"]["BCOLOMBIA"] == "misma"
    assert r["GRUPOARGOS"]["CEMARGOS"] == "grupo" and r["CEMARGOS"]["CELSIA"] == "grupo" and r["PFBCOLOM"]["GRUPOSURA"] == "grupo"
    assert "ECOPETROL" not in r and "EXITO" not in r.get("PFBCOLOM", {}) and "MINEROS" not in r.get("PFBCOLOM", {})      # sin vínculo real no hay contagio
    negra = set(CFG["universe"]["blacklist"])
    assert not (set(r) & negra) and all(not (set(v) & negra) for v in r.values())


def test_el_contagio_da_1_a_1_a_la_otra_serie_y_lo_medido_al_grupo_si_es_firme():
    import numpy as np
    import pandas as pd
    f = Fuentes()
    base = f.acciones["ECOPETROL"]                                                     # se reutiliza como "noticia propia" de Grupo Argos
    ruido = pd.Series(np.random.default_rng(1).normal(0, 0.004, len(base)), index=base.index)
    f.acciones = {"GRUPOARGOS": base, "CEMARGOS": 0.6 * base + ruido}
    c = M.contagio(f, CFG, ["GRUPOARGOS", "CEMARGOS", "CELSIA", "ECOPETROL"])
    assert c["GRUPOARGOS"]["PFGRUPOARG"] == 1.0 and c["GRUPOARGOS"]["CEMARGOS"] == pytest.approx(0.6, abs=0.08)
    assert "CELSIA" not in c["GRUPOARGOS"] and c.get("ECOPETROL", {}) == {}            # Celsia no mostró relación firme; Ecopetrol no tiene grupo


def test_las_afectadas_por_una_noticia_llevan_porcentaje_y_signo():
    a = M.afectadas_por_noticia("PFBCOLOM", -0.018, -1, {"BCOLOMBIA": 1.0, "GRUPOSURA": 0.39, "PFSURA": 0.28, "X": 0.05}, CFG, {"GRUPOSURA"})
    assert [(x["ticker"], round(x["efecto"], 4)) for x in a] == [("PFBCOLOM", -0.018), ("BCOLOMBIA", -0.018), ("GRUPOSURA", -0.007), ("PFSURA", -0.005)]   # X: menos de 0,3 %, no se lista
    assert a[0]["propia"] and all(x["sentido"] == -1 for x in a) and a[2]["tengo"]
    sin = M.afectadas_por_noticia("ECOPETROL", None, 1, {"Y": 1.0}, CFG, set())
    assert sin == [dict(ticker="ECOPETROL", efecto=None, sentido=1, tengo=False, propia=True)]      # sin cifra medida sólo se dice el sentido de la propia


def test_el_bloque_de_afectadas_se_lee_facil():
    a = M.afectadas_por_noticia("PFBCOLOM", -0.018, -1, {"BCOLOMBIA": 1.0, "GRUPOSURA": 0.39}, CFG, {"GRUPOSURA"})
    t = F.plano("\n".join(F.bloque_afectadas(a)))
    assert "Acciones de la BVC afectadas: 🔴 PFBCOLOM -1,8% · 🔴 BCOLOMBIA -1,8% · 🔴 GRUPOSURA -0,7% (la tienes)" in t and "mismo grupo" in t
    sola = F.plano("\n".join(F.bloque_afectadas(M.afectadas_por_noticia("ECOPETROL", 0.02, 1, {}, CFG, set()))))
    assert sola == "📊 Acciones de la BVC afectadas: 🟢 ECOPETROL +2,0%"
    assert "sube (+)" in F.plano(F.bloque_afectadas(M.afectadas_por_noticia("ECOPETROL", None, 1, {}, CFG, set()))[0]) and F.bloque_afectadas(None) == []


def test_los_dos_avisos_de_noticias_incluyen_las_afectadas():
    from tests.test_noticias_bvc import senal
    a = M.afectadas_por_noticia("ECOPETROL", 0.0178, 1, {}, CFG, set())
    assert "Acciones de la BVC afectadas: 🟢 ECOPETROL +1,8%" in F.plano(F.msg_noticia_bvc(senal(), AHORA, CFG, None, a))
    corto = senal(0.004)
    corto.nivel = "medio"
    assert "Acciones de la BVC afectadas: 🟢 ECOPETROL +1,8%" in F.plano(F.msg_noticia_relevante(corto, AHORA, CFG, a))


def test_un_titular_macro_dice_a_quien_le_pega_aunque_el_mercado_aun_no_se_mueva():
    f = Fuentes()                                                                      # día quieto: ningún factor se mueve
    tab = M.tablero(f, CFG)
    sens = M.sensibilidades(f, CFG, ["ECOPETROL", "NUCO", "ISA"])
    uno = {x["clave"]: M.impacto(dict(x, cambio=0.01), sens.get(x["clave"], {}), CFG, {"ECOPETROL"}, minimo=0.001) for x in tab}
    g = dict(tema="petroleo", nombre="petróleo", items=[item("OPEP+ acuerda recortar producción", "Portafolio")], fuentes=["Portafolio", "Semana"], factores=["petroleo"])
    t = F.plano(F.msg_macro_titular(g, tab, {}, AHORA, uno))
    assert "el petróleo (Brent) se mueve dentro de lo normal" in t and "Acciones de la BVC afectadas (cuánto suele moverse cada una por cada 1% del factor)" in t
    assert "Si el petróleo (Brent) sube 1%: 🟢 ECOPETROL (la tienes) +0,5% (si baja 1%, lo contrario)" in t
    oro = F.plano(F.msg_macro_titular(dict(g, factores=["oro"]), tab, {}, AHORA, uno))
    assert "Ninguna acción de la BVC ha mostrado una relación firme" in oro


def test_el_efecto_por_cada_1_por_ciento_usa_un_minimo_mas_bajo():
    f = Fuentes()
    sens = M.sensibilidades(f, CFG, ["ECOPETROL"])
    factor = dict(clave="petroleo", nombre="x", cambio=0.004, z=0.2, en_puntos=False)
    assert M.impacto(factor, sens["petroleo"], CFG, set())["beneficiadas"] == []                    # 0,5 × 0,4 % = 0,2 %: por debajo del mínimo normal
    assert [x["ticker"] for x in M.impacto(factor, sens["petroleo"], CFG, set(), minimo=0.001)["beneficiadas"]] == ["ECOPETROL"]
