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
    assert "EXITO" not in r.get("PFBCOLOM", {}) and "MINEROS" not in r.get("PFBCOLOM", {}) and "MINEROS" not in r      # sin vínculo real no hay contagio
    assert r["ECOPETROL"]["ISA"] == "grupo" and r["ISA"]["ECOPETROL"] == "grupo"                                        # Ecopetrol es dueña de la mayoría de ISA
    assert r["PFBCOLOM"]["BOGOTA"] == "sector" and r["ISA"]["GEB"] == "sector" and r["ECOPETROL"]["TERPEL"] == "sector"  # competidoras del mismo sector
    assert r["AVAL"]["BOGOTA"] == "grupo" and r["PFBCOLOM"]["BCOLOMBIA"] == "misma"                                     # si hay dos vínculos queda el más fuerte
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
    assert sin == [dict(ticker="ECOPETROL", efecto=None, sentido=1, tengo=False, propia=True, via="directo", motivo="")]      # sin cifra medida sólo se dice el sentido de la propia


def test_el_bloque_de_afectadas_se_lee_facil():
    a = M.afectadas_por_noticia("PFBCOLOM", -0.018, -1, {"BCOLOMBIA": 1.0, "GRUPOSURA": 0.39}, CFG, {"GRUPOSURA"})
    t = F.plano("\n".join(F.bloque_afectadas(a)))
    assert "Acciones de la BVC afectadas: 🔴 PFBCOLOM -1,8% · 🔴 BCOLOMBIA -1,8%" in t                                  # directo: la empresa y su otra serie
    assert "↪️ Efecto indirecto: 🔴 GRUPOSURA -0,7% (la tienes) (mismo grupo empresarial)" in t and "Es menos seguro que el efecto directo" in t
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


# =============================== efecto indirecto: mismo grupo y mismo sector ===============================
VEC_BANCO = {"BCOLOMBIA": 1.0, "GRUPOSURA": 0.39, "BOGOTA": 0.30, "AVAL": 0.10}


def test_los_resultados_de_un_banco_tambien_le_pegan_a_los_otros_bancos_pero_una_compra_no():
    a = M.afectadas_por_noticia("PFBCOLOM", 0.02, 1, VEC_BANCO, CFG, {"BOGOTA"}, "resultados")
    assert [(x["ticker"], x["via"]) for x in a] == [("PFBCOLOM", "directo"), ("BCOLOMBIA", "misma"), ("GRUPOSURA", "grupo"), ("BOGOTA", "sector")]     # AVAL: 0,2 %, no llega al mínimo
    bogota = a[-1]
    assert bogota["motivo"] == "mismo sector: bancos" and bogota["efecto"] == pytest.approx(0.006) and bogota["sentido"] == 1 and bogota["tengo"]
    compra = M.afectadas_por_noticia("PFBCOLOM", 0.02, 1, VEC_BANCO, CFG, set(), "fusion")
    assert [x["ticker"] for x in compra] == ["PFBCOLOM", "BCOLOMBIA", "GRUPOSURA"]            # que Bancolombia compre algo no dice nada del Banco de Bogotá
    assert [x["ticker"] for x in M.afectadas_por_noticia("PFBCOLOM", 0.02, 1, VEC_BANCO, CFG, set())] == ["PFBCOLOM", "BCOLOMBIA", "GRUPOSURA"]   # sin tipo: tampoco


def test_el_aviso_separa_el_efecto_directo_del_indirecto_y_dice_por_que():
    t = F.plano("\n".join(F.bloque_afectadas(M.afectadas_por_noticia("PFBCOLOM", 0.02, 1, VEC_BANCO, CFG, {"BOGOTA"}, "resultados"))))
    directo, indirecto, nota = t.split("\n")
    assert directo == "📊 Acciones de la BVC afectadas: 🟢 PFBCOLOM +2,0% · 🟢 BCOLOMBIA +2,0%"
    assert indirecto == "↪️ Efecto indirecto: 🟢 GRUPOSURA +0,8% (mismo grupo empresarial) · 🟢 BOGOTA +0,6% (la tienes) (mismo sector: bancos)"
    assert "mismo grupo o del mismo sector" in nota
    neutra = F.plano("\n".join(F.bloque_afectadas(M.afectadas_por_noticia("ECOPETROL", 0.01, 0, {"ISA": 0.4}, CFG, set(), "otra"))))
    assert "⚪ ECOPETROL ±1,0%" in neutra and "↪️ Efecto indirecto: ⚪ ISA ±0,4% (mismo grupo empresarial)" in neutra


def test_la_relacion_de_sector_se_mide_y_tiene_tope():
    """El par del sector sólo aparece si la relación medida es firme, y nunca recibe más de la mitad del impacto."""
    import numpy as np
    import pandas as pd
    from tests.test_macro import IDX, precios
    rng = np.random.default_rng(3)
    comun = pd.Series(rng.normal(0, 0.012, len(IDX)), index=IDX)
    f = Fuentes()
    f.acciones = {"PFBCOLOM": comun + pd.Series(rng.normal(0, 0.002, len(IDX)), index=IDX),
                  "BOGOTA": comun * 0.9 + pd.Series(rng.normal(0, 0.002, len(IDX)), index=IDX),      # se mueve casi igual que Bancolombia
                  "BHI": pd.Series(rng.normal(0, 0.012, len(IDX)), index=IDX)}                         # mismo sector, pero sin relación medida
    v = M.contagio(f, CFG, ["PFBCOLOM", "BOGOTA", "BHI"])["PFBCOLOM"]
    assert v["BCOLOMBIA"] == 1.0 and v["BOGOTA"] == CFG["noticias_bvc"]["indirectos"]["tope_sector"] and "BHI" not in v


def test_la_lista_de_noticias_dice_a_que_otras_acciones_les_pega():
    a = M.afectadas_por_noticia("PFBCOLOM", -0.02, -1, VEC_BANCO, CFG, set(), "resultados")
    assert F.plano(F.indirectas_txt(a)) == "↪️ también: BCOLOMBIA -2,0% (la misma empresa) · GRUPOSURA -0,8% (mismo grupo empresarial) · BOGOTA -0,6% (mismo sector: bancos)"
    assert F.indirectas_txt(a[:1]) == "" and F.indirectas_txt(None) == ""
    fila = dict(ts=dt.datetime(2026, 10, 6, 14, 0, tzinfo=dt.timezone.utc), emisor="Bancolombia", ticker="PFBCOLOM", titulo="Bancolombia reporta menor utilidad", fuente="Valora Analitik",
                oficial=False, cat="resultados", sentido=-1, impacto=-0.02, tengo=False, afectadas=a)
    t = F.plano(F.msg_noticias_actualizadas([fila], {"Superfinanciera": True}, AHORA, 12))
    assert "🔴 PFBCOLOM · -2,0%" in t and "↪️ también: BCOLOMBIA -2,0% (la misma empresa)" in t and "BOGOTA -0,6% (mismo sector: bancos)" in t
