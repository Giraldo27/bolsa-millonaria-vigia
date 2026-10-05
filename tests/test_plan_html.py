"""El documento del plan completo se genera desde la configuración, sin huecos y sin cifras fijas que contradigan la config."""
import copy

import run_plan as RP
from src.regla_maestra import multiplo_requerido
from tests.helpers import make_cfg

CFG = make_cfg()


def cuerpo(h: str) -> str:
    return h.split("</style>")[1]


def test_el_plan_se_genera_completo_con_todas_las_secciones():
    h = RP.construir(CFG)
    assert h.startswith("<!doctype html>") and h.count("<section") == len(RP.NAV) == 17
    for i, _ in RP.NAV:
        assert f"id='{i}'" in h and f"href='#{i}'" in h
    for esperado in ("Regla Maestra", "Semáforo", "Banco de relevo", "/informe", "CONCONCRET", "TSLA al 97 %", "vie 30/10", "nunca envía órdenes", "+100 %"):
        assert esperado in h, esperado
    c = cuerpo(h)
    assert "None" not in c and "nan %" not in c and ">nan<" not in c and "{" not in c                   # ningún hueco sin rellenar


def test_los_numeros_salen_de_la_configuracion_no_de_texto_fijo():
    cfg = copy.deepcopy(CFG)
    cfg["regla_maestra"]["costo_cambio_pp"] = 2.5
    cfg["semaforo"]["z_caida"] = -2.7
    cfg["posicion_base"]["peso"] = 0.90
    h = RP.construir(cfg)
    assert "(g + 2,5) / g" in h and "-2,7" in h and "TSLA al 90 %" in h
    assert RP.n(multiplo_requerido(6, 2.5), 2) + "×" in RP.sec_regla(cfg)


def test_la_tabla_de_multiplos_es_la_formula_real():
    h = RP.sec_regla(CFG)
    for g in (1, 2, 6, 10, 20):
        assert RP.n(multiplo_requerido(g, 1.3), 2) + "×" in h


def test_el_documento_advierte_los_riesgos_y_no_promete_ganancias():
    h = RP.construir(CFG).lower()
    for frase in ("no hay ninguna regla que garantice", "revoca el token", "supuestos sin verificar", "no se pudo probar desde este pc"):
        assert frase in h, frase
