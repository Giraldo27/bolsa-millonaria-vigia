"""Pruebas de los avisos automáticos de novedades (noticia nueva, Regla Maestra, calendario, subida fuerte, base): sin repetir y sin spam."""
import copy
from types import SimpleNamespace

import pytest

from src import eventos as EV
from src import formato as F
from src import servicio as S
from src.regla_maestra import CAMBIAR, MANTENER, VENDER_TODO, Decision
from src.relevo import Candidato
from src.semaforo import ROJO, VERDE
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg
from tests.test_fase4 import FakeF, banco_vacio, bog, evaluador, mk_res, sin_jerga

CFG = make_cfg()


def k(titulo, puntaje, ts="2026-10-06T14:00:00+00:00", palabras=None, **extra):
    return dict(titulo=titulo, puntaje=puntaje, ts=ts, proveedor="Yahoo", palabras=palabras or [], idioma="en", **extra)


MALA = k("Tesla recalls 100,000 vehicles amid probe", -0.62, palabras=["probe", "recall"])
AHORA = bog(2026, 10, 6, 10, 0)                                                                    # 15:00 UTC


# =============================== noticias nuevas ===============================
def test_la_primera_corrida_solo_toma_la_linea_base_y_no_avisa():
    ev = {}
    assert EV.noticias_nuevas(ev, [MALA], AHORA, CFG) == [] and len(ev["noticias_vistas"]) == 1


def test_avisa_una_noticia_importante_nueva_una_sola_vez():
    ev = {}
    EV.noticias_nuevas(ev, [k("Titular viejo", -0.7, "2026-10-05T10:00:00+00:00")], AHORA, CFG)
    nueva = k("Tesla faces SEC investigation", -0.7, palabras=["sec", "investigation"])
    assert EV.noticias_nuevas(ev, [nueva], AHORA, CFG) == [nueva]
    assert EV.noticias_nuevas(ev, [nueva], bog(2026, 10, 6, 11), CFG) == []                           # ya avisada


def test_ignora_lo_intrascendente_y_lo_viejo_pero_lo_da_por_visto():
    ev = {}
    EV.noticias_nuevas(ev, [], AHORA, CFG)
    neutral, vieja = k("Tesla opens a new store", 0.1), k("Hace diez horas", -0.8, "2026-10-06T04:00:00+00:00")
    assert EV.noticias_nuevas(ev, [neutral, vieja], AHORA, CFG) == []
    assert len(ev["noticias_vistas"]) == 2 and "noticias_ultimo_aviso" not in ev                      # no se avisó, pero tampoco se reciclará


def test_una_noticia_muy_buena_tambien_se_avisa_y_dice_que_no_se_persiga_el_precio():
    ev = {}
    EV.noticias_nuevas(ev, [], AHORA, CFG)
    buena = k("Tesla beats delivery estimates, record quarter", 0.85)
    assert EV.noticias_nuevas(ev, [buena], AHORA, CFG) == [buena]
    t = F.plano(F.msg_noticia_nueva("TSLA", [buena], mk_res(VERDE), AHORA))
    assert "no persigas el precio" in t and "no vendas" not in t


def test_el_enfriamiento_no_pierde_la_noticia_se_avisa_despues():
    ev = {}
    EV.noticias_nuevas(ev, [], AHORA, CFG)
    a = k("Primera mala noticia", -0.7)
    assert EV.noticias_nuevas(ev, [a], AHORA, CFG) == [a]
    b = k("Segunda mala noticia", -0.7, "2026-10-06T14:55:00+00:00")
    assert EV.noticias_nuevas(ev, [a, b], bog(2026, 10, 6, 10, 5), CFG) == []                       # sólo pasaron 5 min < 20
    assert EV.noticias_nuevas(ev, [a, b], bog(2026, 10, 6, 10, 30), CFG) == [b]                      # pasó el enfriamiento: ahora sí, no se perdió


def test_el_tope_por_aviso_limita_y_ordena_por_gravedad():
    ev = {}
    EV.noticias_nuevas(ev, [], AHORA, CFG)
    lote = [k(f"Mala {i}", -0.5 - i / 10) for i in range(5)]
    sal = EV.noticias_nuevas(ev, lote, AHORA, CFG)
    assert len(sal) == CFG["monitor"]["eventos"]["noticias"]["max_por_aviso"] and sal[0]["puntaje"] == min(x["puntaje"] for x in lote)
    assert EV.noticias_nuevas(ev, lote, bog(2026, 10, 6, 12), CFG) == []                              # las que sobraron tampoco se repiten después


def test_palabra_de_riesgo_sin_tono_positivo_cuenta_como_importante():
    assert EV.es_importante(k("SEC probe", 0.0, palabras=["sec", "probe"]), CFG) and not EV.es_importante(k("Tesla beats probe expectations", 0.4, palabras=["probe"]), CFG)


def test_el_historial_de_vistas_no_crece_sin_limite():
    ev = {}
    EV.noticias_nuevas(ev, [], AHORA, CFG)
    for i in range(30):
        EV.noticias_nuevas(ev, [k(f"neutral {i}-{j}", 0.0) for j in range(30)], AHORA, CFG)
    assert len(ev["noticias_vistas"]) <= 400


# =============================== cambio de opinión de la Regla Maestra ===============================
def dec(accion=CAMBIAR, cand="META", codigo="cambiar"):
    return Decision(accion, g=6.0, multiplo=1.22, candidato=cand if accion == CAMBIAR else None, ratio=1.4, codigo=codigo)


def test_avisa_cuando_la_regla_pasa_a_recomendar_y_cuando_deja_de_hacerlo():
    ev = {}
    assert EV.cambio_de_decision(ev, dec(MANTENER, codigo="ninguno_cumple"), AHORA, CFG) is None      # mantener desde el inicio: sin novedad
    assert EV.cambio_de_decision(ev, dec(), AHORA, CFG) == "nueva"
    assert EV.cambio_de_decision(ev, dec(), bog(2026, 10, 6, 10, 15), CFG) is None                    # sigue igual
    assert EV.cambio_de_decision(ev, dec(MANTENER, codigo="ninguno_cumple"), bog(2026, 10, 6, 10, 30), CFG) == "retirada"
    assert EV.cambio_de_decision(ev, dec(MANTENER, codigo="ninguno_cumple"), bog(2026, 10, 6, 10, 45), CFG) is None


def test_sin_ranking_o_sin_mee_no_concluye_nada_ni_borra_lo_guardado():
    ev = {}
    EV.cambio_de_decision(ev, dec(), AHORA, CFG)
    assert EV.cambio_de_decision(ev, Decision("SIN_DATOS", codigo="sin_ranking"), bog(2026, 10, 6, 11), CFG) is None
    assert ev["decision"]["accion"] == CAMBIAR


def test_cambiar_de_candidato_a_menudo_no_marea_con_avisos():
    ev = {}
    assert EV.cambio_de_decision(ev, dec(cand="META"), AHORA, CFG) == "nueva"
    assert EV.cambio_de_decision(ev, dec(cand="AMZN"), bog(2026, 10, 6, 10, 10), CFG) is None         # dentro del enfriamiento
    assert EV.cambio_de_decision(ev, dec(cand="NVDA"), bog(2026, 10, 6, 11, 30), CFG) == "nueva"     # pasada la hora, sí


def test_vender_todo_cuenta_como_recomendacion():
    ev = {}
    assert EV.cambio_de_decision(ev, Decision(VENDER_TODO, codigo="vender_todo"), AHORA, CFG) == "nueva"


def test_el_mensaje_de_decision_esta_en_palabras_sencillas():
    t = F.msg_decision_cambio("nueva", dec(), "TSLA", "08:45")
    assert "ahora recomienda moverte" in t and "cambia a META" in t and "/informe" in t
    sin_jerga(t)
    r = F.msg_decision_cambio("retirada", dec(MANTENER, codigo="ninguno_cumple"), "TSLA")
    assert "ya no recomienda cambiar" in r
    assert "Mantén TSLA" in F.plano(r) or "Por ahora" in F.plano(r)


# =============================== calendario ===============================
def test_avisa_el_reporte_de_hoy_y_el_de_la_proxima_sesion_una_sola_vez():
    ev = {}
    rep = [{"fecha": "2026-10-06", "hora": "amc", "fuente": "finnhub"}, {"fecha": "2026-10-06", "hora": "amc", "fuente": "yfinance"}, {"fecha": "2026-10-20", "hora": "amc"}]
    a = EV.avisos_calendario(ev, "TSLA", rep, AHORA, CFG)
    assert [x["tipo"] for x in a] == ["reporte"] and a[0]["es_hoy"] and a[0]["ticker"] == "TSLA"       # dos fuentes de la misma fecha = un aviso
    assert EV.avisos_calendario(ev, "TSLA", rep, bog(2026, 10, 6, 10, 15), CFG) == []
    manana = EV.avisos_calendario(ev, "TSLA", [{"fecha": "2026-10-07", "hora": "bmo"}], AHORA, CFG)
    assert manana[0]["es_hoy"] is False


def test_avisa_el_evento_macro_del_dia_y_marca_si_la_fecha_esta_por_confirmar():
    ev = {}
    a = EV.avisos_calendario(ev, "TSLA", [], bog(2026, 10, 28, 8, 30), CFG)
    assert [x["tipo"] for x in a] == ["macro"] and "FED" in a[0]["evento"] and EV.avisos_calendario(ev, "TSLA", [], bog(2026, 10, 28, 9), CFG) == []
    dudoso = EV.avisos_calendario({}, "TSLA", [], bog(2026, 10, 14, 8, 30), CFG)
    assert dudoso[0]["verificada"] is False and "por confirmar" in F.plano(F.msg_calendario(dudoso[0]))


def test_avisa_que_el_corte_es_manana_o_hoy_una_sola_vez():
    ev = {}
    a = EV.avisos_calendario(ev, "TSLA", [], bog(2026, 10, 8, 9, 0), CFG)
    assert [x["tipo"] for x in a] == ["corte"] and a[0]["es_hoy"] is False
    t = F.plano(F.msg_calendario(a[0]))
    assert "El corte es mañana" in t and "top 50%" in t and "ROJO se trata como NEGRO" in t
    assert EV.avisos_calendario(ev, "TSLA", [], bog(2026, 10, 8, 12), CFG) == []
    hoy = EV.avisos_calendario({}, "TSLA", [], bog(2026, 10, 9, 9, 0), CFG)
    assert hoy[0]["es_hoy"] is True and "El corte es hoy" in F.plano(F.msg_calendario(hoy[0]))


def test_un_dia_normal_sin_nada_no_genera_avisos_de_calendario():
    assert EV.avisos_calendario({}, "TSLA", [{"fecha": "2026-10-20"}], bog(2026, 10, 6, 9), CFG) == []


def test_el_registro_del_calendario_no_crece_sin_limite():
    ev = {"calendario_avisado": [f"x{i}" for i in range(500)]}
    EV.avisos_calendario(ev, "TSLA", [], AHORA, CFG)
    assert len(ev["calendario_avisado"]) <= 80


# =============================== subida fuerte y base ===============================
def test_la_subida_fuerte_se_avisa_una_vez_al_dia_y_no_con_datos_invalidos():
    ev = {}
    assert not EV.alza_fuerte(ev, 1.5, AHORA, CFG) and not EV.alza_fuerte(ev, float("nan"), AHORA, CFG) and not EV.alza_fuerte(ev, None, AHORA, CFG)
    assert EV.alza_fuerte(ev, 2.4, AHORA, CFG) and not EV.alza_fuerte(ev, 3.0, bog(2026, 10, 6, 14), CFG)
    assert EV.alza_fuerte(ev, 2.4, bog(2026, 10, 7, 10), CFG)                                         # otro día, otro aviso
    t = F.plano(F.msg_alza(mk_res(VERDE, r_hoy=0.047, z=2.4), AHORA))
    assert "sube fuerte hoy: +4,7%" in t and "una subida fuerte no es señal de compra" in t.lower()


def test_el_estado_de_la_base_avisa_solo_cuando_cambia():
    ev = {}
    assert EV.estado_base(ev, "mantener", []) is None
    assert EV.estado_base(ev, "revisar", [{"ticker": "META"}]) == "revisar"
    assert EV.estado_base(ev, "revisar", [{"ticker": "META"}]) is None
    assert EV.estado_base(ev, "revisar", [{"ticker": "AMZN"}]) == "revisar"                          # cambió cuál es la que supera
    assert EV.estado_base(ev, "mantener", []) == "ok"
    assert EV.estado_base(ev, "sin_datos", []) is None and ev["base"]["veredicto"] == "sin_datos"


# =============================== integración con el monitor ===============================
class FuentesConNoticias(FakeF):
    def __init__(self):
        super().__init__()
        self.titulares, self.rep = [], []

    def noticias(self, t, d):
        return [SimpleNamespace(ts=x[1], titulo=x[0], proveedor="Yahoo", idioma="en", resumen="") for x in self.titulares]

    def reportes(self, t, dias_adelante=60):
        return self.rep

    def traducir(self, texto):
        return None


def monitor(tmp_path, f, ahora, secuencia, banco=banco_vacio):
    ctx = S.Contexto(CFG, f, puntuar_vader, "vader", ahora, True, tmp_path / "s.json", salida=lambda s: None)
    return S.correr_monitor(ctx, evaluar=evaluador(secuencia), banco=banco)


def test_el_monitor_avisa_una_noticia_nueva_importante_sin_repetirla(tmp_path):
    f = FuentesConNoticias()
    f.titulares = [("Tesla opens new factory", "2026-10-06T10:00:00+00:00")]
    assert monitor(tmp_path, f, bog(2026, 10, 6, 10, 0), [VERDE]) == []                               # línea base
    f.titulares.append(("Tesla recalls 100,000 vehicles amid investigation, SEC probe", "2026-10-06T14:30:00+00:00"))
    out = monitor(tmp_path, f, bog(2026, 10, 6, 10, 15), [VERDE])
    assert len(out) == 1 and "Noticia nueva de TSLA" in out[0] and "no vendas solo por un titular" in F.plano(out[0])
    assert monitor(tmp_path, f, bog(2026, 10, 6, 10, 30), [VERDE]) == []                             # ya avisada


def test_las_novedades_conviven_con_la_alerta_de_color_sin_duplicar_la_decision(tmp_path):
    est = Estado(CFG, tmp_path / "s.json")
    est.d["rentabilidad"].update(mia=6.0, umbral=11.0)
    est.guardar()
    cands = [Candidato("META", "mgc", 0.50, "IV", corr=0.2, valor_negociado_mm=9999)]
    f = FuentesConNoticias()
    out1 = monitor(tmp_path, f, bog(2026, 10, 6, 10, 0), [VERDE], banco=lambda *a, **kw: cands)
    assert len(out1) == 1 and "ahora recomienda moverte" in out1[0]                                   # verde pero con ranking atrás: la regla recomienda
    assert monitor(tmp_path, f, bog(2026, 10, 6, 10, 15), [VERDE], banco=lambda *a, **kw: cands) == []   # mismo estado: nada nuevo
    out3 = monitor(tmp_path, f, bog(2026, 10, 6, 10, 30), [ROJO], banco=lambda *a, **kw: cands)
    assert len(out3) == 1 and "ALERTA" in out3[0] and "ahora recomienda moverte" not in out3[0]      # la alerta de color ya trae la decisión


def test_el_tope_por_corrida_evita_inundar_el_chat(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["monitor"]["eventos"]["max_por_corrida"] = 1
    est = Estado(cfg, tmp_path / "s.json")
    est.d["rentabilidad"].update(mia=6.0, umbral=11.0)
    est.guardar()
    cands = [Candidato("META", "mgc", 0.50, "IV", corr=0.2, valor_negociado_mm=9999)]
    f = FuentesConNoticias()
    f.rep = [{"fecha": "2026-10-06", "hora": "amc"}]
    ctx = S.Contexto(cfg, f, puntuar_vader, "vader", bog(2026, 10, 6, 10), True, tmp_path / "s.json", salida=lambda s: None)
    assert len(S.correr_monitor(ctx, evaluar=evaluador([mk_res(VERDE, z=2.6)]), banco=lambda *a, **kw: cands)) == 1    # decisión + reporte + alza: sólo el primero


def test_una_novedad_que_falla_no_tumba_el_monitor(tmp_path):
    class Roto(FuentesConNoticias):
        def noticias(self, t, d):
            raise RuntimeError("boom")

        def reportes(self, t, dias_adelante=60):
            raise RuntimeError("boom")
    out = monitor(tmp_path, Roto(), bog(2026, 10, 6, 10), [mk_res(VERDE, z=2.6)])
    assert len(out) == 1 and "sube fuerte" in out[0]                                                  # lo demás siguió funcionando


def test_con_los_eventos_apagados_el_monitor_se_comporta_como_antes(tmp_path):
    cfg = copy.deepcopy(CFG)
    cfg["monitor"]["eventos"]["activo"] = False
    ctx = S.Contexto(cfg, FuentesConNoticias(), puntuar_vader, "vader", bog(2026, 10, 6, 10), True, tmp_path / "s.json", salida=lambda s: None)
    assert S.correr_monitor(ctx, evaluar=evaluador([mk_res(VERDE, z=3.0)]), banco=banco_vacio) == []


def test_el_calendario_se_avisa_en_el_monitor_una_sola_vez_en_el_dia(tmp_path):
    f = FuentesConNoticias()
    f.rep = [{"fecha": "2026-10-06", "hora": "amc"}]
    a = monitor(tmp_path, f, bog(2026, 10, 6, 9, 0), [VERDE])
    assert len(a) == 1 and "TSLA reporta resultados hoy" in a[0]
    assert monitor(tmp_path, f, bog(2026, 10, 6, 9, 15), [VERDE]) == []


def test_alerta_base_avisa_al_cambiar_la_comparacion(tmp_path, monkeypatch):
    import src.seleccion as Q
    filas = [dict(ticker="TSLA", grupo="mgc", es_base=True, mee=0.14, vs_base=1.0, iv=None, real=0.5, liquidez_mm=1e4, supera=False, liquida=True),
             dict(ticker="META", grupo="mgc", es_base=False, mee=0.19, vs_base=1.36, iv=None, real=0.6, liquidez_mm=1e4, supera=True, liquida=True)]
    monkeypatch.setattr(Q, "ranking_base", lambda *a, **k: filas)
    ctx = S.Contexto(CFG, SimpleNamespace(), puntuar_vader, "vader", bog(2026, 10, 6, 19, 30), True, tmp_path / "s.json", salida=lambda s: None)
    m = S.alerta_base(ctx)
    assert m and "Novedad en la comparación de base" in m and "META" in m and "Revisa la base" in F.plano(m)
    assert S.alerta_base(ctx) is None                                                                  # igual que antes: silencio
    filas[1]["supera"] = False
    assert "Ya ninguna acción supera a TSLA" in F.plano(S.alerta_base(ctx))
