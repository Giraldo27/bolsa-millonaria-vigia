"""Pruebas de la liquidez REAL en trii: se mide con lo que se negocia en la Bolsa de Colombia (no en Nueva York) y se revisa antes de dar un resultado."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import formato as F
from src import liquidez as LQ
from src import servicio as S
from src.sentimiento import puntuar_vader
from tests.helpers import make_cfg
from tests.test_cartera import FuentesCartera
from tests.test_fase4 import bog

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)


def serie(valores_mm, precio=1000.0):
    """Valor negociado diario (millones de pesos) → (valor en pesos, volumen en acciones)."""
    v = pd.Series(np.array(valores_mm, dtype=float) * 1e6, index=pd.bdate_range(end="2026-10-05", periods=len(valores_mm)))
    return v, v / precio


def test_trii_negocia_las_globales_en_su_propio_libro_no_en_nueva_york():
    assert LQ.simbolo_trii("TSLA", CFG) == "TSLACO.CL" and LQ.simbolo_trii("NVDA", CFG) == "NVDACO.CL" and LQ.simbolo_trii("NUCO", CFG) == "NUCO.CL"
    assert LQ.simbolo_trii("ECOPETROL", CFG) == "ECOPETROL.CL" and LQ.simbolo_trii("PFBCOLOM", CFG) == "PFCIBEST.CL"
    assert LQ.simbolo_trii("NO_EXISTE", CFG) is None


def test_los_niveles_usan_la_mediana_y_los_dias_sin_negociar():
    assert LQ.clasificar(*serie([50000] * 60), CFG)["nivel"] == LQ.BUENA                              # como Ecopetrol
    assert LQ.clasificar(*serie([500] * 19 + [900]), CFG)["nivel"] == LQ.JUSTA                        # como METACO: apenas
    tsla = LQ.clasificar(*serie([184] * 20), CFG)
    assert tsla["nivel"] == LQ.MALA and tsla["mediana_mm"] == 184                                     # como TSLACO: unos $184 millones al día
    bloque = LQ.clasificar(*serie([100] * 19 + [90000]), CFG)                                         # un día de bloque enorme no la vuelve líquida
    assert bloque["nivel"] == LQ.MALA and bloque["mediana_mm"] == 100
    huecos = LQ.clasificar(*serie([5000, 0, 5000, 0, 5000] * 4), CFG)                                 # se negocia mucho, pero casi la mitad de los días no hay nada
    assert huecos["nivel"] == LQ.MALA and huecos["dias_sin_negociar"] == 8


def test_el_filtro_endurecido_exige_cuatro_cosas_para_decir_buena():
    """6-oct-2026: el usuario terminó con acciones poco líquidas que el sistema había aprobado. Buena = día normal ≥ 3.000, estable en 3 meses, días flojos ≥ 1.500, cero días sin negociar."""
    assert LQ.clasificar(*serie([3500] * 60), CFG)["nivel"] == LQ.BUENA
    apenas = LQ.clasificar(*serie([2500] * 60), CFG)                                                  # con el umbral anterior (2.000) pasaba
    assert apenas["nivel"] == LQ.JUSTA and "pido 3.000" in apenas["motivos"][0]
    moda = LQ.clasificar(*serie([800] * 40 + [5000] * 20), CFG)                                       # sólo lleva un mes negociándose bien
    assert moda["nivel"] == LQ.JUSTA and "puede no durar" in " ".join(moda["motivos"])
    irregular = LQ.clasificar(*serie([9000, 9000, 700] * 20), CFG)                                    # día normal alto, pero uno de cada tres días casi no hay nadie
    assert irregular["nivel"] == LQ.JUSTA and "días flojos" in " ".join(irregular["motivos"])
    un_hueco = LQ.clasificar(*serie([9000] * 59 + [0]), CFG)
    assert un_hueco["nivel"] == LQ.JUSTA and "no se negoció" in " ".join(un_hueco["motivos"])         # antes se toleraba un día sin negociar
    c = CFG["liquidez"]
    assert c["buena_cop_mm"] == 3000 and c["max_dias_sin_negociar"] == 0 and c["buena_60_cop_mm"] == 2000 and c["buena_dia_flojo_cop_mm"] == 1500


def test_sin_volumen_fiable_es_sin_dato_nunca_mala_y_nunca_apta():
    """Caso real: el bot dijo 'NUCO: MALA, 20 de 20 días sin negociar' cuando NUCO era de las más negociadas; Yahoo simplemente no trae su volumen."""
    nuco = LQ.clasificar(*serie([0] * 20), CFG)
    assert nuco["nivel"] == LQ.SIN_DATO and nuco["mediana_mm"] is None
    f = LQ.frase(dict(nuco, ticker="NUCO"))
    assert "NO LA PUEDO MEDIR" in f and "MALA" not in f and "No significa que sea mala" in f and "no la recomiendo" in f
    assert LQ.veredicto(nuco, es_bvc=False)[0] is False and "no puedo comprobar" in LQ.veredicto(nuco, False)[1]
    buena = LQ.clasificar(*serie([50000] * 60), CFG)
    assert LQ.veredicto(buena, es_bvc=True) == (True, "✅ APTA: pasa el filtro de liquidez (y es de la BVC).")
    assert LQ.veredicto(buena, es_bvc=False)[0] is False and LQ.veredicto(LQ.clasificar(*serie([184] * 60), CFG), True)[0] is False


def test_con_historial_incompleto_no_se_inventa_un_nivel():
    nuco = LQ.clasificar(*serie([0] * 19 + [81000]), CFG)                                             # Yahoo no trae el historial de NUCO, pero hoy movió $81.000 M
    assert nuco["nivel"] == LQ.SIN_DATO and nuco["hoy_mm"] == 81000
    assert LQ.clasificar(*serie([0] * 20), CFG)["nivel"] == LQ.SIN_DATO                               # todo en cero: o no se negocia o no hay dato; no se puede saber
    assert LQ.clasificar(*serie([5000] * 4), CFG)["nivel"] == LQ.SIN_DATO                             # muy pocos días


class FuentesLiq(FuentesCartera):
    """Precios de siempre + el libro local de cada acción en Colombia."""
    def __init__(self, libros, **k):
        super().__init__(**k)
        self.pedidos = []

        def download(sym, **kw):
            self.pedidos.append(sym)
            if sym not in libros:
                return pd.DataFrame()
            v, vol = serie(libros[sym])
            return pd.DataFrame({"Close": 1000.0, "Volume": vol})
        self.yf = SimpleNamespace(Ticker=self.yf.Ticker, download=download)


LIBROS = {"TSLACO.CL": [184] * 60, "ECOPETROL.CL": [50000] * 60, "NVDACO.CL": [500] * 60, "NUCO.CL": [0] * 60}
PRECIOS = {"TSLA": (384.0, 380.5), "ECOPETROL": (2775.0, 2760.0), "NVDA": (240.0, 238.0), "NUCO": (15.2, 13.4), "META": (792.0, 790.0)}


def test_medir_consulta_el_libro_de_colombia_y_nunca_falla():
    f = FuentesLiq(LIBROS)
    assert LQ.medir(f, "TSLA", CFG)["nivel"] == LQ.MALA and f.pedidos == ["TSLACO.CL"]                # NO el símbolo de Nueva York
    assert LQ.medir(f, "ECOPETROL", CFG)["nivel"] == LQ.BUENA and LQ.medir(f, "NVDA", CFG)["nivel"] == LQ.JUSTA
    assert LQ.medir(f, "META", CFG)["nivel"] == LQ.SIN_DATO                                           # Yahoo devolvió vacío: no se inventa
    assert LQ.medir(FuentesCartera(), "TSLA", CFG)["nivel"] == LQ.SIN_DATO                            # fuente sin esa consulta: tampoco se cae


def test_la_frase_dice_cuanto_se_negocia_y_que_hacer():
    f = FuentesLiq(LIBROS)
    mala = LQ.frase(LQ.medir(f, "TSLA", CFG), 20e6)
    assert "MALA" in mala and "$ 184 millones al día (pido 3.000)" in mala and "serían el 11% de lo que se negocia en un día" in mala and "No la operes" in mala
    assert "Es una posición grande para esta acción" in LQ.frase(LQ.medir(f, "TSLA", CFG), 20e6, CFG)
    justa = LQ.frase(LQ.medir(f, "NVDA", CFG))
    assert "JUSTA" in justa and "orden límite" in justa and "a mercado" in justa and "No la recomiendo" in justa
    assert "buena" in LQ.frase(LQ.medir(f, "ECOPETROL", CFG)) and "NO LA PUEDO MEDIR" in LQ.frase(LQ.medir(f, "NUCO", CFG))


def ctx(tmp_path, f):
    return S.Contexto(CFG, f, puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)


def test_al_registrar_una_compra_se_avisa_la_liquidez_en_trii(tmp_path):
    c = ctx(tmp_path, FuentesLiq(LIBROS, precios=PRECIOS, trm=3200.0))
    t = F.plano(S.resp_texto(c, "compré 20 tesla a 384")["texto"])
    assert "Compra de TSLA registrada" in t and "Liquidez de TSLA en trii: MALA" in t and "No la operes" in t
    e = F.plano(S.resp_texto(c, "compré 10000 ecopetrol a 2775")["texto"])
    assert "Liquidez de ECOPETROL en trii: buena" in e


def test_la_cartera_y_el_semaforo_recuerdan_lo_que_se_negocia_poco_y_callan_lo_que_esta_bien(tmp_path, monkeypatch):
    from tests.test_fase4 import mk_res
    from src.semaforo import VERDE
    c = ctx(tmp_path, FuentesLiq(LIBROS, precios=PRECIOS, trm=3200.0))
    for frase in ("compré 20 tesla a 384", "compré 10000 ecopetrol a 2775", "compré 24 nvda a 240", "compré 925 nuco a 15,2"):
        S.resp_texto(c, frase)
    t = F.plano(S.resp_cartera(c))
    assert "Liquidez de TSLA en trii: MALA" in t and "Liquidez de NVDA en trii: JUSTA" in t
    assert "Liquidez de ECOPETROL" not in t and "Liquidez de NUCO" not in t                           # lo bueno y lo que no se puede medir no hacen ruido
    monkeypatch.setattr(S, "evaluar_activo", lambda f, tk, *a, **k: (mk_res(VERDE, ticker=tk), None, {}))
    s = F.plano(S.resp_semaforo(c))
    assert "TSLA: VERDE" in s and "Liquidez de TSLA en trii: MALA" in s and s.count("Liquidez de") == 2


def test_una_accion_de_la_bvc_con_dias_sin_negociar_no_entra_a_las_recomendaciones():
    from src.relevo import Candidato, liquidez_ok
    lq = LQ.clasificar(*serie([5000, 0, 5000, 0, 5000] * 4), CFG)
    valor = lq["mediana_mm"] if lq["dias_sin_negociar"] <= CFG["liquidez"]["max_dias_sin_negociar"] else 0.0   # la misma regla de construir_candidato
    assert not liquidez_ok(Candidato("X", "local", 0.05, valor_negociado_mm=valor), CFG)
    assert liquidez_ok(Candidato("ECOPETROL", "local", 0.05, valor_negociado_mm=50000), CFG)
    assert CFG["liquidez"]["buena_cop_mm"] == CFG["banco"]["liquidez_min_cop_mm_locales"] == CFG["noticias_bvc"]["recomendacion"]["liquidez_min_cop_mm"]


def test_que_comprar_muestra_la_liquidez_y_pide_orden_limite():
    from src.regla_maestra import Decision, SIN_DATOS
    from src.relevo import Candidato, EntradaBanco
    import datetime as dt
    banco = [EntradaBanco(Candidato("ECOPETROL", "local", 0.05, "σ20", valor_negociado_mm=50000))]
    t = F.plano(F.msg_que_comprar(banco, "NUCO", 0.04, Decision(SIN_DATOS, codigo="sin_ranking"), dict(fecha=dt.date(2026, 10, 9), top_pct=50), 4, [], "08:45", CFG,
                                  liquidez_txt="💧 Liquidez de ECOPETROL en trii: buena (se negocian unos $ 50.000 millones al día)."))
    assert "Liquidez de ECOPETROL en trii: buena" in t and "siempre con orden límite" in t and 'nunca "a mercado"' in t and "liquidez revisada" in t


def test_el_comando_liquidez_aprueba_o_rechaza_antes_de_comprar(tmp_path):
    c = ctx(tmp_path, FuentesLiq(LIBROS, precios=PRECIOS, trm=3200.0))
    t = F.plano(S.resp_liquidez(c, ["tesla"]))
    assert "TSLA — ⛔ NO APTA: no pasa el filtro de liquidez en trii." in t and "MALA" in t and "día normal de al menos $ 3.000 millones, estable en 3 meses" in t
    assert "ECOPETROL — ✅ APTA" in F.plano(S.resp_liquidez(c, ["ecopetrol"]))
    assert "NUCO — ⛔ NO APTA: no puedo comprobar su liquidez" in F.plano(S.resp_liquidez(c, ["nuco"]))
    assert "Cómo usar /liquidez" in S.resp_liquidez(c, []) and "❌" in S.resp_liquidez(c, ["fabricato"])
    S.resp_texto(c, "compré 20 tesla a 384")
    S.resp_texto(c, "compré 10000 ecopetrol a 2775")
    todo = F.plano(S.resp_liquidez(c, []))                                                           # sin argumento: revisa tu cartera
    assert "TSLA — ⛔" in todo and "ECOPETROL — ✅ APTA" in todo
    assert S.resp_texto(c, "¿puedo comprar tesla?") == {"consulta": "liquidez", "ticker": "TSLA"}


# =============================== segunda fuente: lo negociado en la Bolsa de Colombia ===============================
import datetime as _dt

from src import liquidez as _LQ

_CFG_L = {"liquidez": dict(dias=20, buena_cop_mm=3000, buena_60_cop_mm=2000, buena_dia_flojo_cop_mm=1500, max_dias_sin_negociar=0, justa_cop_mm=500,
                           orden_max_pct=0.02, libro_bvc=dict(activo=True, concentrado_veces=2.5))}
_CERRADA = _dt.datetime(2026, 10, 6, 23, 0, tzinfo=_dt.timezone.utc)                    # 6 p. m. en Bogotá: bolsa cerrada
_ABIERTA = _dt.datetime(2026, 10, 6, 15, 0, tzinfo=_dt.timezone.utc)                    # 10 a. m. en Bogotá: bolsa abierta


def _fila(p10, p30, p60, ult, acciones=1000):
    return dict(simbolo="X", precio=1000.0, acciones_ult=acciones, valor_ult_mm=ult, prom10_mm=p10, prom30_mm=p30, prom60_mm=p60)


def _sin_dato():
    return dict(ticker="NUCO", simbolo="NUCO.CL", nivel=_LQ.SIN_DATO, mediana_mm=None, hoy_mm=None, acciones_hoy=None, dias_sin_negociar=None, dias=0)


def test_lo_que_no_se_podia_medir_ahora_se_mide_con_la_bolsa_y_con_prudencia():
    """Caso real: NUCO promedia mucho, pero en 2 semanas negoció 2,6 veces lo de 3 meses y su última sesión fue floja → no se aprueba."""
    l = _LQ.cruzar(_sin_dato(), _fila(17855, 9700, 6908, 826, 16746), _CFG_L, _CERRADA)
    assert l["nivel"] == _LQ.JUSTA and l["fuente"] == "bvc" and len(l["motivos"]) == 2
    t = _LQ.frase(l)
    assert "JUSTA" in t and "2,6 veces" in t and "última sesión: 16.746 acciones, $ 826 millones" in t and "No la recomiendo" in t
    assert _LQ.veredicto(l, es_bvc=False)[0] is False


def test_con_la_bolsa_solo_se_aprueba_lo_estable_y_nunca_se_sube_lo_ya_medido():
    assert _LQ.cruzar(_sin_dato(), _fila(4000, 4200, 3900, 3500), _CFG_L, _CERRADA)["nivel"] == _LQ.BUENA
    assert _LQ.cruzar(_sin_dato(), _fila(300, 280, 260, 200), _CFG_L, _CERRADA)["nivel"] == _LQ.MALA
    mala = dict(_sin_dato(), nivel=_LQ.MALA, mediana_mm=180.0, motivos=["poco"])
    assert _LQ.cruzar(mala, _fila(9000, 9000, 9000, 9000), _CFG_L, _CERRADA)["nivel"] == _LQ.MALA      # la bolsa no "rescata" una acción que ya se midió mala
    assert _LQ.cruzar(_sin_dato(), None, _CFG_L)["nivel"] == _LQ.SIN_DATO                                # sin la segunda fuente todo sigue igual


def test_si_las_dos_fuentes_no_coinciden_gana_la_mas_prudente():
    buena = dict(_sin_dato(), nivel=_LQ.BUENA, mediana_mm=3200.0, motivos=[])
    l = _LQ.cruzar(buena, _fila(1200, 1500, 1800, 900), _CFG_L, _CERRADA)
    assert l["nivel"] == _LQ.JUSTA and "me quedo con la más prudente" in l["motivos"][0]
    assert _LQ.cruzar(buena, _fila(2300, 6400, 4700, 20), _CFG_L, _ABIERTA)["nivel"] == _LQ.BUENA


def test_con_la_bolsa_abierta_lo_negociado_hoy_no_se_compara_con_un_dia_completo():
    assert _LQ.en_sesion(_ABIERTA) and not _LQ.en_sesion(_CERRADA)
    assert _LQ.en_sesion(_dt.datetime(2026, 10, 6, 13, 45, tzinfo=_dt.timezone.utc))                     # 8:45 de Bogotá: en octubre la bolsa abre 8:30
    assert _LQ.cruzar(_sin_dato(), _fila(4000, 4200, 3900, 50), _CFG_L, _ABIERTA)["nivel"] == _LQ.BUENA
    assert _LQ.cruzar(_sin_dato(), _fila(4000, 4200, 3900, 50), _CFG_L, _CERRADA)["nivel"] == _LQ.JUSTA


def test_el_simbolo_de_la_bolsa_se_encuentra_con_o_sin_el_co_final():
    tabla = {"NU": _fila(1, 1, 1, 1), "TSLACO": _fila(2, 2, 2, 2), "META": _fila(3, 3, 3, 3), "PFCIBEST": _fila(4, 4, 4, 4)}
    f = SimpleNamespace(libro_bvc=lambda: tabla)
    assert {t: _LQ.fila_libro(f, t, CFG)["prom10_mm"] for t in ("NUCO", "TSLA", "META", "PFBCOLOM")} == {"NUCO": 1, "TSLA": 2, "META": 3, "PFBCOLOM": 4}
    assert _LQ.fila_libro(f, "ECOPETROL", CFG) is None                                    # no está en la tabla
    assert _LQ.fila_libro(SimpleNamespace(), "NUCO", CFG) is None                         # fuente sin libro (pruebas antiguas): no se cruza nada
