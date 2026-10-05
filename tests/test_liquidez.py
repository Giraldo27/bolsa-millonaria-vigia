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
    assert LQ.clasificar(*serie([50000] * 20), CFG)["nivel"] == LQ.BUENA                              # como Ecopetrol
    assert LQ.clasificar(*serie([500] * 19 + [900]), CFG)["nivel"] == LQ.JUSTA                        # como NVDACO: apenas
    tsla = LQ.clasificar(*serie([184] * 20), CFG)
    assert tsla["nivel"] == LQ.MALA and tsla["mediana_mm"] == 184                                     # como TSLACO: unos $184 millones al día
    bloque = LQ.clasificar(*serie([100] * 19 + [90000]), CFG)                                         # un día de bloque enorme no la vuelve líquida
    assert bloque["nivel"] == LQ.MALA and bloque["mediana_mm"] == 100
    huecos = LQ.clasificar(*serie([5000, 0, 5000, 0, 5000] * 4), CFG)                                 # se negocia mucho, pero la mitad de los días no hay nada
    assert huecos["nivel"] == LQ.MALA and huecos["dias_sin_negociar"] == 8


def test_con_historial_incompleto_no_se_inventa_un_nivel():
    nuco = LQ.clasificar(*serie([0] * 19 + [81000]), CFG)                                             # Yahoo no trae el historial de NUCO, pero hoy movió $81.000 M
    assert nuco["nivel"] == LQ.SIN_DATO and nuco["hoy_mm"] == 81000
    assert LQ.clasificar(*serie([0] * 20), CFG)["nivel"] == LQ.MALA                                   # nunca se negocia: mala
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


LIBROS = {"TSLACO.CL": [184] * 20, "ECOPETROL.CL": [50000] * 20, "NVDACO.CL": [500] * 20, "NUCO.CL": [0] * 19 + [81000]}
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
    assert "MALA" in mala and "$ 184 millones al día" in mala and "sería el 11% de lo que se negocia en un día" in mala and "Mejor no operarla" in mala
    justa = LQ.frase(LQ.medir(f, "NVDA", CFG))
    assert "JUSTA" in justa and "orden límite" in justa and "a mercado" in justa
    assert "buena" in LQ.frase(LQ.medir(f, "ECOPETROL", CFG)) and "no tengo un historial fiable" in LQ.frase(LQ.medir(f, "NUCO", CFG))
    assert "Hoy negoció cerca de $ 81.000 millones" in LQ.frase(LQ.medir(f, "NUCO", CFG))


def ctx(tmp_path, f):
    return S.Contexto(CFG, f, puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)


def test_al_registrar_una_compra_se_avisa_la_liquidez_en_trii(tmp_path):
    c = ctx(tmp_path, FuentesLiq(LIBROS, precios=PRECIOS, trm=3200.0))
    t = F.plano(S.resp_texto(c, "compré 20 tesla a 384")["texto"])
    assert "Compra de TSLA registrada" in t and "Liquidez de TSLA en trii: MALA" in t and "Mejor no operarla" in t
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
