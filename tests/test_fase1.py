"""Pruebas de la Fase 1 (sin red): calendario oficial, estado, caché, reintentos, normalización de noticias y opciones."""
import datetime as dt
import json
import os

import numpy as np
import pandas as pd
import pytest

from src import concurso as C
from src.data_sources import (CacheDisco, FuenteError, FuentesDatos, Limitador, Noticia, con_reintentos, deduplicar, elegir_vencimiento,
                              iv_atm, noticias_de_alphavantage, noticias_de_finnhub, noticias_de_yfinance)
from src.notify import enviar, partir
from src.state import Estado, ErrorEstado, universo_permitido
from tests.helpers import make_cfg

CFG = make_cfg()
BOG = dt.timezone(dt.timedelta(hours=-5))


def bog(y, m, d, h=12, mi=0):
    return dt.datetime(y, m, d, h, mi, tzinfo=BOG)


# =============================== calendario ===============================
def test_sesiones_son_23_sin_festivos():
    s = C.sesiones(CFG)
    assert len(s) == 23 and s[0] == dt.date(2026, 10, 5) and s[-1] == dt.date(2026, 11, 6)
    assert dt.date(2026, 10, 12) not in s and dt.date(2026, 11, 2) not in s


def test_semanas_del_concurso():
    assert C.semana_concurso(dt.date(2026, 10, 5), CFG) == 1 and C.semana_concurso(dt.date(2026, 10, 9), CFG) == 1
    assert C.semana_concurso(dt.date(2026, 10, 13), CFG) == 2 and C.semana_concurso(dt.date(2026, 10, 30), CFG) == 4
    assert C.semana_concurso(dt.date(2026, 11, 3), CFG) == 5 and C.semana_concurso(dt.date(2026, 10, 1), CFG) is None
    assert C.clave_semana(dt.date(2026, 10, 20), CFG) == "S3"


def test_horario_bvc_octubre_noviembre_y_festivos():
    assert C.horario(dt.date(2026, 10, 6), CFG) == (dt.time(8, 30), dt.time(15, 0))
    assert C.horario(dt.date(2026, 11, 3), CFG) == (dt.time(9, 30), dt.time(16, 0))
    assert C.horario(dt.date(2026, 10, 12), CFG) is None and C.horario(dt.date(2026, 11, 2), CFG) is None      # festivos
    assert C.horario(dt.date(2026, 10, 10), CFG) is None                                                       # sábado
    assert C.mercado_abierto(bog(2026, 10, 6, 8, 30), CFG) and not C.mercado_abierto(bog(2026, 10, 6, 8, 29), CFG)
    assert not C.mercado_abierto(bog(2026, 10, 6, 15, 0), CFG) and C.mercado_abierto(bog(2026, 11, 3, 15, 59), CFG)


def test_cortes_oficiales_y_final():
    c = C.cortes(CFG)
    assert [(x["fecha"], x["top_pct"]) for x in c[:4]] == [(dt.date(2026, 10, 9), 50), (dt.date(2026, 10, 16), 33),
                                                          (dt.date(2026, 10, 23), 10), (dt.date(2026, 10, 30), 10)]
    assert c[-1]["tipo"] == "final" and c[-1]["fecha"] == dt.date(2026, 11, 6)
    assert C.proximo_corte(dt.date(2026, 10, 9), CFG)["fecha"] == dt.date(2026, 10, 9)          # el día del corte es ese corte
    assert C.proximo_corte(dt.date(2026, 10, 10), CFG)["fecha"] == dt.date(2026, 10, 16)
    assert C.proximo_corte(dt.date(2026, 10, 31), CFG)["tipo"] == "final"
    assert C.proximo_corte(dt.date(2026, 11, 7), CFG) is None


def test_sesiones_restantes_hasta_el_corte():
    assert C.sesiones_restantes(bog(2026, 10, 5, 19, 30), CFG) == 4                 # lun noche: faltan mar-vie
    assert C.sesiones_restantes(bog(2026, 10, 5, 10, 0), CFG) == 5                  # lun en la mañana: cuenta la de hoy
    assert C.sesiones_restantes(bog(2026, 10, 8, 19, 30), CFG) == 1                 # jue noche: sólo falta el viernes
    assert C.sesiones_restantes(bog(2026, 10, 9, 19, 30), CFG) == 4                 # vie noche: ya es el corte 2 (lun 12 festivo)
    assert C.sesiones_restantes(bog(2026, 10, 30, 19, 30), CFG) == 4                # tras el corte clave: 3,4,5,6 de nov


def test_dias_calendario_y_semana_final():
    assert C.dias_calendario_al_corte(bog(2026, 10, 5, 19), CFG) == 4
    assert C.dias_calendario_al_corte(bog(2026, 10, 9, 10), CFG) == 0.5
    assert C.semana_final(bog(2026, 11, 3), CFG) and not C.semana_final(bog(2026, 10, 29), CFG)
    assert C.ultimo_dia(bog(2026, 11, 6), CFG) and C.fase(bog(2026, 10, 4), CFG) == "antes" and C.fase(bog(2026, 11, 9), CFG) == "despues"
    assert C.hora_orden_manana(dt.date(2026, 10, 6), CFG) == "08:45" and C.hora_orden_manana(dt.date(2026, 11, 3), CFG) == "09:45"


# =============================== estado ===============================
@pytest.fixture
def est(tmp_path):
    return Estado(CFG, tmp_path / "state.json")


def test_estado_inicial_97_en_tsla(est):
    assert est.activo == "TSLA" and est.d["posicion"]["monto_cop"] == 97_000_000
    assert est.cambios_usados() == 0 and est.cambios_restantes() == 4 and est.ops_total() == 0


def test_entrada_inicial_cuenta_como_operacion_pero_no_como_cambio(est):
    msg = est.registrar_cambio("TSLA", 97_000_000, 380.5, bog(2026, 10, 5, 8, 45))
    assert "Entrada inicial" in msg and est.cambios_usados() == 0
    assert est.ops_semana(dt.date(2026, 10, 5)) == 1 and est.d["posicion"]["precio_entrada"] == 380.5


def test_cambio_de_activo_limites(est):
    est.registrar_cambio("TSLA", 97e6, 380, bog(2026, 10, 5, 8, 45))
    est.registrar_cambio("META", 95e6, 700, bog(2026, 10, 15, 8, 45))
    assert est.activo == "META" and est.cambios_usados() == 1
    with pytest.raises(ErrorEstado, match="1 cambio por día"):
        est.registrar_cambio("NVDA", 95e6, 200, bog(2026, 10, 15, 9, 30))
    for i, (d, t) in enumerate([(16, "NVDA"), (19, "AMZN"), (20, "GOOGL")]):
        est.registrar_cambio(t, 94e6, 100, bog(2026, 10, d, 8, 45))
    assert est.cambios_usados() == 4 and est.cambios_restantes() == 0
    with pytest.raises(ErrorEstado, match="4 cambios"):
        est.registrar_cambio("MSFT", 94e6, 400, bog(2026, 10, 21, 8, 45))


@pytest.mark.parametrize("t", ["CONCONCRET", "ELCONDOR", "ETB", "FABRICATO", "OCCIDENTE", "POPULAR", "VILLAS"])
def test_lista_negra_nunca_se_registra(est, t):
    with pytest.raises(ErrorEstado, match="LISTA NEGRA"):
        est.registrar_cambio(t, 1e6, 10, bog(2026, 10, 6))
    assert t not in universo_permitido(CFG)


def test_activo_fuera_del_universo_se_rechaza(est):
    with pytest.raises(ErrorEstado, match="universo"):
        est.registrar_cambio("AMD", 1e6, 10, bog(2026, 10, 6))


def test_operaciones_por_semana_y_faltantes(est):
    est.registrar_op(bog(2026, 10, 6, 10))
    est.registrar_op(bog(2026, 10, 7, 10))
    est.registrar_op(bog(2026, 10, 14, 10))
    assert est.ops_semana(dt.date(2026, 10, 8)) == 2 and est.ops_semana(dt.date(2026, 10, 15)) == 1
    assert est.ops_faltantes_semana(dt.date(2026, 10, 8)) == 2 and est.ops_total() == 3 and est.operacion_hoy(dt.date(2026, 10, 7))


def test_guardado_atomico_y_relectura(tmp_path):
    e = Estado(CFG, tmp_path / "s.json")
    e.actualizar_rank(6.5, 11.0, bog(2026, 10, 27, 19, 30), primero=14.2)
    e2 = Estado(CFG, tmp_path / "s.json")
    assert e2.rent["mia"] == 6.5 and e2.rent["umbral"] == 11.0 and e2.rent["primero"] == 14.2
    assert not list(tmp_path.glob("*.tmp"))


def test_estado_danado_da_error_claro(tmp_path):
    (tmp_path / "s.json").write_text("{no es json", encoding="utf-8")
    with pytest.raises(ErrorEstado, match="dañado"):
        Estado(CFG, tmp_path / "s.json")


def test_respaldo_automatico_restaura_la_ultima_version_buena(tmp_path):
    e = Estado(CFG, tmp_path / "s.json")
    e.actualizar_rank(5.0, 10.0, bog(2026, 10, 6))
    e.actualizar_rank(6.0, 11.0, bog(2026, 10, 7))                     # la 2.ª escritura deja la 1.ª como .bak
    (tmp_path / "s.json").write_text("{corrupto", encoding="utf-8")
    e2 = Estado(CFG, tmp_path / "s.json")
    assert e2.recuperado_de_respaldo and e2.rent["mia"] == 5.0


def test_estado_danado_y_respaldo_danado_da_error(tmp_path):
    e = Estado(CFG, tmp_path / "s.json")
    e.guardar(); e.guardar()
    (tmp_path / "s.json").write_text("{x", encoding="utf-8")
    (tmp_path / "s.json.bak").write_text("{y", encoding="utf-8")
    with pytest.raises(ErrorEstado, match="respaldo"):
        Estado(CFG, tmp_path / "s.json")


def test_transaccion_lee_datos_frescos_y_guarda(tmp_path):
    p = tmp_path / "s.json"
    Estado(CFG, p).guardar()
    with Estado.transaccion(CFG, p) as e1:
        e1.registrar_op(bog(2026, 10, 6, 10))
    with Estado.transaccion(CFG, p) as e2:                              # otro proceso: ve lo que guardó el primero
        assert e2.ops_total() == 1
        e2.registrar_op(bog(2026, 10, 7, 10))
    assert Estado(CFG, p).ops_total() == 2 and not list(tmp_path.glob("*.lock"))


def test_transaccion_no_pisa_escrituras_concurrentes(tmp_path):
    import threading
    p = tmp_path / "s.json"
    Estado(CFG, p).guardar()

    def trabajo():
        for _ in range(10):
            with Estado.transaccion(CFG, p) as e:
                e.registrar_op(bog(2026, 10, 6, 10))
    hilos = [threading.Thread(target=trabajo) for _ in range(4)]
    [h.start() for h in hilos]
    [h.join() for h in hilos]
    assert Estado(CFG, p).ops_total() == 40                               # ninguna escritura se perdió


def test_candado_huerfano_se_limpia_y_candado_vivo_da_error(tmp_path):
    import time as _t
    cfg = make_cfg()
    cfg["estado"]["lock_timeout_s"] = 1
    p = tmp_path / "s.json"
    lock = tmp_path / "s.json.lock"
    lock.write_text("x")
    os.utime(lock, (_t.time() - 100, _t.time() - 100))                    # candado viejo (proceso muerto)
    with Estado.transaccion(cfg, p) as e:
        e.registrar_op(bog(2026, 10, 6))
    lock.write_text("x")                                                  # candado reciente (otro proceso lo usa)
    with pytest.raises(ErrorEstado, match="candado"):
        with Estado.transaccion(cfg, p):
            pass


def test_excepcion_dentro_de_la_transaccion_no_guarda_y_libera_el_candado(tmp_path):
    p = tmp_path / "s.json"
    Estado(CFG, p).guardar()
    with pytest.raises(RuntimeError):
        with Estado.transaccion(CFG, p) as e:
            e.d["operaciones"]["total"] = 99
            raise RuntimeError("falla a mitad")
    assert Estado(CFG, p).ops_total() == 0 and not list(tmp_path.glob("*.lock"))


def test_salud_de_fuentes_avisa_caida_una_vez_y_la_recuperacion(est):
    t0 = bog(2026, 10, 6, 10)
    r = [est.registrar_fuente("precios", False, t0 + dt.timedelta(minutes=15 * i), umbral=3, cada_min=60) for i in range(5)]
    assert r == [None, None, "CAIDA", None, None]                        # avisa al 3.er fallo y no repite en la hora
    assert est.registrar_fuente("precios", False, t0 + dt.timedelta(minutes=15 * 8), 3, 60) == "CAIDA"   # pasada la hora, recuerda
    assert est.registrar_fuente("precios", True, t0 + dt.timedelta(hours=3), 3, 60) == "RECUPERADA"
    assert est.registrar_fuente("precios", True, t0 + dt.timedelta(hours=4), 3, 60) is None


def test_alertas_pendientes_se_guardan_con_tope_y_se_entregan(est):
    for i in range(25):
        est.agregar_pendiente(f"msg {i}", bog(2026, 10, 6, 10, i))
    assert len(est.d["alertas_pendientes"]) == CFG["estado"]["alertas_pendientes_max"]
    p = est.tomar_pendientes()
    assert p[-1]["texto"] == "msg 24" and est.d["alertas_pendientes"] == []


def test_enfriamiento_de_alertas(est):
    est.set_semaforo("TSLA", "ROJO", bog(2026, 10, 6, 10), avisado=True)
    assert not est.puede_alertar("TSLA", bog(2026, 10, 6, 10, 20), 30)
    assert est.puede_alertar("TSLA", bog(2026, 10, 6, 10, 31), 30) and est.puede_alertar("NVDA", bog(2026, 10, 6, 10, 1), 30)


def test_rentabilidad_invalida_se_rechaza(est):
    with pytest.raises(ErrorEstado):
        est.actualizar_rank("abc", 5, bog(2026, 10, 6))
    with pytest.raises(ErrorEstado):
        est.actualizar_rank(5, 99999, bog(2026, 10, 6))


def test_semaforo_guarda_color_y_desde(est):
    est.set_semaforo("TSLA", "VERDE", bog(2026, 10, 6, 9), avisado=True)
    d1 = est.d["semaforo"]["TSLA"]["desde"]
    est.set_semaforo("TSLA", "VERDE", bog(2026, 10, 6, 10))
    assert est.d["semaforo"]["TSLA"]["desde"] == d1                      # mismo color: 'desde' no cambia
    est.set_semaforo("TSLA", "ROJO", bog(2026, 10, 6, 11))
    assert est.color_previo("TSLA") == "ROJO" and est.d["semaforo"]["TSLA"]["desde"] != d1


# =============================== reintentos, caché, límites ===============================
def test_reintentos_con_backoff_y_exito():
    esperas, n = [], {"i": 0}

    def f():
        n["i"] += 1
        if n["i"] < 3:
            raise RuntimeError("falla")
        return "ok"
    assert con_reintentos(f, 3, 1.0, 2.0, esperas.append) == "ok" and esperas == [1.0, 2.0]


def test_reintentos_agotados_lanza_fuente_error():
    with pytest.raises(FuenteError, match="boom"):
        con_reintentos(lambda: (_ for _ in ()).throw(RuntimeError("boom")), 2, 0, 1, lambda s: None, "x")


def test_errores_definitivos_no_se_reintentan():
    n = {"i": 0}

    def f():
        n["i"] += 1
        raise FuenteError("clave inválida")
    with pytest.raises(FuenteError):
        con_reintentos(f, 3, 0, 1, lambda s: None)
    assert n["i"] == 1


def test_cache_ttl(tmp_path):
    t = {"now": dt.datetime(2026, 10, 5, 12, tzinfo=dt.timezone.utc)}
    c = CacheDisco(tmp_path, lambda: t["now"])
    c.escribir("k", {"a": 1})
    os.utime(next(tmp_path.glob("*.json")), (t["now"].timestamp(),) * 2)
    assert c.leer("k", 10) == {"a": 1}
    t["now"] += dt.timedelta(minutes=11)
    assert c.leer("k", 10) is None and c.leer_vencida("k") == {"a": 1}
    c.escribir("df", pd.DataFrame({"x": [1, 2]}))
    assert isinstance(c.leer_vencida("df"), pd.DataFrame)


def test_limitador_respeta_n_por_minuto():
    t, dormidos = {"now": 0.0}, []
    lim = Limitador(3, reloj=lambda: t["now"], dormir=lambda s: (dormidos.append(s), t.__setitem__("now", t["now"] + s)))
    for _ in range(3):
        lim.esperar()
    assert dormidos == []
    lim.esperar()                                                          # la 4.ª llamada debe esperar ~60 s
    assert len(dormidos) == 1 and 59 < dormidos[0] < 61


# =============================== noticias ===============================
def test_noticias_finnhub_yfinance_y_alphavantage():
    fh = noticias_de_finnhub([{"datetime": 1790971000, "headline": "Tesla beats", "summary": "s", "source": "Yahoo", "url": "u"},
                              {"datetime": 0, "headline": "", "source": "x"}], "TSLA")
    assert len(fh) == 1 and fh[0].proveedor == "finnhub" and fh[0].ts.endswith("+00:00")
    yf_nuevo = noticias_de_yfinance([{"content": {"title": "Nuevo formato", "pubDate": "2026-10-04T15:00:01Z", "summary": "r",
                                                  "canonicalUrl": {"url": "http://a"}, "provider": {"displayName": "Reuters"}}}], "TSLA")
    yf_viejo = noticias_de_yfinance([{"title": "Formato viejo", "providerPublishTime": 1790971000, "publisher": "AP", "link": "http://b"}], "TSLA")
    assert yf_nuevo[0].fuente == "Reuters" and yf_viejo[0].url == "http://b"
    av = noticias_de_alphavantage({"feed": [{"title": "AV", "time_published": "20261004T150000", "source": "S", "url": "u",
                                             "ticker_sentiment": [{"ticker": "TSLA", "ticker_sentiment_score": "-0.42"}]}]}, "TSLA")
    assert av[0].sentimiento_proveedor == -0.42


def test_deduplicar_ignora_mayusculas_y_signos():
    a = Noticia("2026-10-04T10:00:00+00:00", "Tesla: Recalls 100,000 cars!", proveedor="finnhub")
    b = Noticia("2026-10-04T11:00:00+00:00", "tesla recalls 100 000 cars", proveedor="yfinance")
    c = Noticia("2026-10-04T09:00:00+00:00", "Otra noticia", proveedor="yfinance")
    out = deduplicar([a, b, c])
    assert len(out) == 2 and out[0].ts.startswith("2026-10-04T11")


# =============================== opciones ===============================
def cadena(strikes, iv, bid=1.0, ask=1.1, oi=100):
    return pd.DataFrame({"strike": strikes, "impliedVolatility": iv, "bid": bid, "ask": ask, "openInterest": oi})


def test_iv_atm_promedia_strikes_validos_y_descarta_malos():
    calls = cadena([95, 100, 105], [0.50, 0.40, 0.45])
    puts = cadena([95, 100, 105], [0.52, 0.42, 0.47])
    r = iv_atm(calls, puts, 100.0, 10, 0.5, vecinos=0)
    assert r["calidad"] == "ok" and r["iv"] == pytest.approx(0.41) and r["strikes"] == [100.0]
    malos = cadena([100], [0.40], bid=0.0, ask=0.0)
    assert iv_atm(malos, malos, 100.0, 10, 0.5)["calidad"] == "sin_datos"
    ilíquido = cadena([100], [0.40], oi=1)
    assert iv_atm(ilíquido, ilíquido, 100.0, 10, 0.5)["iv"] is None
    ancho = cadena([100], [0.40], bid=0.1, ask=2.0)
    assert iv_atm(ancho, ancho, 100.0, 10, 0.5)["iv"] is None


def test_elegir_vencimiento_posterior_al_corte():
    ven = ["2026-10-07", "2026-10-09", "2026-10-12", "2026-10-16"]
    assert elegir_vencimiento(ven, dt.date(2026, 10, 9)) == "2026-10-12"
    assert elegir_vencimiento(ven, dt.date(2026, 12, 1)) == "2026-10-16" and elegir_vencimiento([], dt.date(2026, 10, 9)) is None


# =============================== FuentesDatos con fuentes simuladas ===============================
class FakeResp:
    def __init__(self, data, status=200):
        self.data, self.status_code, self.text = data, status, json.dumps(data)

    def json(self):
        return self.data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttp:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def get(self, url, params=None, timeout=None, headers=None):
        self.calls.append((url, dict(params or {})))
        for k, v in self.routes.items():
            if k in url:
                return v(params) if callable(v) else v
        return FakeResp({}, 404)


class FakeYF:
    def __init__(self, df=None, news=None, fallar=0):
        self.df, self._news, self.fallar, self.n_download = df, news or [], fallar, 0
        outer = self

        class Ticker:
            def __init__(s, sym):
                s.sym = sym
                s.fast_info = {"last_price": 100.0, "previous_close": 98.0}
                s.options = ("2026-10-07", "2026-10-12")
                s.news = outer._news

            def option_chain(s, v):
                class Ch:
                    calls, puts = cadena([100], [0.45]), cadena([100], [0.47])
                return Ch()

            def get_earnings_dates(s, limit=8):
                return pd.DataFrame(index=pd.DatetimeIndex(["2026-10-21 16:00"], tz="America/New_York"))
        self.Ticker = Ticker

    def download(self, *a, **k):
        self.n_download += 1
        if self.n_download <= self.fallar:
            raise RuntimeError("Yahoo caído")
        return self.df.copy()


def precios(n=30):
    idx = pd.bdate_range(end="2026-10-02", periods=n)
    return pd.DataFrame({"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1e6}, index=idx)


@pytest.fixture
def ahora_fija():
    return lambda: dt.datetime(2026, 10, 4, 15, tzinfo=dt.timezone.utc)


def fuentes(tmp_path, yf=None, http=None, env=None, ahora=None):
    return FuentesDatos(CFG, env=env if env is not None else {"FINNHUB_API_KEY": "k"}, yf=yf or FakeYF(precios()), http=http or FakeHttp({}),
                        ahora=ahora or (lambda: dt.datetime(2026, 10, 4, 15, tzinfo=dt.timezone.utc)), dormir=lambda s: None, cache_dir=tmp_path)


def test_diario_usa_cache_en_la_segunda_llamada(tmp_path):
    yf = FakeYF(precios())
    f = fuentes(tmp_path, yf)
    a, b = f.diario("TSLA", 20), f.diario("TSLA", 20)
    assert len(a) == 20 and len(b) == 20 and yf.n_download == 1


def test_diario_corrige_ticks_erroneos_y_lo_avisa(tmp_path):
    p = precios(30)
    p.iloc[15, p.columns.get_loc("High")] = 500.0                   # tick absurdo: máximo 5× el precio
    f = fuentes(tmp_path, FakeYF(p))
    d = f.diario("TSLA", 30)
    assert d["High"].max() <= 101.0 and any("ticks erróneos" in a for a in f.avisos)


def test_diario_reintenta_y_se_recupera(tmp_path):
    yf = FakeYF(precios(), fallar=2)
    f = fuentes(tmp_path, yf)
    assert len(f.diario("TSLA", 10)) == 10 and yf.n_download == 3


def test_diario_sin_datos_devuelve_vacio_y_avisa(tmp_path):
    f = fuentes(tmp_path, FakeYF(precios(), fallar=99))
    assert f.diario("TSLA", 10).empty and any("yfinance diario" in a for a in f.avisos)


def test_cache_vencida_como_respaldo_cuando_falla_la_fuente(tmp_path):
    reloj = {"t": dt.datetime(2026, 10, 4, 15, tzinfo=dt.timezone.utc)}
    yf = FakeYF(precios())
    f = fuentes(tmp_path, yf, ahora=lambda: reloj["t"])
    assert len(f.diario("TSLA", 10)) == 10
    for p in tmp_path.glob("*.parquet"):
        os.utime(p, (reloj["t"].timestamp(),) * 2)                       # la caché se escribió "a las 15:00" del reloj simulado
    reloj["t"] += dt.timedelta(hours=5)
    yf.fallar = 99
    assert len(f.diario("TSLA", 10)) == 10 and any("caché vencida" in a for a in f.avisos)


def test_cotizacion_y_opciones_ok_y_locales_sin_opciones(tmp_path):
    f = fuentes(tmp_path)
    assert f.cotizacion("TSLA")["precio"] == 100.0
    o = f.opciones_iv("TSLA", dt.date(2026, 10, 9))
    assert o["vencimiento"] == "2026-10-12" and o["iv"] == pytest.approx(0.46) and o["calidad"] in ("ok", "degradada")
    loc = f.opciones_iv("ECOPETROL", dt.date(2026, 10, 9))
    assert loc["iv"] is None and loc["calidad"] == "sin_datos"


def test_noticias_combina_fuentes_deduplica_y_filtra_por_fecha(tmp_path):
    ahora = int(dt.datetime(2026, 10, 3, 12, tzinfo=dt.timezone.utc).timestamp())
    viejo = int(dt.datetime(2026, 9, 1, 12, tzinfo=dt.timezone.utc).timestamp())
    http = FakeHttp({"company-news": FakeResp([{"datetime": ahora, "headline": "Tesla recall", "source": "A", "url": "u"},
                                               {"datetime": viejo, "headline": "Muy vieja", "source": "A", "url": "u2"}])})
    yf = FakeYF(precios(), news=[{"content": {"title": "TESLA RECALL!", "pubDate": "2026-10-03T13:00:00Z"}},
                                 {"content": {"title": "Otra distinta", "pubDate": "2026-10-04T09:00:00Z"}}])
    n = fuentes(tmp_path, yf, http).noticias("TSLA", 3)
    assert [x.titulo for x in n] == ["Otra distinta", "TESLA RECALL!"] or len(n) == 2
    assert all("Muy vieja" != x.titulo for x in n)


def test_noticias_ecopetrol_usa_adr_y_locales_sin_adr_no_tienen_cobertura(tmp_path):
    http = FakeHttp({"company-news": FakeResp([])})
    f = fuentes(tmp_path, http=http)
    f.noticias("ECOPETROL", 3)
    assert http.calls[0][1]["symbol"] == "EC"
    assert f.noticias("ISA", 3) is None and any("ISA" in a for a in f.avisos)


def test_noticias_sin_clave_de_finnhub_avisa_y_no_inventa(tmp_path):
    f = fuentes(tmp_path, env={})
    assert f.noticias("TSLA", 3) is None and any("FINNHUB_API_KEY" in a for a in f.avisos)


def test_alphavantage_solo_si_todo_lo_demas_esta_vacio_y_con_limite_diario(tmp_path):
    av = {"feed": [{"title": "AV news", "time_published": "20261003T150000", "source": "S", "url": "u"}]}
    http = FakeHttp({"company-news": FakeResp([]), "alphavantage": FakeResp(av)})
    f = fuentes(tmp_path, FakeYF(precios(), news=[]), http, env={"FINNHUB_API_KEY": "k", "ALPHAVANTAGE_API_KEY": "z"})
    n = f.noticias("TSLA", 3)
    assert n and n[0].proveedor == "alphavantage"
    (tmp_path / "av_contador.json").write_text(json.dumps({"fecha": "2026-10-04", "n": 25}))
    f2 = fuentes(tmp_path / "otra", FakeYF(precios(), news=[]), http, env={"FINNHUB_API_KEY": "k", "ALPHAVANTAGE_API_KEY": "z"})
    f2.cache.dir.mkdir(exist_ok=True)
    (f2.cache.dir / "av_contador.json").write_text(json.dumps({"fecha": "2026-10-04", "n": 25}))
    assert f2.noticias("TSLA", 3) is not None and any("límite diario" in a for a in f2.avisos)


YAHOO_RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Tesla recalls 100,000 vehicles</title><link>http://y/1</link><pubDate>Sun, 04 Oct 2026 12:00:00 +0000</pubDate></item>
<item><title>Tesla stock jumps</title><link>http://y/2</link><pubDate>Sun, 04 Oct 2026 10:00:00 +0000</pubDate></item></channel></rss>"""
GOOGLE_RSS = """<?xml version="1.0"?><rss><channel>
<item><title>Ecopetrol enfrenta demanda por derrame - Pulzo</title><link>http://g/1</link><pubDate>Sat, 03 Oct 2026 22:25:41 GMT</pubDate><source>Pulzo</source></item>
<item><title>Ecopetrol sube en bolsa - Portafolio</title><link>http://g/2</link><pubDate>Sat, 03 Oct 2026 20:00:00 GMT</pubDate></item></channel></rss>"""


class FakeRss(FakeResp):
    def __init__(self, xml, status=200):
        super().__init__({}, status)
        self.content = xml.encode()
        self.text = xml


def test_rss_se_parsea_y_separa_el_medio_de_los_titulares_de_google():
    from src.data_sources import noticias_de_rss
    g = noticias_de_rss(GOOGLE_RSS, "ECOPETROL", "google_rss", "es")
    assert g[0].titulo == "Ecopetrol enfrenta demanda por derrame" and g[0].fuente == "Pulzo" and g[0].idioma == "es"
    y = noticias_de_rss(YAHOO_RSS, "TSLA", "yahoo_rss")
    assert len(y) == 2 and y[0].ts.startswith("2026-10-04T12:00") and y[0].proveedor == "yahoo_rss"
    with pytest.raises(RuntimeError):
        noticias_de_rss("no es xml", "TSLA", "yahoo_rss")


def test_respaldo_si_finnhub_cae_se_usa_yahoo_rss(tmp_path):
    http = FakeHttp({"company-news": FakeResp({}, 500), "feeds.finance.yahoo.com": FakeRss(YAHOO_RSS)})
    f = fuentes(tmp_path, http=http)
    n = f.noticias("TSLA", 3)
    assert n is not None and {x.proveedor for x in n} == {"yahoo_rss"} and len(n) == 2
    assert f.proveedores_estado["finnhub"] is False and f.proveedores_estado["yahoo_rss"] is True


def test_respaldo_si_finnhub_y_yahoo_caen_se_usa_google_news(tmp_path):
    g = GOOGLE_RSS.replace("Ecopetrol", "Tesla").replace("2026", "2026")
    http = FakeHttp({"company-news": FakeResp({}, 500), "feeds.finance.yahoo.com": FakeRss("<x", 200), "news.google.com": FakeRss(g)})
    n = fuentes(tmp_path, http=http).noticias("TSLA", 3)
    assert n and all(x.proveedor == "google_rss" for x in n)


def test_si_todas_las_fuentes_caen_noticias_es_none_no_vacio(tmp_path):
    http = FakeHttp({"company-news": FakeResp({}, 500), "feeds.finance.yahoo.com": FakeRss("", 500), "news.google.com": FakeRss("", 500)})
    f = fuentes(tmp_path, http=http)
    assert f.noticias("TSLA", 3) is None and any("Google News RSS" in a for a in f.avisos)


def test_si_las_fuentes_caen_se_usa_la_cache_vencida_de_noticias(tmp_path):
    reloj = {"t": dt.datetime(2026, 10, 4, 15, tzinfo=dt.timezone.utc)}
    ok = int(dt.datetime(2026, 10, 4, 12, tzinfo=dt.timezone.utc).timestamp())
    http = FakeHttp({"company-news": FakeResp([{"datetime": ok, "headline": "Tesla news", "source": "A", "url": "u"}])})
    f = fuentes(tmp_path, http=http, ahora=lambda: reloj["t"])
    assert len(f.noticias("TSLA", 3)) == 1
    for p in tmp_path.glob("*.json"):
        os.utime(p, (reloj["t"].timestamp(),) * 2)
    reloj["t"] += dt.timedelta(minutes=30)                                   # el TTL (10 min) venció y Finnhub cae
    http.routes["company-news"] = FakeResp({}, 500)
    n = f.noticias("TSLA", 3)
    assert n and n[0].titulo == "Tesla news" and any("caché vencida" in a for a in f.avisos)


def test_acciones_locales_usan_google_news_en_espanol_con_su_consulta(tmp_path):
    http = FakeHttp({"news.google.com": FakeRss(GOOGLE_RSS)})
    f = fuentes(tmp_path, http=http)
    n = f.noticias("ISA", 3)
    assert n and all(x.idioma == "es" and x.proveedor == "google_rss" for x in n)
    consulta = http.calls[0][1]["q"]
    assert "ISA Interconexión Eléctrica" in consulta and http.calls[0][1]["gl"] == "CO"
    assert not any("finnhub" in c[0] for c in http.calls)                    # las locales sin ADR no se consultan en Finnhub


def test_local_con_google_sin_titulares_no_es_none(tmp_path):
    http = FakeHttp({"news.google.com": FakeRss("<rss><channel></channel></rss>")})
    assert fuentes(tmp_path, http=http).noticias("GRUPOSURA", 3) == []         # respondió sin titulares ≠ sin datos


def test_reportes_combina_yfinance_y_finnhub(tmp_path):
    http = FakeHttp({"calendar/earnings": FakeResp({"earningsCalendar": [{"date": "2026-10-21", "hour": "amc"}]})})
    r = fuentes(tmp_path, http=http).reportes("TSLA")
    assert len(r) == 1 and r[0]["fecha"] == "2026-10-21" and r[0]["hora"] == "amc" and r[0]["fuente"] == "finnhub"


def test_salud_reporta_cada_fuente(tmp_path):
    http = FakeHttp({"company-news": FakeResp([]), "calendar/earnings": FakeResp({"earningsCalendar": []})})
    s = fuentes(tmp_path, http=http).salud("TSLA")
    assert len(s) == 6 and s[0]["ok"] and any(x["fuente"].startswith("Opciones") for x in s)


# =============================== Telegram ===============================
def test_partir_respeta_el_limite():
    largo = "\n".join(["x" * 100] * 80)
    partes = partir(largo, 1000)
    assert all(len(p) <= 1000 for p in partes) and "".join(partes).replace("\n", "") == largo.replace("\n", "")


def test_enviar_sin_credenciales_devuelve_false():
    assert enviar("hola", CFG, env={}) is False


def test_enviar_no_filtra_el_token_en_los_errores(capsys):
    class Http:
        def post(self, *a, **k):
            raise RuntimeError("falló https://api.telegram.org/botSECRETO123/sendMessage")
    assert enviar("hola", CFG, env={"TELEGRAM_BOT_TOKEN": "SECRETO123", "TELEGRAM_CHAT_ID": "1"}, http=Http()) is False
    assert "SECRETO123" not in capsys.readouterr().err
