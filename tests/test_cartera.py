"""Pruebas de la cartera de varias acciones: /compra, /venta, /cartera, vigilancia de cada posición y Regla Maestra con el conjunto."""
import copy
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import bot
from src import cartera as CA
from src import formato as F
from src import servicio as S
from src.regla_maestra import CAMBIAR, ContextoRegla, regla_maestra
from src.relevo import Candidato
from src.semaforo import AMARILLO, ROJO, VERDE
from src.sentimiento import puntuar_vader
from src.state import Estado, ErrorEstado
from tests.helpers import make_cfg
from tests.test_fase4 import FAKE_BOT, banco_vacio, bog, mk_res, sin_jerga, update_de

CFG = make_cfg()
AHORA = bog(2026, 10, 5, 9, 0)
TRM = 3900.0


def estado(tmp_path):
    return Estado(CFG, tmp_path / "s.json")


# =============================== estado ===============================
def test_las_tres_compras_del_usuario_se_registran_como_cartera_sin_gastar_cambios(tmp_path):
    e = estado(tmp_path)
    e.registrar_compra("TSLA", 55, 380.5, AHORA, 55 * 380.5 * TRM)
    e.registrar_compra("META", 8, 720.5, AHORA, 8 * 720.5 * TRM)
    e.registrar_compra("GRUPOARGOS", 500, 21500, AHORA, 500 * 21500)
    assert {p["ticker"] for p in e.d["cartera"]} == {"TSLA", "META", "GRUPOARGOS"} and e.tenidos() == {"TSLA", "META", "GRUPOARGOS"}
    assert e.cambios_usados() == 0 and e.ops_total() == 3 and e.ops_semana(AHORA.date()) == 3                     # tres operaciones de actividad, ningún cambio
    assert e.activo == "TSLA" and e.d["posicion"]["precio_entrada"] == 380.5                                       # la principal es la de mayor monto


def test_la_principal_cambia_si_otra_posicion_pasa_a_ser_la_mayor(tmp_path):
    e = estado(tmp_path)
    e.registrar_compra("TSLA", 10, 380, AHORA, 10 * 380 * TRM)
    e.registrar_compra("GRUPOARGOS", 5000, 21500, AHORA, 5000 * 21500)
    assert e.activo == "GRUPOARGOS"


def test_comprar_dos_veces_la_misma_accion_promedia_el_precio(tmp_path):
    e = estado(tmp_path)
    e.registrar_compra("META", 8, 700, AHORA, 8 * 700 * TRM)
    e.registrar_compra("META", 2, 800, AHORA, 2 * 800 * TRM)
    p = e.d["cartera"][0]
    assert len(e.d["cartera"]) == 1 and p["cantidad"] == 10 and p["precio"] == pytest.approx(720.0) and p["monto_cop"] == pytest.approx(10 * 720 * TRM, rel=1e-6)


def test_no_se_pueden_registrar_la_lista_negra_ni_activos_desconocidos_ni_cifras_invalidas(tmp_path):
    e = estado(tmp_path)
    with pytest.raises(ErrorEstado, match="LISTA NEGRA"):
        e.registrar_compra("ETB", 10, 100, AHORA)
    with pytest.raises(ErrorEstado, match="universo"):
        e.registrar_compra("AMD", 10, 100, AHORA)
    for c, p in ((0, 10), (5, 0), (-1, 10)):
        with pytest.raises(ErrorEstado):
            e.registrar_compra("TSLA", c, p, AHORA)
    assert e.d["cartera"] == [] and e.ops_total() == 0


def test_venta_parcial_y_total(tmp_path):
    e = estado(tmp_path)
    e.registrar_compra("TSLA", 55, 380, AHORA, 55 * 380 * TRM)
    e.registrar_compra("META", 8, 720, AHORA, 8 * 720 * TRM)
    e.registrar_venta("TSLA", 20, AHORA)
    t = next(p for p in e.d["cartera"] if p["ticker"] == "TSLA")
    assert t["cantidad"] == 35 and t["monto_cop"] == pytest.approx(55 * 380 * TRM * 35 / 55) and e.ops_total() == 3
    e.registrar_venta("META", None, AHORA)
    assert e.tenidos() == {"TSLA"}
    e.registrar_venta("TSLA", 1000, AHORA)                                                                        # más de lo que tiene = vende todo
    assert e.d["cartera"] == [] and e.d["posicion"]["precio_entrada"] is None and e.tenidos() == set()
    with pytest.raises(ErrorEstado, match="no está en tu cartera"):
        e.registrar_venta("TSLA", None, AHORA)


def test_un_cambio_con_pos_reemplaza_toda_la_cartera_y_gasta_un_cambio(tmp_path):
    e = estado(tmp_path)
    e.registrar_compra("TSLA", 55, 380, AHORA, 21e6)
    e.registrar_compra("META", 8, 720, AHORA, 22e6)
    e.registrar_cambio("AMZN", 95e6, 230.0, bog(2026, 10, 6, 9))
    assert e.tenidos() == {"AMZN"} and e.cambios_usados() == 1 and e.d["cartera"][0]["cantidad"] is None


def test_un_state_viejo_sin_cartera_sigue_funcionando(tmp_path):
    import json
    e = estado(tmp_path)
    d = e.copia()
    d.pop("cartera")
    (tmp_path / "s.json").write_text(json.dumps(d), encoding="utf-8")
    e2 = estado(tmp_path)
    assert e2.d["cartera"] == [] and e2.tenidos() == set()


# =============================== valoración y TRM ===============================
class FuentesCartera:
    def __init__(self, precios=None, trm=TRM, sigmas=None):
        self.precios = precios or {"TSLA": (390.25, 380.5), "META": (730.0, 720.5), "GRUPOARGOS": (21600.0, 21500.0)}
        self.avisos, self.trm_valor, self.sigmas = [], trm, sigmas or {}
        self.yf = SimpleNamespace(Ticker=lambda s: SimpleNamespace(fast_info={"last_price": self.trm_valor}))

    def _con_cache(self, clave, ttl, fn, nombre):
        return fn()

    def cotizacion(self, t):
        p = self.precios.get(t)
        return {"precio": p[0], "cierre_previo": p[1], "fuente": "falsa"} if p else None

    def diario(self, t, dias=None):
        n = 80
        r = self.sigmas.get(t, 0.02) * np.array([(-1) ** i for i in range(n)])
        return pd.DataFrame({"Close": 100 * np.cumprod(1 + r)}, index=pd.bdate_range(end="2026-10-02", periods=n))

    def opciones_iv(self, t, fecha):
        return {"calidad": "sin_datos"}

    def traducir(self, t):
        return None


def cartera_de(e):
    return e.d["cartera"]


def estado_con_tres(tmp_path):
    e = estado(tmp_path)
    e.registrar_compra("TSLA", 55, 380.5, AHORA, 55 * 380.5 * TRM)
    e.registrar_compra("META", 8, 720.5, AHORA, 8 * 720.5 * TRM)
    e.registrar_compra("GRUPOARGOS", 500, 21500, AHORA, 500 * 21500)
    return e


def test_valorar_usa_la_trm_para_acciones_de_ee_uu_y_pesos_para_la_bvc(tmp_path):
    e = estado_con_tres(tmp_path)
    filas = {x["ticker"]: x for x in CA.valorar(FuentesCartera(), cartera_de(e), CFG)}
    assert filas["TSLA"]["valor_cop"] == pytest.approx(55 * 390.25 * TRM) and filas["GRUPOARGOS"]["valor_cop"] == pytest.approx(500 * 21600)
    assert filas["TSLA"]["rent"] == pytest.approx(390.25 / 380.5 - 1) and filas["META"]["moneda"] == "USD" and filas["GRUPOARGOS"]["moneda"] == "COP"
    assert sum(x["peso"] for x in filas.values()) == pytest.approx(1.0) and filas["TSLA"]["peso"] > filas["META"]["peso"] > filas["GRUPOARGOS"]["peso"]
    total = CA.rent_total(list(filas.values()))
    costo = sum(p["monto_cop"] for p in cartera_de(e))
    assert total == pytest.approx(sum(x["valor_cop"] for x in filas.values()) / costo - 1)


def test_una_accion_sin_cotizacion_no_se_inventa(tmp_path):
    e = estado_con_tres(tmp_path)
    filas = CA.valorar(FuentesCartera(precios={"TSLA": (390.0, 380.5)}), cartera_de(e), CFG)
    meta = next(x for x in filas if x["ticker"] == "META")
    assert meta["precio"] is None and meta["valor_cop"] != meta["valor_cop"] and meta["rent"] is None
    assert CA.rent_total(filas) is None                                                                          # con datos incompletos no hay rentabilidad total
    assert "sin precio ahora" in F.plano(F.msg_cartera(filas, None, TRM))


def test_la_trm_rechaza_ticks_erroneos_y_usa_el_ultimo_cierre_guardado(monkeypatch):
    assert CA.trm(FuentesCartera(trm=3900.0), CFG) == 3900.0
    monkeypatch.setattr("src.data_loader.load_macro", lambda n, cfg: pd.DataFrame({"Close": [3800.0, 3810.0]}))
    assert CA.trm(FuentesCartera(trm=39.0), CFG) == 3810.0                                                       # ×0,01 (error conocido de Yahoo)
    assert CA.trm(FuentesCartera(trm=39000.0), CFG) == 3810.0                                                    # ×10
    monkeypatch.setattr("src.data_loader.load_macro", lambda n, cfg: pd.DataFrame({"Close": [50.0]}))
    assert CA.trm(FuentesCartera(trm=39.0), CFG) is None


def test_monto_en_pesos_segun_la_moneda():
    assert CA.monto_cop("META", 8, 720.5, TRM, CFG) == pytest.approx(8 * 720.5 * TRM)
    assert CA.monto_cop("GRUPOARGOS", 500, 21500, None, CFG) == 500 * 21500                                      # la BVC no necesita TRM
    assert CA.monto_cop("META", 8, 720.5, None, CFG) is None


def test_el_movimiento_esperado_de_la_cartera_es_menor_que_el_de_su_acccion_mas_volatil(tmp_path):
    e = estado_con_tres(tmp_path)
    f = FuentesCartera(sigmas={"TSLA": 0.03, "META": 0.02, "GRUPOARGOS": 0.02})
    filas = CA.valorar(f, cartera_de(e), CFG)
    mee, met = CA.mee_cartera(f, filas, AHORA, CFG)
    from src import concurso as Cc
    mee_tsla = 0.03 * np.sqrt(252) * np.sqrt(Cc.dias_calendario_al_corte(AHORA, CFG) / 365)
    mee_menor = 0.02 * np.sqrt(252) * np.sqrt(Cc.dias_calendario_al_corte(AHORA, CFG) / 365)
    assert met == "σ20 de la cartera" and mee_menor * 0.95 < mee < mee_tsla                                       # entre el menos y el más volátil (aquí se mueven juntas: promedio ponderado)
    assert CA.mee_cartera(f, filas[:1], AHORA, CFG) == (None, "sin_datos")                                        # con una sola acción no hay "cartera"


# =============================== Regla Maestra con cartera ===============================
def cand(t, mee, **k):
    return Candidato(t, "mgc", mee, "σ20", corr=0.2, valor_negociado_mm=9999, **k)


def test_lo_que_ya_tienes_no_se_recomienda_como_cambio():
    ctx = dict(mia=6.0, objetivo=11.0, activo="TSLA", color_actual=VERDE, mee_actual=0.065, cambios_restantes=4, cambio_hoy=False)
    cs = [cand("META", 0.20), cand("AMZN", 0.09)]
    d, banco = regla_maestra(ContextoRegla(**ctx), cs, CFG)
    assert d.candidato == "META"
    d2, banco2 = regla_maestra(ContextoRegla(**ctx, tenidos=frozenset({"META"})), cs, CFG)
    assert d2.accion == CAMBIAR and d2.candidato == "AMZN" and "META" not in [x.c.ticker for x in banco2]


def test_decidir_compara_contra_el_movimiento_esperado_del_conjunto(tmp_path):
    from src.construir import decidir
    e = estado_con_tres(tmp_path)
    e.d["rentabilidad"].update(mia=6.0, umbral=11.0)
    f = FuentesCartera(sigmas={"TSLA": 0.03, "META": 0.015, "GRUPOARGOS": 0.01})
    f.opciones_iv = lambda t, fecha: {"iv": 0.9, "calidad": "ok", "vencimiento": "2026-10-12"}
    ent = SimpleNamespace(cierres=pd.Series(dtype=float), precio=390.0)
    r = decidir(f, e, mk_res(VERDE), ent, [cand("NVDA", 0.08), cand("META", 0.5)], AHORA, CFG)
    assert r.metodo_mee == "σ20 de la cartera" and r.iv == {} and "META" not in [x.c.ticker for x in r.banco]


# =============================== comandos ===============================
def ctx_bot(tmp_path, f=None, ahora=AHORA):
    return S.Contexto(CFG, f or FuentesCartera(), puntuar_vader, "vader", ahora, False, tmp_path / "s.json", salida=lambda s: None)


def test_compra_registra_y_muestra_la_cartera_en_palabras_sencillas(tmp_path):
    c = ctx_bot(tmp_path)
    t1 = S.resp_compra(c, ["TSLA", "55", "380,5"])
    t2 = S.resp_compra(c, ["meta", "8", "720,5"])
    t3 = S.resp_compra(c, ["GRUPOARGOS", "500", "21.500"])                                                        # "21.500" en pesos = veintiún mil quinientos
    assert "Compra de TSLA registrada" in t1 and "llevas 1 de 4" in t1 and "llevas 3 de 4" in t3
    p = F.plano(t3)
    for esperado in ("Tu cartera", "TSLA", "META", "GRUPOARGOS", "55 acciones", "500 acciones", "de la cartera", "Rentabilidad del conjunto", "Acción principal que vigila el sistema: TSLA"):
        assert esperado in p, esperado
    e = Estado(CFG, c.estado_path)
    assert next(x for x in e.d["cartera"] if x["ticker"] == "GRUPOARGOS")["precio"] == 21500 and e.cambios_usados() == 0 and e.ops_total() == 3
    sin_jerga(t3)
    assert t2 and "t2" not in p


def test_compra_con_datos_malos_explica_y_no_guarda(tmp_path):
    c = ctx_bot(tmp_path)
    assert "Cómo usar /compra" in S.resp_compra(c, ["TSLA"])
    assert "LISTA NEGRA" in S.resp_compra(c, ["ETB", "10", "100"]) and "universo" in S.resp_compra(c, ["AMD", "10", "100"])
    assert "❌" in S.resp_compra(c, ["TSLA", "mucho", "380"]) and "❌" in S.resp_compra(c, ["TSLA", "0", "380"])
    assert Estado(CFG, c.estado_path).d["cartera"] == []


def test_si_no_hay_trm_pide_el_monto_en_pesos_y_lo_acepta(tmp_path, monkeypatch):
    monkeypatch.setattr("src.data_loader.load_macro", lambda n, cfg: pd.DataFrame({"Close": [50.0]}))
    c = ctx_bot(tmp_path, FuentesCartera(trm=39.0))
    assert "No pude obtener la TRM" in S.resp_compra(c, ["META", "8", "720,5"]) and Estado(CFG, c.estado_path).d["cartera"] == []
    assert "Compra de META registrada" in S.resp_compra(c, ["META", "8", "720,5", "23M"])
    assert Estado(CFG, c.estado_path).d["cartera"][0]["monto_cop"] == 23e6


def test_venta_y_cartera(tmp_path):
    c = ctx_bot(tmp_path)
    S.resp_compra(c, ["TSLA", "55", "380,5"])
    S.resp_compra(c, ["META", "8", "720,5"])
    v = F.plano(S.resp_venta(c, ["META"]))
    assert "Venta total de META registrada" in v and "llevas 3 de 4" in v and "META" not in v.split("Tu cartera")[1]
    assert "Cómo usar /venta" in S.resp_venta(c, []) and "no está en tu cartera" in S.resp_venta(c, ["META"])
    assert "aún no has registrado compras" in F.plano(S.resp_cartera(ctx_bot(tmp_path / "otra")))
    assert "TSLA" in F.plano(S.resp_cartera(c))


def test_estado_incluye_la_cartera_cuando_hay_compras(tmp_path, monkeypatch):
    c = ctx_bot(tmp_path)
    S.resp_compra(c, ["TSLA", "55", "380,5"])
    monkeypatch.setattr(S, "evaluar_activo", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("sin datos")))
    t = F.plano(S.resp_estado(c))
    assert "Tu cartera" in t and "TSLA" in t


def test_semaforo_sin_ticker_muestra_todas_las_acciones_de_la_cartera(tmp_path, monkeypatch):
    c = ctx_bot(tmp_path)
    for t, n, p in (("TSLA", "55", "380,5"), ("META", "8", "720,5"), ("GRUPOARGOS", "500", "21500")):
        S.resp_compra(c, [t, n, p])
    colores = {"TSLA": VERDE, "META": AMARILLO, "GRUPOARGOS": ROJO}
    pedidos = []

    def falso(f, t, ahora, cfg, est, pun, mot, **k):
        pedidos.append(t)
        return mk_res(colores[t], ticker=t), None, {}
    monkeypatch.setattr(S, "evaluar_activo", falso)
    texto = F.plano(S.resp_semaforo(c))
    assert pedidos == ["TSLA", "GRUPOARGOS", "META"] or set(pedidos) == {"TSLA", "META", "GRUPOARGOS"} and pedidos[0] == "TSLA"       # la principal va primero
    for esperado in ("TSLA: VERDE", "META: AMARILLO", "GRUPOARGOS: ROJO"):
        assert esperado in texto, esperado
    pedidos.clear()
    solo = F.plano(S.resp_semaforo(c, "meta"))
    assert pedidos == ["META"] and "META: AMARILLO" in solo and "TSLA" not in solo.split("¿Qué pasó?")[0]


def test_semaforo_sin_compras_registradas_muestra_la_principal_y_una_falla_no_oculta_las_demas(tmp_path, monkeypatch):
    c = ctx_bot(tmp_path)
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, *a, **k: (mk_res(VERDE, ticker=t), None, {}))
    assert F.plano(S.resp_semaforo(c)).count("➖") == 0 and "TSLA: VERDE" in F.plano(S.resp_semaforo(c))
    S.resp_compra(c, ["TSLA", "55", "380,5"])
    S.resp_compra(c, ["META", "8", "720,5"])

    def con_falla(f, t, *a, **k):
        if t == "META":
            raise RuntimeError("sin datos")
        return mk_res(VERDE, ticker=t), None, {}
    monkeypatch.setattr(S, "evaluar_activo", con_falla)
    t = F.plano(S.resp_semaforo(c))
    assert "TSLA: VERDE" in t and "META: no pude calcular su semáforo" in t


def test_compra_sin_precio_usa_el_de_hoy_y_lo_avisa(tmp_path):
    c = ctx_bot(tmp_path)
    t = F.plano(S.resp_compra(c, ["META", "8"]))
    assert "Compra de META registrada" in t and "APROXIMADO" in t
    e = Estado(CFG, c.estado_path)
    assert e.d["cartera"][0]["precio"] == 730.0 and e.d["cartera"][0]["monto_cop"] == pytest.approx(8 * 730.0 * TRM)
    assert "APROXIMADO" not in F.plano(S.resp_compra(c, ["TSLA", "55", "380,5"]))
    assert "Cómo usar /compra" in S.resp_compra(c, ["TSLA"]) and "universo" in S.resp_compra(c, ["AMD", "5"])


# =============================== monitor: vigila cada posición ===============================
def evaluador_por_ticker(colores):
    def ev(f, t, ahora, cfg, est, pun, mot, **k):
        return mk_res(colores[t], ticker=t), SimpleNamespace(cierres=pd.Series([100.0] * 70, index=pd.bdate_range(end="2026-10-02", periods=70)), precio=370.0), {}
    return ev


class FuentesMonitor(FuentesCartera):
    def noticias(self, t, d):
        return []

    def reportes(self, t, dias_adelante=60):
        return []

    proveedores_estado = {"finnhub": True}


def correr(tmp_path, colores, banco=banco_vacio, ahora=bog(2026, 10, 6, 10)):
    ctx = S.Contexto(CFG, FuentesMonitor(), puntuar_vader, "vader", ahora, True, tmp_path / "s.json", salida=lambda s: None)
    return S.correr_monitor(ctx, evaluar=evaluador_por_ticker(colores), banco=banco)


def test_el_monitor_avisa_cuando_una_accion_de_tu_cartera_que_no_es_la_principal_se_pone_en_rojo(tmp_path):
    estado_con_tres(tmp_path)
    base = {"TSLA": VERDE, "META": VERDE, "GRUPOARGOS": VERDE}
    assert correr(tmp_path, base) == []
    out = correr(tmp_path, {**base, "META": ROJO}, ahora=bog(2026, 10, 6, 10, 15))
    assert len(out) == 1 and "META" in out[0] and "ROJO" in out[0] and "ALERTA" in out[0]
    assert correr(tmp_path, {**base, "META": ROJO}, ahora=bog(2026, 10, 6, 10, 30)) == []                       # ya avisado
    assert Estado(CFG, tmp_path / "s.json").color_previo("META") == ROJO


def test_lo_que_tienes_no_genera_avisos_de_relevo(tmp_path):
    estado_con_tres(tmp_path)
    base = {"TSLA": VERDE, "META": VERDE, "GRUPOARGOS": VERDE}
    c1 = [Candidato("META", "mgc", 0.09, "IV", color=VERDE, corr=0.2, valor_negociado_mm=9999), Candidato("AMZN", "mgc", 0.08, "IV", color=VERDE, corr=0.3, valor_negociado_mm=9999)]
    correr(tmp_path, base, banco=lambda *a, **k: c1)
    assert Estado(CFG, tmp_path / "s.json").d["banco_top"] == ["AMZN"]                                          # META ya es tuya: no es relevo
    c2 = [Candidato("META", "mgc", 0.09, "IV", color=ROJO, corr=0.2, valor_negociado_mm=9999), c1[1]]
    out = correr(tmp_path, {**base, "META": ROJO}, banco=lambda *a, **k: c2, ahora=bog(2026, 10, 6, 10, 15))
    assert all("Un relevo" not in m for m in out) and any("ALERTA: META" in F.plano(m) for m in out)           # la avisa como TU acción, no como relevo


def test_una_accion_de_la_cartera_sin_datos_no_tumba_el_monitor(tmp_path):
    estado_con_tres(tmp_path)

    def ev(f, t, ahora, cfg, est, pun, mot, **k):
        if t == "META":
            raise RuntimeError("sin datos")
        return mk_res(VERDE, ticker=t), SimpleNamespace(cierres=pd.Series([100.0] * 70, index=pd.bdate_range(end="2026-10-02", periods=70)), precio=370.0), {}
    ctx = S.Contexto(CFG, FuentesMonitor(), puntuar_vader, "vader", bog(2026, 10, 6, 10), True, tmp_path / "s.json", salida=lambda s: None)
    assert S.correr_monitor(ctx, evaluar=ev, banco=banco_vacio) == []


# =============================== bot ===============================
def app(**bot_cfg):
    cfg = copy.deepcopy(CFG)
    cfg["bot"].update(bot_cfg)
    return bot.construir_app("123456:ABCDEF-token-falso-para-pruebas", 123456789, cfg)


def test_el_bot_tiene_los_comandos_de_cartera_y_los_de_escritura_se_pueden_reservar():
    handlers = {next(iter(h.commands)): h for g in app().handlers.values() for h in g if hasattr(h, "commands")}      # sólo los comandos (también hay botones y texto libre)
    assert {"compra", "venta", "cartera", "pos", "rank", "op"} <= set(handlers)
    for h in handlers.values():
        assert h.check_update(update_de(-100777, f"/{next(iter(h.commands))}", FAKE_BOT)) not in (None, False)
    assert "/cartera" in F.AYUDA and "compré 300 argos a 21500" in F.AYUDA and "vendí tesla" in F.AYUDA and "me equivoqué" in F.AYUDA      # ya no hace falta el formato de comando
