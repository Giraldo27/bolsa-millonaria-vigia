"""Pruebas de la Fase 4: mensajes, catalizadores, monitor, radar y bot (sin red; las fuentes se simulan)."""
import copy
import datetime as dt
from types import SimpleNamespace

import pandas as pd
import pytest

from src import formato as F
from src import servicio as S
from src.catalizadores import agrupar_reportes, catalizadores, ventana
from src.construir import ResultadoMotor
from src.data_sources import FuentesDatos
from src.regla_maestra import CAMBIAR, MANTENER, Decision
from src.relevo import Candidato, construir_banco
from src.semaforo import AMARILLO, AZUL, NEGRO, ROJO, VERDE, ResultadoSemaforo
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg

CFG = make_cfg()
BOG = dt.timezone(dt.timedelta(hours=-5))


def bog(y, m, d, h=10, mi=0):
    return dt.datetime(y, m, d, h, mi, tzinfo=BOG)


def mk_res(color=VERDE, **k):
    base = dict(ticker="TSLA", color=color, color_calculado=color, z=-0.5, z_res=-0.4, vol_rel=1.1, beta=1.5, sigma20=0.025, sigma_res=0.012,
                r_hoy=-0.012, resid5=-0.01, umbral5=-0.07, noticia_negativa=False, sentimiento=-0.1, n_24h=5, base_diaria=20.0,
                ratio_noticias=0.8, claves=[], sin_noticias=False, motor_sentimiento="vader", motivo="movimiento normal", accion="Mantener.",
                episodio=None, advertencias=[])
    base.update(k)
    return ResultadoSemaforo(**base)


CLAVE = dict(titulo="Tesla recalls 100,000 vehicles amid probe", puntaje=-0.62, palabras=["probe", "recall"], proveedor="finnhub", ts="2026-10-06T14:00:00+00:00")


# =============================== formato ===============================
JERGA = ("z_res", "MEE", "σ", "beta", "sigma", "múltiplo requerido", "vol_rel")           # nada de esto debe verse en los mensajes normales
AHORA_UTC = dt.datetime(2026, 10, 6, 16, 0, tzinfo=dt.timezone.utc)


def sin_jerga(t: str) -> None:
    p = F.plano(t)
    for j in JERGA:
        assert j not in p, f"jerga '{j}' en el mensaje:\n{p}"
    assert t.count("<b>") == t.count("</b>") and t.count("<i>") == t.count("</i>"), "etiquetas HTML sin cerrar"
    assert len(t) < 3900


def test_formato_numeros_con_coma_decimal():
    assert F.n(1.234, 2) == "1,23" and F.n(-2.5, 1, True) == "-2,5" and F.pct(0.0523, 1, True) == "+5,2%" and F.n(None) == "n. d."
    assert F.cop(97_000_000) == "$97.000.000" and F.fecha(dt.date(2026, 10, 5)) == "lun 05/10"
    assert F.puntos(1.0) == "1,0 punto" and F.puntos(5.5) == "5,5 puntos"


def test_plano_quita_etiquetas_y_entidades_y_esc_protege_el_texto():
    assert F.plano("<b>Hola</b> <i>mundo</i> &amp; &lt;x&gt;") == "Hola mundo & <x>"
    assert F.esc("a < b & c") == "a &lt; b &amp; c" and F.b("x<y") == "<b>x&lt;y</b>"


def test_alerta_de_color_en_palabras_sencillas_con_titular_traducido_y_regla_maestra():
    r = mk_res(ROJO, z=-2.6, z_res=-3.1, vol_rel=2.1, noticia_negativa=True, claves=[CLAVE], r_mercado=-0.004, resid_hoy=-0.021, r_hoy=-0.025)
    d = Decision(CAMBIAR, g=6.0, multiplo=1.22, candidato="META", ratio=1.38, codigo="cambiar")
    t = F.msg_alerta_color(r, AMARILLO, bog(2026, 10, 6, 10, 15), mee=0.065, decision=d, traductor=lambda s: "Tesla retira 100.000 vehículos por una investigación",
                           activo="TSLA")
    p = F.plano(t)
    for esperado in ("ALERTA: TSLA ahora en ROJO (alerta)", "antes: AMARILLO", "¿Qué pasó?", "¿Qué significa?", "Qué hacer", "Tesla retira 100.000 vehículos", "😟",
                     "veces más fuerte de lo normal", "El mercado en general explica -0,4%", "2,1 veces el volumen normal", "Recomendación: cambia a META",
                     "1,38 veces", "/detalle"):
        assert esperado in p, esperado
    assert "(en inglés)" not in p                                                              # el titular se tradujo
    sin_jerga(t)


def test_titular_sin_traduccion_se_muestra_en_ingles_y_lo_dice():
    t = F.msg_alerta_color(mk_res(ROJO, noticia_negativa=True, claves=[CLAVE]), AMARILLO, bog(2026, 10, 6), traductor=lambda s: None)
    assert "Tesla recalls 100,000 vehicles" in t and "(en inglés)" in t
    es = dict(CLAVE, titulo="Tesla retira vehículos", idioma="es")
    t2 = F.msg_semaforo(mk_res(ROJO, noticia_negativa=True, claves=[es]), bog(2026, 10, 6), traductor=lambda s: pytest.fail("no se traduce lo que ya está en español"))
    assert "Tesla retira vehículos" in t2 and "(en inglés)" not in t2


def test_noticias_aclara_que_muestra_los_titulares_mas_preocupantes_y_el_tono_general():
    p = F.plano(F.msg_semaforo(mk_res(VERDE, n_24h=13, sentimiento=-0.11, claves=[CLAVE]), bog(2026, 10, 6)))
    assert "13 en las últimas 24 horas, tono general mixto" in p and "El titular más preocupante:" in p
    pos = dict(CLAVE, puntaje=0.6, palabras=[])
    p2 = F.plano(F.msg_semaforo(mk_res(VERDE, n_24h=13, sentimiento=0.4, claves=[pos]), bog(2026, 10, 6)))
    assert "tono general positivo" in p2 and "El titular destacado:" in p2 and "más preocupantes" not in p2
    p3 = F.plano(F.msg_semaforo(mk_res(ROJO, n_24h=1, noticia_negativa=True, claves=[CLAVE]), bog(2026, 10, 6)))
    assert "tono general negativo" in p3 and "titulares" not in p3.split("tono general")[1].split("\n")[0]       # con 1 sola noticia no hace falta el rótulo


def test_alerta_sin_noticias_dice_que_no_pudo_consultarlas_y_menciona_el_plan_b():
    t = F.plano(F.msg_alerta_color(mk_res(ROJO, sin_noticias=True, proxy_noticias=True), AMARILLO, bog(2026, 10, 6)))
    assert "no pude consultarlas" in t and "plan B" in t
    assert "primera lectura" in F.plano(F.msg_alerta_color(mk_res(ROJO), None, bog(2026, 10, 6)))


def test_alerta_de_recuperacion_celebra_y_no_pide_hacer_nada():
    t = F.plano(F.msg_alerta_color(mk_res(VERDE), ROJO, bog(2026, 10, 6)))
    assert "se recuperó" in t and "no hace falta hacer nada" in t and "ALERTA" not in t


def test_cada_color_tiene_explicacion_nombre_y_accion():
    for c in (VERDE, AZUL, AMARILLO, ROJO, NEGRO):
        assert F.EXPLICA_COLOR[c] and F.NOMBRE_COLOR[c] and F.ACCION_COLOR[c]
        sin_jerga(F.msg_semaforo(mk_res(c), bog(2026, 10, 6, 12)))
    assert "No vendas por pánico" in F.ACCION_COLOR[ROJO]


def test_semaforo_corte_manana_explica_por_que_rojo_se_trata_como_negro():
    t = F.plano(F.msg_semaforo(mk_res(NEGRO, color_calculado=ROJO), bog(2026, 10, 8, 19, 30)))
    assert "El corte es mañana" in t


def test_texto_de_decision_para_cada_codigo():
    casos = {
        "sin_ranking": (Decision("SIN_DATOS", codigo="sin_ranking"), "voy 6,5 y el corte está en 11"),
        "g_cero": (Decision(MANTENER, g=-1.5, codigo="g_cero", datos={"adelante": 2.5}), "por encima del objetivo (+2,5 puntos)"),
        "primero_mantener": (Decision(MANTENER, codigo="primero_mantener"), "Vas de primero"),
        "ultimo_dia_mantener": (Decision(MANTENER, codigo="ultimo_dia_mantener"), "8 puntos"),
        "vender_todo": (Decision("VENDER_TODO", codigo="vender_todo", datos={"ventaja": 9.0}), "VENDE TODO mañana"),
        "negro_adelante": (Decision(MANTENER, codigo="negro_adelante", datos={"adelante": 4.0, "rend_entrada": -3.0}), "15%"),
        "limite_dia": (Decision(MANTENER, codigo="limite_dia"), "uno por día"),
        "limite_total": (Decision(MANTENER, codigo="limite_total"), "4 cambios"),
        "sin_mee": (Decision("SIN_DATOS", codigo="sin_mee"), "no puedo evaluar"),
        "banco_vacio": (Decision(MANTENER, codigo="banco_vacio"), "ningún candidato bueno"),
        "ninguno_cumple": (Decision(MANTENER, g=6.0, multiplo=1.22, ratio=1.1, codigo="ninguno_cumple", datos={"mejor": "AMZN"}), "AMZN"),
        "cambiar": (Decision(CAMBIAR, g=6.0, multiplo=1.22, candidato="META", ratio=1.38, codigo="cambiar"), "cambia a META"),
        "cambiar_negro": (Decision(CAMBIAR, g=6.0, multiplo=1.1, candidato="META", ratio=1.38, codigo="cambiar", modo_negro=True), "en NEGRO"),
    }
    for nombre, (d, esperado) in casos.items():
        t = "\n".join(F.texto_decision(d, "TSLA", "08:45"))
        assert esperado in F.plano(t), (nombre, t)
        sin_jerga(t)
    assert "08:45" in "\n".join(F.texto_decision(casos["cambiar"][0], "TSLA", "08:45"))
    assert "razón clara" in "\n".join(F.texto_decision(Decision(MANTENER), "TSLA"))                 # decisión sin código: respuesta genérica, sin romperse


def test_la_regla_maestra_real_siempre_deja_un_codigo_que_el_texto_entiende():
    from src.regla_maestra import ContextoRegla, regla_maestra
    base = dict(mia=6.0, objetivo=11.0, activo="TSLA", color_actual=VERDE, mee_actual=0.06, cambios_restantes=4, cambio_hoy=False)
    cands = [Candidato("META", "mgc", 0.11, "IV", corr=0.2, valor_negociado_mm=9999)]
    escenarios = [dict(), dict(mia=None), dict(mia=12.0), dict(cambio_hoy=True), dict(cambios_restantes=0), dict(mee_actual=None),
                  dict(color_actual=NEGRO), dict(semana_final=True, soy_primero=True), dict(ultimo_dia=True, soy_primero=True, ventaja_pp=9.0)]
    for esc in escenarios:
        d, _ = regla_maestra(ContextoRegla(**{**base, **esc}), cands, CFG)
        assert d.codigo, esc
        sin_jerga("\n".join(F.texto_decision(d, "TSLA")))
    assert regla_maestra(ContextoRegla(**base), [], CFG)[0].codigo == "banco_vacio"


def test_detalle_tecnico_si_trae_los_numeros_que_los_mensajes_normales_ocultan():
    r = mk_res(ROJO, z=-2.6, z_res=-3.1, vol_rel=2.1, noticia_negativa=True, claves=[CLAVE], r_mercado=-0.004, resid_hoy=-0.021, motivo="caída propia")
    t = F.plano(F.msg_detalle(r, 0.065, {"iv": 0.4, "vencimiento": "2026-10-12"}, CFG))
    for esperado in ("Detalle técnico de TSLA", "z): -2,60", "z_res): -3,10", "beta", "2,10×", "6,50%", "según opciones", "caída propia"):
        assert esperado in t, esperado


def estado_tmp(tmp_path, **ranking):
    e = Estado(CFG, tmp_path / "s.json")
    e.d["rentabilidad"].update(ranking)
    return e


def test_msg_estado_con_y_sin_ranking(tmp_path):
    e = estado_tmp(tmp_path)
    t = F.msg_estado(e, CFG, bog(2026, 10, 5, 12))
    p = F.plano(t)
    assert "aún sin comprar" in p and "voy 6,5 y el corte está en 11" in p and "has usado 0 de 4" in p and "0 en todo el concurso (el mínimo es 15: faltan 15)" in p and "llevas 0 esta semana (el mínimo es 4: faltan 4)" in p
    sin_jerga(t)
    e2 = estado_tmp(tmp_path, mia=6.0, umbral=11.0, actualizado="2026-10-06T19:30")
    t2 = F.plano(F.msg_estado(e2, CFG, bog(2026, 10, 6, 12), mk_res(AMARILLO), 0.065))
    assert "vas +6,0% y el objetivo del corte es +11,0%" in t2 and "Te faltan 5,0 puntos" in t2 and "AMARILLO (vigilar)" in t2 and "semana 1 de 5" in t2
    t3 = F.plano(F.msg_estado(estado_tmp(tmp_path, mia=12.0, umbral=11.0), CFG, bog(2026, 10, 6, 12)))
    assert "1,0 punto por encima del objetivo" in t3


def test_msg_estado_avisa_cuando_el_corte_es_manana(tmp_path):
    assert "¡El corte es mañana!" in F.plano(F.msg_estado(estado_tmp(tmp_path), CFG, bog(2026, 10, 8, 19, 30)))
    assert "¡El corte es mañana!" not in F.plano(F.msg_estado(estado_tmp(tmp_path), CFG, bog(2026, 10, 6, 19, 30)))


def test_msg_estado_resume_la_salud_de_las_fuentes(tmp_path):
    assert "todo funciona bien" in F.plano(F.msg_estado(estado_tmp(tmp_path), CFG, bog(2026, 10, 6, 12), fuentes_ok={"precios": True, "noticias": True}))
    assert "problemas con noticias" in F.plano(F.msg_estado(estado_tmp(tmp_path), CFG, bog(2026, 10, 6, 12), fuentes_ok={"precios": True, "noticias": False}))


def test_recordatorio_actividad_en_cada_escenario(tmp_path):
    e = estado_tmp(tmp_path)
    dom = F.plano(F.recordatorio_actividad(e, CFG, bog(2026, 10, 4, 20)))
    assert "COMPRA INICIAL" in dom and "08:45" in dom and "$97.000.000" in dom and "no necesitas micro-compra" in dom
    e.registrar_cambio("TSLA", 97e6, 380, bog(2026, 10, 5, 8, 45))                          # la compra inicial cuenta como la operación del lunes
    mar = F.plano(F.recordatorio_actividad(e, CFG, bog(2026, 10, 5, 19, 30)))
    assert "toca micro-compra" in mar and "mar 06/10" in mar and "1 de 4" in mar and "$100.000" in mar and "ECOPETROL" in mar
    for d in (6, 7, 8):
        e.registrar_op(bog(2026, 10, d, 10))
    assert "no hace falta micro-compra" in F.plano(F.recordatorio_actividad(e, CFG, bog(2026, 10, 8, 19, 30)))
    sem2 = F.plano(F.recordatorio_actividad(e, CFG, bog(2026, 10, 9, 19, 30)))               # viernes noche: mañana es sábado → la próxima sesión es el martes 13
    assert "mar 13/10" in sem2 and "toca" in sem2
    assert "terminó" in F.recordatorio_actividad(e, CFG, bog(2026, 11, 6, 19, 30))


def test_msg_banco_explica_en_palabras_cuanto_se_mueve_y_que_tan_parecido_es():
    cs = [Candidato("META", "mgc", 0.09, "IV", corr=0.26, valor_negociado_mm=9999, reporte_antes_corte="2026-10-28"),
          Candidato("AMZN", "mgc", 0.085, "IV", corr=0.75, valor_negociado_mm=9999, sin_noticias=True)]
    t = F.msg_banco(construir_banco(cs, "TSLA", CFG), "TSLA", 0.065, {"NVDA"})
    p = F.plano(t)
    for esperado in ("Relevos posibles para TSLA", "META se mueve 1,4 veces lo que ya tienes", "se espera ±9,0% hasta el corte (1,4 veces lo tuyo)", "se mueve distinto a TSLA ✅",
                     "se mueve casi igual que TSLA", "reporta resultados el 28/10, antes del corte", "sin datos de noticias", "Excluidos esta semana: NVDA"):
        assert esperado in p, esperado
    sin_jerga(t)
    assert "no hay ninguno que cumpla" in F.plano(F.msg_banco([], "TSLA", 0.06))


def test_msg_banco_dice_primero_si_vale_la_pena_y_ordena_de_mas_a_menos_movimiento():
    """Caso real (5-oct-2026): la lista mostraba 'mejores relevos' que se movían MENOS que la cartera, sin decirlo, y desordenados."""
    cs = [Candidato("ECOPETROL", "local", 0.032, "σ20", corr=0.05, valor_negociado_mm=42000, r5=0.004), Candidato("PFSURA", "local", 0.042, "σ20", corr=0.2, valor_negociado_mm=8300, r5=-0.019),
          Candidato("ISA", "local", 0.035, "σ20", corr=0.1, valor_negociado_mm=8400, r5=0.031)]
    p = F.plano(F.msg_banco(construir_banco(cs, "NUCO", CFG), "NUCO", 0.053))
    assert "Ninguna se mueve más que lo que ya tienes (la que más, PFSURA: 0,8 veces lo tuyo)" in p and "Cambiar no te ayuda a remontar" in p and "se negocia poco en trii" in p
    assert p.index("1. 🟢 PFSURA") < p.index("2. 🟢 ISA") < p.index("3. 🟢 ECOPETROL")                    # de la que más se mueve a la que menos
    assert "esta semana -1,9%" in p and "esta semana +3,1%" in p and "mejor la que viene quieta" in p
    casi = F.plano(F.msg_banco(construir_banco(cs, "NUCO", CFG), "NUCO", 0.040))
    assert "Ninguna se mueve claramente más" in casi and "no paga su costo" in casi
    assert F.conclusion_banco([], 0.05) is None and F.conclusion_banco(construir_banco(cs, "NUCO", CFG), None) is None


def test_msg_noticias_traduce_marca_el_tono_y_dice_cuando_no_hay():
    assert "no pude consultar" in F.msg_noticias("TSLA", None, [], 5) and "no hay noticias recientes" in F.msg_noticias("TSLA", [], [], 5)
    t = F.plano(F.msg_noticias("TSLA", [1], [CLAVE | {"titulo": "Tesla recalls cars"}], 5, traductor=lambda s: "Tesla retira autos", ahora_utc=AHORA_UTC))
    assert "😟 hace 2 h: Tesla retira autos" in t and "(en inglés)" not in t
    assert "(en inglés)" in F.plano(F.msg_noticias("TSLA", [1], [CLAVE], 5, ahora_utc=AHORA_UTC))


def test_msg_salud_en_palabras_sencillas():
    t = F.plano(F.msg_salud("noticias", "CAIDA", bog(2026, 10, 6), "pruebo otras fuentes", {"finnhub": False, "google_rss": True}))
    assert "Falló una fuente de datos: las noticias" in t and "pruebo otras fuentes" in t and "finnhub ❌" in t and "google_rss ✅" in t
    assert "Ya funciona de nuevo: los precios" in F.plano(F.msg_salud("precios", "RECUPERADA", bog(2026, 10, 6), ""))


def test_ayuda_explica_los_colores_y_lista_el_comando_detalle():
    p = F.plano(F.AYUDA)
    for esperado in ("/detalle", "voy 3,5", "/comprar", "/noticias", "🟢 todo normal", "🔵 cae el mercado", "⚫ no se recupera", "nunca compro ni vendo"):
        assert esperado in p, esperado
    assert F.AYUDA.count("<b>") == F.AYUDA.count("</b>")


# =============================== envío: HTML con respaldo a texto plano ===============================
class FakeTelegram:
    def __init__(self, respuestas):
        self.respuestas, self.posts = list(respuestas), []

    def post(self, url, data=None, timeout=None):
        self.posts.append(dict(data))
        r = self.respuestas.pop(0)
        return SimpleNamespace(json=lambda: r)


ENV_TG = {"TELEGRAM_BOT_TOKEN": "123:token-falso", "TELEGRAM_CHAT_ID": "123456789"}


def test_enviar_usa_html_y_si_telegram_rechaza_el_formato_reenvia_en_texto_plano():
    from src.notify import enviar
    http = FakeTelegram([{"ok": False, "description": "Bad Request: can't parse entities: Unexpected end tag"}, {"ok": True}])
    assert enviar("<b>Hola</b> &amp; adiós", CFG, ENV_TG, http) is True
    assert http.posts[0]["parse_mode"] == "HTML" and http.posts[0]["text"] == "<b>Hola</b> &amp; adiós"
    assert "parse_mode" not in http.posts[1] and http.posts[1]["text"] == "Hola & adiós"
    ok = FakeTelegram([{"ok": True}])
    assert enviar("<i>x</i>", CFG, ENV_TG, ok) and len(ok.posts) == 1 and ok.posts[0]["parse_mode"] == "HTML"


def test_enviar_no_reintenta_en_plano_por_errores_que_no_son_de_formato():
    from src.notify import enviar
    http = FakeTelegram([{"ok": False, "description": "Forbidden: bot was blocked by the user"}])
    assert enviar("<b>x</b>", CFG, ENV_TG, http) is False and len(http.posts) == 1


def test_partir_no_corta_etiquetas_porque_parte_por_lineas():
    from src.notify import partir
    texto = "\n".join(f"<b>línea {i}</b> " + "x" * 80 for i in range(100))
    partes = partir(texto, 1000)
    assert len(partes) > 1 and all(len(p) <= 1000 for p in partes)
    assert all(p.count("<b>") == p.count("</b>") for p in partes) and "\n".join(partes) == texto


def test_traducir_usa_mymemory_cachea_y_respeta_el_limite_diario(tmp_path):
    from tests.test_fase1 import FakeHttp, FakeResp, fuentes
    ok = FakeResp({"responseStatus": 200, "responseData": {"translatedText": "Tesla retira autos"}})
    http = FakeHttp({"mymemory": ok})
    f = fuentes(tmp_path, http=http)
    assert f.traducir("Tesla recalls cars") == "Tesla retira autos" and f.traducir("Tesla recalls cars") == "Tesla retira autos"
    assert len(http.calls) == 1 and http.calls[0][1]["langpair"] == "en|es"                    # la segunda salió de la caché
    cuota = FakeHttp({"mymemory": FakeResp({"responseStatus": 200, "responseData": {"translatedText": "MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS"}})})
    assert fuentes(tmp_path / "b", http=cuota).traducir("Other headline") is None              # sin cupo: se muestra el original
    caido = FakeHttp({})                                                                       # 404
    assert fuentes(tmp_path / "c", http=caido).traducir("Another headline") is None
    cfg_chico = dict(CFG, fuentes=dict(CFG["fuentes"], traduccion=dict(CFG["fuentes"]["traduccion"], limite_diario_chars=10)))
    g = FuentesDatos(cfg_chico, env={}, yf=None, http=FakeHttp({"mymemory": ok}), ahora=lambda: dt.datetime(2026, 10, 4, 15, tzinfo=dt.timezone.utc),
                     dormir=lambda s: None, cache_dir=tmp_path / "d")
    assert g.traducir("Una frase mucho más larga que el límite") is None and any("límite diario" in a for a in g.avisos)
    inactivo = dict(CFG, fuentes=dict(CFG["fuentes"], traduccion=dict(CFG["fuentes"]["traduccion"], activo=False)))
    assert FuentesDatos(inactivo, env={}, yf=None, http=FakeHttp({}), cache_dir=tmp_path / "e").traducir("x y z") is None


# =============================== catalizadores ===============================
def test_ventana_de_10_dias_habiles():
    ini, fin = ventana(bog(2026, 10, 5), 10)
    assert ini == dt.date(2026, 10, 5) and fin == dt.date(2026, 10, 19)


def test_agrupar_reportes_marca_fecha_incierta_cuando_las_fuentes_discrepan():
    filas = [{"fecha": "2026-10-20", "hora": "amc", "fuente": "finnhub"}, {"fecha": "2026-10-21", "hora": "amc", "fuente": "yfinance"},
             {"fecha": "2027-01-25", "hora": "amc", "fuente": "finnhub"}]
    g = agrupar_reportes(filas)
    assert len(g) == 2 and len(g[0]["fechas"]) == 2 and "finnhub 20/10" in g[0]["fuentes"] and len(g[1]["fechas"]) == 1


def test_catalizadores_reune_reportes_macro_y_cortes():
    f = SimpleNamespace(reportes=lambda t, dias_adelante=60: {"TSLA": [{"fecha": "2026-10-20", "hora": "amc", "fuente": "finnhub"},
                                                                     {"fecha": "2026-10-21", "hora": "amc", "fuente": "yfinance"}],
                                                              "JPM": [{"fecha": "2026-10-13", "hora": "bmo", "fuente": "finnhub"}],
                                                              "KO": [{"fecha": "2026-12-01", "hora": "bmo", "fuente": "finnhub"}]}.get(t, []))
    items = catalizadores(f, CFG, bog(2026, 10, 12, 19, 30), ["TSLA", "JPM", "KO"], 10)
    det = [(i["fecha"].isoformat(), i["detalle"]) for i in items]
    assert ("2026-10-13", "JPM reporta resultados (antes de abrir)") in det
    assert any("TSLA" in d and "no coinciden" in d for _, d in det) and any(i.get("incierta") for i in items)
    assert not any("KO" in d for _, d in det)                                                # fuera de la ventana
    assert any("Corte del concurso" in d and "33" in d for _, d in det) and any("IPC" in d for _, d in det)
    assert [i["fecha"] for i in items] == sorted(i["fecha"] for i in items)


# =============================== monitor ===============================
class FakeF:
    def __init__(self):
        self.avisos, self.proveedores_estado = [], {"finnhub": True}
        self.cache = SimpleNamespace(dir=None)

    def opciones_iv(self, t, fecha):
        return {"iv": 0.40, "calidad": "ok", "vencimiento": "2026-10-12"}

    def traducir(self, texto):
        return None


def ent_falsa():
    return SimpleNamespace(cierres=pd.Series([100.0] * 70, index=pd.bdate_range(end="2026-10-05", periods=70)), precio=370.0)


def ctx_monitor(tmp_path, ahora=None, dry=True):
    return S.Contexto(CFG, FakeF(), puntuar_vader, "vader", ahora or bog(2026, 10, 6, 10), dry, tmp_path / "s.json", salida=lambda s: None)


def evaluador(secuencia):
    """Devuelve un evaluador que va entregando los colores de la secuencia (una corrida cada uno)."""
    it = iter(secuencia)

    def ev(f, t, ahora, cfg, est, pun, mot, **k):
        x = next(it)
        if isinstance(x, Exception):
            raise x
        return (x if isinstance(x, ResultadoSemaforo) else mk_res(x)), ent_falsa(), {}
    return ev


def banco_vacio(*a, **k):
    return []


def corre(ctx, secuencia, banco=banco_vacio, forzar=False):
    return S.correr_monitor(ctx, forzar=forzar, evaluar=evaluador(secuencia), banco=banco)


def test_monitor_fuera_de_horario_no_hace_nada(tmp_path):
    ctx = ctx_monitor(tmp_path, ahora=bog(2026, 10, 4, 12))                                     # domingo
    assert S.correr_monitor(ctx, evaluar=evaluador([ROJO]), banco=banco_vacio) == []
    ctx2 = ctx_monitor(tmp_path, ahora=bog(2026, 10, 6, 15, 1))                                 # tras el cierre de las 15:00
    assert S.correr_monitor(ctx2, evaluar=evaluador([ROJO]), banco=banco_vacio) == []
    ctx3 = ctx_monitor(tmp_path, ahora=bog(2026, 10, 12, 10))                                   # festivo
    assert S.correr_monitor(ctx3, evaluar=evaluador([ROJO]), banco=banco_vacio) == []
    assert len(S.correr_monitor(ctx3, forzar=True, evaluar=evaluador([ROJO]), banco=banco_vacio)) == 1   # --forzar ignora el horario


def test_primera_corrida_verde_o_azul_no_avisa_pero_guarda_el_color(tmp_path):
    ctx = ctx_monitor(tmp_path)
    assert corre(ctx, [VERDE]) == []
    assert Estado(CFG, ctx.estado_path).color_previo("TSLA") == VERDE
    ctx2 = ctx_monitor(tmp_path / "b")
    assert corre(ctx2, [AZUL]) == []


def test_primera_corrida_en_color_de_alerta_si_avisa(tmp_path):
    msgs = corre(ctx_monitor(tmp_path), [AMARILLO])
    assert len(msgs) == 1 and "primera lectura" in msgs[0] and "AMARILLO (vigilar)" in msgs[0]


def test_solo_avisa_cuando_cambia_el_color(tmp_path):
    ctx = ctx_monitor(tmp_path)
    m = corre(ctx, [VERDE, VERDE, AMARILLO, AMARILLO])
    assert m == []                                                                              # sólo la primera llamada (la del VERDE) no avisó
    ctx = ctx_monitor(tmp_path / "x")
    ev = evaluador([VERDE, VERDE, ROJO, ROJO])
    out = [S.correr_monitor(ctx, evaluar=ev, banco=banco_vacio) for _ in range(4)]
    assert [len(o) for o in out] == [0, 0, 1, 0] and "antes: VERDE" in out[2][0] and "ROJO (alerta)" in out[2][0]


def test_la_alerta_roja_trae_la_regla_maestra_y_se_guarda_el_aviso(tmp_path):
    ctx = ctx_monitor(tmp_path)
    est = Estado(CFG, ctx.estado_path)
    est.d["rentabilidad"].update(mia=6.0, umbral=11.0)
    est.guardar()
    cands = [Candidato("META", "mgc", 0.11, "IV", corr=0.2, valor_negociado_mm=9999)]
    ev = evaluador([VERDE, ROJO])
    S.correr_monitor(ctx, evaluar=ev, banco=lambda *a, **k: cands)
    out = S.correr_monitor(ctx, evaluar=ev, banco=lambda *a, **k: cands)
    assert len(out) == 1 and "¿Cambio de acción?" in out[0] and "META" in out[0]
    assert Estado(CFG, ctx.estado_path).d["semaforo"]["TSLA"]["ultimo_aviso"] is not None
    sin_jerga(out[0])


def test_enfriamiento_suprime_rebotes_pero_las_escaladas_pasan(tmp_path):
    ctx = ctx_monitor(tmp_path)
    ev = evaluador([AMARILLO, VERDE, AMARILLO, ROJO])
    out = [S.correr_monitor(ctx, evaluar=ev, banco=banco_vacio) for _ in range(4)]
    # 1: alerta (primera en amarillo) · 2: VERDE (baja de gravedad, dentro de 30 min) suprimida · 3: AMARILLO (sube) pasa · 4: ROJO (sube) pasa
    assert [len(o) for o in out] == [1, 0, 1, 1]


def test_aviso_de_fuente_de_precios_caida_una_vez_y_recuperada(tmp_path):
    ctx = ctx_monitor(tmp_path)
    falla = RuntimeError("Yahoo caído")
    out = [S.correr_monitor(ctx, evaluar=evaluador([falla]), banco=banco_vacio) for _ in range(4)]
    assert [len(o) for o in out] == [0, 0, 1, 0] and "Falló una fuente de datos: los precios" in out[2][0] and "guardados" in out[2][0]
    rec = S.correr_monitor(ctx, evaluar=evaluador([VERDE]), banco=banco_vacio)
    assert len(rec) == 1 and "Ya funciona de nuevo: los precios" in rec[0]


def test_aviso_de_noticias_caidas_describe_el_respaldo_y_los_proveedores(tmp_path):
    ctx = ctx_monitor(tmp_path)
    ctx.f.proveedores_estado = {"finnhub": False, "yahoo_rss": False, "google_rss": True}
    out = [S.correr_monitor(ctx, evaluar=evaluador([mk_res(VERDE, sin_noticias=True)]), banco=banco_vacio) for _ in range(3)]
    avisos = [m for o in out for m in o if "Falló una fuente de datos: las noticias" in m]
    assert len(avisos) == 1 and "volumen" in avisos[0] and "Yahoo y Google" in avisos[0]
    assert "finnhub ❌" in avisos[0] and "google_rss ✅" in avisos[0]


def test_si_telegram_falla_el_mensaje_queda_pendiente_y_se_reenvia(tmp_path, monkeypatch):
    enviados, estado_red = [], {"ok": False}

    def falso(texto, cfg, env=None, http=None):
        if estado_red["ok"]:
            enviados.append(texto)
        return estado_red["ok"]
    monkeypatch.setattr(S, "enviar", falso)
    ctx = ctx_monitor(tmp_path, dry=False)
    out = S.correr_monitor(ctx, evaluar=evaluador([ROJO]), banco=banco_vacio)
    assert len(out) == 1 and enviados == [] and len(Estado(CFG, ctx.estado_path).d["alertas_pendientes"]) == 1
    estado_red["ok"] = True
    S.correr_monitor(ctx_monitor(tmp_path, ahora=bog(2026, 10, 6, 10, 15), dry=False), evaluar=evaluador([ROJO]), banco=banco_vacio)
    assert len(enviados) == 1 and "Aviso atrasado" in enviados[0] and "antes: VERDE" not in enviados[0] and "primera lectura" in enviados[0]
    assert Estado(CFG, ctx.estado_path).d["alertas_pendientes"] == []


def test_alertas_del_banco_al_entrar_o_salir_de_rojo(tmp_path):
    ctx = ctx_monitor(tmp_path)
    c1 = [Candidato("META", "mgc", 0.09, "IV", color=VERDE, corr=0.2, valor_negociado_mm=9999), Candidato("AMZN", "mgc", 0.08, "IV", color=VERDE, corr=0.3, valor_negociado_mm=9999)]
    S.correr_monitor(ctx, evaluar=evaluador([VERDE]), banco=lambda *a, **k: c1)
    assert Estado(CFG, ctx.estado_path).d["banco_top"] == ["META", "AMZN"]
    c2 = [Candidato("META", "mgc", 0.09, "IV", color=ROJO, corr=0.2, valor_negociado_mm=9999), c1[1]]
    out = S.correr_monitor(ctx, evaluar=evaluador([VERDE]), banco=lambda *a, **k: c2)
    assert len(out) == 1 and "Un relevo se puso en alerta: META" in out[0] and "Ya no lo recomiendo" in out[0]
    out2 = S.correr_monitor(ctx, evaluar=evaluador([VERDE]), banco=lambda *a, **k: c1)
    assert len(out2) == 1 and "Un relevo se recuperó: META" in out2[0] and "Puede volver a la lista" in out2[0]
    assert S.correr_monitor(ctx, evaluar=evaluador([VERDE]), banco=lambda *a, **k: c1) == []   # sin cambios: silencio


def test_si_el_banco_falla_el_monitor_sigue_funcionando(tmp_path):
    def roto(*a, **k):
        raise RuntimeError("boom")
    ctx = ctx_monitor(tmp_path)
    assert S.correr_monitor(ctx, evaluar=evaluador([AMARILLO]), banco=roto)[0].startswith("🟡")


# =============================== radar ===============================
def motor_falso(res, banco=None, decision=None):
    return ResultadoMotor(bog(2026, 10, 27, 19, 30), "TSLA", res, ent_falsa(), 0.065, "IV", {"iv": 0.4}, [], banco or [],
                          decision or Decision(MANTENER, g=6.0, multiplo=1.22, ratio=1.1, codigo="ninguno_cumple", datos={"mejor": "META"}), None, set(), [])


def test_radar_arma_las_cinco_secciones_en_palabras_sencillas(tmp_path):
    e = estado_tmp(tmp_path, mia=6.0, umbral=11.0)
    e.registrar_cambio("TSLA", 97e6, 380, bog(2026, 10, 5, 8, 45))
    ahora = bog(2026, 10, 27, 19, 30)
    cs = [Candidato("META", "mgc", 0.09, "IV", corr=0.26, valor_negociado_mm=9999)]
    r = motor_falso(mk_res(AMARILLO), construir_banco(cs, "TSLA", CFG), Decision(CAMBIAR, g=6.0, multiplo=1.22, candidato="META", ratio=1.38, codigo="cambiar"))
    r.corte = {"fecha": dt.date(2026, 10, 30), "top_pct": 10, "liga": "Corte clave", "tipo": "corte"}
    t = S.armar_radar(e, r, [{"fecha": dt.date(2026, 10, 28), "hora": "13:00", "detalle": "Decisión de la FED", "tipo": "macro", "verificada": True}], CFG, ahora)
    p = F.plano(t)
    for esperado in ("RADAR DE LA NOCHE", "Semana 4 de 5", "pasa el top 10%", "Hoy es noche de decisión", "1️⃣ Tu acción", "AMARILLO (vigilar)", "2️⃣ ¿Te conviene cambiar?",
                     "vas +6,0% y el objetivo del corte es +11,0%", "Recomendación: cambia a META", "3️⃣ Relevos posibles", "se espera ±9,0%", "4️⃣ Qué viene",
                     "FED", "5️⃣ Tus tareas de mañana", "toca micro-compra", "Cambios de acción usados: 0 de 4", "/ayuda"):
        assert esperado in p, esperado
    posiciones = [p.index(s) for s in ("1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣")]
    assert posiciones == sorted(posiciones)
    sin_jerga(t)


def test_radar_muestra_un_aviso_corto_si_fallaron_fuentes_pero_no_los_detalles_tecnicos(tmp_path):
    r = motor_falso(mk_res(VERDE))
    r.avisos = ["Yahoo caído — se usa caché vencida", "TSLA: 3 barras diarias con ticks erróneos corregidas"]
    t = F.plano(S.armar_radar(estado_tmp(tmp_path), r, [], CFG, bog(2026, 10, 6, 19, 30)))
    assert "plan B" in t and "caché vencida" not in t and "ticks" not in t
    r.avisos = ["TSLA: 3 barras diarias con ticks erróneos corregidas", "Traducción: límite diario agotado",
                "yfinance intradía CNEC: RuntimeError: Yahoo devolvió vacío — sin datos"]                  # fallos menores: no asustan
    assert "plan B" not in F.plano(S.armar_radar(estado_tmp(tmp_path), r, [], CFG, bog(2026, 10, 6, 19, 30)))


def test_radar_sin_ranking_pide_el_comando_rank(tmp_path):
    r = motor_falso(mk_res(VERDE), decision=Decision("SIN_DATOS", codigo="sin_ranking"))
    t = F.plano(S.armar_radar(estado_tmp(tmp_path), r, [], CFG, bog(2026, 10, 6, 19, 30)))
    assert "Todavía no sé cómo vas" in t and "Hoy no hay decisión programada" in t
    assert t.count("voy 6,5 y el corte está en 11") == 1                                         # se pide en palabras (sin comando) y una sola vez


def test_radar_noche_antes_del_corte_avisa_que_rojo_se_trata_como_negro(tmp_path):
    r = motor_falso(mk_res(NEGRO, color_calculado=ROJO))
    r.corte = {"fecha": dt.date(2026, 10, 9), "top_pct": 50, "liga": "Bronce → Plata", "tipo": "corte"}
    t = F.plano(S.armar_radar(estado_tmp(tmp_path), r, [], CFG, bog(2026, 10, 8, 19, 30)))
    assert "¡El corte es mañana!" in t and "un ROJO se trata como NEGRO" in t


# =============================== bot: parseo y comandos ===============================
def test_numero_y_monto_aceptan_formatos_colombianos():
    assert S.numero("6,5") == 6.5 and S.numero("+6.5") == 6.5 and S.numero("-3") == -3 and S.numero("6,5%") == 6.5 and S.numero("1.234,5") == 1234.5
    assert S.monto("97000000") == 97e6 and S.monto("97.000.000") == 97e6 and S.monto("97M") == 97e6 and S.monto("97,5M") == 97.5e6 and S.monto("$ 95mm") == 95e6
    with pytest.raises(ValueError):
        S.numero("abc")


def ctx_bot(tmp_path, ahora=None):
    return ctx_monitor(tmp_path, ahora=ahora or bog(2026, 10, 6, 21), dry=False)


def test_rank_guarda_y_calcula_g_y_multiplo(tmp_path):
    c = ctx_bot(tmp_path)
    t = F.plano(S.resp_rank(c, ["6,5", "11"]))
    assert "vas +6,5% y el objetivo es +11,0%" in t and "Te faltan 4,5 puntos" in t and "1,24 veces" in t
    assert Estado(CFG, c.estado_path).rent["mia"] == 6.5
    assert "lo mejor es mantener" in F.plano(S.resp_rank(c, ["12", "11"]))
    assert "❌" in S.resp_rank(c, ["abc", "11"])
    assert "vas +6,0% y el objetivo es +11,0%" in F.plano(S.resp_rank(c, ["6"]))                         # con un solo número se conserva el corte que ya habías dado
    assert "Cómo usar /rank" in S.resp_rank(ctx_bot(tmp_path / "nuevo"), ["6"])                          # la primera vez sí hacen falta los dos
    sin_jerga(S.resp_rank(c, ["6,5", "11"]))


def test_rank_semana_final_pide_el_segundo_si_voy_primero(tmp_path):
    c = ctx_bot(tmp_path, bog(2026, 11, 4, 21))
    assert "ventaja sobre el segundo" in S.resp_rank(c, ["20", "20"])
    assert "ventaja sobre el segundo" not in S.resp_rank(c, ["20", "20", "11"])
    assert Estado(CFG, c.estado_path).rent["segundo"] == 11.0 and Estado(CFG, c.estado_path).rent["primero"] == 20.0


def test_pos_registra_cambio_con_validaciones(tmp_path):
    c = ctx_bot(tmp_path)
    assert "Entrada inicial registrada" in S.resp_pos(c, ["TSLA", "97M", "380,5"])
    t = F.plano(S.resp_pos(c, ["META", "95.000.000", "720,5"]))
    assert "Cambio TSLA → META registrado" in t and "Ahora tienes META: $95.000.000" in t and "te quedan: 3" in t
    assert "1 cambio por día" in S.resp_pos(c, ["NVDA", "95M", "200"])
    assert "LISTA NEGRA" in S.resp_pos(c, ["ETB", "10M", "5"]) and "universo" in S.resp_pos(c, ["AMD", "10M", "5"])
    assert "Cómo usar /pos" in S.resp_pos(c, ["META"]) and "❌" in S.resp_pos(c, ["META", "mucho", "5"])


def test_op_cuenta_la_operacion_y_celebra_la_meta(tmp_path):
    c = ctx_bot(tmp_path)
    respuestas = [F.plano(S.resp_op(c)) for _ in range(4)]
    assert "llevas 1 de 4" in respuestas[0] and "cumpliste la meta" not in respuestas[2]
    assert "llevas 4 de 4" in respuestas[3] and "cumpliste la meta" in respuestas[3]
    assert Estado(CFG, c.estado_path).ops_total() == 4


def test_semaforo_y_noticias_rechazan_activos_prohibidos(tmp_path):
    c = ctx_bot(tmp_path)
    assert "❌" in S.resp_semaforo(c, "ETB") and "LISTA NEGRA".lower() not in "" and "❌" in S.resp_noticias(c, "popular")


def test_purgar_cache_borra_todo_menos_el_contador_de_alpha_vantage(tmp_path):
    c = ctx_bot(tmp_path)
    c.f.cache = SimpleNamespace(dir=tmp_path / "cache")
    c.f.cache.dir.mkdir()
    for nombre in ("a.json", "b.parquet", "av_contador.json"):
        (c.f.cache.dir / nombre).write_text("x")
    assert S.purgar_cache(c) == 2 and [p.name for p in c.f.cache.dir.iterdir()] == ["av_contador.json"]


# =============================== bot: seguridad ===============================
FAKE_BOT = SimpleNamespace(username="BolsaMillonariaClaudeBot")           # check_update sólo necesita el nombre de usuario del bot


def update_de(chat_id: int, texto: str, bot=None):
    from telegram import Chat, Message, MessageEntity, Update, User
    msg = Message(message_id=1, date=dt.datetime.now(dt.timezone.utc), chat=Chat(id=chat_id, type="private"),
                  from_user=User(id=chat_id, first_name="x", is_bot=False), text=texto,
                  entities=[MessageEntity(type="bot_command", offset=0, length=len(texto.split()[0]))])
    upd = Update(update_id=1, message=msg)
    if bot is not None:
        msg.set_bot(bot)
        upd.set_bot(bot)
    return upd


def test_el_bot_registra_los_comandos_y_solo_responde_a_mi_chat():
    import bot
    cfg = copy.deepcopy(CFG)
    cfg["bot"]["responder_a"] = "solo_mi_chat"                                                          # por defecto el bot es abierto (ver tests/test_fase6.py); aquí se prueba el modo cerrado
    app = bot.construir_app("123456:ABCDEF-token-falso-para-pruebas", 123456789, cfg)
    handlers = [h for g in app.handlers.values() for h in g if hasattr(h, "commands")]                  # sólo los comandos (también hay botones y texto libre)
    nombres = {next(iter(h.commands)) for h in handlers}
    assert {"estado", "semaforo", "detalle", "banco", "noticias", "catalizadores", "actualizar", "rank", "pos", "op", "ayuda"} <= nombres
    for h in handlers:
        assert h.check_update(update_de(123456789, f"/{next(iter(h.commands))}", FAKE_BOT)) not in (None, False)     # mi chat: responde
        assert h.check_update(update_de(999999, f"/{next(iter(h.commands))}", FAKE_BOT)) in (None, False)             # cualquier otro: silencio


def test_el_bot_responde_en_html_y_si_telegram_lo_rechaza_reenvia_en_texto_plano():
    import asyncio

    import bot
    from telegram.error import BadRequest

    class Mensaje:
        def __init__(self, rechazar_html):
            self.rechazar_html, self.enviados = rechazar_html, []

        async def reply_text(self, texto, parse_mode=None, disable_web_page_preview=None):
            if parse_mode == "HTML" and self.rechazar_html:
                raise BadRequest("Can't parse entities: unsupported start tag")
            self.enviados.append((texto, parse_mode))

    ok, roto = Mensaje(False), Mensaje(True)
    asyncio.run(bot.responder(SimpleNamespace(message=ok), "<b>Hola</b> &amp; adiós"))
    asyncio.run(bot.responder(SimpleNamespace(message=roto), "<b>Hola</b> &amp; adiós"))
    assert ok.enviados == [("<b>Hola</b> &amp; adiós", "HTML")] and roto.enviados == [("Hola & adiós", None)]
