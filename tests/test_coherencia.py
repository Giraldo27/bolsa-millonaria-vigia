"""Pruebas de COHERENCIA: el bot no puede recomendar en una respuesta lo que en otra desaconseja, ni dar dos nombres distintos para "la que compraría".

Caso que las motivó (6-oct-2026): /comprar decía "Si vas a comprar una: ECOPETROL", /banco decía que la mejor era BCOLOMBIA, /ganar nombraba PFGRUPOARG,
/revisar decía que Ecopetrol tenía "más en contra que a favor", y el efecto de la macro se contaba a favor aunque la acción iba en la dirección contraria."""
import datetime as dt
from types import SimpleNamespace

import pytest

from src import analisis as A
from src import formato as F
from src import liquidez as LQ
from src import macro as M
from src import servicio as S
from src.regla_maestra import Decision
from src.relevo import Candidato, EntradaBanco
from src.semaforo import VERDE
from tests.helpers import make_cfg
from tests.test_fase4 import bog, mk_res, sin_jerga
from tests.test_macro import ctx

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)
BANCO = [EntradaBanco(Candidato(t, "local", m, "σ20", valor_negociado_mm=v)) for t, m, v in (("BCOLOMBIA", 0.032, 10100), ("ECOPETROL", 0.029, 42500), ("CEMARGOS", 0.028, 7200))]
CORTE = dict(fecha=dt.date(2026, 10, 9), top_pct=50)
SIN_RANK = Decision("SIN_DATOS", codigo="sin_ranking")


def liq(nivel):
    return lambda f, t, cfg: dict(ticker=t, simbolo=t + ".CL", nivel=nivel, mediana_mm=5000.0, hoy_mm=4000.0, acciones_hoy=1000.0, dias_sin_negociar=0, dias=20, motivos=[])


@pytest.fixture
def c(tmp_path, monkeypatch):
    x = ctx(tmp_path, SimpleNamespace(avisos=[]))
    with S.transaccion(x) as e:
        e.d["posicion"]["activo"] = "NUCO"
    monkeypatch.setattr(LQ, "medir", liq(LQ.BUENA))
    monkeypatch.setattr(S, "mee_de", lambda f, t, s, ahora, cfg: (0.03, "historia", None))
    monkeypatch.setattr(S, "_tickers_macro", lambda ctx_, est: [])
    monkeypatch.setattr(S, "buscar_noticias", lambda ctx_, horas=12, lector=None: ([], {}))
    monkeypatch.setattr(M, "tablero", lambda f, cfg, ahora=None: [])
    monkeypatch.setattr(M, "sensibilidades", lambda f, cfg, tickers: {})
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(VERDE, ticker=t, r_hoy=0.004, z=0.2), None, None))
    return x


def petroleo(cambio, beta=0.3):
    return (lambda f, cfg, ahora=None: [dict(clave="petroleo", nombre="el petróleo (Brent)", cambio=cambio, cuando="hoy")]), (lambda f, cfg, tickers: {"petroleo": {"ECOPETROL": {"beta": beta}}})


# =============================== la macro sólo cuenta si la acción la está siguiendo ===============================
def test_si_el_petroleo_sube_pero_ecopetrol_baja_no_se_cuenta_a_favor(c, monkeypatch):
    """El error que el usuario vio: "compra Ecopetrol porque subió el petróleo" cuando la acción iba para abajo."""
    tab, sens = petroleo(+0.03)
    monkeypatch.setattr(M, "tablero", tab)
    monkeypatch.setattr(M, "sensibilidades", sens)
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(VERDE, ticker=t, r_hoy=-0.009, z=-0.4), None, None))
    ch = A.chequeo(c, "ECOPETROL", S.leer_estado(c))
    assert not any("macro" in p for p in ch["pros"]) and not any("macro" in x for x in ch["contras"])
    t = F.plano("\n".join(ch["lineas"]))
    assert "el petróleo (Brent) sube 3,0% → a ECOPETROL suele sumarle 0,9%" in t and "Pero hoy ECOPETROL no lo está siguiendo (baja 0,9%)" in t
    assert "No lo cuento ni a favor ni en contra" in t and "(a las 10:00, frente al cierre de ayer)" in t


def test_si_la_accion_si_sigue_a_la_macro_entonces_cuenta(c, monkeypatch):
    tab, sens = petroleo(+0.03)
    monkeypatch.setattr(M, "tablero", tab)
    monkeypatch.setattr(M, "sensibilidades", sens)
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(VERDE, ticker=t, r_hoy=0.012, z=0.5), None, None))
    sube = A.chequeo(c, "ECOPETROL", S.leer_estado(c))
    assert any("la macro de hoy la favorece" in p and "la acción lo está siguiendo" in p for p in sube["pros"]) and sube["veredicto"] == "sin_contras"
    tab, sens = petroleo(-0.03)
    monkeypatch.setattr(M, "tablero", tab)
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(VERDE, ticker=t, r_hoy=-0.012, z=-0.5), None, None))
    baja = A.chequeo(c, "ECOPETROL", S.leer_estado(c))
    assert any("la macro de hoy la empuja hacia abajo" in x for x in baja["contras"]) and baja["veredicto"] == "en_contra"


# =============================== /comprar nunca nombra una acción que /revisar desaconseja ===============================
def test_comprar_y_revisar_dan_siempre_el_mismo_veredicto(c, monkeypatch):
    def evaluar(f, t, ahora, cfg, est, p, m):                                             # BCOLOMBIA viene de un salto fuerte: tiene algo en contra
        return (mk_res(VERDE, ticker=t, r_hoy=0.06 if t == "BCOLOMBIA" else 0.004, z=3.0 if t == "BCOLOMBIA" else 0.2), None, None)
    monkeypatch.setattr(S, "evaluar_activo", evaluar)
    buenas, ch = A.banco_revisado(c, BANCO)
    assert [e.c.ticker for e in buenas] == ["ECOPETROL", "CEMARGOS"] and ch["BCOLOMBIA"]["veredicto"] == "en_contra"
    t = F.plano(F.msg_que_comprar(BANCO, "NUCO", 0.046, SIN_RANK, CORTE, 3, [], "08:45", CFG, eleccion=dict(ticker=buenas[0].c.ticker, chequeos=ch)))
    assert "Si vas a comprar una: ECOPETROL." in t and "Descartadas hoy: BCOLOMBIA: viene de una subida fuerte" in t
    assert t.index("1. 🟢 BCOLOMBIA") < t.index("2. 🟢 ECOPETROL")                         # la lista sigue ordenada por movimiento: sin "empates" que la reordenen
    for ticker, veredicto in (("BCOLOMBIA", "Hoy tiene más en contra que a favor"), ("ECOPETROL", "Hoy no tiene nada serio en contra")):
        assert veredicto in F.plano(A.resp_revisar(c, [ticker]))                           # /revisar dice lo mismo de cada una
    sin_jerga(t)


def test_si_todas_tienen_algo_en_contra_no_recomienda_ninguna(c, monkeypatch):
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(VERDE, ticker=t, r_hoy=0.05, z=2.6), None, None))
    buenas, ch = A.banco_revisado(c, BANCO)
    assert buenas == []
    t = F.plano(F.msg_que_comprar(BANCO, "NUCO", 0.046, SIN_RANK, CORTE, 3, [], "08:45", CFG, eleccion=dict(ticker=None, chequeos=ch)))
    assert "Hoy no compraría ninguna" in t and "Si vas a comprar una" not in t and "BCOLOMBIA: viene de una subida fuerte" in t and "Nada obliga a entrar hoy" in t


def test_el_orden_del_banco_es_estricto_y_el_mejor_candidato_es_el_primero():
    """Con la tolerancia en 0 no hay "empatadas" que se reordenen: /comprar, /banco, los cambios sugeridos y la Regla Maestra nombran a la misma."""
    from src.relevo import construir_banco
    assert CFG["banco"]["tolerancia_mee"] == 0
    cs = [Candidato("ECOPETROL", "local", 0.029, corr=0.05, valor_negociado_mm=42500), Candidato("BCOLOMBIA", "local", 0.032, corr=0.60, valor_negociado_mm=10100)]
    assert [e.c.ticker for e in construir_banco(cs, "NUCO", CFG)] == ["BCOLOMBIA", "ECOPETROL"]


# =============================== una sola postura sobre lo que se negocia poco ===============================
def test_lo_que_se_negocia_poco_se_dice_en_una_linea_y_con_la_misma_postura(c, monkeypatch):
    niveles = {"META": LQ.MALA, "TSLA": LQ.MALA, "NUCO": LQ.JUSTA, "GRUPOARGOS": LQ.JUSTA, "ECOPETROL": LQ.BUENA}
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: dict(ticker=t, nivel=niveles[t]))
    (linea,) = S.resumen_liquidez(c, ["ECOPETROL", "GRUPOARGOS", "META", "NUCO", "TSLA"])
    t = F.plano(linea)
    assert t == ("💧 Liquidez en trii: META y TSLA casi no se negocian; GRUPOARGOS y NUCO se negocian poco o a ratos. No compres más de estas; "
                 "si vendes, que sea con orden límite y sin prisa. Detalle: /liquidez")
    assert S.resumen_liquidez(c, ["ECOPETROL"]) == []
    for frase in (LQ.frase(dict(ticker="TSLA", nivel=LQ.MALA, mediana_mm=184.0, acciones_hoy=42.0, motivos=["poco"])), linea):
        assert "No la operes" not in frase and "orden límite" in frase                    # nunca "no la operes" de algo que ya tienes


# =============================== el ranking viejo o que no cuadra se avisa ===============================
def test_un_ranking_viejo_o_que_no_cuadra_con_la_cartera_se_avisa(c, monkeypatch):
    from src import cartera as CA
    monkeypatch.setattr(CA, "valorar", lambda f, cartera, cfg, tasa=None: [])
    monkeypatch.setattr(CA, "trm", lambda f, cfg: 3200.0)
    monkeypatch.setattr(CA, "rent_total", lambda filas: 0.029)
    assert S.nota_ranking(c, S.leer_estado(c)) == ""                                      # sin ranking no hay nada que revisar
    with S.transaccion(c) as e:
        e.actualizar_rank(-0.3, 10.0, bog(2026, 10, 5, 17, 0))
    t = F.plano(S.nota_ranking(c, S.leer_estado(c)))
    assert "Revisa tu ranking" in t and "(-0,3%) es del lun 05/10" in t and "Por mis cuentas tu cartera va hoy cerca de +2,9%" in t and "voy X y el corte está en Y" in t
    with S.transaccion(c) as e:
        e.actualizar_rank(2.5, 10.0, AHORA)
    assert S.nota_ranking(c, S.leer_estado(c)) == ""                                      # de hoy y cuadra con la cartera: no se molesta


# =============================== los titulares del semáforo son de la empresa, no del mercado ===============================
def test_el_semaforo_no_muestra_titulares_generales_como_si_fueran_de_la_empresa(c):
    clave = lambda t: dict(titulo=t, puntaje=0.1, palabras=[], idioma="es")               # noqa: E731
    argos = mk_res(VERDE, ticker="GRUPOARGOS", n_24h=3, claves=[clave("CEO de Saudi Aramco: inventarios de petróleo al límite"), clave("Acciones de Marvell suben tras ambiciosas metas")])
    S.titulares_propios(c, argos)
    t = F.plano("\n".join(F.bloque_noticias(argos)))
    assert argos.claves == [] and "ningún titular de las últimas 24 horas habla directamente de GRUPOARGOS" in t and "Aramco" not in t
    meta = mk_res(VERDE, ticker="META", n_24h=3, claves=[clave("Meta se enfrenta a una investigación en el Reino Unido"), clave("Marvell fija ambiciosas metas"), clave("Instagram cambia sus reglas")])
    S.titulares_propios(c, meta)
    assert [k["titulo"][:9] for k in meta.claves] == ["Meta se e", "Instagram"]            # "metas" no es Meta
    eco = mk_res(VERDE, ticker="ECOPETROL", n_24h=1, claves=[clave("Ecopetrol anuncia hallazgo en el Caribe")])
    assert len(S.titulares_propios(c, eco).claves) == 1 and not getattr(eco, "solo_generales", False)


# =============================== el silencio no parece una falla ===============================
def test_con_la_bolsa_abierta_y_mucho_rato_sin_avisos_se_manda_un_sigo_vigilando(c, monkeypatch):
    enviados = []
    monkeypatch.setattr(S, "enviar_o_encolar", lambda ctx_, texto, botones=None: enviados.append(texto) or True)
    mem = SimpleNamespace(d=dict(salud={"Superfinanciera": True, "Valora Analitik": True, "Semana": False},
                                 senales=[dict(ts="2026-10-06T14:40+00:00", ticker="ECOPETROL")]))
    monkeypatch.setattr(S, "ULTIMO_AVISO", S.time.monotonic() - 30 * 60)
    assert S.latido(c, mem, 120) is None                                                  # sólo 30 minutos de silencio: no se molesta
    monkeypatch.setattr(S, "ULTIMO_AVISO", S.time.monotonic() - 121 * 60)
    t = F.plano(S.latido(c, mem, 120))
    assert "Sigo vigilando" in t and "en las últimas 2 horas" in t and "La última fue a las 09:40 (ECOPETROL)" in t and "Fuentes de noticias respondiendo: 2 de 3" in t
    assert len(enviados) == 1 and S.latido(c, mem, 0) is None                             # 0 = apagado
    noche = ctx(c.estado_path.parent, SimpleNamespace(avisos=[]), ahora=bog(2026, 10, 6, 20, 0))
    assert S.latido(noche, mem, 120) is None                                              # con la bolsa cerrada no se manda


def test_los_avisos_automaticos_van_al_chat_principal_y_a_los_grupos_suscritos(c, monkeypatch):
    destinos = []
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "111")
    monkeypatch.setattr(S, "enviar", lambda texto, cfg, env=None, botones=None, **k: destinos.append((env or {}).get("TELEGRAM_CHAT_ID", "111")) or True)
    with S.transaccion(c) as e:
        e.d["suscriptores"] = [-1009999, 111]
    c.dry = False
    assert S.enviar_o_encolar(c, "noticia") and sorted(destinos) == ["-1009999", "111"]    # una vez a cada uno, sin repetir el principal


# =============================== /ganar y /reemplazo dicen lo mismo de quién se mueve más ===============================
def test_ganar_y_reemplazo_no_se_contradicen_sobre_cual_se_mueve_mas(c, monkeypatch):
    ratos = dict(mediana=0.63, flojo=0.59, pausa=25.0, dias=21)
    def fila(t, corte, nivel, grupo="local", cont=None, a_ratos=False, base=False):
        return dict(ticker=t, grupo=grupo, mee_corte=corte, mee=corte * 2.6, vs_base=corte / 0.040, color="VERDE", nivel=nivel, liquida=nivel == LQ.BUENA, a_ratos=a_ratos,
                    continuidad=cont, liquidez_mm=5000.0, es_base=base, supera=False)
    filas = [fila("GRUPOARGOS", 0.044, LQ.JUSTA, cont=ratos, a_ratos=True), fila("NUCO", 0.040, LQ.JUSTA, "mgc", base=True), fila("BCOLOMBIA", 0.032, LQ.BUENA)]
    monkeypatch.setattr(A, "_filas", lambda ctx_, base: [dict(f, es_base=f["ticker"] == base) for f in filas])
    monkeypatch.setattr(A, "_valores", lambda ctx_, est: {})
    monkeypatch.setattr(S, "panorama_macro", lambda ctx_: ([], {}))
    monkeypatch.setattr(S, "senales_recientes", lambda horas, ahora=None: [])
    monkeypatch.setattr(LQ, "medir", lambda f, t, cfg: dict(ticker=t, nivel=LQ.BUENA if t == "BCOLOMBIA" else LQ.JUSTA, mediana_mm=5000.0))
    g, r = F.plano(A.resp_ganar(c)), F.plano(A.resp_reemplazo(c, ["nuco"]))
    assert "GRUPOARGOS se mueve incluso más que NUCO" in r and "se queda fuera del filtro de liquidez por poco" in r
    assert "Ninguna de las que se negocian bien en trii se mueve más" in g and "GRUPOARGOS se mueve un poco más —1,10 veces—, pero se negocia a ratos" in g
    assert "Es la que más puede moverse" not in g                                         # la frase que contradecía a /reemplazo
    for texto in (g, r):                                                                   # el mismo número y la misma candidata líquida en los dos
        assert "±4,0% hasta el corte" in texto and "BCOLOMBIA" in texto
    assert "/revisar bcolombia" in g and "/revisar bcolombia" in r


def test_sin_precio_de_hoy_la_macro_tampoco_se_cuenta(c, monkeypatch):
    """Caso real (7-oct-2026, 8:46): el dato de la acción era de ayer y aun así se decía "la macro la favorece y la acción lo está siguiendo"."""
    tab, sens = petroleo(+0.03)
    monkeypatch.setattr(M, "tablero", tab)
    monkeypatch.setattr(M, "sensibilidades", sens)
    monkeypatch.setattr(S, "evaluar_activo", lambda f, t, ahora, cfg, est, p, m: (mk_res(VERDE, ticker=t, r_hoy=-0.009, z=-0.4, fecha=dt.date(2026, 10, 5)), None, None))
    ch = A.chequeo(c, "ECOPETROL", S.leer_estado(c))
    t = F.plano("\n".join(ch["lineas"]))
    assert not any("macro" in x for x in ch["pros"] + ch["contras"])
    assert "bajó 0,9% en la última sesión (lun 05/10)" in t and "Todavía no tengo el precio de hoy de ECOPETROL para confirmar que lo esté siguiendo" in t


# =============================== nunca la misma noticia dos veces ===============================
# Casos reales del 8 al 12 de octubre de 2026: lo que el bot envió repetido.
def _mem(tmp_path):
    from src import noticias_bvc as N
    m = N.Memoria(tmp_path / "rep.json")
    m.d["mercado"] = dict(base="x", vistos={}, temas={}, titulos=[], ultimas=[], historias=[])
    return m


class _Lec:
    """Lector de prueba: entrega los titulares que se le pongan y "traduce" con un diccionario fijo."""
    en_ingles = {"Bloomberg (tecnología)", "Investing.com"}
    TRAD = {"Tencent Kicks Off $5 Billion Bond Sale to Finance AI Spending": "Tencent inicia la venta de bonos por valor de 5.000 millones de dólares para financiar el gasto en IA",
            "Tencent sells dollar, dim sum bonds amid AI spending spree": "Tencent vende bonos en dólares y dim sum en medio de una ola de gastos en IA",
            "Gold slips near $4,160 as Middle East tensions fuel rate-hike concerns": "El oro cae cerca de 4.160 dólares mientras las tensiones en Oriente Medio alimentan la preocupación por una subida de tasas"}

    def __init__(self):
        self.items = []

    def todo(self, ahora, completo=False):
        return list(self.items), {"x": True}

    def traducir(self, t):
        return self.TRAD.get(t)


def _it(titulo, fuente, minutos_despues=0):
    from src import noticias_bvc as N
    return N.Item(AHORA.astimezone(dt.timezone.utc) + dt.timedelta(minutes=minutos_despues - 1), titulo, "", fuente, "https://e.co/" + str(abs(hash(titulo)) % 9999), "rss")


def _pasan(tmp_path, entregas):
    """Va entregando titulares (titulo, fuente, minutos) uno por uno, como llegan en la realidad, y devuelve los que el bot avisaría."""
    from src import mercado as MK
    mem, lec, salen = _mem(tmp_path), _Lec(), []
    for titulo, fuente, minuto in entregas:
        lec.items.append(_it(titulo, fuente, minuto))
        nuevas, _ = MK.ronda(lec, mem, AHORA + dt.timedelta(minutes=minuto), CFG)
        salen += [a["item"].titulo for a in nuevas]
    return salen


def test_la_misma_noticia_en_tres_medios_y_dos_idiomas_sale_una_sola_vez(tmp_path):
    salen = _pasan(tmp_path, [("Tencent prepara una emisión de bonos por US$5.000 millones para financiar sus inversiones en IA", "Bloomberg Línea", 0),
                              ("Tencent Kicks Off $5 Billion Bond Sale to Finance AI Spending", "Bloomberg (tecnología)", 69),
                              ("Tencent sells dollar, dim sum bonds amid AI spending spree", "Investing.com", 210)])
    assert salen == ["Tencent prepara una emisión de bonos por US$5.000 millones para financiar sus inversiones en IA"]


def test_la_misma_cifra_del_mismo_tema_es_la_misma_noticia_aunque_cambie_el_idioma(tmp_path):
    salen = _pasan(tmp_path, [("El oro cae cerca de los 4.160 dólares por las tensiones en Oriente Próximo", "Investing.com (español)", 0),
                              ("Gold slips near $4,160 as Middle East tensions fuel rate-hike concerns", "Investing.com", 2)])
    assert len(salen) == 1


def test_el_mismo_tema_contado_otra_vez_no_sale_pero_un_giro_si(tmp_path):
    salen = _pasan(tmp_path, [("Las acciones de chips caen en Asia tras las pérdidas de los semiconductores en EE. UU.", "Bloomberg Línea", 0),
                              ("Los fabricantes de chips coreanos se hunden por el temor a rivales chinos", "Valora Analitik", 33),      # misma dirección, 33 minutos después
                              ("La liquidación de semiconductores provoca nerviosismo en Seúl", "La República", 86),                    # sin dirección clara: también es la misma
                              ("Los chips rebotan con fuerza: Nvidia y TSMC se disparan tras nuevos pedidos", "Valora Analitik", 100)])   # GIRO: de caer a subir
    assert len(salen) == 2 and "caen en Asia" in salen[0] and "rebotan" in salen[1]


def test_un_tema_distinto_o_una_noticia_distinta_del_mismo_tema_horas_despues_si_salen(tmp_path):
    salen = _pasan(tmp_path, [("Petróleo Brent se dispara 5% por la tensión en Medio Oriente", "Bloomberg Línea", 0),
                              ("Dólar en Colombia cae y la TRM rompe los $3.200", "Valora Analitik", 5),                                # otro tema: sale
                              ("El crudo sube otra vez tras nuevos ataques", "La República", 30),                                        # mismo tema y dirección, 30 min: no
                              ("La OPEP anuncia un recorte sorpresa y el barril salta a US$112", "Portafolio", 200)])                    # 3 horas después y con hechos nuevos: sale
    assert len(salen) == 3 and not any("otra vez" in t for t in salen)


@pytest.mark.parametrize("primera, segunda", [
    ("Bancolombia acuerda aumento salarial del 10,15 % y más beneficios en salud y vivienda para 11.000 empleados",
     "Bancolombia presentó nuevos beneficios para sus empleados tras conversaciones sindicales"),                                         # de qué trata: lo laboral
    ("Bancolombia acuerda aumento salarial del 10,15 % y más beneficios en salud y vivienda para 11.000 empleados",
     "Paz laboral en Bancolombia: ¿de cuánto será el incremento salarial en el banco más grande del país?"),
    ("Economía colombiana pierde tracción en el tercer trimestre de 2026: NowCast Bancolombia proyecta un PIB de 2,1 %",
     "Bancolombia estimó que la economía colombiana creció apenas 2,1% en el tercer trimestre"),                                          # la misma cifra
    ("Bancolombia aportó 46,6% de la financiación a Altia para adquisición de portafolio a Isagen",
     "Bancolombia impulsa la adquisición de activos de energía renovable de Altia a ISAGEN en Colombia"),
    ("Senador Enrique Gómez pide al presidente de Ecopetrol investigar si la compañía dio beneficios especiales a la firma Química Comercial Andina",
     "Senador Enrique Gómez pide a Ecopetrol investigar ventas de polietileno a empresa Química Comercial Andina"),
])
def test_las_noticias_de_empresa_que_llegaron_repetidas_ya_no_se_repiten(primera, segunda):
    from src import noticias_bvc as N
    f = N.firma(primera)
    entradas = [["2026-10-08T22:12:00+00:00", f["pal"], f["con"], f["cif"], f["ent"]]]
    assert N.repetida_emisor(segunda, entradas, CFG["noticias_bvc"]["parecido_repetida"], "2026-10-09T16:00:00+00:00")


def test_dos_noticias_distintas_de_la_misma_empresa_si_salen():
    from src import noticias_bvc as N
    f = N.firma("SIC sanciona a Celsia con multa millonaria")
    entradas = [["2026-10-06T15:00:00+00:00", f["pal"], f["con"], f["cif"], f["ent"]]]
    ahora = "2026-10-06T16:30:00+00:00"
    assert not N.repetida_emisor("Superservicios impone sanción a Celsia y abre investigación", entradas, 0.5, ahora)       # otra entidad sanciona: es otra noticia
    assert not N.repetida_emisor("Celsia gana contrato de energía solar en Panamá", entradas, 0.5, ahora)                  # otro asunto
    assert N.repetida_emisor("La SIC multa a Celsia", entradas, 0.5, ahora)                                               # la misma, más corta
    assert N.cifras_de("bonos por US$5.000 millones") == N.cifras_de("a $5 billion bond sale") == N.cifras_de("5 mil millones de dólares") == {"5000M"}
    assert N.cifras_de("creció 2,1 % en 2026") == {"21%"} and N.cifras_de("fecha límite el 31 de diciembre de 2026") == set()


def test_al_actualizar_el_bot_lo_que_ya_habia_avisado_tampoco_se_repite(tmp_path):
    """La memoria vieja no traía el registro de historias: se arma con los avisos ya enviados, para que el primer minuto tras una actualización no los repita."""
    from src import mercado as MK
    from src import noticias_bvc as N
    mem = N.Memoria(tmp_path / "vieja.json")
    hace_1h = (AHORA.astimezone(dt.timezone.utc) - dt.timedelta(hours=1)).isoformat(timespec="seconds")
    mem.d["mercado"] = dict(base="x", vistos={}, temas={}, titulos=[], ultimas=[dict(ts=hace_1h, titulo="Tencent prepara una emisión de bonos por US$5.000 millones para financiar sus inversiones en IA",
                                                                                   fuente="Bloomberg Línea", url="", nombres=["Inteligencia artificial y chips"], empresas=[], direccion=0)])
    lec = _Lec()
    lec.items = [_it("Tencent Kicks Off $5 Billion Bond Sale to Finance AI Spending", "Bloomberg (tecnología)")]
    assert MK.ronda(lec, mem, AHORA, CFG)[0] == [] and len(mem.d["mercado"]["historias"]) == 1
