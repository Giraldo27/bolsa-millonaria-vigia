"""Pruebas del análisis "¿esta noticia es buena o mala para la acción?" cuando el titular no lo dice a simple vista."""
import json
from types import SimpleNamespace

import pytest

from src import analisis_noticia as A
from src import formato as F
from src import noticias_bvc as N
from tests.helpers import make_cfg
from tests.test_fase4 import bog

CFG = make_cfg()
SIN = {"analisis_noticias": {"leer_articulo": False}}
AHORA = bog(2026, 10, 6, 10, 0)


@pytest.mark.parametrize("titulo, sentido", [
    ("Ecopetrol refuerza su apuesta por el gas en el Caribe y pone a Venezuela en el radar", 1),      # el titular de la captura del usuario
    ("ISA gana contrato de transmisión en Perú", 1), ("Grupo Sura anunció el inicio del programa de readquisición de acciones", 1),
    ("Ecopetrol une petróleo y energía solar en proyecto aprobado de US$7.400 millones", 1), ("Celsia reduce su deuda en $300.000 millones", 1),
    ("Presidente de Ecopetrol renuncia en medio de investigación", -1), ("Ecopetrol no pagará dividendo extraordinario este año", -1),
    ("Presidente de la junta de Ecopetrol cuestiona a un medio", -1), ("Sindicato anuncia paro en refinería de Ecopetrol", -1),
    ("Fitch rebaja la calificación de Grupo Argos", -1), ("Cementos Argos retrasa la venta de su negocio en EE. UU.", -1),
    ("Celsia presenta resultados del tercer trimestre", 0), ("Grupo Argos y Nicky Jam donarán 100 viviendas", 0),
    ("Ecopetrol y Petrobras culminan exploración gasífera offshore en el Caribe", 0),
])
def test_el_analisis_de_frases_reconoce_lo_bueno_lo_malo_y_lo_neutro(titulo, sentido):
    r = A.analizar("X", titulo, cfg=SIN, env={})
    assert r["sentido"] == sentido, (r, titulo)
    assert (r["motivos"] != []) == (sentido != 0) or sentido == 0


def test_las_negaciones_invierten_el_sentido_y_se_explica_el_motivo():
    total, hall = A.puntos("Ecopetrol no pagará dividendo extraordinario")
    assert total < 0 and ("no dividendo extraordinario", -2.0) in hall
    assert A.puntos("La empresa cerró el año sin pérdidas")[0] > 0
    r = A.analizar("X", "Ecopetrol refuerza su apuesta por el gas", cfg=SIN, env={})
    assert r["motivos"] == ["refuerza o impulsa", "apuesta por crecer"] and r["metodo"] == "texto" and 0 < r["confianza"] <= 1


def test_si_el_titular_no_alcanza_se_lee_el_articulo():
    pagina = "<html><nav>menú con caída y crisis</nav><p>La compañía informó que la junta aprobó un nuevo contrato por US$200 millones y que sus ingresos crecieron 15 % frente al año anterior.</p></html>"
    http = SimpleNamespace(get=lambda url, **k: SimpleNamespace(status_code=200, text=pagina))
    r = A.analizar("Celsia", "Celsia presenta novedades", url="https://medio.co/nota", cfg=CFG, http=http, env={})
    assert r["sentido"] == 1 and r["metodo"] == "articulo" and "nuevo proyecto o contrato" in r["motivos"]      # el menú de la página no cuenta
    assert A.leer_articulo("https://news.google.com/rss/articles/x", http) == "" and A.leer_articulo("https://x.co/a.pdf", http) == ""
    roto = SimpleNamespace(get=lambda url, **k: (_ for _ in ()).throw(RuntimeError("sin red")))
    assert A.analizar("Celsia", "Celsia presenta novedades", url="https://medio.co/nota", cfg=CFG, http=roto, env={})["sentido"] == 0   # sin artículo: neutra, no se inventa


def test_con_clave_el_modelo_decide_y_sin_clave_o_si_falla_se_usan_las_frases():
    llamadas = []

    def post(url, **k):
        llamadas.append((url, k))
        return SimpleNamespace(json=lambda: {"content": [{"type": "text", "text": 'Claro: {"sentido": "-", "confianza": 0.8, "razon": "Aumenta el riesgo regulatorio"}'}]})
    http = SimpleNamespace(post=post, get=lambda url, **k: SimpleNamespace(status_code=404, text=""))
    r = A.analizar("Ecopetrol", "Ecopetrol refuerza su apuesta por el gas", cfg=CFG, http=http, env={"ANTHROPIC_API_KEY": "clave-falsa"})
    assert r == dict(sentido=-1, confianza=0.8, motivos=["Aumenta el riesgo regulatorio"], metodo="modelo")
    url, k = llamadas[0]
    assert url == "https://api.anthropic.com/v1/messages" and k["headers"]["x-api-key"] == "clave-falsa" and k["json"]["model"] == CFG["analisis_noticias"]["modelo"]
    assert "Ecopetrol refuerza su apuesta por el gas" in k["json"]["messages"][0]["content"]
    assert A.analizar("Ecopetrol", "Ecopetrol refuerza su apuesta por el gas", cfg=CFG, http=http, env={})["metodo"] == "texto" and len(llamadas) == 1     # sin clave no se llama
    basura = SimpleNamespace(post=lambda url, **k: SimpleNamespace(json=lambda: {"error": "x"}), get=http.get)
    assert A.analizar("Ecopetrol", "Ecopetrol refuerza su apuesta por el gas", cfg=CFG, http=basura, env={"ANTHROPIC_API_KEY": "k"})["sentido"] == 1        # respuesta rara: frases


def test_el_aviso_explica_el_analisis_y_pone_el_signo():
    from tests.test_noticias_bvc import senal
    s = senal(0.004)
    s.nivel, s.analisis = "bajo", dict(sentido=1, confianza=0.75, motivos=["refuerza o impulsa", "apuesta por crecer"], metodo="texto")
    t = F.plano(F.msg_noticia_relevante(s, AHORA, CFG))
    assert "Análisis: positiva (+) para la acción · seguridad alta · por: refuerza o impulsa; apuesta por crecer" in t and "es automático y puede equivocarse" in t
    assert "Impacto estimado: +1,8%" in t
    s.analisis = dict(sentido=-1, confianza=0.3, motivos=["El mercado lo verá como más riesgo"], metodo="modelo")
    assert "negativa (−) para la acción · seguridad baja" in F.plano("\n".join(F.analisis_txt(s))) and "modelo de lenguaje" in F.plano("\n".join(F.analisis_txt(s)))
    s.analisis = None
    assert F.analisis_txt(s) == []


def test_en_la_ronda_el_titular_de_la_captura_sale_como_positiva_con_su_porcentaje(tmp_path, monkeypatch):
    from tests.test_noticias_bvc import ESTUDIO, Fuentes, LectorFalso, item
    monkeypatch.setattr(N, "cargar_estudio", lambda ruta=None: ESTUDIO)
    mem = N.Memoria(tmp_path / "n.json")
    lec = LectorFalso()
    N.ronda(lec, mem, Fuentes(), AHORA, CFG)
    lec.items = [item("Ecopetrol refuerza su apuesta por el gas en el Caribe y pone a Venezuela en el radar", fuente="CONtexto Ganadero")]
    (s,), _ = N.ronda(lec, mem, Fuentes({"ECOPETROL": 0.026}, vol=5e7), AHORA, CFG)
    assert s.sentido == 1 and s.impacto == pytest.approx(0.003) and s.analisis["sentido"] == 1 and s.nivel == "bajo"
    t = F.plano(F.msg_noticia_relevante(s, AHORA, CFG))
    assert "Análisis: positiva (+) para la acción" in t and "Impacto estimado: +0,3%" in t and "±" not in t      # positiva, pero no es un hecho del negocio: impacto pequeño
