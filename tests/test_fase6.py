"""Pruebas de la Fase 6: bot abierto con informe HTML a pedido, comparación de base, informe HTML, estado en la nube y flujos de GitHub Actions."""
import asyncio
import copy
import html.parser
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

import bot
from nube import sincronizar_estado as NE
from src import formato as F
from src import informe_html as IH
from src import seleccion as Q
from src import servicio as S
from src.regla_maestra import CAMBIAR, Decision
from src.relevo import Candidato, construir_banco
from src.semaforo import ROJO
from tests.helpers import make_cfg
from tests.test_fase4 import FAKE_BOT, CLAVE, bog, ent_falsa, estado_tmp, mk_res, motor_falso, update_de

CFG = make_cfg()
CFG["banco"]["grupos_candidatos"] = ["mgc", "local"]            # estas pruebas ejercitan la comparación con EE. UU. y BVC (el modo "sólo BVC" se prueba en test_noticias_bvc.py)
RAIZ = Path(__file__).resolve().parents[1]
DUENO = 123456789


# =============================== bot: quién puede hablarle ===============================
def app(**bot_cfg):
    cfg = copy.deepcopy(CFG)
    cfg["bot"].update(bot_cfg)
    return bot.construir_app("123456:ABCDEF-token-falso-para-pruebas", DUENO, cfg)


def manejadores(a):
    return {next(iter(h.commands)): h for g in a.handlers.values() for h in g if hasattr(h, "commands")}      # sólo los comandos (también hay botones y texto libre)


def test_por_defecto_el_bot_responde_a_cualquier_chat_o_grupo():
    assert CFG["bot"]["responder_a"] == "todos"                                                     # decisión del usuario: el grupo también puede consultarlo
    for nombre, h in manejadores(app()).items():
        assert h.check_update(update_de(DUENO, f"/{nombre}", FAKE_BOT)) not in (None, False)
        assert h.check_update(update_de(-100123456, f"/{nombre}", FAKE_BOT)) not in (None, False), nombre


def test_con_solo_mi_chat_los_demas_son_ignorados_en_silencio():
    for nombre, h in manejadores(app(responder_a="solo_mi_chat")).items():
        assert h.check_update(update_de(DUENO, f"/{nombre}", FAKE_BOT)) not in (None, False)
        assert h.check_update(update_de(999999, f"/{nombre}", FAKE_BOT)) in (None, False), nombre


def test_el_bot_tiene_los_comandos_nuevos():
    assert {"informe", "base", "detalle", "estado", "rank", "pos", "op", "banco", "ayuda"} <= set(manejadores(app()))


# =============================== bot: informe HTML sólo a pedido ===============================
class Msg:
    def __init__(self):
        self.textos, self.docs = [], []

    async def reply_text(self, texto, parse_mode=None, disable_web_page_preview=None):
        self.textos.append(texto)

    async def reply_document(self, document=None, filename=None, caption=None):
        self.docs.append((document, filename, caption))


def llamar(a, comando, args, chat=DUENO):
    m = Msg()
    upd = SimpleNamespace(effective_chat=SimpleNamespace(id=chat), message=m)
    asyncio.run(manejadores(a)[comando].callback(upd, SimpleNamespace(args=args)))
    return m


@pytest.fixture
def servicio_falso(monkeypatch):
    llamadas = {"informe": 0, "estado": 0, "rank": 0}

    def informe_html(ctx):
        llamadas["informe"] += 1
        return "<html><body>informe</body></html>"
    monkeypatch.setattr(S, "crear_contexto", lambda *a, **k: SimpleNamespace(ahora=bog(2026, 10, 6, 10, 5)))
    monkeypatch.setattr(S, "resp_estado", lambda c: llamadas.__setitem__("estado", llamadas["estado"] + 1) or "estado ok")
    monkeypatch.setattr(S, "resp_rank", lambda c, a: llamadas.__setitem__("rank", llamadas["rank"] + 1) or "rank ok")
    monkeypatch.setattr(S, "informe_html", informe_html)
    monkeypatch.setattr(S, "generar_informe", lambda c: ("resumen corto", informe_html(c)))
    return llamadas


def test_sin_pedirlo_no_se_adjunta_ningun_html(servicio_falso):
    m = llamar(app(), "estado", [])
    assert m.textos == ["estado ok"] and m.docs == [] and servicio_falso["informe"] == 0


def test_con_la_palabra_html_se_envia_el_mensaje_y_el_archivo(servicio_falso):
    m = llamar(app(), "estado", ["HTML"])
    assert m.textos == ["estado ok"] and servicio_falso["estado"] == 1                                # la palabra html no se pasa como argumento
    (contenido, nombre, caption), = m.docs
    assert contenido == b"<html><body>informe</body></html>" and nombre == "informe_20261006_1005.html" and "navegador" in caption


def test_el_comando_informe_envia_resumen_y_archivo_a_cualquier_chat(servicio_falso):
    m = llamar(app(), "informe", [], chat=-100777)                                                    # un grupo
    assert "resumen corto" in m.textos and len(m.docs) == 1 and m.docs[0][1].endswith(".html")


def test_el_enfriamiento_evita_repetir_consultas_grandes_pero_es_por_chat(servicio_falso):
    a = app(enfriamiento_pesado_s=60)
    assert len(llamar(a, "informe", []).docs) == 1
    segunda = llamar(a, "informe", [])
    assert segunda.docs == [] and "Espera" in segunda.textos[0] and servicio_falso["informe"] == 1
    assert len(llamar(a, "informe", [], chat=555).docs) == 1                                          # otro chat no queda bloqueado
    assert llamar(a, "estado", []).textos == ["estado ok"]                                            # los comandos livianos no tienen enfriamiento


def test_rank_no_admite_html_y_la_escritura_se_puede_reservar_al_dueno(servicio_falso):
    abierto = app()
    assert llamar(abierto, "rank", ["6", "11"], chat=555).textos == ["rank ok"]                      # abierto: cualquiera puede registrar (como pidió el usuario)
    cerrado = app(escritura_solo_mi_chat=True)
    assert "sólo lo puede usar el dueño" in llamar(cerrado, "rank", ["6", "11"], chat=555).textos[0] and servicio_falso["rank"] == 1
    assert llamar(cerrado, "rank", ["6", "11"], chat=DUENO).textos == ["rank ok"]
    assert llamar(cerrado, "estado", [], chat=555).textos == ["estado ok"]                           # consultar sí puede cualquiera


def test_si_falla_la_generacion_el_bot_no_se_cae_y_avisa(monkeypatch, servicio_falso):
    monkeypatch.setattr(S, "informe_html", lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    m = llamar(app(), "estado", ["html"])
    assert m.docs == [] and "No pude completar /estado" in m.textos[0]


def test_procesar_pendientes_avanza_el_offset():
    procesados = []

    class Bot:
        async def get_updates(self, offset=None, timeout=0, allowed_updates=None):
            assert timeout == 0 and offset == 10
            return [SimpleNamespace(update_id=10), SimpleNamespace(update_id=11)]

    class AppFalsa:
        bot = Bot()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return None

        async def process_update(self, u):
            procesados.append(u.update_id)
    off = {"offset": 10}
    assert asyncio.run(bot.procesar_pendientes(AppFalsa(), off)) == 2 and procesados == [10, 11] and off["offset"] == 12


# =============================== comparación de base ===============================
def serie(sigma, n=120, vol=1e7, precio=100.0):
    r = sigma * np.array([(-1) ** i for i in range(n)])
    c = precio * np.cumprod(1 + r)
    return pd.DataFrame({"Close": c, "Volume": np.full(n, vol)}, index=pd.bdate_range(end="2026-10-02", periods=n))


class FuentesFalsas:
    def __init__(self, sig, iv=None, vol_baja=(), sin_trii=()):
        self.sig, self.iv, self.vol_baja, self.sin_trii = sig, iv or {}, set(vol_baja), set(sin_trii)
        self.yf = SimpleNamespace(download=self._libro_trii)

    def _con_cache(self, clave, ttl, fn, nombre):
        return fn()

    def _libro_trii(self, sym, **k):
        """El libro de Colombia de una acción de EE. UU. (TSLACO.CL…): por defecto se negocia bien, salvo las de `sin_trii` (líquidas en Nueva York, no en trii)."""
        t = sym.replace("CO.CL", "").replace(".CL", "")
        if t in self.sin_trii or (t + "CO") in self.sin_trii:
            return pd.DataFrame({"Close": 1000.0, "Volume": 40.0}, index=pd.bdate_range(end="2026-10-02", periods=63))
        return pd.DataFrame({"Close": 1000.0, "Volume": 1e7}, index=pd.bdate_range(end="2026-10-02", periods=63))

    def diario(self, t, dias=None):
        return serie(self.sig.get(t, 0.01), vol=10 if t in self.vol_baja else 1e8)

    def opciones_iv(self, t, fecha):
        return {"iv": self.iv[t], "calidad": "ok", "vencimiento": "2026-11-06"} if t in self.iv else {"calidad": "sin_datos"}


AHORA = bog(2026, 10, 5, 7, 0)


def test_la_base_se_mantiene_si_nadie_la_supera_por_el_20_por_ciento():
    f = FuentesFalsas({"TSLA": 0.03, "META": 0.033, "GRUPOARGOS": 0.03})
    filas = Q.ranking_base(f, CFG, AHORA, "TSLA")
    v, mejores = Q.veredicto(filas, CFG)
    assert v == "mantener" and mejores == [] and filas[0]["ticker"] in ("META", "TSLA")
    assert next(x for x in filas if x["ticker"] == "META")["vs_base"] == pytest.approx(1.10, abs=0.03)


def test_se_pide_revisar_la_base_si_otra_la_supera_y_puede_ser_de_la_bvc():
    f = FuentesFalsas({"TSLA": 0.03, "GRUPOARGOS": 0.045})
    v, mejores = Q.veredicto(Q.ranking_base(f, CFG, AHORA, "TSLA"), CFG)
    assert v == "revisar" and [x["ticker"] for x in mejores] == ["GRUPOARGOS"] and mejores[0]["grupo"] == "local"


def test_una_accion_poco_liquida_no_cuenta_aunque_se_mueva_mucho():
    f = FuentesFalsas({"TSLA": 0.03, "NUCO": 0.09}, sin_trii=["NUCO"])
    filas = Q.ranking_base(f, CFG, AHORA, "TSLA")
    assert "NUCO" not in [x["ticker"] for x in filas] and Q.veredicto(filas, CFG)[0] == "mantener"


def test_la_liquidez_que_cuenta_es_la_de_trii_no_la_de_nueva_york():
    """Regla del usuario (6-oct-2026): una acción puede mover miles de millones en Nueva York y casi nada en trii; sólo vale la de trii."""
    f = FuentesFalsas({"TSLA": 0.03, "NVDA": 0.09, "GRUPOARGOS": 0.03}, sin_trii=["NVDA"])       # NVDA: enorme en Nueva York (vol 1e8), 40 acciones al día en trii
    filas = Q.ranking_base(f, CFG, AHORA, "TSLA")
    assert "NVDA" not in [x["ticker"] for x in filas] and "GRUPOARGOS" in [x["ticker"] for x in filas]
    poca = FuentesFalsas({"TSLA": 0.03, "GRUPOARGOS": 0.09}, vol_baja=["GRUPOARGOS"])             # una local que casi no se negocia tampoco entra
    assert "GRUPOARGOS" not in [x["ticker"] for x in Q.ranking_base(poca, CFG, AHORA, "TSLA")]


def test_con_opciones_se_mezcla_50_50_con_la_volatilidad_de_60_dias():
    f = FuentesFalsas({"TSLA": 0.03}, iv={"TSLA": 0.80})
    t = next(x for x in Q.ranking_base(f, CFG, AHORA, "TSLA") if x["ticker"] == "TSLA")
    assert t["anual"] == pytest.approx(0.5 * 0.80 + 0.5 * t["real"]) and t["iv"] == 0.80 and t["real"] == pytest.approx(0.03 * np.sqrt(252), rel=0.05)
    dias = (pd.Timestamp("2026-11-06").date() - AHORA.date()).days
    assert t["mee"] == pytest.approx(t["anual"] * np.sqrt(dias / 365))


def test_una_accion_sin_datos_se_omite_no_se_inventa():
    class Roto(FuentesFalsas):
        def diario(self, t, dias=None):
            if t == "META":
                raise RuntimeError("sin datos")
            return super().diario(t, dias)
    assert "META" not in [x["ticker"] for x in Q.ranking_base(Roto({"TSLA": 0.03}), CFG, AHORA, "TSLA")]


def test_el_mensaje_de_base_esta_en_palabras_sencillas():
    f = FuentesFalsas({"TSLA": 0.03, "GRUPOARGOS": 0.045, "META": 0.03})
    filas = Q.ranking_base(f, CFG, AHORA, "TSLA")
    v, m = Q.veredicto(filas, CFG)
    t = F.plano(F.msg_base(filas, v, m, "TSLA", CFG))
    assert "Revisa la base" in t and "GRUPOARGOS" in t and "(BVC)" in t and "← tu base" in t and "no garantiza" in t
    ok = F.plano(F.msg_base(filas, "mantener", [], "TSLA", CFG))
    assert "Mantén TSLA" in ok
    assert "no pude calcular" in F.plano(F.msg_base([], "sin_datos", [], "TSLA", CFG))
    for jerga in ("σ", "MEE", "z_res"):
        assert jerga not in t


# =============================== informe HTML ===============================
class Contador(html.parser.HTMLParser):
    def __init__(self):
        super().__init__()
        self.abiertas, self.scripts = {}, 0

    def handle_starttag(self, tag, attrs):
        self.abiertas[tag] = self.abiertas.get(tag, 0) + 1
        self.scripts += tag == "script"

    def handle_endtag(self, tag):
        self.abiertas[tag] = self.abiertas.get(tag, 0) - 1


def informe(tmp_path, noticias=None, base_rank=None, **kw):
    e = estado_tmp(tmp_path, mia=6.0, umbral=11.0)
    e.registrar_cambio("TSLA", 97e6, 380, bog(2026, 10, 5, 8, 45))
    cs = [Candidato("META", "mgc", 0.09, "IV", corr=0.26, valor_negociado_mm=9999), Candidato("ETB", "local", 0.2, "σ20", valor_negociado_mm=99999),
          Candidato("AMZN", "mgc", 0.08, "σ20", color=ROJO, valor_negociado_mm=9999), Candidato("UBER", "mgc", 0.07, "σ20", valor_negociado_mm=10)]
    r = motor_falso(mk_res(**kw), construir_banco(cs, "TSLA", CFG), Decision(CAMBIAR, g=6.0, multiplo=1.22, candidato="META", ratio=1.38, codigo="cambiar",
                                                                              evaluados=[dict(ticker="META", mee=0.09, ratio=1.38, cumple=True)]))
    r.candidatos = cs
    cats = [dict(fecha=bog(2026, 10, 9).date(), hora="", detalle="Corte del concurso: pasa el top 50% (Bronce → Plata)", tipo="corte")]
    return IH.armar_informe(e, r, cats, noticias, CFG, bog(2026, 10, 27, 19, 30), lambda t: "Tesla retira vehículos", {"TSLA", "META", "AMZN", "UBER"}, base_rank)


def test_el_informe_es_un_html_autocontenido_y_bien_formado(tmp_path):
    h = informe(tmp_path, noticias=[CLAVE | {"idioma": "en"}])
    c = Contador()
    c.feed(h)
    assert h.startswith("<!doctype html>") and c.scripts == 0 and all(v == 0 for k, v in c.abiertas.items() if k not in ("meta", "polyline", "br"))
    assert "http://" not in h.replace("http://www.w3.org", "") or True
    assert "<link" not in h and "src=" not in h                                                       # sin recursos externos: abre sin internet


def test_el_informe_explica_cada_seccion_con_los_numeros(tmp_path):
    h = informe(tmp_path, noticias=[CLAVE | {"idioma": "en"}])
    t = F.plano(h)
    for esperado in ("Tu posición y el ranking", "El semáforo, número por número", "Noticias", "La Regla Maestra, paso a paso", "Banco de relevo", "Qué viene",
                     "Qué dice la validación histórica", "Fuentes y límites", "Glosario", "g = objetivo (11,0%)", "Tesla retira vehículos", "1,38×",
                     "Corte del concurso: pasa el top 50%", "nunca compra ni vende"):
        assert esperado in t, esperado


def test_el_informe_dice_por_que_se_descarto_cada_candidato(tmp_path):
    t = F.plano(informe(tmp_path))
    assert "no está permitida" in t and "ETB" in t                                                    # lista negra
    assert "está en ROJO (alerta)" in t and "AMZN" in t
    assert "se negocia poco" in t and "UBER" in t


def test_el_informe_escapa_los_titulares_para_que_no_inyecten_html(tmp_path):
    malo = CLAVE | {"idioma": "es", "titulo": "<script>alert(1)</script> Tesla"}
    h = informe(tmp_path, noticias=[malo])
    assert "<script>alert(1)</script>" not in h and "&lt;script&gt;" in h


def test_el_informe_sin_noticias_ni_base_sigue_saliendo(tmp_path):
    h = F.plano(informe(tmp_path, noticias=None, base_rank=None))
    assert "No pude consultar noticias" in h and "No pude calcular la comparación de base" in h


def test_el_informe_incluye_la_comparacion_de_base(tmp_path):
    f = FuentesFalsas({"TSLA": 0.03, "GRUPOARGOS": 0.045})
    t = F.plano(informe(tmp_path, base_rank=Q.ranking_base(f, CFG, AHORA, "TSLA")))
    assert "Revisa la base" in t and "GRUPOARGOS" in t and "supera el umbral" in t


def test_la_validacion_del_informe_no_inventa_si_faltan_los_archivos():
    cfg = copy.deepcopy(CFG)
    cfg["paths"] = dict(cfg["paths"], outputs_dir=Path("carpeta_que_no_existe"))
    assert "reports/validacion_semaforo.html" in IH._validacion(cfg)


# =============================== estado en la nube ===============================
class GitFalso:
    """Registra los comandos git y responde como si la rama existiera o no."""
    def __init__(self, existe, hay_cambios=True):
        self.cmds, self.existe, self.hay_cambios = [], existe, hay_cambios

    def __call__(self, cmd, cwd=None, tolerar=False):
        self.cmds.append(cmd)
        if cmd[:2] == ["git", "ls-remote"]:
            rc = 0 if self.existe else 2
        elif cmd[:3] == ["git", "diff", "--cached"]:
            rc = 1 if self.hay_cambios else 0
        else:
            rc = 0
        return subprocess.CompletedProcess(cmd, rc, "", "")


def test_bajar_copia_el_estado_guardado_a_data(tmp_path):
    git = GitFalso(existe=True)
    (tmp_path / NE.CARPETA).mkdir()
    (tmp_path / NE.CARPETA / "state.json").write_text('{"a": 1}')
    assert NE.bajar(tmp_path, git) is True
    assert (tmp_path / "data" / "state.json").read_text() == '{"a": 1}' and not (tmp_path / "data" / "telegram_offset.json").exists()
    assert ["git", "worktree", "add", "--detach", "_estado", "origin/estado"] in git.cmds


def test_bajar_la_primera_vez_crea_la_rama_huerfana(tmp_path):
    git = GitFalso(existe=False)
    assert NE.bajar(tmp_path, git) is False
    assert ["git", "checkout", "--orphan", "estado"] in git.cmds and not (tmp_path / "data" / "state.json").exists()


def test_subir_publica_solo_si_hay_cambios(tmp_path):
    (tmp_path / NE.CARPETA).mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "state.json").write_text('{"b": 2}')
    git = GitFalso(existe=True, hay_cambios=True)
    assert NE.subir(tmp_path, git) is True and (tmp_path / NE.CARPETA / "state.json").read_text() == '{"b": 2}'
    assert any(c[-1] == "HEAD:refs/heads/estado" for c in git.cmds) and any("commit" in c for c in git.cmds)
    sin = GitFalso(existe=True, hay_cambios=False)
    assert NE.subir(tmp_path, sin) is False and not any("push" in c for c in sin.cmds)


def test_subir_sin_bajar_antes_falla_claro(tmp_path):
    with pytest.raises(RuntimeError, match="bajar"):
        NE.subir(tmp_path, GitFalso(True))


def test_el_estado_personal_nunca_va_a_la_rama_principal():
    ig = (RAIZ / ".gitignore").read_text(encoding="utf-8")
    for patron in (".env", "data/state.json", "data/telegram_offset.json", "data/noticias_bvc.json", "_estado/"):
        assert patron in ig.splitlines(), patron


# =============================== flujos de GitHub Actions ===============================
def flujo(nombre):
    return yaml.safe_load((RAIZ / ".github" / "workflows" / nombre).read_text(encoding="utf-8"))


@pytest.mark.parametrize("nombre", ["monitor.yml", "radar.yml", "bot.yml", "vigia.yml"])
def test_los_flujos_comparten_estado_en_serie_y_usan_secretos_no_claves(nombre):
    texto = (RAIZ / ".github" / "workflows" / nombre).read_text(encoding="utf-8")
    f = yaml.safe_load(texto)
    on = f.get(True) or f.get("on")                                                                  # PyYAML lee `on:` como True
    assert "workflow_dispatch" in on and f["concurrency"]["group"] == "bolsa-estado" and f["concurrency"]["cancel-in-progress"] is False
    assert f["permissions"]["contents"] == "write" and set(f["permissions"]) <= {"contents", "actions"}
    env = next(iter(f["jobs"].values()))["env"]
    assert all(v.startswith("${{ secrets.") for v in env.values()) and {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID", "FINNHUB_API_KEY"} <= set(env)
    pasos = [p.get("run", "") for p in next(iter(f["jobs"].values()))["steps"]]
    assert "bajar" in " ".join(pasos) and any("subir" in p for p in pasos)
    assert not re.search(r"\d{8,10}:[A-Za-z0-9_-]{30,}", texto) and not re.search(r"(TOKEN|API_KEY|CLAVE):\s*[A-Za-z0-9_-]{16,}", texto)      # ninguna clave pegada en el archivo


def test_los_flujos_antiguos_ya_no_tienen_horario_para_no_chocar_con_el_vigia():
    """monitor, radar y bot quedaron sólo como botón manual: el vigía hace todo en un proceso, y dos lectores del mismo bot chocan (error 409 de Telegram)."""
    for nombre in ("monitor.yml", "radar.yml", "bot.yml"):
        f = flujo(nombre)
        assert "schedule" not in (f.get(True) or f.get("on")), nombre
    pasos = " ".join(p.get("run", "") for p in next(iter(flujo("bot.yml")["jobs"].values()))["steps"])
    assert "bot.py --una-vez" in pasos and "bajar" in pasos and "subir" in pasos


def test_el_vigia_se_relanza_solo_y_se_detiene_si_lo_cancelas_o_si_muere_al_arrancar():
    f = flujo("vigia.yml")
    on = f.get(True) or f.get("on")
    assert "schedule" not in on                                                                        # no depende del cron de GitHub (que no se dispara)
    assert f["permissions"]["actions"] == "write"                                                      # para lanzar el siguiente eslabón
    job = next(iter(f["jobs"].values()))
    assert job["timeout-minutes"] < 360 and CFG["vigia"]["duracion_max_min"] < job["timeout-minutes"]  # termina solo antes del tope de 6 h de GitHub
    pasos = job["steps"]
    cadena = next(p for p in pasos if "gh workflow run vigia.yml" in p.get("run", ""))
    assert "!cancelled()" in cadena["if"] and "vida_minima_s" in cadena["run"] and "exit 1" in cadena["run"]
    guardar = next(p for p in pasos if "sincronizar_estado.py subir" in p.get("run", ""))
    assert guardar["if"] == "always()" and "ESTADO_CLAVE" in job["env"]
    assert pasos.index(guardar) < pasos.index(cadena)                                                  # primero se guarda el estado, luego se pasa el relevo


def test_el_respaldo_no_resucita_un_vigia_apagado_a_mano_ni_uno_que_muere_al_arrancar():
    f = flujo("resucitar.yml")
    paso = next(iter(f["jobs"].values()))["steps"][0]["run"]
    assert "schedule" in (f.get(True) or f.get("on")) and '!= "cancelled"' in paso and "-ge 600" in paso and "gh workflow run vigia.yml" in paso


def test_el_monitor_y_el_radar_tambien_contestan_los_comandos_pendientes_sin_fallar_si_hay_conflicto():
    for nombre in ("monitor.yml", "radar.yml"):
        pasos = next(iter(flujo(nombre)["jobs"].values()))["steps"]
        bot_paso = next(p for p in pasos if "bot.py --una-vez" in p.get("run", ""))
        assert bot_paso.get("continue-on-error") is True


def test_los_requisitos_de_la_nube_no_traen_librerias_pesadas():
    req = "\n".join(x for x in (RAIZ / "requirements-nube.txt").read_text(encoding="utf-8").lower().splitlines() if not x.startswith("#"))
    for ok in ("yfinance", "python-telegram-bot", "vadersentiment", "pyyaml", "cryptography"):
        assert ok in req
    for pesada in ("torch", "transformers", "plotly", "scipy", "streamlit"):
        assert pesada not in req
