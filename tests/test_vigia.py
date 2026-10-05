"""Pruebas del proceso continuo (vigia.py): tareas en paralelo que no se tumban entre sí, radar una vez por noche, bienvenida una sola vez,
respaldo del estado y cifrado para repositorio público."""
import asyncio
import subprocess
import time

import pytest

import vigia as V
from nube import sincronizar_estado as NE
from src import servicio as S
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg
from tests.test_fase4 import bog

CFG = make_cfg()


def ctx_en(tmp_path, ahora, enviados):
    c = S.Contexto(CFG, None, puntuar_vader, "vader", ahora, False, tmp_path / "s.json", salida=lambda s: None)
    return c


@pytest.fixture
def mundo(tmp_path, monkeypatch):
    """Servicio falso: reloj controlable, mensajes capturados y nada de red."""
    m = {"ahora": bog(2026, 10, 6, 19, 35), "enviados": [], "radares": 0}
    monkeypatch.setattr(S, "crear_contexto", lambda *a, **k: ctx_en(tmp_path, m["ahora"], m["enviados"]))
    monkeypatch.setattr(S, "enviar_o_encolar", lambda ctx, texto, botones=None: m["enviados"].append((texto, botones)) or True)
    monkeypatch.setattr(S, "enviar_pendientes", lambda ctx: 0)
    monkeypatch.setattr(S, "alerta_base", lambda ctx: None)

    def radar(ctx):
        m["radares"] += 1
        return "radar de la noche"
    monkeypatch.setattr(S, "generar_radar", radar)
    real = V.N.Memoria
    monkeypatch.setattr(V.N, "Memoria", lambda *a, **k: real(tmp_path / "noticias.json"))            # sin tocar data/noticias_bvc.json real
    m["tmp"] = tmp_path
    return m


def nuevo(minutos=1):
    return V.Vigia(CFG, minutos)


def test_el_radar_sale_una_sola_vez_por_noche_de_lunes_a_jueves(mundo):
    v = nuevo()
    mundo["ahora"] = bog(2026, 10, 6, 19, 20)                                                        # martes, antes de las 19:30
    v.radar()
    assert mundo["radares"] == 0
    mundo["ahora"] = bog(2026, 10, 6, 19, 31)
    v.radar()
    v.radar()
    assert mundo["radares"] == 1 and mundo["enviados"][0][0] == "radar de la noche"
    assert Estado(CFG, mundo["tmp"] / "s.json").d["vigia"]["radar"] == "2026-10-06"
    assert nuevo().radar() is None and mundo["radares"] == 1                                         # otro eslabón de la cadena esa misma noche: tampoco repite
    mundo["ahora"] = bog(2026, 10, 7, 21, 0)                                                         # miércoles: otro radar
    v.radar()
    assert mundo["radares"] == 2
    mundo["ahora"] = bog(2026, 10, 9, 19, 40)                                                        # viernes: no hay radar
    v.radar()
    mundo["ahora"] = bog(2026, 10, 8, 23, 30)                                                        # jueves pero ya muy tarde
    v.radar()
    mundo["ahora"] = bog(2026, 11, 10, 19, 40)                                                       # terminó el concurso
    v.radar()
    assert mundo["radares"] == 2


def test_si_el_radar_falla_avisa_y_no_lo_repite_en_bucle(mundo, monkeypatch):
    monkeypatch.setattr(S, "generar_radar", lambda ctx: (_ for _ in ()).throw(RuntimeError("sin datos")))
    v = nuevo()
    v.radar()
    v.radar()
    assert len(mundo["enviados"]) == 1 and "El radar de esta noche falló" in mundo["enviados"][0][0] and "/comprar" in mundo["enviados"][0][0]


def test_la_bienvenida_se_manda_una_sola_vez_en_la_vida(mundo):
    nuevo().bienvenida()
    nuevo().bienvenida()                                                                             # otro eslabón, horas después
    assert len(mundo["enviados"]) == 1
    texto, botones = mundo["enviados"][0]
    assert "encendido de forma permanente" in texto and "cada 2 minutos" in texto and botones == V.bot.MENU
    V.Vigia(CFG, 1, prueba=True).bienvenida()
    assert len(mundo["enviados"]) == 1


def test_una_prueba_corta_no_anuncia_que_quedo_encendido(mundo):
    V.Vigia(CFG, 1, prueba=True).bienvenida()
    assert mundo["enviados"] == [] and "vigia" not in Estado(CFG, mundo["tmp"] / "s.json").d


def test_una_tarea_que_falla_no_tumba_a_las_demas_y_el_vigia_para_a_tiempo(mundo):
    v = nuevo()
    cuenta = {"buena": 0, "mala": 0}

    def mala():
        cuenta["mala"] += 1
        raise RuntimeError("boom")

    async def correr():
        t = [asyncio.create_task(v.bucle("mala", 0.01, mala)), asyncio.create_task(v.bucle("buena", 0.01, lambda: cuenta.__setitem__("buena", cuenta["buena"] + 1)))]
        await asyncio.sleep(1.3)
        v.parar.set()
        t0 = time.monotonic()
        await asyncio.gather(*t)
        return time.monotonic() - t0
    assert asyncio.run(correr()) < 2.5 and cuenta["mala"] >= 1 and cuenta["buena"] >= 1               # el bucle espera ≥ 1 s entre vueltas y se detiene al pedirlo


def test_fuera_de_la_nube_no_se_toca_git(mundo, monkeypatch):
    monkeypatch.setattr(V, "EN_NUBE", False)
    monkeypatch.setattr(NE, "subir", lambda raiz: (_ for _ in ()).throw(AssertionError("no debe llamarse")))
    nuevo().guardar(True)


def test_en_la_nube_lo_que_registras_se_respalda_enseguida_y_lo_demas_cada_cierto_tiempo(mundo, monkeypatch):
    subidas = []
    monkeypatch.setattr(V, "EN_NUBE", True)
    monkeypatch.setattr(NE, "subir", lambda raiz: subidas.append(1) or True)
    monkeypatch.setattr(V, "ROOT", mundo["tmp"])
    cfg = dict(CFG, estado=dict(CFG["estado"], archivo="s.json"))
    (mundo["tmp"] / "s.json").write_text('{"cartera": []}', encoding="utf-8")
    v = V.Vigia(cfg, 1)
    v.guardar()
    assert subidas == []                                                                             # lo que había al arrancar ya está respaldado
    (mundo["tmp"] / "s.json").write_text('{"cartera": []}', encoding="utf-8")                        # se reescribió igual (pasa en cada revisión): no es un cambio
    v.guardar()
    assert subidas == []
    (mundo["tmp"] / "s.json").write_text('{"cartera": [1]}', encoding="utf-8")                       # registraste una compra
    v.guardar()
    v.guardar()
    assert len(subidas) == 1
    v._ultimo_respaldo -= cfg["vigia"]["guardar_cada_s"] + 1                                         # pasó el intervalo: respaldo de la memoria de noticias
    v.guardar()
    assert len(subidas) == 2
    v.guardar(True)                                                                                  # al apagarse, siempre
    assert len(subidas) == 3


def test_el_vigia_reparte_las_tareas_del_config():
    assert CFG["noticias_bvc"]["cada_s"] == 120 and CFG["vigia"]["monitor_cada_min"] == 15           # lo que pidió el usuario: noticias cada 2 minutos
    assert CFG["vigia"]["duracion_max_min"] * 60 > CFG["vigia"]["vida_minima_s"] * 10


# =============================== estado cifrado (repositorio público) ===============================
class GitFalso:
    def __init__(self, existe=True):
        self.cmds, self.existe = [], existe

    def __call__(self, cmd, cwd=None, tolerar=False):
        self.cmds.append(cmd)
        rc = (0 if self.existe else 2) if cmd[:2] == ["git", "ls-remote"] else (1 if cmd[:3] == ["git", "diff", "--cached"] else 0)
        return subprocess.CompletedProcess(cmd, rc, "", "")


def test_con_clave_el_estado_viaja_cifrado_y_se_recupera_igual(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet
    monkeypatch.setenv("ESTADO_CLAVE", Fernet.generate_key().decode())
    (tmp_path / NE.CARPETA).mkdir()
    (tmp_path / "data").mkdir()
    secreto = '{"cartera": [{"ticker": "NUCO", "cantidad": 1200}]}'
    (tmp_path / "data" / "state.json").write_text(secreto, encoding="utf-8")
    (tmp_path / NE.CARPETA / "state.json").write_text("copia vieja sin cifrar", encoding="utf-8")
    assert NE.subir(tmp_path, GitFalso()) is True
    rama = tmp_path / NE.CARPETA
    assert not (rama / "state.json").exists()                                                        # la copia legible desaparece de la rama
    cifrado = (rama / "state.json.enc").read_bytes()
    assert b"NUCO" not in cifrado and b"cartera" not in cifrado and b"1200" not in (rama / "state.json.firma").read_bytes()
    NE.subir(tmp_path, GitFalso())
    assert (rama / "state.json.enc").read_bytes() == cifrado                                         # sin cambios no se vuelve a cifrar (no ensucia la rama)
    (tmp_path / "data" / "state.json").unlink()
    assert NE.bajar(tmp_path, GitFalso()) is True and (tmp_path / "data" / "state.json").read_text(encoding="utf-8") == secreto


def test_sin_clave_sigue_funcionando_como_antes_y_una_clave_mala_falla_claro(tmp_path, monkeypatch):
    from cryptography.fernet import Fernet, InvalidToken
    monkeypatch.delenv("ESTADO_CLAVE", raising=False)
    (tmp_path / NE.CARPETA).mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "noticias_bvc.json").write_text('{"vistos": {}}', encoding="utf-8")
    assert NE.subir(tmp_path, GitFalso()) and (tmp_path / NE.CARPETA / "noticias_bvc.json").read_text(encoding="utf-8") == '{"vistos": {}}'
    monkeypatch.setenv("ESTADO_CLAVE", Fernet.generate_key().decode())
    NE.subir(tmp_path, GitFalso())
    monkeypatch.setenv("ESTADO_CLAVE", Fernet.generate_key().decode())                               # otra clave: no debe "recuperar" basura
    with pytest.raises(InvalidToken):
        NE.bajar(tmp_path, GitFalso())


def test_la_memoria_de_noticias_tambien_se_respalda_y_nunca_va_a_la_rama_principal():
    assert "noticias_bvc.json" in NE.ARCHIVOS and "state.json" in NE.ARCHIVOS
