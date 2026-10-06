"""Pruebas de /avisos: otros chats (otro celular, un grupo) también pueden recibir los avisos automáticos."""
from src import formato as F
from src import servicio as S
from src.sentimiento import puntuar_vader
from src.state import Estado
from tests.helpers import make_cfg
from tests.test_fase4 import bog

CFG = make_cfg()
AHORA = bog(2026, 10, 6, 10, 0)
PRINCIPAL = "123456789"


def ctx(tmp_path):
    return S.Contexto(CFG, None, puntuar_vader, "vader", AHORA, False, tmp_path / "s.json", salida=lambda s: None)


def test_un_chat_se_suscribe_y_se_da_de_baja(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", PRINCIPAL)
    c = ctx(tmp_path)
    assert "chat principal" in S.suscribir(c, int(PRINCIPAL), True) and not (tmp_path / "s.json").exists()
    assert "también recibirá los avisos automáticos" in F.plano(S.suscribir(c, 555, True)) and "/silencio" in S.suscribir(c, -100777, True)
    S.suscribir(c, 555, True)                                                                         # repetir no duplica
    assert Estado(CFG, c.estado_path).d["suscriptores"] == [-100777, 555]
    assert "ya no recibirá" in S.suscribir(c, 555, False) and Estado(CFG, c.estado_path).d["suscriptores"] == [-100777]


def test_cada_aviso_automatico_se_copia_a_los_suscritos_sin_afectar_al_principal(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", PRINCIPAL)
    enviados = []

    def falso(texto, cfg, env=None, http=None, botones=None):
        chat = (env or {}).get("TELEGRAM_CHAT_ID", PRINCIPAL)
        if chat == "555":
            raise RuntimeError("ese chat bloqueó al bot")
        enviados.append((chat, texto, botones))
        return True
    monkeypatch.setattr(S, "enviar", falso)
    c = ctx(tmp_path)
    S.suscribir(c, 555, True)
    S.suscribir(c, -100777, True)
    assert S.enviar_o_encolar(c, "aviso de prueba", [[("🌍 Macro ahora", "c:macro")]]) is True
    assert [(x[0], x[1]) for x in enviados] == [("-100777", "aviso de prueba"), (PRINCIPAL, "aviso de prueba")] and enviados[0][2] == enviados[1][2]
    assert Estado(CFG, c.estado_path).d["alertas_pendientes"] == []


def test_el_bot_tiene_los_comandos_avisos_y_silencio_y_la_ayuda_los_menciona():
    import bot
    app = bot.construir_app("123456:ABCDEF-token-falso-para-pruebas", int(PRINCIPAL), CFG)
    nombres = {next(iter(h.commands)) for g in app.handlers.values() for h in g if hasattr(h, "commands")}
    assert {"avisos", "silencio", "macro"} <= nombres and "/avisos" in F.AYUDA and "impacto estimado en %" in F.AYUDA


def test_un_grupo_queda_suscrito_solo_y_respeta_el_silencio(tmp_path, monkeypatch):
    monkeypatch.setenv("TELEGRAM_CHAT_ID", PRINCIPAL)
    c = ctx(tmp_path)
    assert S.suscribir_grupo(c, -100777) is True and S.suscribir_grupo(c, -100777) is False          # la segunda vez ya estaba
    assert Estado(CFG, c.estado_path).d["suscriptores"] == [-100777]
    S.suscribir(c, -100777, False)                                                                    # /silencio en el grupo
    assert S.suscribir_grupo(c, -100777) is False and Estado(CFG, c.estado_path).d["suscriptores"] == []   # no se vuelve a suscribir solo
    assert "también recibirá" in F.plano(S.suscribir(c, -100777, True)) and S.suscribir_grupo(c, -100777) is False
    assert Estado(CFG, c.estado_path).d["suscriptores"] == [-100777] and Estado(CFG, c.estado_path).d["silenciados"] == []


def test_usar_el_bot_en_un_grupo_lo_suscribe_a_los_avisos(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace
    import bot
    monkeypatch.setenv("TELEGRAM_CHAT_ID", PRINCIPAL)
    c = ctx(tmp_path)
    monkeypatch.setattr(S, "crear_contexto", lambda *a, **k: c)
    monkeypatch.setattr(S, "resp_estado", lambda ctx_: "estado ok")
    app = bot.construir_app("123456:ABCDEF-token-falso-para-pruebas", int(PRINCIPAL), CFG)
    enviados = []

    async def send_message(chat, texto, **k):
        enviados.append((chat, texto))
    monkeypatch.setattr(type(app.bot), "send_message", lambda self, chat, texto, **k: send_message(chat, texto, **k))
    estado = next(h for g in app.handlers.values() for h in g if hasattr(h, "commands") and "estado" in h.commands)

    class Msg:
        textos = []

        async def reply_text(self, texto, **k):
            self.textos.append(texto)

    def usar(chat, tipo):
        asyncio.run(estado.callback(SimpleNamespace(effective_chat=SimpleNamespace(id=chat, type=tipo), message=Msg()), SimpleNamespace(args=[])))
    usar(-100777, "supergroup")
    usar(-100777, "supergroup")
    usar(555, "private")                                                                             # un chat privado ajeno NO se suscribe solo
    assert Estado(CFG, c.estado_path).d["suscriptores"] == [-100777]
    assert len(enviados) == 1 and enviados[0][0] == -100777 and "Este grupo recibirá los avisos automáticos" in enviados[0][1]
    assert "my_chat_member" in bot.ACTUALIZACIONES
