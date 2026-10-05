"""Las claves copiadas con un BOM invisible (pasó al subir los secretos a GitHub desde PowerShell) no deben romper el bot en la nube."""
from src.config import limpiar_valor, sanear_entorno
from src.notify import enviar
from tests.helpers import make_cfg
from tests.test_fase4 import FakeTelegram

CFG = make_cfg()


def test_limpiar_valor_quita_bom_espacios_saltos_y_comillas():
    assert limpiar_valor("﻿123456789") == "123456789"
    assert limpiar_valor("  abc123 \r\n") == "abc123" and limpiar_valor('"tok:en"') == "tok:en" and limpiar_valor("'x'") == "x" and limpiar_valor("") == ""


def test_sanear_entorno_solo_toca_las_claves_conocidas():
    env = {"TELEGRAM_BOT_TOKEN": "﻿123:abc", "TELEGRAM_CHAT_ID": " 77 ", "FINNHUB_API_KEY": "﻿key\n", "ALPHAVANTAGE_API_KEY": "", "OTRA": "﻿x"}
    sanear_entorno(env)
    assert env == {"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "77", "FINNHUB_API_KEY": "key", "ALPHAVANTAGE_API_KEY": "", "OTRA": "﻿x"}


def test_enviar_funciona_aunque_el_token_y_el_chat_traigan_bom():
    http = FakeTelegram([{"ok": True}])
    assert enviar("hola", CFG, {"TELEGRAM_BOT_TOKEN": "﻿123:token", "TELEGRAM_CHAT_ID": "﻿123456789"}, http) is True
    assert http.posts[0]["chat_id"] == "123456789" and "﻿" not in http.posts[0]["chat_id"]


def test_la_clave_de_finnhub_con_bom_se_limpia_antes_de_usarse():
    from src.data_sources import FuentesDatos
    visto = {}

    class Http:
        def get(self, url, params=None, timeout=None, headers=None):
            visto.update(params or {})
            return type("R", (), {"status_code": 200, "text": "[]", "json": lambda s: [], "raise_for_status": lambda s: None})()
    f = FuentesDatos(CFG, env={"FINNHUB_API_KEY": "﻿abc"}, yf=object(), http=Http(), dormir=lambda s: None)
    f._finnhub("quote", symbol="TSLA")
    assert visto["token"] == "abc"
