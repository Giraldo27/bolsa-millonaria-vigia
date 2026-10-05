"""Envío de mensajes por Telegram (sólo avisos; trii no tiene API y NUNCA se envían órdenes).

Los mensajes usan el HTML de Telegram (<b>, <i>) para que se lean fácil. Si Telegram rechaza el formato (etiqueta rota, texto raro),
se reenvía el mismo mensaje como texto plano: nunca se pierde un aviso por un problema de formato."""
from __future__ import annotations

import html
import json
import os
import re
import sys
from typing import Any

import requests

_ETIQUETAS = re.compile(r"</?(b|i|code)>")


def a_plano(texto: str) -> str:
    """Quita las etiquetas HTML y deshace las entidades (&lt; → <)."""
    return html.unescape(_ETIQUETAS.sub("", texto))


def partir(texto: str, largo: int = 3900) -> list[str]:
    """Parte el mensaje por líneas (las etiquetas nunca cruzan líneas) para respetar el límite de Telegram (4096 caracteres)."""
    if len(texto) <= largo:
        return [texto]
    trozos, actual = [], ""
    for linea in texto.split("\n"):
        if len(actual) + len(linea) + 1 > largo and actual:
            trozos.append(actual)
            actual = ""
        actual += ("\n" if actual else "") + linea[:largo]
    if actual:
        trozos.append(actual)
    return trozos


def _post(http: Any, token: str, chat: str, texto: str, html_mode: bool, timeout: float, teclado: str | None = None) -> dict[str, Any]:
    datos = {"chat_id": chat, "text": texto, "disable_web_page_preview": "true"}
    if html_mode:
        datos["parse_mode"] = "HTML"
    if teclado:
        datos["reply_markup"] = teclado
    return http.post(f"https://api.telegram.org/bot{token}/sendMessage", data=datos, timeout=timeout).json()


def teclado_json(botones: list[list[tuple[str, str]]] | None) -> str | None:
    """Filas de botones (texto, dato) → el JSON que espera Telegram (inline_keyboard). Al tocarlos, el bot recibe el dato y contesta."""
    if not botones:
        return None
    return json.dumps({"inline_keyboard": [[{"text": t, "callback_data": d} for t, d in fila] for fila in botones]}, ensure_ascii=False)


def enviar(texto: str, cfg: dict[str, Any], env: dict[str, str] | None = None, http: Any = None, botones: list[list[tuple[str, str]]] | None = None) -> bool:
    env = env if env is not None else dict(os.environ)
    token, chat = env.get("TELEGRAM_BOT_TOKEN"), env.get("TELEGRAM_CHAT_ID")
    token, chat = (token or "").replace("﻿", "").strip(), (chat or "").replace("﻿", "").strip()               # un BOM invisible en el secreto rompe el envío
    if not token or not chat:
        print("⚠️ Falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID en .env: no se envió el mensaje.", file=sys.stderr)
        return False
    http = http or requests
    c = cfg["fuentes"]["telegram"]
    ok = True
    partes = partir(texto, c["max_largo"])
    for k, parte in enumerate(partes):
        teclado = teclado_json(botones) if k == len(partes) - 1 else None                       # los botones van bajo el último trozo
        try:
            resp = _post(http, token, chat, parte, True, c["timeout_s"], teclado)
            if not resp.get("ok") and "parse" in str(resp.get("description", "")).lower():      # "can't parse entities": reenviar sin formato
                resp = _post(http, token, chat, a_plano(parte), False, c["timeout_s"], teclado)
            ok &= bool(resp.get("ok"))
        except Exception as e:                      # noqa: BLE001
            print(f"⚠️ Error enviando a Telegram: {type(e).__name__}", file=sys.stderr)   # sin imprimir el token
            ok = False
    return ok
