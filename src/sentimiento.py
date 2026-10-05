"""Sentimiento de titulares en inglés: FinBERT (si está instalado) o VADER (liviano, siempre disponible) + palabras clave.

El puntaje va de -1 (muy negativo) a +1 (muy positivo). Es un indicador de ayuda: no se opera por noticias (cada cambio cuesta ~1,3 %)."""
from __future__ import annotations

import functools
import re
from typing import Any, Callable


@functools.lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    return SentimentIntensityAnalyzer()


def puntuar_vader(texto: str) -> float:
    return float(_vader().polarity_scores(texto)["compound"])


@functools.lru_cache(maxsize=1)
def _finbert(modelo: str):
    from transformers import pipeline          # import perezoso: torch/transformers sólo se cargan si se pide FinBERT
    return pipeline("text-classification", model=modelo, top_k=None, truncation=True)


def puntuar_finbert(texto: str, modelo: str = "ProsusAI/finbert") -> float:
    """P(positivo) − P(negativo)."""
    res = _finbert(modelo)([texto])[0]
    p = {r["label"].lower(): r["score"] for r in res}
    return float(p.get("positive", 0.0) - p.get("negative", 0.0))


def crear_puntuador(cfg: dict[str, Any]) -> tuple[Callable[[str], float], str]:
    """Devuelve (función texto→puntaje, nombre del motor realmente usado). Si FinBERT no está, usa VADER."""
    motor = cfg["sentimiento"]["motor"]
    if motor == "finbert":
        try:
            _finbert(cfg["sentimiento"]["finbert_modelo"])
            return (lambda t: puntuar_finbert(t, cfg["sentimiento"]["finbert_modelo"])), "finbert"
        except Exception:                       # noqa: BLE001 — sin torch/transformers o sin descarga del modelo
            pass
    return puntuar_vader, "vader"


def puntuar_es(texto: str, cfg: dict[str, Any]) -> float:
    """Sentimiento de un titular en español con un léxico simple: (positivas − negativas) / (positivas + negativas + 1), en [-1, 1]."""
    lex = cfg["sentimiento"]["lexico_es"]
    t = texto.lower()
    pos = len(_patron(lex["positivas"]).findall(t))
    neg = len(_patron(lex["negativas"]).findall(t))
    return (pos - neg) / (pos + neg + 1)


def _patron(palabras: list[str]) -> re.Pattern[str]:
    return re.compile(r"\b(" + "|".join(re.escape(p) for p in sorted(palabras, key=len, reverse=True)) + r")\b", re.IGNORECASE)


def palabras_clave(titulo: str, palabras: list[str]) -> list[str]:
    """Palabras clave negativas presentes en el titular (sin distinguir mayúsculas; palabra completa)."""
    return sorted({m.group(1).lower() for m in _patron(palabras).finditer(titulo)})


def texto_de(noticia: Any, usar_resumen: bool) -> str:
    t = noticia.titulo if hasattr(noticia, "titulo") else noticia["titulo"]
    r = (noticia.resumen if hasattr(noticia, "resumen") else noticia.get("resumen", "")) if usar_resumen else ""
    return f"{t}. {r}".strip(". ") if r else t
