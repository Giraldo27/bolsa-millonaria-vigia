"""¿La noticia es buena o mala para la acción? Análisis del texto cuando el titular no lo dice a simple vista.

Tres capas, de la más barata a la más fina:
  1. **Frases del titular y del resumen** con peso: no sólo "sube / cae", también lo que un inversionista lee como bueno ("refuerza su apuesta", "invierte",
     "nuevo contrato", "alianza", "reduce deuda") o malo ("retrasa", "cuestiona", "protesta", "riesgo", "renuncia", "investigan"). Respeta las negaciones
     ("no pagará", "sin pérdidas").
  2. **El comienzo del artículo** (se descarga la página) cuando el titular y el resumen no alcanzan.
  3. **Un modelo de lenguaje (Claude)**, sólo si existe la variable ANTHROPIC_API_KEY: lee titular y texto y responde si es positiva, negativa o neutra para
     la acción, con su razón. Es la capa que de verdad "entiende"; sin clave se usan las dos primeras.

Es un análisis automático, NO exacto: las capas 1 y 2 cuentan frases y no entienden ironías ni contexto; el modelo también se equivoca. Por eso el aviso
muestra siempre POR QUÉ se concluyó el sentido y con cuánta seguridad, y si nada es claro se dice "neutra" en vez de inventar."""
from __future__ import annotations

import html
import json
import os
import re
from typing import Any

from .noticias_bvc import norm

# (patrón sobre texto sin tildes, peso, cómo se le muestra al usuario)
POSITIVAS: list[tuple[str, float, str]] = [
    (r"\b(recompra|readquisicion)\b", 2.0, "recompra de acciones"), (r"dividendo extraordinario", 2.0, "dividendo extraordinario"),
    (r"\b(record|historic[oa]s? (maximo|utilidad|resultado))", 1.5, "récord"), (r"\b(crec(e|io|ieron|en|imiento)|aument(a|o|an|aron|ara)|sub(e|io|ieron|en)|repunt(a|o)|dispar(a|o))\b", 1.5, "crece o sube"),
    (r"\b(super(a|o|an|aron)|mejor(a|o|an|aron)|duplic(a|o)|triplic(a|o))\b", 1.5, "supera o mejora"), (r"\b(gan(a|o|an|aron)|adjudic\w+|logr(a|o|an))\b", 1.5, "gana o logra"),
    (r"\b(hallazgo|descubr\w+)\b", 1.5, "hallazgo"), (r"nuev[oa]s? (contrato|proyecto|planta|linea|mercado|negocio|mina|pozo|tienda)", 1.5, "nuevo proyecto o contrato"),
    (r"\b(refuerza|fortalece|consolida|impulsa|acelera)\b", 1.0, "refuerza o impulsa"), (r"\bapuesta por\b|\bapuesta (al|a la|en)\b", 1.0, "apuesta por crecer"),
    (r"\b(invert(ira|ir|ira)|invierte|inversion(es)? (de|por))\b", 1.0, "inversión"), (r"\b(expan(de|sion|dira)|ampli(a|o|ara|acion)|crecimiento)\b", 1.0, "expansión"),
    (r"\b(alianza|se alia|acuerdo (con|de|para)|convenio|firma(n|ron)?)\b", 1.0, "alianza o acuerdo"), (r"\b(aprueb(a|an)|aprob(o|aron|ada|ado)|luz verde|licencia ambiental|autoriza(n|ron)?)\b", 1.0, "aprobación"),
    (r"(reduc(e|ira|io)|baja|prepag(a|o)|pag(a|o)) (su |la |sus )?deuda", 1.5, "reduce deuda"), (r"\b(inaugur(a|o)|pone en marcha|entra en operacion|inicia operacion)", 1.0, "pone en marcha"),
    (r"(mejora|sube|eleva) (la |su )?(calificacion|perspectiva|recomendacion|precio objetivo)|perspectiva positiva|sobreponderar", 1.5, "mejor calificación o recomendación"),
    (r"\b(recuper(a|o|acion)|optimis\w+|exitos[oa]s?|solid[oa]s?|robust[oa]s?)\b", 1.0, "recuperación o solidez"), (r"\b(culmin(a|o|an)|complet(a|o|an)|termin(a|o|an)) (con exito|la campana|el proyecto)", 0.5, "completa un proyecto"),
    (r"\b(lidera|lider en|reconocimiento|premi[oa])\b", 0.5, "liderazgo o reconocimiento"), (r"\b(lanz(a|o|an)|estrena|presenta nuev)", 0.5, "lanzamiento"),
    (r"\b(ahorr(a|o|os)|eficiencia|reduc(e|ira|io) costos)\b", 1.0, "ahorro o eficiencia"), (r"\b(beneficia|favorece|oportunidad)\b", 0.5, "le favorece"),
]
NEGATIVAS: list[tuple[str, float, str]] = [
    (r"\b(sancion\w*|multa\w*|pliego de cargos)\b", 2.0, "sanción o multa"), (r"\b(fraude|corrupcion|escandalo|soborno|lavado)\b", 2.0, "escándalo"),
    (r"\b(atentado|derrame|accidente|incendio|explosion|ciberataque|voladura)\b", 2.0, "accidente o ataque"), (r"\b(quiebra|insolvencia|reorganizacion|liquidacion|crisis)\b", 2.0, "crisis"),
    (r"\b(desplom\w+|derrumb\w+|se hunde|colaps\w+)\b", 2.0, "desplome"), (r"(rebaja|baja|recorta) (la |su )?(calificacion|perspectiva|recomendacion|precio objetivo)|perspectiva negativa|infraponderar|degrad\w+", 2.0, "peor calificación o recomendación"),
    (r"\b(ca(e|yo|yeron|en)|baj(a|o|an|aron)|disminuy\w+|reduj\w+|retroced\w+|se contra(e|jo))\b", 1.5, "cae o baja"), (r"\b(perdidas?|pierde|perdio|perdieron)\b", 1.5, "pérdidas"),
    (r"\b(investig\w+|imput\w+|allan\w+|captur\w+)\b", 1.5, "investigación"), (r"\bdemanda(n|ron|do|da)? (contra|a |por )|\bdemandas?\b", 1.5, "demanda"),
    (r"\b(huelga|paro|bloqueos?|protestas?|toma de)\b", 1.5, "paro o protesta"), (r"\b(suspend\w+|cancel\w+|retras\w+|aplaz\w+|fren(a|o|an)|paraliz\w+|posterg\w+)\b", 1.5, "suspende o retrasa"),
    (r"\b(recort(a|o|an|ara|es?)|despid\w+|cierr(a|e) (de |sus )?(planta|tienda|mina|oficina|campo)s?)\b", 1.5, "recortes o cierres"), (r"\bembarg\w+|intervencion|toma de posesion", 1.5, "embargo o intervención"),
    (r"\b(riesgos?|alerta\w*|advierte\w*|preocup\w+|incertidumbre|amenaza\w*|temor\w*)\b", 1.0, "riesgo o alerta"), (r"\b(critic\w+|cuestion\w+|polemica|rechaz\w+|denunci\w+|pelea|choque|tension\w*|conflicto)\b", 1.0, "críticas o conflicto"),
    (r"\b(renunci\w+|destitu\w+|sale (el|la) presidente|salida del presidente)\b", 1.0, "salida de directivos"), (r"\b(endeud\w+|sobreendeud\w+|mas deuda|sube (su |la )?deuda|deficit)\b", 1.0, "más deuda"),
    (r"\b(emision de acciones|aumento de capital|dilu\w+)\b", 1.0, "emisión de acciones"), (r"\b(nuevo impuesto|sobretasa|gravamen|mas impuestos|expropi\w+|control de precios)\b", 1.5, "impuestos o regulación en contra"),
    (r"\b(incumpl\w+|impago|mora|no pagara|sin dividendo)\b", 1.5, "incumplimiento"), (r"\b(menor(es)?|menos|debil\w*|flojo|dificil\w*)\b", 0.5, "menor o débil"),
]
NEGACION = r"\b(no|sin|nunca|descarta|niega|desmiente)\s+(\w+\s+){0,2}$"
_P = [(re.compile(p), w, e) for p, w, e in POSITIVAS]
_N = [(re.compile(p), w, e) for p, w, e in NEGATIVAS]


def puntos(texto: str) -> tuple[float, list[tuple[str, float]]]:
    """Suma de pesos de las frases buenas menos las malas, y qué frases se encontraron (etiqueta, peso con signo). Una negación justo antes invierte el sentido."""
    t = norm(texto)
    total, hallazgos = 0.0, []
    vistos = set()
    for lista, signo in ((_P, 1.0), (_N, -1.0)):
        for patron, peso, etiqueta in lista:
            m = patron.search(t)
            if not m or etiqueta in vistos:
                continue
            vistos.add(etiqueta)
            s = -signo if re.search(NEGACION, t[max(0, m.start() - 25): m.start()]) else signo
            total += s * peso
            hallazgos.append((etiqueta if s == signo else "no " + etiqueta, s * peso))
    return total, hallazgos


def leer_articulo(url: str, http: Any = None, timeout: float = 6.0, max_chars: int = 1800) -> str:
    """Primeros párrafos del artículo (texto plano). '' si no se pudo (los enlaces de Google News no llevan al artículo sin un navegador: se omiten)."""
    if not url or "news.google.com" in url or url.lower().endswith(".pdf") or "superfinanciera" in url:
        return ""
    try:
        if http is None:
            import requests
            http = requests
        r = http.get(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126 Safari/537.36"}, timeout=timeout)
        if getattr(r, "status_code", 200) != 200:
            return ""
        cuerpo = r.text if hasattr(r, "text") else r.content.decode("utf-8", "replace")
    except Exception:                                                                  # noqa: BLE001 — sin artículo se analiza sólo el titular
        return ""
    cuerpo = re.sub(r"(?is)<(script|style|nav|header|footer|aside|form)[^>]*>.*?</\1>", " ", cuerpo)
    parrafos = [" ".join(html.unescape(re.sub(r"<[^>]+>", " ", p)).split()) for p in re.findall(r"(?is)<p[^>]*>(.*?)</p>", cuerpo)]
    return " ".join(p for p in parrafos if len(p) > 60)[:max_chars]


def con_modelo(empresa: str, titulo: str, cuerpo: str, cfg: dict[str, Any], http: Any = None, env: dict[str, str] | None = None) -> dict[str, Any] | None:
    """Capa 3: le pregunta a Claude si la noticia es buena, mala o neutra para la acción. None si no hay clave, si falla o si la respuesta no se entiende."""
    env = env if env is not None else dict(os.environ)
    clave = (env.get("ANTHROPIC_API_KEY") or "").replace("﻿", "").strip()
    a = cfg.get("analisis_noticias", {})
    if not clave or not a.get("usar_modelo", True):
        return None
    pregunta = (f"Empresa (acción de la Bolsa de Colombia): {empresa}\nTitular: {titulo}\n" + (f"Comienzo del artículo: {cuerpo[:1500]}\n" if cuerpo else "")
                + "\n¿Esta noticia es positiva, negativa o neutra para el PRECIO de la acción de esa empresa en los próximos días? Responde SÓLO un JSON así: "
                  '{"sentido": "+", "confianza": 0.7, "razon": "una frase corta en español"} donde sentido es "+", "-" o "0". '
                  'Usa "0" si no tiene efecto claro sobre el negocio o las utilidades (política, opinión, notas de color).')
    try:
        if http is None:
            import requests
            http = requests
        r = http.post("https://api.anthropic.com/v1/messages", timeout=a.get("timeout_s", 10),
                      headers={"x-api-key": clave, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                      json={"model": a.get("modelo", "claude-haiku-4-5-20251001"), "max_tokens": 150, "messages": [{"role": "user", "content": pregunta}]})
        j = r.json()
        texto = "".join(b.get("text", "") for b in j.get("content", []) if b.get("type") == "text")
        d = json.loads(re.search(r"\{.*\}", texto, re.S).group(0))
        sentido = {"+": 1, "-": -1, "0": 0}[str(d["sentido"]).strip()]
        return dict(sentido=sentido, confianza=max(0.0, min(1.0, float(d.get("confianza", 0.6)))), motivos=[str(d.get("razon", ""))[:160]], metodo="modelo")
    except Exception:                                                                  # noqa: BLE001 — sin modelo se usan las otras capas
        return None


def analizar(empresa: str, titulo: str, resumen: str = "", url: str = "", cfg: dict[str, Any] | None = None, http: Any = None,
             env: dict[str, str] | None = None) -> dict[str, Any]:
    """{sentido: +1|-1|0, confianza 0–1, motivos: [frases], metodo: modelo|texto|articulo}. Nunca lanza."""
    cfg = cfg or {}
    a = cfg.get("analisis_noticias", {})
    umbral = a.get("umbral", 1.0)
    p_t, h_t = puntos(titulo)
    p_r, h_r = puntos(resumen) if resumen else (0.0, [])
    total, hall, metodo = 2.0 * p_t + p_r, h_t + [x for x in h_r if x[0] not in {y[0] for y in h_t}], "texto"   # el titular pesa el doble
    cuerpo = ""
    if (abs(total) < umbral or (env if env is not None else os.environ).get("ANTHROPIC_API_KEY")) and a.get("leer_articulo", True):
        cuerpo = leer_articulo(url, http)
        if cuerpo and abs(total) < umbral:
            p_c, h_c = puntos(cuerpo)
            total += p_c
            hall += [x for x in h_c if x[0] not in {y[0] for y in hall}]
            metodo = "articulo"
    m = con_modelo(empresa, titulo, cuerpo or resumen, cfg, http, env)
    if m is not None:
        return m
    sentido = 1 if total >= umbral else (-1 if total <= -umbral else 0)
    motivos = [e for e, w in sorted(hall, key=lambda x: -abs(x[1])) if sentido == 0 or (w > 0) == (sentido > 0)][:3]
    return dict(sentido=sentido, confianza=min(1.0, abs(total) / 4.0), motivos=motivos, metodo=metodo)
