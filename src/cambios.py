"""Avisos automáticos "cambia esta acción por esta".

Cambiar cuesta ≈ 1,3 % (dos comisiones más la diferencia de precios), así que sólo se sugiere con un motivo que lo justifique:
  · **liquidez**: tienes una acción que casi no se negocia en trii (salir de ella puede costar mucho más que el 1,3 %) → pasar a una de la BVC que sí se negocia;
  · la **Regla Maestra** (tu ranking pide más movimiento del que tiene tu cartera) ya tiene su propio aviso en el monitor.

Lo que NO se hace, a propósito: sugerir "cámbiate a la que va subiendo más". Se midió (run_estudio_relativo.py): en la BVC las que más subieron en 1, 3, 5 y 10
días rindieron MENOS que las rezagadas en el período siguiente. Perseguir a la ganadora pierde dos veces: por la reversión y por el costo del cambio."""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

from .config import ROOT

ESTUDIO_REL = "data/estudio_relativo_bvc.json"


def estudio_relativo(k: int = 3) -> dict[str, Any] | None:
    """Fila del estudio "ganadoras contra rezagadas" para el horizonte k. None si no se ha corrido."""
    try:
        t = json.loads((ROOT / ESTUDIO_REL).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    f = next((x for x in t.get("filas", []) if x["k"] == k), None)
    return f if f and f.get("n", 0) >= 30 else None


def sugerencias(tenidos: list[str], liquidez: dict[str, dict[str, Any]], banco: list[Any], montos: dict[str, float], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Cambios que vale la pena proponer ahora. `liquidez` = {ticker: resultado de liquidez.medir}; `banco` = candidatas de la BVC ya filtradas (líquidas, sin alerta),
    de mayor a menor movimiento esperado. Devuelve [{de, a, tipo, motivo, por_que_a}]."""
    c = cfg["cambios"]
    if not c.get("activo") or not banco:
        return []
    destino = banco[0].c
    out = []
    for t in tenidos:
        l = liquidez.get(t)
        if not l or l["nivel"] not in c["niveles"] or l.get("mediana_mm") is None:
            continue
        med = f"$ {l['mediana_mm']:,.0f} millones".replace(",", ".")
        tuyo = montos.get(t)
        parte = ""
        if tuyo and l["mediana_mm"]:
            pct = tuyo / (l["mediana_mm"] * 1e6) * 100
            parte = f" Lo que tienes (≈ $ {tuyo / 1e6:,.1f} millones) es {'menos del 1' if pct < 1 else f'el {pct:,.0f}'} % de todo lo que se negocia en un día.".replace(",", "§").replace(".", ",").replace("§", ".")
        out.append(dict(de=t, a=destino.ticker, tipo="liquidez",
                        motivo=f"{t} casi no se negocia en trii (unos {med} al día; hoy {l['acciones_hoy']:,.0f} acciones).".replace(",", ".") + parte
                               + " Si necesitas salir rápido puedes quedarte sin comprador o vender muy por debajo.",
                        por_que_a=f"es la acción de la BVC con liquidez buena que más puede moverse hasta el próximo corte (±{destino.mee * 100:.1f} %)".replace(".", ",")
                                  + (f" y negocia unos $ {destino.valor_negociado_mm:,.0f} millones al día.".replace(",", ".") if destino.valor_negociado_mm else ".")))
    return out


def nuevos(sugs: list[dict[str, Any]], memoria: dict[str, Any], ahora: dt.datetime, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Los que no se han avisado en las últimas `repetir_h` horas (clave: tipo y acción de la que se sale). Actualiza `memoria`."""
    hechos = memoria.setdefault("cambios_avisados", {})
    out = []
    for s in sugs:
        k = f"{s['tipo']}|{s['de']}"
        ult = hechos.get(k)
        if ult and (ahora - dt.datetime.fromisoformat(ult)).total_seconds() / 3600 < cfg["cambios"]["repetir_h"]:
            continue
        hechos[k] = ahora.isoformat(timespec="minutes")
        out.append(s)
    vivos = {f"{s['tipo']}|{s['de']}" for s in sugs}
    for k in [k for k in hechos if k not in vivos]:                                    # ya no la tienes (o ya no aplica): se olvida, por si vuelve a pasar
        del hechos[k]
    return out
