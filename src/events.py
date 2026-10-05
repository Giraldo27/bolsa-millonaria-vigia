"""Calendario de eventos de reporte sobre el calendario de sesiones de un activo."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass
class EventCal:
    pre_exit: set = field(default_factory=set)        # última sesión antes de que salga el reporte (vender al cierre)
    reaction: set = field(default_factory=set)        # primera sesión que ya refleja el reporte
    b_entry: dict = field(default_factory=dict)       # fecha de entrada del Setup B -> fecha de salida


def build_event_cal(cal: pd.DatetimeIndex, events: pd.DataFrame, entry_before: int = 2) -> EventCal:
    """events: columnas report_date, timing (BMO/AMC). Reporte BMO en D: sale antes de la apertura de D, así que la
    última sesión previa es D-1. Reporte AMC en D: sale tras el cierre de D, la última sesión previa es D.
    Setup B compra al cierre de la sesión D-entry_before. Eventos posteriores al último dato se ignoran."""
    out = EventCal()
    n = len(cal)
    for _, ev in events.iterrows():
        d, timing = pd.Timestamp(ev["report_date"]), ev["timing"]
        i = int(cal.searchsorted(d))
        if i >= n:
            continue
        is_session = cal[i] == d
        if timing == "BMO":
            exit_i, react_i = i - 1, i
        else:
            exit_i, react_i = (i, i + 1) if is_session else (i - 1, i)
        entry_i = i - entry_before
        if exit_i < 0 or exit_i >= n:
            continue
        out.pre_exit.add(cal[exit_i])
        if react_i < n:
            out.reaction.add(cal[react_i])
        if entry_i >= 0 and exit_i > entry_i:
            out.b_entry[cal[entry_i]] = cal[exit_i]
    return out
