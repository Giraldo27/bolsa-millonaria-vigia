"""Motor de backtest diario, event-driven, sin look-ahead.

Línea de tiempo de cada fecha d (en este orden):
  1. Se ejecutan las entradas pendientes a la APERTURA de d (Setup C, señal del día anterior).
  2. Se gestiona cada posición abierta con la vela de d (SL, TP, reporte, Brent, tiempo; luego trailing y regla USD/COP).
  3. Se evalúan las señales de CIERRE de d (B y A) con datos hasta d y se compra al cierre.
  4. Se evalúa la señal de Brent de d: compra de ECOPETROL a la apertura de la sesión siguiente.
  5. Se marca a mercado el portafolio y se descuenta la micro-compra de actividad si no hubo compra real.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .costs import commission, entry_price, gross_from_budget, market_exit_price
from .data_loader import assert_not_blacklisted
from .events import EventCal
from .indicators import indicator_frame, passes_filter


@dataclass
class AssetData:
    ohlc: pd.DataFrame                 # Open, High, Low, Close en moneda de cotización
    fx: pd.Series | None               # COP por unidad de moneda de cotización (None = COP)
    own: EventCal                      # reportes propios (regla 3, Setup A)
    cat: EventCal                      # eventos de catalizador (Setup B): propios o proxy


@dataclass
class Inputs:
    assets: dict[str, AssetData]
    brent_ret: pd.Series               # variación diaria del Brent, indexada por fecha
    fx_ret: pd.Series                  # variación diaria del USD/COP
    fomc: list[pd.Timestamp]
    activity_cal: pd.DatetimeIndex     # sesiones del concurso (para el costo de actividad)


@dataclass
class Position:
    asset: str
    setup: str
    entry_date: pd.Timestamp
    entry_ref: float
    entry_px: float
    shares: float
    fx_entry: float
    budget: float
    fee_entry: float
    gross_buy: float
    tp: float
    sl: float
    trigger: float
    trail_sl: float
    max_sessions: int
    b_exit: pd.Timestamp | None = None
    sessions: int = 0
    sl_state: str = "SL"
    sl_state_next: str = "SL"
    pending_sl: float = 0.0
    hold_through: bool = False       # True si se abrió en modo All-In (no vende antes de los reportes)


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    activity_cost: float
    activity_days: int
    skipped: dict[str, int]
    mode: str
    purchase_days: list = field(default_factory=list)   # fechas con al menos una compra real
    exposure: pd.Series | None = None                   # fracción del patrimonio invertida al cierre de cada día


class Backtester:
    def __init__(self, cfg: dict[str, Any], inputs: Inputs, mode: str, start: str | None = None,
                 end: str | None = None, capital: float | None = None, deduct_activity: bool = True, compound: bool = True,
                 liquidate_end: bool = True, controller=None):
        """controller(i, fecha, patrimonio) -> nombre de modo (o None): permite cambiar de modo de riesgo durante la
        corrida (simulador de ligas). Las posiciones abiertas conservan los parámetros con que se abrieron."""
        self.cfg, self.inp = cfg, inputs
        self.deduct_activity = deduct_activity
        self.compound = compound   # False: tamaño fijo sobre el capital inicial (sin reinversión)
        self.liquidate_end = liquidate_end
        self.controller = controller
        self.purchase_days: list[pd.Timestamp] = []
        order = cfg["rules"]["entry_order"]
        self.assets_all = [a for a in order if a in inputs.assets]
        self._set_mode(mode)
        if controller is None:
            self.assets_all = self.assets          # sin cambios de modo sólo se preparan los activos del modo
        self.usd = set(cfg["rules"]["usd_assets"])
        start = pd.Timestamp(start or cfg["backtest"]["start"])
        end = end or cfg["backtest"]["end"]
        end = pd.Timestamp(end) if end else None
        idx = inputs.activity_cal
        for a in self.assets_all:
            idx = idx.union(inputs.assets[a].ohlc.index)
        self.dates = idx[(idx >= start) & ((idx <= end) if end is not None else True)]
        self.cash = float(capital or cfg["capital"])
        self.capital0 = self.cash
        self._prep()
        self.positions: list[Position] = []
        self.trades: list[dict] = []
        self.pending_open: dict[pd.Timestamp, list[str]] = {}
        self.skipped = {"sin_cupo": 0, "sin_capital": 0, "fomc": 0, "reporte_cercano": 0, "misma_posicion": 0}

    def _set_mode(self, name: str) -> None:
        self.mode_name = name
        self.mode = self.cfg["modes"][name]
        self.allin = bool(self.mode.get("hold_through_reports"))
        self.setups = self.mode.get("setups", self.cfg["setups"]["priority"])
        only = self.mode.get("only_asset")
        self.assets = [a for a in self.assets_all if only is None or a == only]

    # ---------- preparación ----------
    def _prep(self) -> None:
        d = self.dates
        self.O, self.H, self.L, self.C, self.fxc, self.fxo, self.ret = {}, {}, {}, {}, {}, {}, {}
        self.closeff, self.ind = {}, {}
        for a in self.assets_all:
            ad = self.inp.assets[a]
            px = ad.ohlc.reindex(d)
            self.O[a], self.H[a], self.L[a], self.C[a] = (px[k].values for k in ("Open", "High", "Low", "Close"))
            self.closeff[a] = px["Close"].ffill().values
            own_ret = ad.ohlc["Close"].pct_change().reindex(d)
            self.ret[a] = own_ret.values
            if ad.fx is None:
                fx = pd.Series(1.0, index=d)
            else:
                fx = ad.fx.reindex(d.union(ad.fx.index)).ffill().reindex(d).bfill()
            self.fxc[a] = fx.values
            self.fxo[a] = fx.shift(1).bfill().values
            self.ind[a] = indicator_frame(ad.ohlc["Close"]).reindex(d)
        self.brent = self.inp.brent_ret.reindex(d).values
        self.fxret = self.inp.fx_ret.reindex(d).values
        self.date_i = {x: i for i, x in enumerate(d)}
        blk = set()
        for f in self.inp.fomc:
            if f in self.date_i:
                i = self.date_i[f]
                blk.update(d[max(0, i - (self.cfg["rules"]["fomc_blackout_days"] - 1)): i + 1])
            else:
                j = int(d.searchsorted(f))
                blk.update(d[max(0, j - self.cfg["rules"]["fomc_blackout_days"]): j])
        self.blackout = blk
        self.activity = set(self.inp.activity_cal)

    # ---------- utilidades ----------
    def _setup_ok(self, setup: str, asset: str, i: int) -> bool:
        """Setup habilitado para el activo (config setups.enabled_assets) y filtros de entrada (config filters)."""
        en = self.cfg["setups"].get("enabled_assets", {}).get(setup)
        if en is not None and asset not in en:
            return False
        flt = self.cfg.get("filters", {}).get(setup)
        if flt and asset in flt:                      # filtro específico por activo: {A: {TSLA: {rsi_max: 40}}}
            flt = flt[asset]
        elif flt and any(isinstance(v, dict) for v in flt.values()):
            return True                               # hay filtros por activo, pero no para éste
        return passes_filter(self.ind[asset].iloc[i].to_dict(), flt) if flt else True

    def _equity(self, i: int) -> float:
        return self.cash + sum(p.shares * self.closeff[p.asset][i] * self.fxc[p.asset][i] for p in self.positions)

    def _params(self, asset: str) -> dict:
        a = self.cfg["assets"]
        return a["TSLA_allin"] if (self.allin and asset == "TSLA") else a[asset]

    def _held(self, asset: str) -> bool:
        return any(p.asset == asset for p in self.positions)

    # ---------- entradas ----------
    def _enter(self, asset: str, setup: str, i: int, when: str) -> bool:
        d = self.dates[i]
        assert_not_blacklisted([asset], self.cfg)
        if asset not in self.assets:                  # el modo vigente no opera este activo (p. ej. All-In sólo TSLA)
            return False
        if self._held(asset):
            self.skipped["misma_posicion"] += 1
            return False
        if len(self.positions) >= self.mode["positions"]:
            self.skipped["sin_cupo"] += 1
            return False
        ad = self.inp.assets[asset]
        if when == "close" and not self.allin and d in ad.own.pre_exit:
            self.skipped["reporte_cercano"] += 1          # mantendría la posición durante un reporte
            return False
        eq = self._equity(i if when == "close" else max(i - 1, 0)) if self.compound else self.capital0
        budget = self.mode["size"] * eq
        if self.compound:
            budget = min(budget, self.cash)
        if budget < self.cfg["costs"]["min_position_cop"]:
            self.skipped["sin_capital"] += 1
            return False
        ref = (self.C if when == "close" else self.O)[asset][i]
        px = entry_price(ref, asset, self.cfg["costs"])
        fx = (self.fxc if when == "close" else self.fxo)[asset][i]
        gross = gross_from_budget(budget, self.cfg["costs"])
        fee = commission(gross, self.cfg["costs"])
        shares = gross / (px * fx)
        p = self._params(asset)
        sset, hold = self.cfg["setups"], self.cfg["setups"]["max_hold_sessions"]
        ts = {"A": sset["A"]["time_stop"], "C": sset["C"]["time_stop"], "B": hold}[setup]
        self.cash -= gross + fee
        self.positions.append(Position(
            asset=asset, setup=setup, entry_date=d, entry_ref=ref, entry_px=px, shares=shares, fx_entry=fx,
            budget=budget, fee_entry=fee, gross_buy=gross, tp=px * (1 + p["tp"]), sl=px * (1 - p["sl"]),
            trigger=px * (1 + p["trail_trigger"]), trail_sl=px * (1 + p["trail_sl"]), max_sessions=min(ts, hold),
            b_exit=None if (self.allin or setup != "B") else ad.cat.b_entry.get(d), pending_sl=px * (1 - p["sl"]),
            hold_through=self.allin))
        if not self.bought_today:
            self.purchase_days.append(d)
        self.bought_today = True
        return True

    # ---------- gestión diaria ----------
    def _close(self, p: Position, i: int, ref: float, reason: str, limit: bool, at_open: bool = False) -> None:
        d = self.dates[i]
        px = ref if limit else market_exit_price(ref, p.asset, self.cfg["costs"])
        fx = (self.fxo if at_open else self.fxc)[p.asset][i]
        proceeds = p.shares * px * fx
        fee = commission(proceeds, self.cfg["costs"])
        self.cash += proceeds - fee
        spread = p.shares * (p.entry_px - p.entry_ref) * p.fx_entry
        slip = p.shares * (ref - px) * fx
        gross = p.shares * (ref * fx - p.entry_ref * p.fx_entry)
        invested = p.gross_buy + p.fee_entry
        net = proceeds - fee - invested
        self.trades.append(dict(
            asset=p.asset, setup=p.setup, entry_date=p.entry_date, exit_date=d, entry_ref=p.entry_ref,
            entry_px=p.entry_px, exit_px=px, exit_reason=reason, sessions=p.sessions, fx_entry=p.fx_entry, fx_exit=fx,
            invested_cop=invested, gross_pnl=gross, spread_cost=spread, slippage_cost=slip, fees=p.fee_entry + fee,
            net_pnl=net, ret_net=net / invested, ret_asset=px / p.entry_px - 1))
        self.positions.remove(p)

    def _manage(self, p: Position, i: int) -> None:
        o, h, l, c = self.O[p.asset][i], self.H[p.asset][i], self.L[p.asset][i], self.C[p.asset][i]
        d = self.dates[i]
        ad, rules = self.inp.assets[p.asset], self.cfg["rules"]
        p.sessions += 1
        sl_hit, tp_hit = l <= p.sl, h >= p.tp
        first = rules["same_candle_assumption"] if (sl_hit and tp_hit) else ("SL" if sl_hit else "TP")
        if sl_hit or tp_hit:
            if first == "SL":
                gap = o < p.sl
                self._close(p, i, o if gap else p.sl, p.sl_state, False, at_open=gap)
            else:
                gap = o > p.tp
                self._close(p, i, o if gap else p.tp, "TP", True, at_open=gap)
            return
        reason = None
        if not p.hold_through and d in ad.own.pre_exit:
            reason = "reporte"
        elif p.setup == "B" and p.b_exit is not None and d == p.b_exit:
            reason = "catalizador"
        elif p.asset == "ECOPETROL" and self.brent[i] <= -rules["brent_drop_exit"]:
            reason = "brent_cae"
        elif p.sessions >= p.max_sessions:
            reason = "tiempo"
        if reason:
            self._close(p, i, c, reason, False)
            return
        # actualizaciones de SL: rigen desde la vela siguiente
        if h >= p.trigger and p.pending_sl < p.trail_sl:
            p.pending_sl, p.sl_state_next = p.trail_sl, "SL_trailing"
        if p.asset in self.usd and self.fxret[i] <= -rules["usdcop_drop_trigger"]:
            lvl = p.entry_px * (1 - rules["usdcop_sl"])
            if lvl > p.pending_sl:
                p.pending_sl, p.sl_state_next = lvl, "SL_usdcop"
        if p.pending_sl > p.sl:
            p.sl, p.sl_state = p.pending_sl, p.sl_state_next

    # ---------- señales ----------
    def _close_signals(self, i: int) -> list[tuple[int, int, str, str]]:
        d, out = self.dates[i], []
        s = self.cfg["setups"]
        att = self.mode.get("attack", False)
        for k, a in enumerate(self.assets):
            if np.isnan(self.C[a][i]):
                continue
            if a in self.usd and d in self.blackout:
                self.skipped["fomc"] += 1
                continue
            ad = self.inp.assets[a]
            if "B" in self.setups and d in ad.cat.b_entry and self._setup_ok("B", a, i):
                out.append((0, k, "B", a))
            if "A" in self.setups and a in s["A"]["drop_threshold"]:
                thr = s["A"]["drop_threshold"][a] - (s["A"]["attack_reduction"] if att else 0.0)
                r = self.ret[a][i]
                if (not np.isnan(r) and r <= -thr and not (s["A"]["skip_day_after_own_report"] and d in ad.own.reaction)
                        and self._setup_ok("A", a, i)):
                    out.append((1, k, "A", a))
        return sorted(out)

    # ---------- bucle principal ----------
    def run(self) -> Result:
        n, eq = len(self.dates), np.full(len(self.dates), np.nan)
        expo = np.zeros(n)
        cs = self.cfg["setups"]["C"]
        act_days = 0
        for i, d in enumerate(self.dates):
            self.bought_today = False
            for a in self.pending_open.pop(d, []):
                if not np.isnan(self.O[a][i]):
                    self._enter(a, "C", i, "open")
            for p in list(self.positions):
                if not np.isnan(self.C[p.asset][i]):
                    self._manage(p, i)
            for _, _, setup, a in self._close_signals(i):
                self._enter(a, setup, i, "close")
            if ("C" in self.setups and cs["asset"] in self.assets and not np.isnan(self.brent[i])
                    and self.brent[i] >= cs["brent_up"] and i + 1 < n and self._setup_ok("C", cs["asset"], i)):
                ca = cs["asset"]
                j = next((k for k in range(i + 1, n) if not np.isnan(self.O[ca][k])), None)
                if j is not None:
                    self.pending_open.setdefault(self.dates[j], []).append(ca)
            if d in self.activity and not self.bought_today:
                act_days += 1
                if self.deduct_activity:
                    self.cash -= self.cfg["costs"]["activity_micro_buy_cop"]
            if i == n - 1 and self.liquidate_end:
                for p in list(self.positions):
                    self._close(p, i, self.closeff[p.asset][i], "fin_datos", False)
            eq[i] = self._equity(i)
            expo[i] = (eq[i] - self.cash) / eq[i] if eq[i] > 0 else 0.0
            if self.controller is not None and i < n - 1:
                new = self.controller(i, d, eq[i])
                if new and new != self.mode_name:
                    self._set_mode(new)
        trades = pd.DataFrame(self.trades)
        if not trades.empty:
            assert_not_blacklisted(list(trades["asset"].unique()), self.cfg)
        return Result(trades, pd.Series(eq, index=self.dates, name="equity"),
                      act_days * self.cfg["costs"]["activity_micro_buy_cop"], act_days, self.skipped, self.mode_name,
                      self.purchase_days, pd.Series(expo, index=self.dates, name="exposure"))
