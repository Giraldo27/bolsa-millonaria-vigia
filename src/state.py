"""Estado persistente del participante (state.json): posición, rentabilidades, cambios y operaciones.

Escritura atómica (archivo temporal + reemplazo) para que el monitor, el bot y la nube no corrompan el archivo."""
from __future__ import annotations

import contextlib
import copy
import datetime as dt
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from . import concurso


class ErrorEstado(ValueError):
    """Operación no permitida por las reglas del plan (límites, activo prohibido, datos inválidos)."""


def universo_permitido(cfg: dict[str, Any]) -> set[str]:
    u = cfg["universe"]
    return (set(u["mgc"]) | set(u["etf"]) | set(u["local"])) - set(u["blacklist"])


def estado_inicial(cfg: dict[str, Any]) -> dict[str, Any]:
    base = cfg["posicion_base"]
    return {
        "posicion": {"activo": base["ticker"], "monto_cop": round(cfg["capital"] * base["peso"]), "precio_entrada": None,
                     "fecha_entrada": None},
        "rentabilidad": {"mia": None, "umbral": None, "primero": None, "segundo": None, "actualizado": None},
        "cambios": {"usados": 0, "maximo": cfg["estado"]["max_cambios"], "historial": []},
        "operaciones": {"total": 0, "por_semana": {}, "historial": []},
        "semaforo": {},          # ticker -> {"color", "desde", "ultimo_aviso", "episodio"}
        "salud": {},             # fuente -> {"fallos", "ultimo_ok", "ultimo_aviso", "caida_avisada"}
        "alertas_pendientes": [],  # mensajes que no se pudieron enviar (se reintentan)
        "cartera": [],           # compras registradas: {"ticker", "cantidad", "precio", "fecha", "monto_cop"} (la principal es la de mayor monto)
        "deshacer": None,        # copia previa al último registro de compra o venta (para "me equivoqué")
        "version": 1,
    }


class Estado:
    def __init__(self, cfg: dict[str, Any], path: str | Path | None = None):
        self.cfg = cfg
        raiz = Path(__file__).resolve().parents[1]
        self.path = Path(path) if path else raiz / cfg["estado"]["archivo"]
        self.d = self._leer()

    # ---------- persistencia ----------
    @property
    def _bak(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".bak")

    def _leer(self) -> dict[str, Any]:
        """Lee state.json. RESPALDO: si está dañado, restaura la copia .bak (la última versión buena) y lo avisa."""
        base = estado_inicial(self.cfg)
        self.recuperado_de_respaldo = False
        if self.path.exists():
            try:
                cargado = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                if not self._bak.exists():
                    raise ErrorEstado(f"state.json está dañado ({e}) y no hay copia de respaldo; revisa o bórralo para empezar de cero") from e
                try:
                    cargado = json.loads(self._bak.read_text(encoding="utf-8"))
                except json.JSONDecodeError as e2:
                    raise ErrorEstado(f"state.json y su respaldo están dañados ({e2})") from e2
                self.recuperado_de_respaldo = True
            for k, v in base.items():
                cargado.setdefault(k, v)
            return cargado
        return base

    def guardar(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(self.d, f, ensure_ascii=False, indent=2)
        if self.path.exists():
            try:
                json.loads(self.path.read_text(encoding="utf-8"))          # sólo se respalda una versión que sea JSON válido
                shutil.copy2(self.path, self._bak)
            except (json.JSONDecodeError, OSError):
                pass
        os.replace(tmp, self.path)

    @classmethod
    @contextlib.contextmanager
    def transaccion(cls, cfg: dict[str, Any], path: str | Path | None = None) -> Iterator["Estado"]:
        """Lectura-modificación-escritura con candado: el monitor, el radar y el bot pueden correr a la vez sin pisarse.
        Uso: `with Estado.transaccion(cfg) as e: e.registrar_op(...)`. Guarda al salir si no hubo error."""
        probe = cls(cfg, path)
        lock = probe.path.with_suffix(probe.path.suffix + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        limite = time.monotonic() + cfg["estado"]["lock_timeout_s"]
        while True:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except (FileExistsError, PermissionError):                      # en Windows, crear el candado justo mientras otro hilo lo borra da "permiso denegado"
                try:                                                        # candado huérfano (un proceso murió): se limpia
                    if time.time() - lock.stat().st_mtime > cfg["estado"]["lock_timeout_s"] * 2:
                        lock.unlink()
                        continue
                except FileNotFoundError:
                    continue
                except PermissionError:                                     # el candado está a medio borrar: se reintenta, pero respetando el tiempo límite
                    pass
                if time.monotonic() > limite:
                    raise ErrorEstado("No se pudo obtener el candado de state.json (otro proceso lo está usando). Intenta de nuevo.")
                time.sleep(0.05)
        try:
            e = cls(cfg, path)                                              # se lee DENTRO del candado: datos frescos
            yield e
            e.guardar()
        finally:
            try:
                lock.unlink()
            except FileNotFoundError:
                pass

    # ---------- salud de las fuentes y alertas pendientes ----------
    def registrar_fuente(self, nombre: str, ok: bool, ahora: dt.datetime, umbral: int, cada_min: float) -> str | None:
        """Cuenta fallos consecutivos de una fuente. Devuelve 'CAIDA' cuando llega al umbral (y no se avisó hace menos de `cada_min`),
        'RECUPERADA' cuando vuelve tras haber avisado la caída, o None."""
        s = self.d["salud"].setdefault(nombre, {"fallos": 0, "ultimo_ok": None, "ultimo_aviso": None, "caida_avisada": False})
        if ok:
            s["fallos"], s["ultimo_ok"] = 0, ahora.isoformat(timespec="minutes")
            if s["caida_avisada"]:
                s["caida_avisada"] = False
                return "RECUPERADA"
            return None
        s["fallos"] = int(s["fallos"]) + 1
        if s["fallos"] < umbral:
            return None
        ult = dt.datetime.fromisoformat(s["ultimo_aviso"]) if s.get("ultimo_aviso") else None
        if not s["caida_avisada"] or ult is None or (ahora - ult).total_seconds() / 60 >= cada_min:
            s["ultimo_aviso"], s["caida_avisada"] = ahora.isoformat(timespec="minutes"), True
            return "CAIDA"
        return None

    def agregar_pendiente(self, texto: str, ahora: dt.datetime) -> None:
        """Guarda un mensaje que no se pudo enviar para reintentarlo en la siguiente corrida (máx. alertas_pendientes_max)."""
        p = self.d["alertas_pendientes"]
        p.append({"cuando": ahora.isoformat(timespec="minutes"), "texto": texto})
        del p[: max(len(p) - self.cfg["estado"]["alertas_pendientes_max"], 0)]

    def tomar_pendientes(self) -> list[dict[str, str]]:
        p, self.d["alertas_pendientes"] = self.d["alertas_pendientes"], []
        return p

    def puede_alertar(self, ticker: str, ahora: dt.datetime, enfriamiento_min: float) -> bool:
        """Enfriamiento: no repetir alertas del mismo activo antes de `enfriamiento_min` minutos."""
        u = self.d["semaforo"].get(ticker, {}).get("ultimo_aviso")
        return u is None or (ahora - dt.datetime.fromisoformat(u)).total_seconds() / 60 >= enfriamiento_min

    # ---------- lectura ----------
    @property
    def activo(self) -> str:
        return self.d["posicion"]["activo"]

    @property
    def rent(self) -> dict[str, Any]:
        return self.d["rentabilidad"]

    def cambios_usados(self) -> int:
        return int(self.d["cambios"]["usados"])

    def cambios_restantes(self) -> int:
        return int(self.d["cambios"]["maximo"]) - self.cambios_usados()

    def cambio_hoy(self, fecha: dt.date) -> bool:
        return any(c["fecha"] == fecha.isoformat() for c in self.d["cambios"]["historial"])

    def ops_semana(self, fecha: dt.date) -> int:
        return int(self.d["operaciones"]["por_semana"].get(concurso.clave_semana(fecha, self.cfg), 0))

    def ops_faltantes_semana(self, fecha: dt.date) -> int:
        """Operaciones que faltan para llegar a la meta semanal (4) — nunca negativo."""
        return max(self.cfg["concurso"]["actividad"]["ops_por_semana"] - self.ops_semana(fecha), 0)

    def ops_total(self) -> int:
        return int(self.d["operaciones"]["total"])

    def operacion_hoy(self, fecha: dt.date) -> bool:
        return any(o["fecha"] == fecha.isoformat() for o in self.d["operaciones"]["historial"])

    # ---------- escritura ----------
    def actualizar_rank(self, mia: float, objetivo: float, ahora: dt.datetime, primero: float | None = None,
                        segundo: float | None = None) -> None:
        """Rentabilidades en puntos porcentuales (p. ej. 6.5 = +6,5 %). 'objetivo' = umbral del próximo corte
        (o la rentabilidad del #1 en la semana 5). 'segundo' (opcional) = rentabilidad del #2, para medir la ventaja si voy #1."""
        for nombre, v in (("mi rentabilidad", mia), ("objetivo", objetivo)) + ((("rentabilidad del #2", segundo),) if segundo is not None else ()):
            if not isinstance(v, (int, float)) or isinstance(v, bool) or abs(v) > 1000:
                raise ErrorEstado(f"{nombre} inválida: {v!r}")
        r = self.d["rentabilidad"]
        r["mia"], r["umbral"], r["actualizado"] = float(mia), float(objetivo), ahora.isoformat(timespec="minutes")
        if concurso.semana_final(ahora, self.cfg):              # en la semana 5 el objetivo es la rentabilidad del #1
            r["primero"] = float(objetivo if primero is None else primero)
        elif primero is not None:
            r["primero"] = float(primero)
        r["segundo"] = float(segundo) if segundo is not None else None
        self.guardar()

    def registrar_op(self, ahora: dt.datetime, detalle: str = "micro-compra ECOPETROL") -> None:
        """Operación de actividad (micro-compra u otra). Los cambios de activo también cuentan (ver registrar_cambio)."""
        self._contar_op(ahora, detalle)
        self.guardar()

    def _contar_op(self, ahora: dt.datetime, detalle: str) -> None:
        o = self.d["operaciones"]
        k = concurso.clave_semana(ahora.date(), self.cfg)
        o["por_semana"][k] = int(o["por_semana"].get(k, 0)) + 1
        o["total"] = int(o["total"]) + 1
        o["historial"].append({"fecha": ahora.date().isoformat(), "hora": ahora.strftime("%H:%M"), "detalle": detalle})

    def registrar_cambio(self, ticker: str, monto: float, precio: float, ahora: dt.datetime) -> str:
        """Registra el cambio de activo. Valida: universo permitido (sin lista negra), máx. 4 cambios y 1 por día."""
        t = ticker.strip().upper()
        if t in set(self.cfg["universe"]["blacklist"]):
            raise ErrorEstado(f"{t} está en la LISTA NEGRA: no se puede operar.")
        if t not in universo_permitido(self.cfg):
            raise ErrorEstado(f"{t} no está en el universo permitido.")
        if monto <= 0 or precio <= 0:
            raise ErrorEstado("El monto y el precio deben ser positivos.")
        base = self.d["posicion"]
        es_entrada_inicial = base["precio_entrada"] is None and t == base["activo"]
        if not es_entrada_inicial:
            if self.cambios_restantes() <= 0:
                raise ErrorEstado(f"Ya usaste los {self.d['cambios']['maximo']} cambios permitidos.")
            if self.cambio_hoy(ahora.date()):
                raise ErrorEstado("Máximo 1 cambio por día: ya registraste uno hoy.")
        anterior = base["activo"]
        self.d["posicion"] = {"activo": t, "monto_cop": float(monto), "precio_entrada": float(precio), "fecha_entrada": ahora.date().isoformat()}
        if not es_entrada_inicial:
            c = self.d["cambios"]
            c["usados"] = int(c["usados"]) + 1
            c["historial"].append({"fecha": ahora.date().isoformat(), "de": anterior, "a": t, "monto": float(monto), "precio": float(precio)})
        self.d["cartera"] = [{"ticker": t, "cantidad": None, "precio": float(precio), "fecha": ahora.date().isoformat(), "monto_cop": float(monto)}]   # un cambio reemplaza TODA la cartera
        self._contar_op(ahora, f"{'entrada' if es_entrada_inicial else 'cambio'} {anterior}→{t}")
        self.guardar()
        return "Entrada inicial registrada." if es_entrada_inicial else f"Cambio {anterior} → {t} registrado."

    # ---------- cartera de varias acciones ----------
    def tenidos(self) -> set[str]:
        """Acciones que realmente tienes (las compras registradas). La base cuenta sólo si ya registraste su compra."""
        t = {p["ticker"] for p in self.d["cartera"]}
        if self.d["posicion"].get("precio_entrada"):
            t.add(self.activo)
        return t

    def _validar_titulo(self, ticker: str) -> str:
        t = ticker.strip().upper()
        if t in set(self.cfg["universe"]["blacklist"]):
            raise ErrorEstado(f"{t} está en la LISTA NEGRA: no se puede operar.")
        if t not in universo_permitido(self.cfg):
            raise ErrorEstado(f"{t} no está en el universo permitido.")
        return t

    def _reasignar_principal(self) -> None:
        """La acción principal (la que vigilan el semáforo y la Regla Maestra) es la de mayor monto en COP de la cartera."""
        if not self.d["cartera"]:
            return
        mayor = max(self.d["cartera"], key=lambda p: p.get("monto_cop") or 0.0)
        self.d["posicion"] = {"activo": mayor["ticker"], "monto_cop": float(mayor.get("monto_cop") or 0.0), "precio_entrada": float(mayor["precio"]),
                              "fecha_entrada": mayor["fecha"]}

    def registrar_compra(self, ticker: str, cantidad: float, precio: float, ahora: dt.datetime, monto_cop: float | None = None) -> str:
        """Agrega una compra a la cartera (no es un 'cambio de acción': no gasta uno de los 4). Si ya tenías esa acción, promedia el precio.
        `monto_cop` es el valor aproximado en pesos (cantidad × precio × TRM para acciones de EE. UU.). Cuenta como una operación de actividad."""
        t = self._validar_titulo(ticker)
        if cantidad <= 0 or precio <= 0:
            raise ErrorEstado("La cantidad y el precio deben ser positivos.")
        self.guardar_punto(f"la compra de {cantidad:g} {t}", ahora)
        monto = float(monto_cop) if monto_cop else float(cantidad * precio)
        previa = next((p for p in self.d["cartera"] if p["ticker"] == t), None)
        if previa and previa.get("cantidad"):
            total = previa["cantidad"] + cantidad
            previa["precio"] = (previa["precio"] * previa["cantidad"] + precio * cantidad) / total
            previa["cantidad"], previa["monto_cop"] = total, float(previa.get("monto_cop") or 0.0) + monto
        elif previa:                                                                                  # venía de /pos (sin cantidad): se reemplaza con los datos reales
            previa.update(cantidad=float(cantidad), precio=float(precio), monto_cop=monto, fecha=ahora.date().isoformat())
        else:
            self.d["cartera"].append({"ticker": t, "cantidad": float(cantidad), "precio": float(precio), "fecha": ahora.date().isoformat(), "monto_cop": monto})
        self._reasignar_principal()
        self._contar_op(ahora, f"compra {t}")
        self.guardar()
        return f"Compra de {t} registrada."

    def registrar_venta(self, ticker: str, cantidad: float | None, ahora: dt.datetime) -> str:
        """Vende todo (cantidad=None) o parte de una acción de tu cartera. Cuenta como una operación de actividad."""
        t = ticker.strip().upper()
        p = next((x for x in self.d["cartera"] if x["ticker"] == t), None)
        if p is None:
            raise ErrorEstado(f"{t} no está en tu cartera registrada.")
        if cantidad is not None and cantidad <= 0:
            raise ErrorEstado("La cantidad debe ser positiva.")
        self.guardar_punto(f"la venta de {t}", ahora)
        if cantidad is None or not p.get("cantidad") or cantidad >= p["cantidad"]:
            self.d["cartera"].remove(p)
            resultado = f"Venta total de {t} registrada."
        else:
            frac = cantidad / p["cantidad"]
            p["monto_cop"] = float(p.get("monto_cop") or 0.0) * (1 - frac)
            p["cantidad"] = p["cantidad"] - cantidad
            resultado = f"Venta de {cantidad:g} acciones de {t} registrada."
        if self.d["cartera"]:
            self._reasignar_principal()
        else:
            self.d["posicion"] = {**self.d["posicion"], "precio_entrada": None, "fecha_entrada": None}
        self._contar_op(ahora, f"venta {t}")
        self.guardar()
        return resultado

    # ---------- deshacer el último registro ----------
    def guardar_punto(self, que: str, ahora: dt.datetime) -> None:
        """Copia de lo que cambia una compra o una venta, para poder deshacerla ("me equivoqué" o /deshacer). Sólo se recuerda el último paso."""
        self.d["deshacer"] = {"que": que, "cuando": ahora.isoformat(timespec="minutes"), "cartera": copy.deepcopy(self.d["cartera"]),
                              "posicion": copy.deepcopy(self.d["posicion"]), "operaciones": copy.deepcopy(self.d["operaciones"])}

    def deshacer(self) -> str:
        """Devuelve la cartera, la acción principal y el conteo de operaciones a como estaban antes del último registro. Devuelve qué se deshizo."""
        p = self.d.get("deshacer")
        if not p:
            raise ErrorEstado("No hay nada que deshacer: sólo recuerdo el último registro de compra o venta.")
        self.d["cartera"], self.d["posicion"], self.d["operaciones"] = p["cartera"], p["posicion"], p["operaciones"]
        self.d["deshacer"] = None
        self.guardar()
        return p["que"]

    def set_semaforo(self, ticker: str, color: str, ahora: dt.datetime, avisado: bool = False) -> None:
        s = self.d["semaforo"].setdefault(ticker, {})
        if s.get("color") != color:
            s["desde"] = ahora.isoformat(timespec="minutes")
        s["color"] = color
        if avisado:
            s["ultimo_aviso"] = ahora.isoformat(timespec="minutes")

    def color_previo(self, ticker: str) -> str | None:
        return self.d["semaforo"].get(ticker, {}).get("color")

    def copia(self) -> dict[str, Any]:
        return copy.deepcopy(self.d)
