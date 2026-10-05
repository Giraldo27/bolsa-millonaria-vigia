"""Guarda y recupera el estado del bot (posición, ranking, colores, memoria de noticias) entre ejecuciones de GitHub Actions.

Cada ejecución de Actions empieza con un disco vacío, así que el estado vive en una rama aparte llamada `estado` del MISMO repositorio. Uso dentro de un flujo:
    python nube/sincronizar_estado.py bajar     # antes de arrancar
    python nube/sincronizar_estado.py subir     # después (sólo hace commit si algo cambió)
Sólo usa git (ya configurado por actions/checkout con credenciales).

CIFRADO (para repositorio PÚBLICO): si existe la variable de entorno ESTADO_CLAVE, los archivos se guardan cifrados (`state.json.enc`) y nadie que mire el
repositorio puede leer tu posición ni tu rentabilidad. Sin esa variable se guardan tal cual (sólo aceptable con el repositorio PRIVADO).
    python nube/sincronizar_estado.py clave     # imprime una clave nueva para guardarla como secreto ESTADO_CLAVE"""
from __future__ import annotations

import hashlib
import hmac
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

ARCHIVOS = ("state.json", "telegram_offset.json", "noticias_bvc.json")
RAMA = "estado"
CARPETA = "_estado"


def _ejecutar(cmd: list[str], cwd: Path | None = None, tolerar: bool = False) -> subprocess.CompletedProcess:
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode and not tolerar:
        raise RuntimeError(f"falló `{' '.join(cmd)}`: {r.stderr.strip()[:300]}")
    return r


def _clave() -> bytes | None:
    k = (os.environ.get("ESTADO_CLAVE") or "").replace("\ufeff", "").strip()
    return k.encode() if k else None


def _fernet(clave: bytes):
    from cryptography.fernet import Fernet
    return Fernet(clave)


def _firma(clave: bytes, datos: bytes) -> str:
    """Huella del contenido (con la clave, para que no sirva de pista): permite saber si cambió sin volver a cifrar (el cifrado da un resultado distinto cada vez)."""
    return hmac.new(clave, datos, hashlib.sha256).hexdigest()


def bajar(raiz: Path, run: Callable[..., subprocess.CompletedProcess] = _ejecutar) -> bool:
    """Deja en data/ el último estado guardado. Devuelve True si existía la rama `estado` (si no, la crea vacía en _estado/ para el primer `subir`)."""
    existe = run(["git", "ls-remote", "--exit-code", "--heads", "origin", RAMA], cwd=raiz, tolerar=True).returncode == 0
    destino = raiz / CARPETA
    if existe:
        run(["git", "fetch", "--depth", "1", "origin", f"{RAMA}:refs/remotes/origin/{RAMA}"], cwd=raiz)
        run(["git", "worktree", "add", "--detach", CARPETA, f"origin/{RAMA}"], cwd=raiz)
        (raiz / "data").mkdir(exist_ok=True)
        clave = _clave()
        for a in ARCHIVOS:
            cifrado = destino / (a + ".enc")
            if cifrado.exists() and clave:
                (raiz / "data" / a).write_bytes(_fernet(clave).decrypt(cifrado.read_bytes()))
            elif (destino / a).exists():
                shutil.copy2(destino / a, raiz / "data" / a)
    else:
        run(["git", "worktree", "add", "--detach", CARPETA], cwd=raiz)
        run(["git", "checkout", "--orphan", RAMA], cwd=destino)
        run(["git", "rm", "-rf", "--quiet", "."], cwd=destino, tolerar=True)
    return existe


def subir(raiz: Path, run: Callable[..., subprocess.CompletedProcess] = _ejecutar) -> bool:
    """Copia el estado de data/ a la rama `estado` (cifrado si hay ESTADO_CLAVE) y hace push sólo si cambió. Devuelve True si publicó algo."""
    destino = raiz / CARPETA
    if not destino.exists():
        raise RuntimeError("falta _estado/: corre primero `bajar`")
    clave = _clave()
    copiado = False
    for a in ARCHIVOS:
        origen = raiz / "data" / a
        if not origen.exists():
            continue
        copiado = True
        if clave:
            datos = origen.read_bytes()
            firma, arch_firma = _firma(clave, datos), destino / (a + ".firma")
            if not (arch_firma.exists() and arch_firma.read_text().strip() == firma and (destino / (a + ".enc")).exists()):
                (destino / (a + ".enc")).write_bytes(_fernet(clave).encrypt(datos))
                arch_firma.write_text(firma)
            (destino / a).unlink(missing_ok=True)                                        # nunca queda la copia legible junto a la cifrada
        else:
            shutil.copy2(origen, destino / a)
    if not copiado:
        return False
    run(["git", "add", "-A"], cwd=destino)
    if run(["git", "diff", "--cached", "--quiet"], cwd=destino, tolerar=True).returncode == 0:
        return False                                                                                   # nada cambió
    run(["git", "-c", "user.name=bolsa-bot", "-c", "user.email=bolsa-bot@users.noreply.github.com", "commit", "-m", "estado", "--quiet"], cwd=destino)
    run(["git", "push", "origin", f"HEAD:refs/heads/{RAMA}"], cwd=destino)
    return True


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("bajar", "subir", "clave"):
        print(__doc__)
        return 2
    if argv[1] == "clave":
        from cryptography.fernet import Fernet
        print(Fernet.generate_key().decode())
        return 0
    raiz = Path(__file__).resolve().parents[1]
    if argv[1] == "bajar":
        print("estado recuperado de la rama 'estado'" if bajar(raiz) else "primera vez: aún no hay estado guardado")
    else:
        print("estado publicado" if subir(raiz) else "sin cambios en el estado")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
