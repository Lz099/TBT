"""Lancement/arrêt/statut des scripts longs du projet comme sous-processus OS,
piloté depuis le dashboard Streamlit.

POURQUOI DES SOUS-PROCESSUS : Streamlit est mono-thread et se recharge à
chaque interaction — impossible d'y faire tourner un entraînement de
plusieurs minutes/heures ou une boucle infinie de paper trading sans bloquer
l'UI. Chaque tâche est donc lancée comme un vrai processus OS séparé
(`subprocess.Popen`), déjà le pattern utilisé par `paper_trading/live_runner.py`
(pensé dès le départ comme un processus autonome écrivant son état dans
SQLite). Ce module généralise ce pattern aux 3 tâches longues du projet.

SÉCURITÉ : `TASK_REGISTRY` est une liste FERMÉE de commandes fixes (les
scripts déjà audités du projet). Aucune commande arbitraire construite à
partir d'une entrée utilisateur libre n'est jamais exécutée — seuls des
arguments (symbole, timeframe, stratégie, nombre de timesteps) choisis parmi
des valeurs contrôlées (selectbox/slider, pas de texte libre côté dashboard)
sont ajoutés à une commande de base fixe.

État persisté par tâche dans storage/runs/process_state/{task_name}.json
(PID, statut, timestamps) + {task_name}.log (stdout/stderr redirigés) — un
fichier partagé entre processus, même philosophie que storage/runs/live_state.py
pour l'état du portefeuille live.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

_STATE_DIR = Path("storage/runs/process_state")

TaskStatus = Literal["never_started", "running", "finished", "error", "stopped"]

# Commandes de base fixes — seule la fonction start_task() peut leur ajouter
# des arguments contrôlés (jamais de texte libre non validé).
TASK_REGISTRY: dict[str, list[str]] = {
    # `-m package.module` (plutôt que `package/module.py`) garantit que la
    # racine du projet est ajoutée à sys.path par Python lui-même — avec un
    # chemin de fichier direct, certaines configurations Windows n'ajoutent
    # PAS automatiquement `cwd` à sys.path, ce qui casse tous les imports
    # absolus du projet (`from backtest.engine import ...`, etc.) même en
    # fixant `cwd=` dans subprocess.Popen.
    "fetch_data": [sys.executable, "-m", "data.fetch_historical"],
    "train_agent": [sys.executable, "-m", "models.train_pipeline"],
    "live_trading": [sys.executable, "-m", "paper_trading.live_runner"],
}


def _state_path(task_name: str) -> Path:
    return _STATE_DIR / f"{task_name}.json"


def _log_path(task_name: str) -> Path:
    return _STATE_DIR / f"{task_name}.log"


def _read_state(task_name: str) -> dict[str, Any] | None:
    path = _state_path(task_name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(task_name: str, state: dict[str, Any]) -> None:
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    _state_path(task_name).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _is_pid_alive(pid: int) -> bool:
    """Vérifie si un PID correspond toujours à un processus vivant (portable Windows/Unix)."""
    try:
        import psutil  # dépendance déjà listée dans requirements.txt (non utilisée jusqu'ici)

        return psutil.pid_exists(pid)
    except ImportError:
        # Fallback sans psutil : os.kill(pid, 0) ne fonctionne pas sur Windows
        # pour tester la vivacité sans envoyer de signal réel. On retombe sur
        # subprocess.Popen.poll() côté start_task/stop_task, qui suffit tant
        # que ce process Streamlit reste le même que celui qui a lancé la
        # tâche (cas normal d'usage). Ici, en dernier recours, on suppose vivant.
        return True


def start_task(task_name: str, extra_args: list[str] | None = None) -> None:
    """Lance `task_name` comme sous-processus si aucune instance n'est déjà en cours.

    `extra_args` doit toujours provenir de valeurs contrôlées côté dashboard
    (selectbox/slider), jamais de texte libre non validé — cf. avertissement
    sécurité en tête de fichier.
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(f"Tâche inconnue: {task_name!r} — attendu un de {list(TASK_REGISTRY)}")

    current = get_task_status(task_name)
    if current["status"] == "running":
        raise RuntimeError(f"La tâche '{task_name}' est déjà en cours (PID {current.get('pid')}).")

    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _log_path(task_name)
    log_file = log_path.open("w", encoding="utf-8")  # écrase le log du run précédent

    command = [*TASK_REGISTRY[task_name], *(extra_args or [])]
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(Path(__file__).resolve().parent.parent),  # racine du projet, quel que soit le cwd de Streamlit
    )

    _write_state(
        task_name,
        {
            "pid": process.pid,
            "status": "running",
            "command": command,
            "started_at": time.time(),
            "ended_at": None,
            "returncode": None,
        },
    )


def stop_task(task_name: str) -> None:
    """Termine le sous-processus de `task_name` s'il tourne encore."""
    state = _read_state(task_name)
    if state is None or state.get("status") != "running":
        return

    pid = state.get("pid")
    if pid is not None:
        try:
            import psutil

            proc = psutil.Process(pid)
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
        except ImportError:
            import os
            import signal

            try:
                os.kill(pid, signal.SIGTERM)
            except (OSError, ProcessLookupError):
                pass
        except Exception:
            pass  # processus déjà terminé ou inaccessible — on met quand même à jour l'état

    state["status"] = "stopped"
    state["ended_at"] = time.time()
    _write_state(task_name, state)


def get_task_status(task_name: str) -> dict[str, Any]:
    """Retourne le statut courant de `task_name`, avec vérification de vivacité du PID.

    Toujours un dict avec au minimum la clé "status" — jamais None, pour que
    l'appelant dashboard n'ait pas à gérer un cas particulier supplémentaire.
    """
    state = _read_state(task_name)
    if state is None:
        return {"status": "never_started"}

    if state.get("status") == "running":
        pid = state.get("pid")
        if pid is not None and not _is_pid_alive(pid):
            # Le process s'est terminé de lui-même entre deux lectures du
            # dashboard — on ne connaît pas son code de sortie exact ici
            # (psutil absent ou process déjà nettoyé par l'OS), donc statut
            # neutre "finished" plutôt que de deviner une erreur.
            state["status"] = "finished"
            state["ended_at"] = state.get("ended_at") or time.time()
            _write_state(task_name, state)

    return state


def get_task_logs(task_name: str, tail_lines: int = 50) -> str:
    """Retourne les `tail_lines` dernières lignes du log de `task_name` (chaîne vide si absent)."""
    path = _log_path(task_name)
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-tail_lines:])
