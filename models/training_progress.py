"""Lecture/écriture de la progression d'entraînement — AUCUNE dépendance lourde
(pas de torch, stable-baselines3, gymnasium, hmmlearn) volontairement.

Historiquement `get_training_progress`/`TrainingProgressCallback` vivaient
dans `models/train_pipeline.py`, qui importe `stable_baselines3` (donc
`torch`) en tête de fichier. Comme `app/dashboard.py` n'a besoin que de LIRE
la progression (pas d'entraîner quoi que ce soit), importer
`train_pipeline` depuis le dashboard forçait Python à charger ~4s de
librairies ML à chaque process — négligeable une fois, mais un point de
fragilité inutile. Ce module sépare la responsabilité "lire/écrire un JSON
de progression" de "entraîner un agent PPO", pour que le dashboard reste léger.

`train_pipeline.py` réexporte ces symboles pour ne rien casser côté
compatibilité (imports existants continuent de fonctionner).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

TRAIN_PROGRESS_PATH = Path("storage/runs/process_state/train_progress.json")


def get_training_progress(progress_path: Path = TRAIN_PROGRESS_PATH) -> dict[str, Any]:
    """Lit la progression d'entraînement écrite par `TrainingProgressCallback`.

    Retourne toujours un dict avec au minimum la clé "status" (jamais None) —
    "never_started" si le fichier n'existe pas encore, pour que l'appelant
    dashboard n'ait pas de cas particulier supplémentaire à gérer.
    """
    if not progress_path.exists():
        return {"status": "never_started", "total_timesteps": 0, "points": []}
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "never_started", "total_timesteps": 0, "points": []}


def write_training_progress(payload: dict[str, Any], progress_path: Path = TRAIN_PROGRESS_PATH) -> None:
    """Écrit l'état de progression — utilisé par `TrainingProgressCallback` (models/rl_agent.py)."""
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    progress_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
