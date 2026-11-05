"""Baseline la plus sévère du projet : politique aléatoire à exposition moyenne
équivalente à une cible donnée.

Si l'agent RL ne bat pas cette baseline, il n'a rien appris d'utile — une
politique qui décide au hasard, avec la même exposition moyenne au marché,
sert de barre de référence minimale (cf. plan, section benchmarks).
"""

from __future__ import annotations

import random
from typing import Any


class RandomWeightedPolicy:
    """Décide HOLD/BUY/SELL au hasard, pondéré pour viser `target_exposure_pct`
    de temps passé en position sur la durée.
    """

    def __init__(self, target_exposure_pct: float = 0.5, seed: int | None = None) -> None:
        if not 0.0 <= target_exposure_pct <= 1.0:
            raise ValueError("target_exposure_pct doit être dans [0, 1]")
        self._target_exposure_pct = target_exposure_pct
        self._seed = seed
        self._rng = random.Random(seed)
        self._in_position = False

    def decide(self, observation: Any) -> str:
        if self._in_position:
            # Probabilité de sortie choisie pour que le temps moyen en position
            # converge vers target_exposure_pct (processus à deux états).
            exit_probability = 1.0 - self._target_exposure_pct
            if self._rng.random() < exit_probability:
                self._in_position = False
                return "SELL"
            return "HOLD"

        entry_probability = self._target_exposure_pct
        if self._rng.random() < entry_probability:
            self._in_position = True
            return "BUY"
        return "HOLD"

    def reset(self) -> None:
        self._rng = random.Random(self._seed)
        self._in_position = False
