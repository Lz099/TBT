"""Stratégie baseline : achète une fois puis conserve indéfiniment.

Interface commune à toutes les stratégies du dossier `models/` :
    decide(observation) -> "HOLD" | "BUY" | "SELL"
    reset() -> None

`observation` est un dict-like (dict ou pandas Series) exposant au moins
les colonnes OHLCV + indicateurs de la bougie courante (ex. observation["close"]).
"""

from __future__ import annotations

from typing import Any


class BuyAndHoldStrategy:
    """Achète au tout premier appel, puis HOLD indéfiniment."""

    def __init__(self) -> None:
        self._has_bought = False

    def decide(self, observation: Any) -> str:
        if not self._has_bought:
            self._has_bought = True
            return "BUY"
        return "HOLD"

    def reset(self) -> None:
        self._has_bought = False
