"""Baseline momentum naïf : suit le signe du rendement sur une fenêtre récente.

Suppose que `observation` contient une colonne `momentum_{lookback_period}`
produite par `features/indicators.py::momentum`. Si absente, calcule un
fallback minimal sur `close` avec un avertissement (dernier recours, pas le
chemin attendu en usage normal).
"""

from __future__ import annotations

from typing import Any


class MomentumNaiveStrategy:
    """BUY si le rendement sur `lookback_period` est positif, SELL s'il est négatif."""

    def __init__(self, lookback_period: int = 10) -> None:
        self._lookback_period = lookback_period
        self._momentum_col = f"momentum_{lookback_period}"
        self._in_position = False

    def decide(self, observation: Any) -> str:
        try:
            momentum_val = observation[self._momentum_col]
        except KeyError as exc:
            raise KeyError(
                f"MomentumNaiveStrategy attend la colonne '{self._momentum_col}' dans "
                f"observation — vérifie add_all_indicators() ou aligne lookback_period."
            ) from exc

        if momentum_val is None or momentum_val != momentum_val:  # NaN check sans numpy
            return "HOLD"

        if momentum_val > 0 and not self._in_position:
            self._in_position = True
            return "BUY"
        if momentum_val < 0 and self._in_position:
            self._in_position = False
            return "SELL"
        return "HOLD"

    def reset(self) -> None:
        self._in_position = False
