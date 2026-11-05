"""Baseline à règles : croisement de moyennes mobiles exponentielles (EMA).

Suppose que `observation` contient déjà les colonnes d'EMA calculées par
`features/indicators.py::add_all_indicators` (ex. "ema_12", "ema_26").
Si les colonnes attendues (selon short_period/long_period) ne sont pas
présentes, la stratégie lève une erreur explicite plutôt que de recalculer
un EMA en double — le calcul d'indicateurs n'est pas sa responsabilité.
"""

from __future__ import annotations

from typing import Any


class EmaCrossoverStrategy:
    """BUY quand EMA courte croise au-dessus de l'EMA longue, SELL à l'inverse."""

    def __init__(self, short_period: int = 12, long_period: int = 26) -> None:
        self._short_period = short_period
        self._long_period = long_period
        self._short_col = f"ema_{short_period}"
        self._long_col = f"ema_{long_period}"
        self._prev_short: float | None = None
        self._prev_long: float | None = None
        self._in_position = False

    def decide(self, observation: Any) -> str:
        try:
            short_val = observation[self._short_col]
            long_val = observation[self._long_col]
        except KeyError as exc:
            raise KeyError(
                f"EmaCrossoverStrategy attend les colonnes '{self._short_col}' et "
                f"'{self._long_col}' dans observation — vérifie add_all_indicators() "
                f"ou passe short_period/long_period correspondant aux colonnes présentes."
            ) from exc

        action = "HOLD"
        if self._prev_short is not None and self._prev_long is not None:
            crossed_up = self._prev_short <= self._prev_long and short_val > long_val
            crossed_down = self._prev_short >= self._prev_long and short_val < long_val
            if crossed_up and not self._in_position:
                self._in_position = True
                action = "BUY"
            elif crossed_down and self._in_position:
                self._in_position = False
                action = "SELL"

        self._prev_short = short_val
        self._prev_long = long_val
        return action

    def reset(self) -> None:
        self._prev_short = None
        self._prev_long = None
        self._in_position = False
