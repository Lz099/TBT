"""Stops adaptatifs (Chandelier Exit) basés sur l'ATR.

Anti-look-ahead : le plus haut/bas glissant et l'ATR utilisés au temps t ne
doivent porter que sur des barres <= t (rolling standard, pas de centre=True).
La barre courante est incluse dans la fenêtre car son high/low/close sont
déjà connus une fois la barre clôturée — c'est la convention utilisée
partout ailleurs dans le projet pour les indicateurs de fenêtre glissante.
"""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from features.indicators import atr


@dataclass(frozen=True)
class ChandelierParams:
    lookback: int
    atr_multiplier: float


def get_chandelier_params(timeframe: str, config: dict[str, Any]) -> ChandelierParams:
    """Choisit lookback/multiplicateur chandelier selon le timeframe.

    `config` est la section `risk` de config.yaml (ou un dict équivalent).
    Le timeframe "1d" utilise les paramètres `_1d`, tout autre timeframe
    (5m, 1h, ...) utilise les paramètres `_intraday`.
    """
    risk_cfg = config["risk"] if "risk" in config else config
    if timeframe == "1d":
        return ChandelierParams(
            lookback=risk_cfg["chandelier_lookback_1d"],
            atr_multiplier=risk_cfg["chandelier_atr_multiplier_1d"],
        )
    return ChandelierParams(
        lookback=risk_cfg["chandelier_lookback_intraday"],
        atr_multiplier=risk_cfg["chandelier_atr_multiplier_intraday"],
    )


def chandelier_exit_long(df: pd.DataFrame, lookback: int, atr_multiplier: float) -> pd.Series:
    """Stop-loss adaptatif pour position longue.

    stop = plus_haut_sur_N_périodes - atr_multiplier * ATR(N)
    """
    highest_high = df["high"].rolling(window=lookback, min_periods=lookback).max()
    atr_series = atr(df, period=lookback)
    stop = highest_high - atr_multiplier * atr_series
    stop.name = "chandelier_stop_long"
    return stop


def chandelier_exit_short(df: pd.DataFrame, lookback: int, atr_multiplier: float) -> pd.Series:
    """Stop-loss adaptatif pour position courte (symétrique du long).

    stop = plus_bas_sur_N_périodes + atr_multiplier * ATR(N)

    Non utilisée activement en V1 : le plan ne prévoit pas de short pour le
    moment. Implémentée pour la complétude de l'API des stops.
    """
    lowest_low = df["low"].rolling(window=lookback, min_periods=lookback).min()
    atr_series = atr(df, period=lookback)
    stop = lowest_low + atr_multiplier * atr_series
    stop.name = "chandelier_stop_short"
    return stop
