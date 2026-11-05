"""Extraction des périodes de stress historiques connues (config.yaml `stress_periods`).

RÈGLE ABSOLUE (même règle que `backtest/validation.py` pour le jeu de test
gelé) : ces périodes sont des jeux de VALIDATION/STRESS uniquement. Jamais
utilisées pour entraîner ou tuner un modèle — sinon la mesure de robustesse
qu'elles sont censées apporter devient sans valeur (le modèle aurait
simplement mémorisé la période au lieu de démontrer une résilience réelle).
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def list_stress_periods(config: dict[str, Any]) -> list[str]:
    """Retourne les noms des périodes de stress disponibles dans config.yaml."""
    periods = config.get("stress_periods", [])
    return [p["name"] for p in periods]


def get_stress_period_data(df: pd.DataFrame, period_name: str, config: dict[str, Any]) -> pd.DataFrame:
    """Extrait la portion de `df` correspondant à la période de stress nommée.

    Lève une erreur explicite si `period_name` ne correspond à aucune entrée
    de `config.yaml` (section `stress_periods`).
    """
    periods = config.get("stress_periods", [])
    matching = [p for p in periods if p["name"] == period_name]

    if not matching:
        available = ", ".join(p["name"] for p in periods) or "(aucune définie)"
        raise ValueError(
            f"Période de stress inconnue: {period_name!r}. Périodes disponibles: {available}"
        )

    period = matching[0]
    start = pd.Timestamp(period["start"], tz="UTC")
    end = pd.Timestamp(period["end"], tz="UTC")

    return df.loc[(df.index >= start) & (df.index <= end)]
