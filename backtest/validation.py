"""Validation walk-forward et protection contre l'overfitting temporel.

RÈGLE ABSOLUE DU PROJET : le jeu de test gelé (`get_frozen_test_set`) et les
périodes de stress (`backtest/stress_tests.py`) ne doivent JAMAIS être
consultés pendant le développement, le tuning d'hyperparamètres, ou
l'entraînement d'un modèle. Ils ne servent qu'à une évaluation finale,
une seule fois, une fois toutes les décisions de conception déjà prises
sur les données walk-forward. Violer cette règle invalide silencieusement
toute mesure de performance produite par le projet.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd


def walk_forward_splits(
    df: pd.DataFrame,
    train_days: int,
    test_days: int,
    step_days: int,
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Découpe chronologique en fenêtres glissantes (train, test) sans chevauchement.

    Le test d'un split commence strictement après la fin du train du même
    split. La fenêtre avance de `step_days` à chaque itération jusqu'à
    épuisement des données disponibles dans `df` (index datetime croissant).
    """
    if df.empty:
        return

    start = df.index.min()
    end = df.index.max()

    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)

    train_start = start
    while True:
        train_end = train_start + train_delta
        test_start = train_end
        test_end = test_start + test_delta

        if test_end > end:
            break

        train_df = df.loc[(df.index >= train_start) & (df.index < train_end)]
        test_df = df.loc[(df.index >= test_start) & (df.index < test_end)]

        if not train_df.empty and not test_df.empty:
            yield train_df, test_df

        train_start = train_start + step_delta


def get_frozen_test_set(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Portion de `df` à partir de `validation.frozen_test_start` (config.yaml).

    JAMAIS utilisée en développement/tuning — voir avertissement en tête
    de fichier.
    """
    validation_cfg = config["validation"] if "validation" in config else config
    frozen_start = pd.Timestamp(validation_cfg["frozen_test_start"], tz="UTC")
    return df.loc[df.index >= frozen_start]


def assert_no_lookahead(df: pd.DataFrame, feature_columns: list[str]) -> None:
    """Vérification heuristique partielle : les colonnes de features glissantes
    doivent avoir leurs NaN en DÉBUT de série (période de warm-up), pas en fin.

    Un pattern inversé (NaN en fin de série alors que le début est rempli)
    est le signe classique d'un rolling/shift mal orienté vers le futur.
    Cette fonction ne prouve PAS l'absence de look-ahead — elle attrape
    seulement une classe d'erreurs grossières et évidentes.
    """
    for col in feature_columns:
        if col not in df.columns:
            raise KeyError(f"Colonne '{col}' absente du DataFrame — impossible de vérifier.")

        series = df[col]
        non_null_mask = series.notna()
        if not non_null_mask.any():
            continue  # colonne entièrement vide : rien à vérifier ici

        first_valid_idx = non_null_mask.idxmax()
        first_valid_pos = df.index.get_loc(first_valid_idx)

        # Heuristique : après le premier point valide, il ne devrait plus y
        # avoir de longue plage de NaN qui réapparaît (signe probable d'un
        # calcul mal aligné plutôt que d'un warm-up normal en début de série).
        tail_after_warmup = series.iloc[first_valid_pos:]
        suspicious_nan_ratio = tail_after_warmup.isna().mean()
        if suspicious_nan_ratio > 0.5:
            raise ValueError(
                f"Colonne '{col}' : plus de 50% de NaN après le warm-up initial "
                f"(premier point valide à l'index {first_valid_idx}). Vérifier "
                f"l'orientation temporelle du calcul (look-ahead potentiel)."
            )
