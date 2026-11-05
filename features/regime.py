"""Détection de régime de marché via HMM Gaussien (hmmlearn).

IMPORTANT — distinction stricte respectée dans ce module : les fonctions
ci-dessous produisent uniquement une FEATURE d'entrée (des probabilités
d'appartenance à chaque état latent) pour l'agent RL entraîné plus tard.
Ce module ne contient et ne doit jamais contenir de règle de décision du
type "si régime == crise alors vendre/réduire l'exposition". L'exposition
des probabilités de régime est descriptive, pas prescriptive ; toute
logique de décision reste la responsabilité exclusive de l'agent RL (env/,
models/).

Anti-look-ahead : les features de régime (rendements log, volatilité
réalisée glissante) utilisent exclusivement des données passées (rolling
avec min_periods, pas de center=True). Le modèle HMM est entraîné avec
`fit_regime_model` sur un DataFrame d'entraînement fourni par l'appelant
(à qui incombe le respect du split walk-forward défini dans config.yaml) ;
ce module ne fait aucun split lui-même.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM


def _regime_features(df: pd.DataFrame, volatility_window: int = 20) -> pd.DataFrame:
    """Construit les features de régime : rendements log + volatilité réalisée glissante (pas le prix brut)."""
    log_return = np.log(df["close"] / df["close"].shift(1))
    realized_vol = log_return.rolling(window=volatility_window, min_periods=volatility_window).std(ddof=0)
    features = pd.DataFrame(
        {"log_return": log_return, "realized_volatility": realized_vol},
        index=df.index,
    )
    return features


def fit_regime_model(
    df: pd.DataFrame,
    n_states: int = 4,
    volatility_window: int = 20,
    random_state: int = 42,
) -> GaussianHMM:
    """Entraîne un GaussianHMM (n_states régimes) sur rendements log + volatilité réalisée glissante de `df`."""
    features = _regime_features(df, volatility_window=volatility_window).dropna()
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="full",
        random_state=random_state,
    )
    model.fit(features.to_numpy())
    return model


def predict_regime_probabilities(
    model: GaussianHMM, df: pd.DataFrame, volatility_window: int = 20
) -> pd.DataFrame:
    """Probabilités posterior par état HMM pour chaque timestamp de `df`.

    Retourne un DataFrame aligné sur df.index avec colonnes
    regime_probability_0 .. regime_probability_{n_states-1}. Les lignes
    initiales sans volatilité glissante disponible (warm-up) ont des NaN :
    à charge de l'appelant de les gérer (dropna ou masquage) comme pour
    tout indicateur avec période de warm-up.
    """
    features = _regime_features(df, volatility_window=volatility_window)
    valid_mask = features.notna().all(axis=1)

    n_states = model.n_components
    columns = [f"regime_probability_{i}" for i in range(n_states)]
    result = pd.DataFrame(np.nan, index=df.index, columns=columns)

    if valid_mask.any():
        posteriors = model.predict_proba(features.loc[valid_mask].to_numpy())
        result.loc[valid_mask, columns] = posteriors

    return result
