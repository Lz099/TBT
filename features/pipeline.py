"""Pipeline d'assemblage central des features.

Ce module est le POINT D'ASSEMBLAGE UNIQUE de toutes les briques de
`features/` (indicateurs techniques, signaux de contexte, régime HMM) en un
seul DataFrame propre, prêt à être consommé par l'environnement Gymnasium
et par l'agent RL. TOUTE nouvelle feature future (nouveau signal de
contexte, nouvel indicateur, nouvelle sortie de modèle) doit transiter par
`build_feature_dataframe` plutôt que d'être branchée éparpillée ailleurs
(env/, models/, app/) : ça garde un seul endroit où auditer la liste
complète des features et leur alignement temporel.

Convention d'index : tous les DataFrames OHLCV manipulés ici (`df_ohlcv`,
`df_eth_ohlcv`, `df_funding`) sont indexés par un DatetimeIndex UTC nommé
`timestamp`, une ligne par bougie fermée — la convention déjà utilisée par
`data/fetch_historical.py` et `data/fetch_derivatives.py`.

Règle anti-look-ahead : chaque étape de merge ci-dessous ne fait que
propager en avant (forward-fill) une valeur déjà publiée/connue au moment
où elle s'applique (Fear & Greed publié une fois par jour à une heure
donnée, funding rate publié toutes les 8h). Aucune étape n'interpole vers
l'arrière ni n'utilise `bfill`.
"""

from __future__ import annotations

import warnings
from typing import Any

import pandas as pd

from features.context_signals import compute_eth_btc_ratio, fetch_fear_greed_index
from features.indicators import add_all_indicators
from features.regime import fit_regime_model, predict_regime_probabilities

# Timeframes pour lesquels le Fear & Greed Index (résolution quotidienne) est
# pertinent comme feature. Voir la règle documentée dans
# features/context_signals.py : jamais de forward-fill artificiel vers le 5m.
_FNG_ELIGIBLE_TIMEFRAMES = frozenset({"1h", "1d"})

# Colonnes de prix/volume bruts : jamais exposées comme features RL (non
# comparables d'une période à l'autre sans normalisation par régime de prix).
_RAW_PRICE_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


def _merge_fear_greed(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Ajoute la colonne `fear_greed_value` par forward-fill jour->timeframe cible.

    N'agit que pour `timeframe` in {"1h", "1d"}. Le F&G Index est publié une
    fois par jour ; on aligne chaque bougie sur la dernière valeur de F&G
    déjà publiée à son timestamp (merge_asof direction="backward" = la
    dernière observation connue au plus tard égale au timestamp courant,
    jamais une observation future).
    """
    if timeframe not in _FNG_ELIGIBLE_TIMEFRAMES:
        return df

    fng = fetch_fear_greed_index()
    if fng.empty:
        out = df.copy()
        out["fear_greed_value"] = pd.NA
        return out

    fng = fng.sort_values("timestamp").rename(columns={"value": "fear_greed_value"})

    left = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    left = left.sort_values("timestamp")

    merged = pd.merge_asof(
        left,
        fng[["timestamp", "fear_greed_value"]],
        on="timestamp",
        direction="backward",
    )
    merged = merged.set_index("timestamp").sort_index()
    return merged


def _merge_funding_rate(df: pd.DataFrame, df_funding: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne `funding_rate` par forward-fill (publication toutes les 8h).

    Utilise merge_asof direction="backward" : chaque bougie récupère le
    dernier funding rate publié à un timestamp <= son propre timestamp.
    Aucune donnée future n'est utilisée.
    """
    funding = df_funding.sort_index().reset_index()
    funding_ts_col = funding.columns[0]
    funding = funding.rename(columns={funding_ts_col: "timestamp"})[["timestamp", "funding_rate"]]

    left = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    left = left.sort_values("timestamp")

    merged = pd.merge_asof(
        left,
        funding,
        on="timestamp",
        direction="backward",
    )
    merged = merged.set_index("timestamp").sort_index()
    return merged


def _merge_eth_btc_ratio(df: pd.DataFrame, df_eth_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la colonne `eth_btc_ratio` (pertinent seulement si `df` = BTC).

    Asymétrie documentée : `compute_eth_btc_ratio` calcule close_ETH/close_BTC.
    Si le symbole principal du pipeline est déjà ETH, ce ratio est
    l'information réciproque (redondante avec le prix ETH lui-même divisé
    par BTC) — elle reste ajoutée telle quelle si l'appelant fournit
    `df_eth_ohlcv`, mais il est de la responsabilité de l'appelant de ne
    passer `df_eth_ohlcv` que lorsque le symbole principal est BTC/USDT
    (cf. docstring de `build_feature_dataframe`).
    """
    df_btc_reset = df.reset_index().rename(columns={df.index.name or "index": "timestamp"})
    df_eth_reset = df_eth_ohlcv.reset_index().rename(
        columns={df_eth_ohlcv.index.name or "index": "timestamp"}
    )

    ratio = compute_eth_btc_ratio(df_eth_reset, df_btc_reset)
    ratio_df = ratio.to_frame()

    out = df.copy()
    out = out.join(ratio_df, how="left")
    return out


def _add_regime_probabilities(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    n_states: int,
    volatility_window: int,
    random_state: int,
) -> pd.DataFrame:
    """Entraîne le HMM sur `train_df` (ou `df` entier si non fourni) puis prédit sur tout `df`."""
    fit_source = train_df if train_df is not None else df
    if train_df is None:
        warnings.warn(
            "build_feature_dataframe: aucun train_df fourni pour fit_regime_model — "
            "le HMM de régime est entraîné sur l'ensemble de df_ohlcv (train+test "
            "confondus). C'est un raccourci acceptable uniquement pour l'exploration "
            "manuelle/prototypage. Pour un entraînement RL réel, appeler ce pipeline "
            "séparément par split walk-forward et passer explicitement `train_df` "
            "(portion train uniquement) afin de ne jamais fitter le HMM sur des "
            "données de test.",
            stacklevel=2,
        )

    model = fit_regime_model(
        fit_source, n_states=n_states, volatility_window=volatility_window, random_state=random_state
    )
    proba_df = predict_regime_probabilities(model, df, volatility_window=volatility_window)

    out = df.copy()
    for col in proba_df.columns:
        out[col] = proba_df[col]
    return out


def build_feature_dataframe(
    df_ohlcv: pd.DataFrame,
    df_funding: pd.DataFrame | None = None,
    df_eth_ohlcv: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
    timeframe: str = "1h",
    train_df: pd.DataFrame | None = None,
    regime_n_states: int = 4,
    regime_volatility_window: int = 20,
    regime_random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Assemble indicateurs + signaux de contexte + probabilités de régime en un seul DataFrame.

    Args:
        df_ohlcv: DataFrame OHLCV du symbole principal, DatetimeIndex UTC
            nommé `timestamp`, colonnes open/high/low/close/volume.
        df_funding: optionnel, DataFrame funding rate déjà chargé depuis le
            Parquet écrit par `data/fetch_derivatives.py` (DatetimeIndex UTC,
            colonne `funding_rate`). Fusionné en forward-fill (publication
            toutes les 8h) : pas de look-ahead, on ne fait que propager une
            valeur déjà connue jusqu'à la prochaine publication.
        df_eth_ohlcv: optionnel, DataFrame OHLCV ETH (même format que
            `df_ohlcv`). Pertinent uniquement si `df_ohlcv` est BTC/USDT —
            voir asymétrie documentée dans `_merge_eth_btc_ratio`. Ajoute la
            colonne `eth_btc_ratio`.
        config: dict de config.yaml (actuellement non utilisé directement ici
            car chaque fonction Vague 1 lit ses propres valeurs par défaut ou
            son propre chemin de config ; conservé au signature pour que
            l'appelant Gymnasium puisse toujours passer sa config sans avoir
            à savoir si elle est utilisée). Réservé pour de futurs paramètres
            (ex: fenêtre HMM configurable via config.yaml).
        timeframe: "5m", "1h" ou "1d". Contrôle uniquement l'éligibilité du
            merge Fear & Greed Index (jamais forward-fill vers le 5m, cf.
            règle documentée dans features/context_signals.py).
        train_df: portion train (walk-forward) sur laquelle fitter le HMM de
            régime. Si None, fit sur `df_ohlcv` entier — raccourci
            d'exploration uniquement, voir avertissement émis à l'exécution.
        regime_n_states, regime_volatility_window, regime_random_state:
            transmis tels quels à `fit_regime_model`/`predict_regime_probabilities`.

    Returns:
        (df_features, feature_columns) où `df_features` contient toutes les
        colonnes (OHLCV brut inclus, conservé pour le moteur de backtest/env
        qui a besoin du prix réel pour exécuter les ordres) et
        `feature_columns` est la liste des colonnes numériques dérivées
        utilisables comme observation RL, EXCLUANT open/high/low/close/volume
        bruts.
    """
    out = add_all_indicators(df_ohlcv)

    out = _merge_fear_greed(out, timeframe)

    if df_funding is not None:
        out = _merge_funding_rate(out, df_funding)

    if df_eth_ohlcv is not None:
        out = _merge_eth_btc_ratio(out, df_eth_ohlcv)

    out = _add_regime_probabilities(
        out,
        train_df,
        n_states=regime_n_states,
        volatility_window=regime_volatility_window,
        random_state=regime_random_state,
    )

    feature_columns = [
        col
        for col in out.columns
        if col not in _RAW_PRICE_COLUMNS and pd.api.types.is_numeric_dtype(out[col])
    ]

    return out, feature_columns


def normalize_features(
    df: pd.DataFrame,
    feature_columns: list[str],
    train_stats: dict[str, tuple[float, float]] | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Normalisation z-score des colonnes `feature_columns`.

    RÈGLE ANTI-LOOK-AHEAD CRITIQUE : si `train_stats` est fourni (calculé au
    préalable sur le train uniquement via un premier appel avec
    `train_stats=None` sur le train), ces statistiques (moyenne, écart-type)
    sont appliquées telles quelles à `df` — que `df` soit le train ou le
    test. On ne recalcule JAMAIS moyenne/écart-type sur train+test combinés,
    ce qui fuiterait de l'information du futur (test) vers la normalisation
    du train.

    Si `train_stats` est None, les statistiques sont calculées sur `df` lui
    -même et retournées (cas d'usage : on calcule les stats sur le train).

    Écart-type nul (colonne constante sur le train) : la colonne normalisée
    est mise à 0.0 partout plutôt que de diviser par zéro.

    Returns:
        (df_normalized, train_stats)
    """
    out = df.copy()
    stats: dict[str, tuple[float, float]] = {}

    for col in feature_columns:
        if train_stats is not None:
            if col not in train_stats:
                raise KeyError(f"Colonne '{col}' absente de train_stats.")
            mean, std = train_stats[col]
        else:
            mean = float(df[col].mean())
            std = float(df[col].std(ddof=0))
            stats[col] = (mean, std)

        if std == 0.0 or pd.isna(std):
            out[col] = 0.0
        else:
            out[col] = (out[col] - mean) / std

    final_stats = train_stats if train_stats is not None else stats
    return out, final_stats
