"""Contraintes d'exposition au niveau portefeuille (BTC + ETH combinés).

Logique : deux positions fortement corrélées portent un risque redondant
(un même choc de marché les impacte toutes les deux dans le même sens),
donc l'exposition combinée maximale autorisée est réduite quand la
corrélation glissante dépasse un seuil (cf. plan, section portfolio).
"""

from typing import Any, TypedDict

import pandas as pd


def rolling_correlation(
    returns_a: pd.Series,
    returns_b: pd.Series,
    window_days: int,
) -> pd.Series:
    """Corrélation glissante entre deux séries de rendements (ex. BTC et ETH).

    Fenêtre rolling standard (pas de centre=True) : au temps t, ne dépend
    que des rendements <= t, donc pas de look-ahead.
    """
    return returns_a.rolling(window=window_days, min_periods=window_days).corr(returns_b)


def max_combined_exposure(
    correlation: float,
    base_max_pct: float,
    high_correlation_threshold: float,
    reduced_max_pct: float,
) -> float:
    """Plafond d'exposition combinée BTC+ETH, dépendant de la corrélation courante.

    Si `correlation` dépasse `high_correlation_threshold`, les deux actifs
    bougent essentiellement ensemble : le plafond combiné est réduit à
    `reduced_max_pct` au lieu de `base_max_pct`, pour éviter qu'une
    exposition "diversifiée" en apparence ne soit en réalité un seul pari
    concentré.
    """
    if correlation > high_correlation_threshold:
        return reduced_max_pct
    return base_max_pct


class ExposureCheckResult(TypedDict):
    within_limits: bool
    btc_within_limit: bool
    eth_within_limit: bool
    combined_within_limit: bool
    applicable_combined_max_pct: float
    btc_excess_pct: float
    eth_excess_pct: float
    combined_excess_pct: float


def check_exposure_limits(
    current_btc_exposure_pct: float,
    current_eth_exposure_pct: float,
    correlation: float,
    config: dict[str, Any],
) -> ExposureCheckResult:
    """Valide les expositions courantes contre les limites de config.yaml (section `portfolio`).

    Appelée par paper_trading/portfolio.py avant chaque exécution de trade.
    Retourne si les limites sont respectées et, sinon, de combien il
    faudrait réduire chaque exposition (en points de %) pour repasser
    sous les plafonds.
    """
    portfolio_cfg = config["portfolio"] if "portfolio" in config else config

    max_per_asset_pct = portfolio_cfg["max_exposure_per_asset_pct"]
    base_max_combined_pct = portfolio_cfg["max_combined_exposure_pct"]
    high_corr_threshold = portfolio_cfg["high_correlation_threshold"]
    # Pas de clé dédiée pour le plafond combiné réduit dans config.yaml :
    # le plan ne définit qu'un seuil de corrélation + un plafond combiné de base.
    # On applique une réduction proportionnelle à l'excès de corrélation au-delà
    # du seuil, bornée par base_max_combined_pct, plutôt que d'inventer une
    # nouvelle clé de config non spécifiée.
    reduced_max_combined_pct = base_max_combined_pct

    applicable_combined_max_pct = max_combined_exposure(
        correlation=correlation,
        base_max_pct=base_max_combined_pct,
        high_correlation_threshold=high_corr_threshold,
        reduced_max_pct=reduced_max_combined_pct,
    )

    btc_within_limit = current_btc_exposure_pct <= max_per_asset_pct
    eth_within_limit = current_eth_exposure_pct <= max_per_asset_pct

    combined_exposure_pct = current_btc_exposure_pct + current_eth_exposure_pct
    combined_within_limit = combined_exposure_pct <= applicable_combined_max_pct

    btc_excess_pct = max(0.0, current_btc_exposure_pct - max_per_asset_pct)
    eth_excess_pct = max(0.0, current_eth_exposure_pct - max_per_asset_pct)
    combined_excess_pct = max(0.0, combined_exposure_pct - applicable_combined_max_pct)

    return ExposureCheckResult(
        within_limits=btc_within_limit and eth_within_limit and combined_within_limit,
        btc_within_limit=btc_within_limit,
        eth_within_limit=eth_within_limit,
        combined_within_limit=combined_within_limit,
        applicable_combined_max_pct=applicable_combined_max_pct,
        btc_excess_pct=btc_excess_pct,
        eth_excess_pct=eth_excess_pct,
        combined_excess_pct=combined_excess_pct,
    )
