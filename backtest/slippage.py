"""Modèle de coûts d'exécution réaliste : slippage dépendant du volume/volatilité
(square-root market impact law) + frais taker fixes.

Sans ce module, un backtest biaise structurellement en faveur des stratégies
qui tradent souvent (elles paieraient un coût irréaliste de quasi zéro),
ce qui invaliderait toute comparaison contre buy-and-hold.
"""

from __future__ import annotations

from typing import Any, Literal


def square_root_slippage(
    order_quantity: float,
    volume_24h: float,
    volatility: float,
    impact_coefficient: float,
) -> float:
    """Impact de marché en fraction du prix : I(Q) = Y * sigma * sqrt(Q / V).

    Y = impact_coefficient, sigma = volatility (ex. écart-type des rendements
    récents), Q = order_quantity (unités de l'actif), V = volume_24h.
    Retourne une fraction du prix (à multiplier par le prix pour obtenir un
    coût en valeur), toujours >= 0.
    """
    if volume_24h <= 0:
        raise ValueError("volume_24h doit être strictement positif")
    if order_quantity < 0:
        raise ValueError("order_quantity ne peut pas être négatif")

    return impact_coefficient * volatility * (order_quantity / volume_24h) ** 0.5


def apply_trading_costs(
    price: float,
    quantity: float,
    side: Literal["BUY", "SELL"],
    volume_24h: float,
    volatility: float,
    config: dict[str, Any],
) -> float:
    """Retourne le prix d'exécution effectif après slippage + frais taker.

    Le slippage dégrade toujours le prix pour le trader (plus cher à l'achat,
    moins cher à la vente) — cohérent avec un carnet d'ordres qui s'épuise
    dans le sens du trade. Les frais taker (config.yaml `costs.taker_fee_pct`)
    s'appliquent en plus, dans le même sens défavorable.
    """
    costs_cfg = config["costs"] if "costs" in config else config
    impact_coefficient = costs_cfg["slippage_impact_coefficient"]
    taker_fee_pct = costs_cfg["taker_fee_pct"]

    impact_fraction = square_root_slippage(
        order_quantity=quantity,
        volume_24h=volume_24h,
        volatility=volatility,
        impact_coefficient=impact_coefficient,
    )

    total_cost_fraction = impact_fraction + taker_fee_pct

    if side == "BUY":
        return price * (1.0 + total_cost_fraction)
    if side == "SELL":
        return price * (1.0 - total_cost_fraction)
    raise ValueError(f"side doit être 'BUY' ou 'SELL', reçu: {side!r}")
