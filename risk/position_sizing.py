"""Dimensionnement de position : ATR volatility targeting (méthode principale)
et Kelly fractionnaire (plafond secondaire optionnel).

Séparation stricte avec la décision RL : l'agent décide Hold/Buy/Sell,
cette couche décide COMBIEN. Les fonctions de calcul restent pures
(aucune lecture de fichier) ; seule `load_risk_config` touche le disque.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class RiskConfig:
    """Section `risk` de config.yaml, typée."""

    atr_period: int
    target_volatility_pct: float
    chandelier_atr_multiplier_1d: float
    chandelier_atr_multiplier_intraday: float
    chandelier_lookback_1d: int
    chandelier_lookback_intraday: int
    kelly_fraction_cap: float


def load_risk_config(config_path: str = "config.yaml") -> RiskConfig:
    """Charge la section `risk` de config.yaml — seul point d'accès disque de ce module."""
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    risk_raw = raw["risk"]
    return RiskConfig(
        atr_period=risk_raw["atr_period"],
        target_volatility_pct=risk_raw["target_volatility_pct"],
        chandelier_atr_multiplier_1d=risk_raw["chandelier_atr_multiplier_1d"],
        chandelier_atr_multiplier_intraday=risk_raw["chandelier_atr_multiplier_intraday"],
        chandelier_lookback_1d=risk_raw["chandelier_lookback_1d"],
        chandelier_lookback_intraday=risk_raw["chandelier_lookback_intraday"],
        kelly_fraction_cap=risk_raw["kelly_fraction_cap"],
    )


def atr_volatility_targeting(
    capital: float,
    atr_value: float,
    price: float,
    target_volatility_pct: float,
    atr_stop_multiplier: float,
) -> float:
    """Taille de position (en unités de l'actif) telle que la perte au stop ATR
    représente `target_volatility_pct` du capital.

    position_size = (capital * target_volatility_pct) / (atr_value * atr_stop_multiplier)

    GARDE-FOU : quand `atr_value` est petit par rapport à `price` (fréquent
    sur des bougies courtes comme le 5m, où l'ATR est minuscule comparé au
    prix du BTC), la formule brute peut renvoyer une quantité représentant
    plusieurs fois — voire des centaines de fois — le capital disponible en
    valeur notionnelle, ce qui équivaut à un effet de levier caché et
    involontaire (aucun levier n'est prévu dans ce projet). On plafonne donc
    la position à 100% du capital en valeur (`capital / price`), qui reste
    la limite économique absolue même avant d'appliquer les contraintes
    d'exposition plus fines de `risk/portfolio_constraints.py` en aval.
    """
    if atr_value <= 0:
        raise ValueError("atr_value doit être strictement positif")
    if price <= 0:
        raise ValueError("price doit être strictement positif")

    risk_budget = capital * target_volatility_pct
    stop_distance = atr_value * atr_stop_multiplier
    raw_quantity = risk_budget / stop_distance

    max_quantity_no_leverage = capital / price
    return min(raw_quantity, max_quantity_no_leverage)


def kelly_fraction(
    win_probability: float,
    win_loss_ratio: float,
    fraction_cap: float,
) -> float:
    """Kelly fractionnaire — PLAFOND SECONDAIRE OPTIONNEL, jamais le calcul principal
    de taille de position (cf. plan : atr_volatility_targeting est la méthode
    principale). Utilisation prévue : min(taille_atr, capital_fraction * taille_kelly).

    f = fraction_cap * ((win_loss_ratio * win_probability - (1 - win_probability)) / win_loss_ratio)
    Résultat clippé dans [0, 1] (fraction du capital).
    """
    if win_loss_ratio <= 0:
        raise ValueError("win_loss_ratio doit être strictement positif")

    raw_kelly = (win_loss_ratio * win_probability - (1 - win_probability)) / win_loss_ratio
    f = fraction_cap * raw_kelly
    return min(max(f, 0.0), 1.0)
