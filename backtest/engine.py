"""Moteur de backtest — assemble stratégie, portefeuille simulé, risk layer,
coûts réalistes et validation walk-forward.

Aucun résultat produit ici n'a de sens hors du cadre walk-forward : voir
`backtest/validation.py` pour la règle sur le jeu de test gelé.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd

from backtest.validation import walk_forward_splits
from features.indicators import atr as compute_atr
from paper_trading.portfolio import SimulatedPortfolio
from risk.position_sizing import atr_volatility_targeting
from risk.stops import chandelier_exit_long, get_chandelier_params


@dataclass
class BacktestResult:
    """Résultat d'un backtest sur une seule période."""

    value_curve: list[tuple[Any, float]]
    metrics: dict[str, float]
    trades: list[Any]


def run_backtest(
    df: pd.DataFrame,
    strategy: Any,
    initial_cash: float,
    config: dict[str, Any],
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
) -> BacktestResult:
    """Rejoue `df` chronologiquement, applique la décision de `strategy` à
    chaque pas, dimensionne via ATR volatility targeting, exécute au travers
    du portefeuille simulé (slippage + limites d'exposition), vérifie les
    stops Chandelier Exit à chaque pas indépendamment de la stratégie.

    `df` doit déjà contenir les colonnes d'indicateurs nécessaires à la
    stratégie (voir `features.indicators.add_all_indicators`), ainsi que
    open/high/low/close/volume.
    """
    strategy.reset()
    portfolio = SimulatedPortfolio(initial_cash=initial_cash, config=config)

    # `config` est déjà chargé en mémoire par l'appelant : on lit directement
    # la section utile plutôt que de forcer un second accès disque via
    # risk.position_sizing.load_risk_config (qui prend un chemin de fichier).
    risk_section = config["risk"]

    # BUG CORRIGÉ : recalculer l'ATR localement sur `df` (le split walk-forward,
    # potentiellement seulement quelques centaines de lignes) réintroduit un
    # faux warm-up NaN sur les ~14 premières lignes DU SPLIT — alors que l'ATR
    # est déjà valide dès le début du split si calculé sur l'historique complet
    # en amont (features.indicators.add_all_indicators, avant le découpage
    # walk-forward). Conséquence concrète observée : BuyAndHoldStrategy (qui
    # n'achète qu'une seule fois, à la première ligne) tombait systématiquement
    # sur un ATR NaN et n'achetait donc JAMAIS — 0% de rendement à chaque split.
    # On réutilise donc la colonne `atr` déjà présente dans `df` si elle existe
    # (chemin normal : `df` vient de add_all_indicators + walk_forward_splits),
    # avec un recalcul de secours uniquement si elle est absente (usage direct
    # de run_backtest sur un df non enrichi, ex. tests).
    if "atr" in df.columns:
        atr_series = df["atr"]
    else:
        atr_series = compute_atr(df, period=risk_section["atr_period"])
    chandelier_params = get_chandelier_params(timeframe, config)
    stop_series = chandelier_exit_long(
        df, lookback=chandelier_params.lookback, atr_multiplier=chandelier_params.atr_multiplier
    )

    for i, (timestamp, row) in enumerate(df.iterrows()):
        current_price = row["close"]
        current_prices = {symbol: current_price}

        # Vérifie le stop AVANT toute nouvelle décision — filet de sécurité
        # non désapprenable, indépendant de ce que la stratégie déciderait.
        portfolio.apply_stop_loss(symbol, current_price, timestamp)

        action = strategy.decide(row)

        if action == "BUY":
            atr_value = atr_series.iloc[i]
            if atr_value and atr_value > 0:
                quantity = atr_volatility_targeting(
                    capital=portfolio.get_total_value(current_prices),
                    atr_value=atr_value,
                    price=current_price,
                    target_volatility_pct=risk_section["target_volatility_pct"],
                    atr_stop_multiplier=chandelier_params.atr_multiplier,
                )
                volume_24h = row.get("volume", 0.0) * 24  # approximation si le volume est par bougie infra-jour
                volatility = (atr_value / current_price) if current_price > 0 else 0.0
                if quantity > 0:
                    portfolio.execute_trade(
                        symbol=symbol,
                        side="BUY",
                        quantity=quantity,
                        price=current_price,
                        timestamp=timestamp,
                        volume_24h=max(volume_24h, 1e-9),
                        volatility=volatility,
                    )
                    stop_value = stop_series.iloc[i]
                    if stop_value == stop_value:  # not NaN
                        portfolio.set_stop_level(symbol, stop_value)

        elif action == "SELL":
            position = portfolio.positions.get(symbol, 0.0)
            if position > 0:
                volume_24h = row.get("volume", 0.0) * 24
                atr_value = atr_series.iloc[i] or 0.0
                volatility = (atr_value / current_price) if current_price > 0 else 0.0
                portfolio.execute_trade(
                    symbol=symbol,
                    side="SELL",
                    quantity=position,
                    price=current_price,
                    timestamp=timestamp,
                    volume_24h=max(volume_24h, 1e-9),
                    volatility=volatility,
                )

        portfolio.record_value(timestamp, current_prices)

    return BacktestResult(
        value_curve=portfolio.value_history,
        metrics=portfolio.get_metrics(),
        trades=portfolio.trades,
    )


def run_walk_forward_backtest(
    df: pd.DataFrame,
    strategy_factory: Callable[[], Any],
    config: dict[str, Any],
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
) -> dict[str, Any]:
    """Lance `run_backtest` sur chaque split de test walk-forward (jamais sur
    le train — le train n'est utile qu'aux stratégies qui ont besoin d'un
    fit, ex. un futur agent RL ; les baselines actuelles n'en ont pas besoin
    mais on respecte quand même le découpage pour la cohérence future).

    Retourne les métriques agrégées (moyenne + écart-type inter-splits, pour
    voir la stabilité de la stratégie d'une période à l'autre) ainsi que le
    détail par split.
    """
    validation_cfg = config["validation"]
    splits = list(
        walk_forward_splits(
            df,
            train_days=validation_cfg["walk_forward_train_days"],
            test_days=validation_cfg["walk_forward_test_days"],
            step_days=validation_cfg["walk_forward_step_days"],
        )
    )

    if not splits:
        raise ValueError(
            "Aucun split walk-forward généré — le DataFrame est probablement trop "
            "court par rapport à train_days + test_days de config.yaml."
        )

    per_split_results: list[BacktestResult] = []
    for _train_df, test_df in splits:
        strategy = strategy_factory()
        result = run_backtest(
            test_df,
            strategy,
            initial_cash=config["portfolio"]["initial_cash_usdt"],
            config=config,
            symbol=symbol,
            timeframe=timeframe,
        )
        per_split_results.append(result)

    metric_names = per_split_results[0].metrics.keys()
    aggregated: dict[str, float] = {}
    for name in metric_names:
        values = [r.metrics[name] for r in per_split_results]
        mean_val = sum(values) / len(values)
        variance = sum((v - mean_val) ** 2 for v in values) / len(values)
        aggregated[f"{name}_mean"] = mean_val
        aggregated[f"{name}_std"] = variance ** 0.5

    return {
        "aggregated_metrics": aggregated,
        "per_split_results": per_split_results,
        "num_splits": len(splits),
    }


def compare_strategies(
    df: pd.DataFrame,
    strategies: dict[str, Callable[[], Any]],
    config: dict[str, Any],
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
) -> pd.DataFrame:
    """Lance `run_walk_forward_backtest` pour chaque stratégie et retourne un
    DataFrame comparatif (une ligne par stratégie, colonnes = métriques
    agrégées). `strategies` associe un nom lisible à une factory (fonction
    sans argument qui retourne une nouvelle instance de stratégie — nécessaire
    pour repartir d'un état propre à chaque split walk-forward).
    """
    rows: dict[str, dict[str, float]] = {}
    for name, factory in strategies.items():
        wf_result = run_walk_forward_backtest(
            df, factory, config, symbol=symbol, timeframe=timeframe
        )
        rows[name] = wf_result["aggregated_metrics"]

    return pd.DataFrame.from_dict(rows, orient="index")
