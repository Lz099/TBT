"""Boucle de PAPER TRADING EN TEMPS REEL — lecture de marché public + simulation.

===========================================================================
LIMITATIONS — A LIRE AVANT UTILISATION
===========================================================================
- Ce script n'est PAS un service de production. C'est une boucle Python
  manuelle (while True + pause), lancée à la main dans un terminal séparé
  du dashboard (`python paper_trading/live_runner.py ...`).
- AUCUNE reprise automatique après crash ou redémarrage du PC : si le
  process meurt ou si la machine s'éteint, le portefeuille en mémoire est
  perdu et rien ne relance la boucle tout seul. Le dernier état connu reste
  visible dans storage/runs/paper_trading.db (lu par le dashboard), mais la
  simulation elle-même doit être relancée manuellement (elle repart alors
  avec un portefeuille neuf — pas de sérialisation du portefeuille entre
  deux lancements dans cette V1).
- AUCUNE garantie de fonctionnement 24/7 : pas de watchdog, pas de
  supervision, pas de retry persistant au-delà de ce qui est codé ici.
- Conçu pour un usage expérimental/personnel : lancer, observer la console
  et/ou le dashboard, arrêter avec Ctrl+C. C'est volontairement simple
  (pas de Celery, pas de cron système) — cf. décisions actées du projet.
===========================================================================
RAPPEL SECURITE — LE PLUS IMPORTANT
===========================================================================
Ce fichier ne passe et ne passera JAMAIS d'ordre réel. Il n'appelle aucun
endpoint d'exécution d'ordre ccxt (pas de `exchange.create_order`, pas de
`createOrder`, pas de clé API secrète/privée nulle part). Toute "exécution"
de trade passe EXCLUSIVEMENT par `paper_trading.portfolio.SimulatedPortfolio`,
100% en mémoire. La boucle ci-dessous ne fait que LIRE des données de marché
publiques via `data.live_feed` (endpoints publics ccxt : klines, order book)
et SIMULER des décisions dessus. Si tu modifies ce fichier, ne casse jamais
cette invariante.
===========================================================================

Anti-look-ahead : à chaque itération, seules des bougies déjà CLÔTURÉES sont
utilisées (data.live_feed.fetch_latest_ohlcv ne retourne que des bougies
fermées, cf. sa docstring). La décision de la stratégie au temps t ne porte
que sur la dernière ligne disponible du DataFrame de features, qui ne dépend
elle-même que de bougies <= t (features/pipeline.py, risk/stops.py).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from data.live_feed import fetch_latest_ohlcv
from features.pipeline import build_feature_dataframe, normalize_features
from models.baseline_rules import EmaCrossoverStrategy
from models.buy_and_hold import BuyAndHoldStrategy
from models.momentum_naive import MomentumNaiveStrategy
from models.random_policy import RandomWeightedPolicy
from models.rl_agent import RLAgentStrategy, load_agent, load_agent_metadata
from paper_trading.portfolio import SimulatedPortfolio
from risk.position_sizing import atr_volatility_targeting
from risk.stops import chandelier_exit_long, get_chandelier_params
from storage.runs.live_state import DEFAULT_DB_PATH, init_db, record_decision, record_snapshot

# Nombre de bougies récupérées à chaque itération : assez pour couvrir le
# warm-up des indicateurs (EMA26, régime HMM, etc.) + la fenêtre glissante de
# l'agent RL (window_size, typiquement 20) avec de la marge.
_FETCH_LIMIT = 200

# Durée approximative (en secondes) d'un timeframe, utilisée pour rythmer la
# boucle. Pas besoin d'une synchronisation parfaite à la seconde près pour ce
# projet expérimental (cf. _sleep_until_next_candle).
_TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
}

_STRATEGY_CHOICES = ["buy_and_hold", "ema_crossover", "momentum", "random", "rl"]


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Charge config.yaml — même convention que les autres modules du projet."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_rl_model_path(symbol: str, timeframe: str) -> Path:
    """Convention de nommage partagée avec models/train_pipeline.py."""
    return Path("storage/models") / f"ppo_{symbol.replace('/', '-')}_{timeframe}"


def build_strategy(
    strategy_name: str,
    symbol: str,
    timeframe: str,
    rl_model_path: str | None = None,
) -> tuple[Any, list[str] | None, int | None]:
    """Construit l'objet stratégie demandé (interface commune decide()/reset()).

    Retourne (strategy, feature_columns, window_size). `feature_columns` et
    `window_size` ne sont non-None que pour strategy_name == "rl" (nécessaires
    pour reconstruire les observations normalisées attendues par le modèle) ;
    les autres stratégies lisent directement les colonnes dont elles ont
    besoin sur `observation` (ex. ema_12/ema_26, momentum_10).
    """
    if strategy_name == "buy_and_hold":
        return BuyAndHoldStrategy(), None, None
    if strategy_name == "ema_crossover":
        return EmaCrossoverStrategy(), None, None
    if strategy_name == "momentum":
        return MomentumNaiveStrategy(), None, None
    if strategy_name == "random":
        return RandomWeightedPolicy(), None, None
    if strategy_name == "rl":
        model_path = Path(rl_model_path) if rl_model_path else _default_rl_model_path(symbol, timeframe)
        metadata = load_agent_metadata(model_path)
        if metadata is None:
            raise FileNotFoundError(
                f"Aucun sidecar JSON de métadonnées trouvé pour le modèle RL '{model_path}' "
                f"(feature_columns/window_size introuvables). Entraîne d'abord un agent via "
                f"models/train_pipeline.py, qui sauvegarde le .zip + le .json à côté."
            )
        agent = load_agent(model_path)
        feature_columns = metadata["feature_columns"]
        window_size = metadata["window_size"]
        strategy = RLAgentStrategy(agent, feature_columns=feature_columns, window_size=window_size)
        return strategy, feature_columns, window_size

    raise ValueError(f"strategy inconnue: {strategy_name!r} — attendu un de {_STRATEGY_CHOICES}")


def _load_train_stats(symbol: str, timeframe: str) -> dict[str, tuple[float, float]] | None:
    """Charge les statistiques de normalisation (moyenne, écart-type) sauvegardées
    au dernier entraînement du modèle RL pour ce couple (symbole, timeframe), si
    disponibles.

    CHOIX DOCUMENTE (important pour la cohérence anti-look-ahead) : en mode
    live, il n'y a pas de split walk-forward actif — on ne peut pas "calculer
    les stats sur le train" puisqu'il n'y a pas de train ici, seulement un flux
    continu de bougies récentes. Les statistiques utilisées pour normaliser les
    features live DOIVENT donc venir du dernier entraînement du modèle
    (mêmes stats que celles vues par le modèle pendant l'entraînement),
    jamais recalculées sur la fenêtre live elle-même (ce qui introduirait un
    décalage de distribution avec ce que le modèle a appris, et techniquement
    un calcul de moyenne/écart-type sur une fenêtre glissante qui inclut des
    données "futures" relativement au tout début de cette même fenêtre).

    Convention de chemin : `storage/models/ppo_{SYMBOL}_{timeframe}_train_stats.json`,
    à côté du .zip/.json déjà sauvegardés par models/rl_agent.save_agent.

    ATTENTION — ACTION REQUISE AILLEURS (documentée aussi dans le résumé de
    cette tâche) : au moment où ce fichier est écrit, `models/train_pipeline.py`
    NE SAUVEGARDE PAS ENCORE ce fichier de train_stats. Ce module sait le LIRE
    s'il existe, mais ne le génère jamais lui-même (il n'a pas accès au train
    split original). Tant que train_pipeline.py n'est pas mis à jour pour
    l'écrire, cette fonction retourne None et le live runner normalise alors
    les features sur leur propre fenêtre récente (fallback documenté ci-dessous
    dans build_live_features), un raccourci acceptable seulement pour
    l'expérimentation.
    """
    path = Path("storage/models") / f"ppo_{symbol.replace('/', '-')}_{timeframe}_train_stats.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {col: (float(mean), float(std)) for col, (mean, std) in raw.items()}


def build_live_features(
    df_ohlcv: pd.DataFrame,
    config: dict[str, Any],
    timeframe: str,
    train_stats: dict[str, tuple[float, float]] | None,
) -> tuple[pd.DataFrame, list[str]]:
    """Construit les features sur la fenêtre live courante et les normalise.

    Pas de `train_df` explicite ici (documenté dans le docstring du module) :
    en mode live il n'y a pas de split walk-forward actif, donc le HMM de
    régime est fit sur la fenêtre live elle-même (`build_feature_dataframe`
    émet un avertissement dans ce cas, attendu et accepté ici — raccourci
    d'exploration, cf. sa docstring).

    Normalisation : si `train_stats` est disponible (dernier entraînement du
    modèle RL), on l'applique telle quelle — c'est le chemin correct anti-
    look-ahead. Sinon (aucun sidecar de stats trouvé), on retombe sur une
    normalisation calculée sur la fenêtre live elle-même ; ce n'est PAS
    équivalent à un train_stats figé et peut légèrement dériver dans le
    temps, mais reste anti-look-ahead au sens strict (aucune donnée future
    utilisée) — acceptable en attendant que train_pipeline.py sauvegarde ses
    propres train_stats (cf. docstring de `_load_train_stats`).
    """
    df_features, feature_columns = build_feature_dataframe(
        df_ohlcv,
        config=config,
        timeframe=timeframe,
    )
    df_norm, _ = normalize_features(df_features, feature_columns, train_stats=train_stats)
    return df_norm, feature_columns


def _extract_reasoning(row: pd.Series) -> dict[str, Any]:
    """Résumé compact des indicateurs clés au moment d'une décision, pour audit a posteriori."""
    reasoning: dict[str, Any] = {}
    for col in ("rsi", "macd_histogram", "momentum_10", "atr"):
        if col in row.index and pd.notna(row[col]):
            reasoning[col] = float(row[col])

    regime_cols = [c for c in row.index if c.startswith("regime_probability_")]
    if regime_cols:
        probs = {c: float(row[c]) for c in regime_cols if pd.notna(row[c])}
        if probs:
            reasoning["dominant_regime"] = max(probs, key=probs.get)
            reasoning["regime_probabilities"] = probs
    return reasoning


def _sleep_until_next_candle(timeframe: str, iteration_started_at: float) -> None:
    """Attend approximativement jusqu'à la prochaine bougie du `timeframe` donné.

    Pas de synchronisation parfaite à la seconde près (acceptable pour ce
    projet expérimental) : on dort simplement (durée du timeframe - temps déjà
    écoulé dans l'itération courante), avec un plancher pour ne jamais dormir
    une durée négative ou nulle si l'itération a été anormalement lente.
    """
    period_seconds = _TIMEFRAME_SECONDS.get(timeframe, 60)
    elapsed = time.monotonic() - iteration_started_at
    sleep_seconds = max(period_seconds - elapsed, 5.0)
    time.sleep(sleep_seconds)


def run_live_loop(
    symbol: str,
    timeframe: str,
    strategy_name: str,
    config_path: str = "config.yaml",
    rl_model_path: str | None = None,
    db_path: str = str(DEFAULT_DB_PATH),
    max_iterations: int | None = None,
) -> None:
    """Boucle principale de paper trading live. Tourne indéfiniment (Ctrl+C pour arrêter),
    sauf si `max_iterations` est fourni (utile pour des tests/démonstrations courtes).
    """
    config = load_config(config_path)
    init_db(db_path)

    strategy, feature_columns, window_size = build_strategy(
        strategy_name, symbol, timeframe, rl_model_path
    )
    train_stats = _load_train_stats(symbol, timeframe) if strategy_name == "rl" else None
    if strategy_name == "rl" and train_stats is None:
        print(
            "[live_runner] AVERTISSEMENT: aucun train_stats sauvegardé trouvé pour ce modèle RL — "
            "normalisation calculée sur la fenêtre live elle-même (cf. docstring "
            "_load_train_stats). Résultats du modèle potentiellement moins fiables."
        )

    # Portefeuille simulé créé UNE SEULE FOIS pour toute la durée du process —
    # ne jamais le recréer dans la boucle, sinon l'historique de position
    # (cash, positions ouvertes, stops) serait perdu à chaque itération.
    initial_cash = config["portfolio"]["initial_cash_usdt"]
    portfolio = SimulatedPortfolio(initial_cash=initial_cash, config=config)

    chandelier_params = get_chandelier_params(timeframe, config)
    risk_cfg = config["risk"]

    print(
        f"[live_runner] démarrage — symbol={symbol} timeframe={timeframe} "
        f"strategy={strategy_name} cash_initial={initial_cash:.2f} USDT"
    )
    print(f"[live_runner] état écrit dans: {db_path} (lu par app/dashboard.py, onglet 'Portefeuille simulé')")

    iteration = 0
    try:
        while max_iterations is None or iteration < max_iterations:
            iteration_started_at = time.monotonic()
            now = datetime.now(timezone.utc)

            try:
                _run_single_iteration(
                    symbol=symbol,
                    timeframe=timeframe,
                    strategy=strategy,
                    portfolio=portfolio,
                    config=config,
                    train_stats=train_stats,
                    chandelier_lookback=chandelier_params.lookback,
                    chandelier_atr_multiplier=chandelier_params.atr_multiplier,
                    target_volatility_pct=risk_cfg["target_volatility_pct"],
                    db_path=db_path,
                    now=now,
                )
            except Exception as exc:  # noqa: BLE001 — la boucle doit survivre à une itération ratée
                print(f"[live_runner] {now.isoformat()} ERREUR pendant l'itération: {exc!r} — on continue.")

            iteration += 1
            if max_iterations is None or iteration < max_iterations:
                _sleep_until_next_candle(timeframe, iteration_started_at)
    except KeyboardInterrupt:
        print("\n[live_runner] interruption clavier (Ctrl+C) reçue — arrêt propre de la boucle.")
        sys.exit(0)


def _run_single_iteration(
    symbol: str,
    timeframe: str,
    strategy: Any,
    portfolio: SimulatedPortfolio,
    config: dict[str, Any],
    train_stats: dict[str, tuple[float, float]] | None,
    chandelier_lookback: int,
    chandelier_atr_multiplier: float,
    target_volatility_pct: float,
    db_path: str,
    now: datetime,
) -> None:
    """Un tick complet : fetch -> features -> décision -> exécution -> stop -> persistance."""
    df_ohlcv = fetch_latest_ohlcv(symbol, timeframe, limit=_FETCH_LIMIT)
    if df_ohlcv.empty:
        print(f"[live_runner] {now.isoformat()} aucune donnée reçue, on saute cette itération.")
        return

    df_features, _ = build_live_features(df_ohlcv, config, timeframe, train_stats)
    df_features = df_features.dropna(subset=["atr"])
    if df_features.empty:
        print(f"[live_runner] {now.isoformat()} pas assez d'historique pour un ATR valide, on saute.")
        return

    latest_row = df_features.iloc[-1]
    candle_timestamp = df_features.index[-1]
    current_price = float(latest_row["close"])
    current_prices = {symbol: current_price}

    # Filet de sécurité indépendant de la décision de la stratégie, vérifié à
    # chaque itération avant toute nouvelle décision (même invariant que
    # TradingEnv.step()).
    stop_triggered = portfolio.apply_stop_loss(symbol, current_price, candle_timestamp)
    if stop_triggered:
        print(f"[live_runner] {now.isoformat()} STOP CHANDELIER déclenché sur {symbol} @ {current_price:.2f}")
        record_decision(db_path, candle_timestamp, symbol, "SELL", current_price, {"reason": "chandelier_stop"})

    action = strategy.decide(latest_row)

    if action == "BUY":
        atr_value = float(latest_row["atr"])
        if atr_value > 0:
            volume_24h = max(float(latest_row.get("volume", 0.0)) * 24, 1e-9)
            volatility = atr_value / current_price if current_price > 0 else 0.0
            quantity = atr_volatility_targeting(
                capital=portfolio.get_total_value(current_prices),
                atr_value=atr_value,
                price=current_price,
                target_volatility_pct=target_volatility_pct,
                atr_stop_multiplier=chandelier_atr_multiplier,
            )
            if quantity > 0:
                portfolio.execute_trade(
                    symbol=symbol,
                    side="BUY",
                    quantity=quantity,
                    price=current_price,
                    timestamp=candle_timestamp,
                    volume_24h=volume_24h,
                    volatility=volatility,
                )
                stop_series = chandelier_exit_long(
                    df_features, lookback=chandelier_lookback, atr_multiplier=chandelier_atr_multiplier
                )
                stop_value = stop_series.iloc[-1]
                if pd.notna(stop_value):
                    portfolio.set_stop_level(symbol, float(stop_value))
        record_decision(db_path, candle_timestamp, symbol, "BUY", current_price, _extract_reasoning(latest_row))

    elif action == "SELL":
        position = portfolio.positions.get(symbol, 0.0)
        if position > 0:
            atr_value = float(latest_row["atr"])
            volume_24h = max(float(latest_row.get("volume", 0.0)) * 24, 1e-9)
            volatility = atr_value / current_price if current_price > 0 else 0.0
            portfolio.execute_trade(
                symbol=symbol,
                side="SELL",
                quantity=position,
                price=current_price,
                timestamp=candle_timestamp,
                volume_24h=volume_24h,
                volatility=volatility,
            )
        record_decision(db_path, candle_timestamp, symbol, "SELL", current_price, _extract_reasoning(latest_row))

    total_value = portfolio.get_total_value(current_prices)
    portfolio.record_value(candle_timestamp, current_prices)

    peak_value = max((v for _, v in portfolio.value_history), default=total_value)
    drawdown_pct = ((peak_value - total_value) / peak_value * 100.0) if peak_value > 0 else 0.0

    record_snapshot(
        db_path,
        candle_timestamp,
        total_value=total_value,
        cash=portfolio.cash,
        positions=dict(portfolio.positions),
        drawdown_pct=drawdown_pct,
    )

    print(
        f"[live_runner] {now.isoformat()} bougie={candle_timestamp} action={action} "
        f"prix={current_price:.2f} valeur_portefeuille={total_value:.2f} USDT "
        f"drawdown={drawdown_pct:.2f}%"
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Boucle de paper trading live (PAPER TRADING UNIQUEMENT, aucun ordre réel). "
            "A lancer manuellement dans un terminal séparé du dashboard."
        )
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole à trader (défaut: BTC/USDT).")
    parser.add_argument("--timeframe", default="1h", help="Timeframe (défaut: 1h).")
    parser.add_argument(
        "--strategy",
        default="buy_and_hold",
        choices=_STRATEGY_CHOICES,
        help="Stratégie à utiliser (défaut: buy_and_hold).",
    )
    parser.add_argument(
        "--rl-model-path",
        default=None,
        help=(
            "Chemin du modèle RL sauvegardé (sans extension), ignoré sauf si --strategy rl. "
            "Défaut: storage/models/ppo_{SYMBOL}_{timeframe}."
        ),
    )
    parser.add_argument("--config", default="config.yaml", help="Chemin de config.yaml.")
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_DB_PATH),
        help=f"Chemin du fichier SQLite d'état (défaut: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Nombre d'itérations avant arrêt automatique (défaut: illimité, boucle infinie).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    run_live_loop(
        symbol=args.symbol,
        timeframe=args.timeframe,
        strategy_name=args.strategy,
        config_path=args.config,
        rl_model_path=args.rl_model_path,
        db_path=args.db_path,
        max_iterations=args.max_iterations,
    )
