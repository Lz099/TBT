"""Environnement Gymnasium pour l'agent PPO — interface standard entre les
données de marché (déjà enrichies/normalisées par features/pipeline.py, en
amont, hors de ce module) et le futur agent.

Découplage volontaire : ce module ne construit ni ne normalise aucune
feature. Il reçoit un DataFrame déjà prêt (`df`) et une liste de noms de
colonnes (`feature_columns`) à extraire pour former l'observation. L'appelant
(un futur script d'entraînement) est responsable d'assembler
features.pipeline.build_feature_dataframe(...) + normalize_features(...) et
de passer le résultat ici.

Action space : Discrete(3) — 0=HOLD, 1=BUY, 2=SELL. Pas de sizing continu
dans l'action de l'agent (décision actée) : le dimensionnement de position
reste délégué à risk.position_sizing.atr_volatility_targeting, appelé en
interne par l'environnement à chaque BUY, jamais exposé à l'agent.

Anti-look-ahead : au pas t, l'observation ne porte que sur les lignes
d'index <= t de `df` (fenêtre glissante se terminant à t inclus, jamais
au-delà). Le stop Chandelier Exit et le dimensionnement ATR utilisent la
colonne "atr" au pas t, qui par construction (features.indicators.atr) ne
dépend que de bougies déjà closes à t. La récompense et le calcul de
drawdown au pas t utilisent la valeur de portefeuille V_t obtenue après
exécution du trade décidé au pas t, jamais une valeur future.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from paper_trading.portfolio import SimulatedPortfolio
from risk.position_sizing import atr_volatility_targeting
from risk.stops import chandelier_exit_long, get_chandelier_params

_ACTION_HOLD = 0
_ACTION_BUY = 1
_ACTION_SELL = 2
_ACTION_NAMES = {_ACTION_HOLD: "HOLD", _ACTION_BUY: "BUY", _ACTION_SELL: "SELL"}


class TradingEnv(gym.Env):
    """Environnement de trading discret (Hold/Buy/Sell) sur un seul symbole.

    L'observation est une fenêtre glissante des `window_size` derniers pas de
    temps sur `feature_columns` (features déjà normalisées en amont). Le
    dimensionnement de position et le stop-loss adaptatif sont gérés en
    interne (couches risk/ et paper_trading/ déjà écrites), jamais par
    l'agent lui-même.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        config: dict[str, Any],
        symbol: str = "BTC/USDT",
        timeframe: str = "1h",
        initial_cash: float | None = None,
        window_size: int = 20,
    ) -> None:
        super().__init__()

        required_columns = {"open", "high", "low", "close", "volume", "atr"}
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(
                f"df doit contenir les colonnes {sorted(required_columns)} "
                f"(atr provient de features.indicators.add_all_indicators) — "
                f"colonnes manquantes: {sorted(missing)}"
            )
        missing_features = set(feature_columns) - set(df.columns)
        if missing_features:
            raise ValueError(
                f"feature_columns contient des colonnes absentes de df: {sorted(missing_features)}"
            )
        if window_size < 1:
            raise ValueError("window_size doit être >= 1")

        self._df = df.reset_index(drop=False)
        # Conserve le nom de la colonne d'index d'origine (souvent "timestamp")
        # pour l'utiliser comme timestamp de trade sans dépendre d'un nom précis.
        self._timestamp_col = self._df.columns[0]

        self._feature_columns = list(feature_columns)
        self._config = config
        self._symbol = symbol
        self._timeframe = timeframe
        self._initial_cash = (
            initial_cash if initial_cash is not None else config["portfolio"]["initial_cash_usdt"]
        )
        self._window_size = window_size

        risk_cfg = config["risk"]
        self._atr_stop_multiplier = get_chandelier_params(timeframe, config).atr_multiplier
        self._target_volatility_pct = risk_cfg["target_volatility_pct"]
        self._cost_penalty_lambda = config["reward"]["cost_penalty_lambda"]
        self._drawdown_penalty_lambda = config["reward"]["drawdown_penalty_lambda"]

        chandelier_params = get_chandelier_params(timeframe, config)
        self._chandelier_stop_series = chandelier_exit_long(
            self._df,
            lookback=chandelier_params.lookback,
            atr_multiplier=chandelier_params.atr_multiplier,
        )

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, len(self._feature_columns)),
            dtype=np.float32,
        )

        self._portfolio: SimulatedPortfolio | None = None
        self._current_step: int = 0
        self._last_portfolio_value: float = self._initial_cash
        self._peak_value: float = self._initial_cash
        self._last_drawdown_pct: float = 0.0

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Réinitialise l'environnement au tout début de `df`.

        Mode simple pour cette version : démarrage toujours au premier pas
        disponible (pas de départ aléatoire). `seed`/`options` sont acceptés
        pour respecter l'API Gymnasium mais ne modifient pas le point de
        départ ici.
        """
        super().reset(seed=seed)

        self._portfolio = SimulatedPortfolio(initial_cash=self._initial_cash, config=self._config)
        self._current_step = 0
        self._last_portfolio_value = self._initial_cash
        self._peak_value = self._initial_cash
        self._last_drawdown_pct = 0.0

        observation = self._get_observation()
        info: dict[str, Any] = {
            "portfolio_value": self._last_portfolio_value,
            "action_taken": None,
            "drawdown_pct": self._last_drawdown_pct,
        }
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Applique `action` au pas courant, avance d'un pas, retourne la transition."""
        if self._portfolio is None:
            raise RuntimeError("step() appelé avant reset()")

        action_name = _ACTION_NAMES.get(int(action))
        if action_name is None:
            raise ValueError(f"action invalide: {action!r} — attendu 0 (HOLD), 1 (BUY) ou 2 (SELL)")

        row = self._df.iloc[self._current_step]
        timestamp = row[self._timestamp_col]
        current_price = float(row["close"])
        current_prices = {self._symbol: current_price}

        # Filet de sécurité indépendant de l'action de l'agent, appliqué à
        # chaque pas avant toute nouvelle décision (cf. règle du projet).
        self._portfolio.apply_stop_loss(self._symbol, current_price, timestamp)

        value_before_trade = self._portfolio.get_total_value(current_prices)
        step_cost = 0.0

        if action_name == "BUY":
            atr_value = float(row["atr"]) if row["atr"] == row["atr"] else 0.0  # NaN check
            if atr_value > 0:
                volume_24h = max(float(row.get("volume", 0.0)) * 24, 1e-9)
                volatility = atr_value / current_price if current_price > 0 else 0.0
                quantity = atr_volatility_targeting(
                    capital=value_before_trade,
                    atr_value=atr_value,
                    price=current_price,
                    target_volatility_pct=self._target_volatility_pct,
                    atr_stop_multiplier=self._atr_stop_multiplier,
                )
                if quantity > 0:
                    trade = self._portfolio.execute_trade(
                        symbol=self._symbol,
                        side="BUY",
                        quantity=quantity,
                        price=current_price,
                        timestamp=timestamp,
                        volume_24h=volume_24h,
                        volatility=volatility,
                    )
                    step_cost = abs(trade.execution_price - current_price) * trade.quantity
                    stop_value = self._chandelier_stop_series.iloc[self._current_step]
                    if stop_value == stop_value:  # not NaN
                        self._portfolio.set_stop_level(self._symbol, float(stop_value))

        elif action_name == "SELL":
            position = self._portfolio.positions.get(self._symbol, 0.0)
            if position > 0:
                atr_value = float(row["atr"]) if row["atr"] == row["atr"] else 0.0
                volume_24h = max(float(row.get("volume", 0.0)) * 24, 1e-9)
                volatility = atr_value / current_price if current_price > 0 else 0.0
                trade = self._portfolio.execute_trade(
                    symbol=self._symbol,
                    side="SELL",
                    quantity=position,
                    price=current_price,
                    timestamp=timestamp,
                    volume_24h=volume_24h,
                    volatility=volatility,
                )
                step_cost = abs(trade.execution_price - current_price) * trade.quantity

        self._portfolio.record_value(timestamp, current_prices)
        value_after_trade = self._portfolio.get_total_value(current_prices)

        self._peak_value = max(self._peak_value, value_after_trade)
        drawdown_pct = (
            (self._peak_value - value_after_trade) / self._peak_value if self._peak_value > 0 else 0.0
        )

        reward = self._compute_reward(
            value_before_step=self._last_portfolio_value,
            value_after_step=value_after_trade,
            step_cost=step_cost,
            drawdown_pct=drawdown_pct,
        )

        self._last_portfolio_value = value_after_trade
        self._last_drawdown_pct = drawdown_pct

        self._current_step += 1
        terminated = self._current_step >= len(self._df)
        truncated = False

        observation = self._get_observation()
        info: dict[str, Any] = {
            "portfolio_value": value_after_trade,
            "action_taken": action_name,
            "drawdown_pct": drawdown_pct,
        }
        return observation, reward, terminated, truncated, info

    def _compute_reward(
        self,
        value_before_step: float,
        value_after_step: float,
        step_cost: float,
        drawdown_pct: float,
    ) -> float:
        """r_t = log(V_t / V_t-1) - lambda_c * coûts_t - lambda_dd * max(0, DD_t - DD_t-1).

        `step_cost` est un coût en valeur (devise), converti ici en fraction
        de V_t-1 pour rester à la même échelle que le rendement log.
        """
        if value_before_step <= 0 or value_after_step <= 0:
            return 0.0

        log_return = float(np.log(value_after_step / value_before_step))
        cost_fraction = step_cost / value_before_step if value_before_step > 0 else 0.0
        drawdown_increase = max(0.0, drawdown_pct - self._last_drawdown_pct)

        return (
            log_return
            - self._cost_penalty_lambda * cost_fraction
            - self._drawdown_penalty_lambda * drawdown_increase
        )

    def _get_observation(self) -> np.ndarray:
        """Fenêtre glissante de `window_size` pas sur `feature_columns`, se
        terminant au pas courant inclus (jamais au-delà — anti-look-ahead).

        Avant que `window_size` pas ne soient disponibles (tout début de la
        série), la première observation disponible est dupliquée vers le
        passé (padding par répétition) plutôt que de lever une erreur ou de
        renvoyer une fenêtre de forme irrégulière. Ce choix maintient une
        forme d'observation constante (requis par observation_space) sans
        inventer de données futures ni de zéros arbitraires qui casseraient
        l'échelle des features normalisées.
        """
        end = min(self._current_step, len(self._df) - 1)
        start = end - self._window_size + 1

        if start < 0:
            available = self._df.iloc[0 : end + 1][self._feature_columns].to_numpy(dtype=np.float32)
            pad_count = -start
            if available.shape[0] == 0:
                # Aucune ligne disponible du tout (df vide) : fenêtre de zéros.
                window = np.zeros((self._window_size, len(self._feature_columns)), dtype=np.float32)
            else:
                pad = np.repeat(available[0:1], pad_count, axis=0)
                window = np.concatenate([pad, available], axis=0)
        else:
            window = self._df.iloc[start : end + 1][self._feature_columns].to_numpy(dtype=np.float32)

        return window.astype(np.float32)
