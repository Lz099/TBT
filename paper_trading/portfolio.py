"""Portefeuille simulé — AUCUNE connexion réseau réelle, AUCUNE clé API de
trading, 100% en mémoire. C'est la seule "exécution" d'ordre de tout le
projet, et elle ne touche jamais un compte réel.

Intègre :
- le slippage/frais réalistes (backtest.slippage.apply_trading_costs)
- les contraintes d'exposition (risk.portfolio_constraints.check_exposure_limits)
- le stop-loss adaptatif Chandelier Exit (risk.stops), appliqué indépendamment
  de ce que déciderait un agent — filet de sécurité non désapprenable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from risk.portfolio_constraints import check_exposure_limits
from backtest.slippage import apply_trading_costs


@dataclass
class Trade:
    timestamp: Any
    symbol: str
    side: Literal["BUY", "SELL"]
    quantity: float
    execution_price: float
    requested_quantity: float


class SimulatedPortfolio:
    """Portefeuille multi-actifs simulé (pensé pour BTC/USDT + ETH/USDT conjoints)."""

    def __init__(self, initial_cash: float, config: dict[str, Any]) -> None:
        self._initial_cash = initial_cash
        self._config = config
        self.cash: float = initial_cash
        self.positions: dict[str, float] = {}
        self.value_history: list[tuple[Any, float]] = []
        self.trades: list[Trade] = []
        self._stop_levels: dict[str, float] = {}

    def execute_trade(
        self,
        symbol: str,
        side: Literal["BUY", "SELL"],
        quantity: float,
        price: float,
        timestamp: Any,
        volume_24h: float,
        volatility: float,
        other_asset_exposure_pct: float = 0.0,
        correlation: float = 0.0,
    ) -> Trade:
        """Exécute un trade simulé après vérification des limites d'exposition.

        Si le trade dépasserait les limites configurées, la quantité est
        réduite automatiquement plutôt que le trade entièrement rejeté —
        choix délibéré : un agent qui a une conviction directionnelle reste
        exposé (à une taille prudente) plutôt que totalement bloqué, ce qui
        est plus représentatif d'une gestion de risque réelle qu'un rejet dur.
        """
        current_prices = {symbol: price}
        total_value = self.get_total_value(current_prices)

        current_symbol_exposure_pct = (
            abs(self.positions.get(symbol, 0.0)) * price / total_value if total_value > 0 else 0.0
        )

        exposure_check = check_exposure_limits(
            current_btc_exposure_pct=(
                current_symbol_exposure_pct if "BTC" in symbol else other_asset_exposure_pct
            ),
            current_eth_exposure_pct=(
                current_symbol_exposure_pct if "ETH" in symbol else other_asset_exposure_pct
            ),
            correlation=correlation,
            config=self._config,
        )

        adjusted_quantity = quantity
        if side == "BUY" and not exposure_check["within_limits"]:
            portfolio_cfg = self._config["portfolio"]
            max_pct = portfolio_cfg["max_exposure_per_asset_pct"]
            max_notional = total_value * max_pct
            max_quantity = max_notional / price if price > 0 else 0.0
            adjusted_quantity = min(quantity, max(max_quantity, 0.0))

        execution_price = apply_trading_costs(
            price=price,
            quantity=adjusted_quantity,
            side=side,
            volume_24h=volume_24h,
            volatility=volatility,
            config=self._config,
        )

        if side == "BUY" and execution_price > 0:
            # GARDE-FOU CRITIQUE : sans ça, un calcul de sizing en amont
            # (ex. risk.position_sizing.atr_volatility_targeting, qui peut
            # légitimement retourner une quantité représentant plusieurs
            # fois le capital si l'ATR est petit par rapport au prix) permet
            # d'acheter à crédit sans limite — self.cash devient négatif et
            # peut s'effondrer sur des dizaines de trades, produisant des
            # rendements impossibles (ex. -1800%) qui n'ont plus rien à voir
            # avec une vraie performance de marché. Jamais d'effet de levier
            # dans ce projet : on ne peut jamais dépenser plus de cash
            # disponible, point final.
            max_affordable_quantity = self.cash / execution_price
            adjusted_quantity = min(adjusted_quantity, max(max_affordable_quantity, 0.0))

        notional = adjusted_quantity * execution_price
        current_position = self.positions.get(symbol, 0.0)

        if side == "BUY":
            self.cash -= notional
            self.positions[symbol] = current_position + adjusted_quantity
        else:
            self.cash += notional
            self.positions[symbol] = current_position - adjusted_quantity

        trade = Trade(
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            quantity=adjusted_quantity,
            execution_price=execution_price,
            requested_quantity=quantity,
        )
        self.trades.append(trade)
        return trade

    def set_stop_level(self, symbol: str, stop_price: float) -> None:
        """Enregistre le niveau de stop Chandelier Exit courant pour `symbol`."""
        self._stop_levels[symbol] = stop_price

    def apply_stop_loss(self, symbol: str, current_price: float, current_timestamp: Any) -> bool:
        """Clôture la position sur `symbol` si `current_price` a franchi le stop
        enregistré. Indépendant de toute décision d'un agent RL. Retourne True
        si un stop a été déclenché (et exécuté).
        """
        stop_price = self._stop_levels.get(symbol)
        position = self.positions.get(symbol, 0.0)

        if stop_price is None or position <= 0:
            return False

        if current_price <= stop_price:
            self.execute_trade(
                symbol=symbol,
                side="SELL",
                quantity=position,
                price=current_price,
                timestamp=current_timestamp,
                volume_24h=1.0,  # stop d'urgence : exécution immédiate, impact non prioritaire ici
                volatility=0.0,
            )
            self._stop_levels.pop(symbol, None)
            return True

        return False

    def get_total_value(self, current_prices: dict[str, float]) -> float:
        """Cash + valeur des positions ouvertes aux prix courants."""
        positions_value = sum(
            qty * current_prices.get(symbol, 0.0) for symbol, qty in self.positions.items()
        )
        return self.cash + positions_value

    def record_value(self, timestamp: Any, current_prices: dict[str, float]) -> None:
        """Enregistre un point de la courbe de valeur du portefeuille."""
        self.value_history.append((timestamp, self.get_total_value(current_prices)))

    def get_metrics(self, risk_free_rate: float = 0.0) -> dict[str, float]:
        """Métriques de performance calculées sur `value_history` et `trades`.

        Formules :
        - rendement total = V_fin / V_initial - 1
        - Sharpe = moyenne(rendements) / écart-type(rendements), annualisé
        - Sortino = moyenne(rendements) / écart-type(rendements négatifs uniquement)
        - Calmar = rendement annualisé / |max drawdown|
        - max drawdown = pire baisse depuis un plus haut historique de la courbe de valeur
        - win rate = proportion de trades de sortie (SELL) clôturés en gain
        """
        if len(self.value_history) < 2:
            return {
                "total_return_pct": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "calmar_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate_pct": 0.0,
                "num_trades": len(self.trades),
            }

        values = [v for _, v in self.value_history]
        returns = [(values[i] / values[i - 1] - 1.0) for i in range(1, len(values))]

        total_return_pct = (values[-1] / values[0] - 1.0) * 100.0

        mean_return = sum(returns) / len(returns)
        variance = sum((r - mean_return) ** 2 for r in returns) / len(returns) if returns else 0.0
        std_return = variance ** 0.5

        sharpe_ratio = ((mean_return - risk_free_rate) / std_return) if std_return > 0 else 0.0

        downside_returns = [r for r in returns if r < 0]
        if downside_returns:
            downside_variance = sum(r**2 for r in downside_returns) / len(downside_returns)
            downside_std = downside_variance ** 0.5
        else:
            downside_std = 0.0
        sortino_ratio = ((mean_return - risk_free_rate) / downside_std) if downside_std > 0 else 0.0

        peak = values[0]
        max_drawdown = 0.0
        for v in values:
            peak = max(peak, v)
            drawdown = (peak - v) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, drawdown)
        max_drawdown_pct = max_drawdown * 100.0

        annualized_return = mean_return * 252  # approximation standard, à affiner selon le timeframe réel
        calmar_ratio = (annualized_return / max_drawdown) if max_drawdown > 0 else 0.0

        sell_trades = [t for t in self.trades if t.side == "SELL"]
        buy_trades_by_symbol: dict[str, list[Trade]] = {}
        winning_trades = 0
        for t in self.trades:
            if t.side == "BUY":
                buy_trades_by_symbol.setdefault(t.symbol, []).append(t)
            elif t.side == "SELL" and buy_trades_by_symbol.get(t.symbol):
                entry = buy_trades_by_symbol[t.symbol].pop(0)
                if t.execution_price > entry.execution_price:
                    winning_trades += 1
        win_rate_pct = (winning_trades / len(sell_trades) * 100.0) if sell_trades else 0.0

        return {
            "total_return_pct": total_return_pct,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate_pct": win_rate_pct,
            "num_trades": len(self.trades),
        }

    def reset(self) -> None:
        self.cash = self._initial_cash
        self.positions = {}
        self.value_history = []
        self.trades = []
        self._stop_levels = {}
