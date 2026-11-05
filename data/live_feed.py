"""Lecture de données de marché en temps quasi-réel via polling REST ccxt (pas de websocket).

Interface préparée pour la Vague 3 (paper_trading/live_runner.py) : ce module n'expose que des
fonctions appelables, sans boucle infinie. La profondeur de carnet complète est hors-scope V1
(mismatch train/inférence) : `fetch_current_spread` ne lit que le meilleur bid/ask.
"""

from __future__ import annotations

import ccxt
import pandas as pd

from data.fetch_historical import OHLCV_COLUMNS, load_config

_SPREAD_ORDER_BOOK_LIMIT = 5


def _build_exchange(config_path: str = "config.yaml") -> ccxt.Exchange:
    config = load_config(config_path)
    exchange_id = config["exchange"]["id"]
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({"enableRateLimit": True})


def fetch_latest_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int = 100,
    exchange: ccxt.Exchange | None = None,
) -> pd.DataFrame:
    """Retourne les `limit` dernières bougies OHLCV pour `symbol`/`timeframe`, indexées par timestamp UTC."""
    exchange = exchange or _build_exchange()
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(raw, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def fetch_current_spread(symbol: str, exchange: ccxt.Exchange | None = None) -> dict[str, float]:
    """Retourne le meilleur bid/ask instantané (et le spread) pour `symbol`, sans profondeur de carnet."""
    exchange = exchange or _build_exchange()
    order_book = exchange.fetch_order_book(symbol, limit=_SPREAD_ORDER_BOOK_LIMIT)

    best_bid = order_book["bids"][0][0] if order_book["bids"] else float("nan")
    best_ask = order_book["asks"][0][0] if order_book["asks"] else float("nan")

    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": best_ask - best_bid,
        "spread_pct": (best_ask - best_bid) / best_bid if best_bid else float("nan"),
    }
