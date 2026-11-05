"""Signaux de contexte hors OHLCV : Fear & Greed Index, ratio ETH/BTC, spread bid/ask.

Fear & Greed Index (alternative.me) : résolution QUOTIDIENNE uniquement (une
valeur par jour). Il est donc pertinent comme feature pour les timeframes
1h/1d, mais PAS pour le 5m : on ne le forward-fill pas artificiellement à une
résolution plus fine ici (ça créerait de faux paliers d'information constante
sur des centaines de bougies 5m). L'alignement/resample vers un timeframe
cible est laissé aux modules consommateurs (ex: features/pipeline), qui
décident explicitement de la stratégie de merge selon le timeframe.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import requests
import yaml

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
_FNG_URL = "https://api.alternative.me/fng/"
_CACHE_MAX_AGE_SECONDS = 24 * 60 * 60  # Fear & Greed est publié une fois par jour


def _storage_dir() -> Path:
    with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    project_root = _CONFIG_PATH.parent
    return project_root / config["history"]["storage_dir"]


def fetch_fear_greed_index(limit: int = 0) -> pd.DataFrame:
    """Récupère l'historique du Fear & Greed Index (limit=0 -> historique complet).

    Retourne un DataFrame avec colonnes timestamp (datetime UTC, résolution
    jour) / value (int 0-100). Résultat mis en cache en Parquet dans
    `{storage_dir}/fear_greed.parquet` ; re-fetch réseau uniquement si le
    cache n'existe pas ou a plus de 24h.
    """
    cache_path = _storage_dir() / "fear_greed.parquet"

    if cache_path.exists():
        cache_age_seconds = time.time() - cache_path.stat().st_mtime
        if cache_age_seconds < _CACHE_MAX_AGE_SECONDS:
            return pd.read_parquet(cache_path)

    response = requests.get(_FNG_URL, params={"limit": limit, "format": "json"}, timeout=10)
    response.raise_for_status()
    payload = response.json()["data"]

    df = pd.DataFrame(payload)[["timestamp", "value"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="s", utc=True)
    df["value"] = df["value"].astype(int)
    df = df.sort_values("timestamp").reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)

    return df


def compute_eth_btc_ratio(df_eth: pd.DataFrame, df_btc: pd.DataFrame) -> pd.Series:
    """Ratio close_ETH/close_BTC aligné sur les timestamps communs aux deux DataFrames (index=timestamp)."""
    eth_close = df_eth.set_index("timestamp")["close"]
    btc_close = df_btc.set_index("timestamp")["close"]
    common_index = eth_close.index.intersection(btc_close.index)
    ratio = eth_close.loc[common_index] / btc_close.loc[common_index]
    ratio.name = "eth_btc_ratio"
    return ratio.sort_index()


def compute_bid_ask_spread_pct(bid: float, ask: float) -> float:
    """Spread relatif (ask-bid)/mid_price. `bid`/`ask` viennent typiquement de data/live_feed.py::fetch_current_spread."""
    mid_price = (bid + ask) / 2.0
    return (ask - bid) / mid_price
