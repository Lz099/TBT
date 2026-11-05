"""Collecte continue de données dérivées (funding rate, open interest) via l'API Binance Futures publique.

Collecte uniquement — ces données ne sont pas encore consommées par le modèle (voir plan).
Aucune clé API n'est requise : ce sont des endpoints publics en lecture seule.
L'open interest historique gratuit de Binance ne couvre que ~30 jours glissants, d'où
la nécessité de faire tourner ce script régulièrement (ex: cron) pour accumuler l'historique.
"""

from __future__ import annotations

from pathlib import Path

import ccxt
import pandas as pd

from data.fetch_historical import load_config

_FUNDING_LIMIT = 1000  # max autorisé par /fapi/v1/fundingRate
_OPEN_INTEREST_LIMIT = 500  # max autorisé par /futures/data/openInterestHist
_OPEN_INTEREST_PERIOD = "5m"  # granularité la plus fine disponible côté Binance


def _symbol_to_futures_id(symbol: str) -> str:
    """"BTC/USDT" -> "BTCUSDT" (format attendu par les endpoints Binance Futures)."""
    return symbol.replace("/", "")


def _funding_path(storage_dir: str, symbol: str) -> Path:
    return Path(storage_dir) / f"{symbol.replace('/', '-')}_funding.parquet"


def _open_interest_path(storage_dir: str, symbol: str) -> Path:
    return Path(storage_dir) / f"{symbol.replace('/', '-')}_open_interest.parquet"


def _append_deduplicated(path: Path, new_df: pd.DataFrame) -> None:
    """Ajoute `new_df` (indexé par timestamp) au Parquet existant, en dédoublonnant par timestamp."""
    if new_df.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
    else:
        combined = new_df
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    combined.to_parquet(path)


def fetch_funding_rate_history(
    exchange: ccxt.binance, symbol: str, since_ms: int | None = None
) -> pd.DataFrame:
    """Récupère l'historique de funding rate pour `symbol` via l'endpoint public /fapi/v1/fundingRate."""
    futures_id = _symbol_to_futures_id(symbol)
    params: dict = {"symbol": futures_id, "limit": _FUNDING_LIMIT}
    if since_ms is not None:
        params["startTime"] = since_ms

    raw = exchange.fapiPublicGetFundingRate(params)
    if not raw:
        return pd.DataFrame(columns=["funding_rate"]).set_index(
            pd.DatetimeIndex([], name="timestamp", tz="UTC")
        )

    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["fundingTime"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    df = df[["timestamp", "funding_rate"]].drop_duplicates(subset="timestamp")
    return df.set_index("timestamp").sort_index()


def fetch_open_interest_history(exchange: ccxt.binance, symbol: str) -> pd.DataFrame:
    """Récupère l'open interest récent (fenêtre glissante ~30j côté Binance) via /futures/data/openInterestHist."""
    futures_id = _symbol_to_futures_id(symbol)
    params = {"symbol": futures_id, "period": _OPEN_INTEREST_PERIOD, "limit": _OPEN_INTEREST_LIMIT}

    raw = exchange.fapiDataGetOpeninteresthist(params)
    if not raw:
        return pd.DataFrame(columns=["open_interest"]).set_index(
            pd.DatetimeIndex([], name="timestamp", tz="UTC")
        )

    df = pd.DataFrame(raw)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df["open_interest"] = df["sumOpenInterest"].astype(float)
    df = df[["timestamp", "open_interest"]].drop_duplicates(subset="timestamp")
    return df.set_index("timestamp").sort_index()


def _last_known_timestamp_ms(path: Path) -> int | None:
    if not path.exists():
        return None
    existing = pd.read_parquet(path)
    if existing.empty:
        return None
    return int(existing.index.max().timestamp() * 1000)


def fetch_and_save_all(config_path: str = "config.yaml") -> list[Path]:
    """Met à jour les fichiers funding rate et open interest de tous les symboles définis dans config.yaml."""
    config = load_config(config_path)

    exchange_id = config["exchange"]["id"]
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    symbols: list[str] = config["symbols"]
    storage_dir: str = config["history"]["storage_dir"]

    updated_paths: list[Path] = []

    for symbol in symbols:
        funding_path = _funding_path(storage_dir, symbol)
        since_ms = _last_known_timestamp_ms(funding_path)
        since_ms = since_ms + 1 if since_ms is not None else None
        funding_df = fetch_funding_rate_history(exchange, symbol, since_ms=since_ms)
        _append_deduplicated(funding_path, funding_df)
        updated_paths.append(funding_path)

        oi_path = _open_interest_path(storage_dir, symbol)
        oi_df = fetch_open_interest_history(exchange, symbol)
        _append_deduplicated(oi_path, oi_df)
        updated_paths.append(oi_path)

    return updated_paths


if __name__ == "__main__":
    for saved_path in fetch_and_save_all():
        print(f"OK: {saved_path}")
