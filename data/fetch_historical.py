"""Télécharge et met à jour l'historique OHLCV (klines) depuis Binance via ccxt.

Idempotent : relance uniquement depuis la dernière bougie connue si un fichier
Parquet existe déjà pour le couple (symbole, timeframe).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import ccxt
import pandas as pd
import yaml

OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]

# Nombre max de bougies renvoyées par appel par la plupart des exchanges ccxt.
_FETCH_LIMIT = 1000

_TIMEFRAME_MS = {
    "1m": 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "1h": 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    return f"{symbol.replace('/', '-')}_{timeframe}.parquet"


def load_config(config_path: str) -> dict:
    """Charge config.yaml (partagé par les autres modules `data/*` pour éviter toute constante en dur)."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _storage_path(storage_dir: str, symbol: str, timeframe: str) -> Path:
    return Path(storage_dir) / _symbol_to_filename(symbol, timeframe)


def _last_known_timestamp_ms(path: Path) -> int | None:
    """Retourne le timestamp (ms, UTC) de la dernière bougie déjà stockée, ou None si absent."""
    if not path.exists():
        return None
    existing = pd.read_parquet(path)
    if existing.empty:
        return None
    last_ts = existing.index.max()
    return int(last_ts.timestamp() * 1000)


def fetch_ohlcv_since(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    since_ms: int,
    until_ms: int,
) -> pd.DataFrame:
    """Pagine `fetch_ohlcv` depuis `since_ms` jusqu'à `until_ms` (exclus) et retourne un DataFrame indexé par timestamp UTC.

    Exclut toute bougie dont la clôture n'est pas encore passée (bougie en cours de formation),
    pour ne jamais persister une valeur provisoire qui changerait à la prochaine relance.
    """
    all_rows: list[list[float]] = []
    cursor = since_ms
    page = 0
    while cursor < until_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=cursor, limit=_FETCH_LIMIT)
        if not batch:
            break
        all_rows.extend(batch)
        page += 1
        if page % 10 == 0:
            # Progression visible pour les longues périodes (ex. 5m sur plusieurs
            # années = des centaines de pages) — sans ça, le script reste
            # silencieux pendant de longues minutes, ce qui donne l'impression
            # (fausse) qu'il est bloqué depuis un terminal ou un log tail.
            current_date = datetime.fromtimestamp(batch[-1][0] / 1000, tz=timezone.utc).date()
            print(f"  [{symbol} {timeframe}] page {page}, {len(all_rows)} bougies, en cours: {current_date}", flush=True)
        last_open_time = batch[-1][0]
        if last_open_time <= cursor:
            # Pas de progression (exchange ne peut pas avancer davantage) : on arrête.
            break
        cursor = last_open_time + 1
        if len(batch) < _FETCH_LIMIT:
            # Dernière page (moins de bougies que la limite demandée).
            break

    if not all_rows:
        return pd.DataFrame(columns=OHLCV_COLUMNS[1:]).set_index(
            pd.DatetimeIndex([], name="timestamp", tz="UTC")
        )

    df = pd.DataFrame(all_rows, columns=OHLCV_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()

    timeframe_ms = _TIMEFRAME_MS.get(timeframe)
    open_time_ms = df.index.view("int64") // 1_000_000
    if timeframe_ms is not None:
        # Ne garder que les bougies dont l'heure de clôture (open + durée) est déjà passée.
        df = df[(open_time_ms + timeframe_ms <= until_ms) & (open_time_ms < until_ms)]
    else:
        df = df[open_time_ms < until_ms]
    return df


def update_symbol_timeframe(
    exchange: ccxt.Exchange,
    symbol: str,
    timeframe: str,
    start_date: str,
    storage_dir: str,
) -> Path:
    """Met à jour (ou crée) le fichier Parquet d'un couple (symbole, timeframe) et retourne son chemin."""
    path = _storage_path(storage_dir, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)

    last_ts_ms = _last_known_timestamp_ms(path)
    if last_ts_ms is not None:
        # Repart juste après la dernière bougie connue.
        since_ms = last_ts_ms + 1
    else:
        since_ms = int(
            datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        )

    until_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    if since_ms >= until_ms:
        return path

    new_df = fetch_ohlcv_since(exchange, symbol, timeframe, since_ms, until_ms)
    if new_df.empty:
        return path

    if last_ts_ms is not None:
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, new_df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = new_df

    combined.to_parquet(path)
    return path


def fetch_and_save_all(config_path: str = "config.yaml") -> list[Path]:
    """Met à jour l'historique OHLCV de tous les symboles/timeframes définis dans config.yaml."""
    config = load_config(config_path)

    exchange_id = config["exchange"]["id"]
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({"enableRateLimit": True})

    symbols: list[str] = config["symbols"]
    timeframes: list[str] = config["timeframes"]
    start_date: str = config["history"]["start_date"]
    storage_dir: str = config["history"]["storage_dir"]

    updated_paths: list[Path] = []
    total = len(symbols) * len(timeframes)
    done = 0
    for symbol in symbols:
        for timeframe in timeframes:
            done += 1
            print(f"[{done}/{total}] {symbol} {timeframe} — démarrage…", flush=True)
            path = update_symbol_timeframe(exchange, symbol, timeframe, start_date, storage_dir)
            print(f"[{done}/{total}] {symbol} {timeframe} — terminé -> {path}", flush=True)
            updated_paths.append(path)
    return updated_paths


if __name__ == "__main__":
    for saved_path in fetch_and_save_all():
        print(f"OK: {saved_path}")
