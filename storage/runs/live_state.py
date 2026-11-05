"""Persistance de l'état de la boucle de paper trading live (SQLite).

Pourquoi SQLite (`sqlite3`, bibliothèque standard) plutôt que SQLAlchemy
(présent dans requirements.txt pour un usage futur éventuel mais pas
nécessaire ici) : ce module n'a besoin que de deux tables simples, d'écritures
peu fréquentes (une par bougie, ex. toutes les 5 minutes à 1 jour) et d'un
seul lecteur concurrent (le dashboard Streamlit). `sqlite3` est inclus dans la
stdlib Python, ne demande aucune dépendance/serveur supplémentaire, et un
fichier `.db` unique est trivial à inspecter/supprimer pendant l'expérimentation.
Pas besoin d'un ORM pour ce volume de données.

Ce module est délibérément séparé de `paper_trading/live_runner.py` pour que
`app/dashboard.py` puisse l'importer sans dépendre du reste du runner (qui,
lui, importe ccxt/torch/stable-baselines3 etc. — inutile pour un simple
lecteur d'état).

Concurrence : SQLite gère nativement un seul writer + plusieurs readers via
son verrouillage de fichier standard ; `live_runner.py` est le seul writer
(un processus), `app/dashboard.py` est un reader occasionnel (rafraîchissement
périodique) — ce schéma reste dans le cas d'usage confortable de SQLite, pas
besoin d'un serveur de base de données pour ce projet expérimental.

Aucune connexion réseau, aucun ordre réel : ce module ne fait que lire/écrire
un fichier local décrivant l'état d'un portefeuille SIMULÉ.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DB_PATH = Path("storage/runs/paper_trading.db")

_CREATE_PORTFOLIO_SNAPSHOTS_SQL = """
CREATE TABLE IF NOT EXISTS portfolio_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    total_value REAL NOT NULL,
    cash REAL NOT NULL,
    positions_json TEXT NOT NULL,
    drawdown_pct REAL NOT NULL
)
"""

_CREATE_DECISIONS_LOG_SQL = """
CREATE TABLE IF NOT EXISTS decisions_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    price REAL NOT NULL,
    reasoning_json TEXT
)
"""

_CREATE_SNAPSHOTS_TIMESTAMP_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_timestamp "
    "ON portfolio_snapshots (timestamp)"
)
_CREATE_DECISIONS_TIMESTAMP_INDEX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_decisions_log_timestamp ON decisions_log (timestamp)"
)


def _connect(db_path: str | Path) -> sqlite3.Connection:
    """Ouvre une connexion et s'assure que le schéma existe.

    Le dashboard (lecteur) peut être ouvert AVANT tout lancement de
    `live_runner.py` (écrivain) — sans ce garde, les fonctions de lecture
    planteraient sur une base/table inexistante plutôt que de retourner un
    état vide. `init_db` est idempotente (CREATE TABLE IF NOT EXISTS), donc
    l'appeler à chaque connexion ne coûte rien de significatif à ce volume
    d'usage (lectures peu fréquentes, un seul writer).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(_CREATE_PORTFOLIO_SNAPSHOTS_SQL)
    conn.execute(_CREATE_DECISIONS_LOG_SQL)
    conn.execute(_CREATE_SNAPSHOTS_TIMESTAMP_INDEX_SQL)
    conn.execute(_CREATE_DECISIONS_TIMESTAMP_INDEX_SQL)
    conn.commit()
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Crée les tables `portfolio_snapshots`/`decisions_log` si elles n'existent pas.

    Idempotent : appelable à chaque démarrage de `live_runner.py` sans risque
    d'écraser des données déjà présentes (`CREATE TABLE IF NOT EXISTS`).
    Conservée comme point d'entrée explicite même si `_connect` fait
    désormais le même travail à chaque connexion (y compris côté lecteurs
    comme le dashboard) — un appel explicite au démarrage du writer reste
    une bonne pratique lisible.
    """
    _connect(db_path).close()


def record_snapshot(
    db_path: str | Path,
    timestamp: Any,
    total_value: float,
    cash: float,
    positions: dict[str, float],
    drawdown_pct: float,
) -> None:
    """Insère un point de la courbe de valeur du portefeuille simulé.

    `timestamp` est converti en chaîne ISO 8601 (via `pd.Timestamp`) pour un
    tri lexicographique correct et une relecture sans ambiguïté de fuseau.
    `positions` (dict symbole -> quantité) est sérialisé en JSON.
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO portfolio_snapshots "
            "(timestamp, total_value, cash, positions_json, drawdown_pct) VALUES (?, ?, ?, ?, ?)",
            (
                pd.Timestamp(timestamp).isoformat(),
                float(total_value),
                float(cash),
                json.dumps(positions),
                float(drawdown_pct),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_decision(
    db_path: str | Path,
    timestamp: Any,
    symbol: str,
    action: str,
    price: float,
    reasoning: dict[str, Any] | None = None,
) -> None:
    """Insère une entrée dans le journal des décisions (BUY/SELL/HOLD).

    `reasoning` (optionnel) est un dict libre sérialisé en JSON — pensé pour
    stocker les valeurs des indicateurs clés (ex. rsi, macd_histogram, régime
    dominant) au moment de la décision, utile pour comprendre a posteriori
    pourquoi le bot a décidé quelque chose. `None` est stocké comme NULL.
    """
    conn = _connect(db_path)
    try:
        conn.execute(
            "INSERT INTO decisions_log (timestamp, symbol, action, price, reasoning_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                pd.Timestamp(timestamp).isoformat(),
                symbol,
                action,
                float(price),
                json.dumps(reasoning) if reasoning is not None else None,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_latest_snapshot(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any] | None:
    """Retourne le dernier snapshot enregistré, ou None si la table est vide.

    Dict retourné avec les clés : `timestamp` (str ISO 8601), `total_value`
    (float), `cash` (float), `positions` (dict[str, float], déjà désérialisé
    depuis `positions_json`), `drawdown_pct` (float).
    """
    conn = _connect(db_path)
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT timestamp, total_value, cash, positions_json, drawdown_pct "
            "FROM portfolio_snapshots ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        return None

    return {
        "timestamp": row["timestamp"],
        "total_value": row["total_value"],
        "cash": row["cash"],
        "positions": json.loads(row["positions_json"]),
        "drawdown_pct": row["drawdown_pct"],
    }


def get_snapshot_history(db_path: str | Path = DEFAULT_DB_PATH, limit: int = 500) -> pd.DataFrame:
    """Retourne les `limit` derniers snapshots (ordre chronologique croissant).

    Colonnes retournées : `timestamp` (pd.Timestamp, tz-aware si l'original
    l'était), `total_value` (float), `cash` (float), `positions` (dict[str,
    float], désérialisé depuis `positions_json`), `drawdown_pct` (float).
    DataFrame vide (mêmes colonnes, 0 ligne) si la table est vide — jamais
    None, pour que l'appelant (dashboard) puisse toujours tracer sans
    vérification supplémentaire.
    """
    conn = _connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, total_value, cash, positions_json, drawdown_pct "
            "FROM portfolio_snapshots ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    finally:
        conn.close()

    df = df.iloc[::-1].reset_index(drop=True)  # ordre chronologique croissant
    if df.empty:
        return pd.DataFrame(
            columns=["timestamp", "total_value", "cash", "positions", "drawdown_pct"]
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["positions"] = df["positions_json"].apply(json.loads)
    return df[["timestamp", "total_value", "cash", "positions", "drawdown_pct"]]


def get_recent_decisions(db_path: str | Path = DEFAULT_DB_PATH, limit: int = 50) -> pd.DataFrame:
    """Retourne les `limit` décisions les plus récentes (ordre chronologique croissant).

    Colonnes retournées : `timestamp` (pd.Timestamp), `symbol` (str),
    `action` (str, "HOLD"/"BUY"/"SELL"), `price` (float), `reasoning` (dict ou
    None, désérialisé depuis `reasoning_json`). DataFrame vide (mêmes
    colonnes, 0 ligne) si la table est vide.
    """
    conn = _connect(db_path)
    try:
        df = pd.read_sql_query(
            "SELECT timestamp, symbol, action, price, reasoning_json "
            "FROM decisions_log ORDER BY id DESC LIMIT ?",
            conn,
            params=(limit,),
        )
    finally:
        conn.close()

    df = df.iloc[::-1].reset_index(drop=True)  # ordre chronologique croissant
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "symbol", "action", "price", "reasoning"])

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["reasoning"] = df["reasoning_json"].apply(lambda v: json.loads(v) if v is not None else None)
    return df[["timestamp", "symbol", "action", "price", "reasoning"]]
