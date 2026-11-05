"""Script d'entraînement de bout en bout : données -> features -> env -> PPO -> backtest.

Point d'entrée exécutable (`python -m models.train_pipeline` ou
`python models/train_pipeline.py` depuis la racine du projet).

SIMPLIFICATION ASSUMÉE POUR CETTE V1 : l'entraînement complet ré-entraînerait
un agent PPO sur CHAQUE split walk-forward (cf. `backtest.validation.walk_forward_splits`),
ce qui est la façon rigoureuse de valider une stratégie RL dans le temps mais
peut être très long (un agent par split, plusieurs dizaines/centaines de
milliers de timesteps chacun). Pour ce premier run de bout en bout, on
n'entraîne et n'évalue que sur LE PREMIER split walk-forward, pour vérifier
que le pipeline entier (chargement -> features -> env -> PPO -> backtest)
fonctionne sans erreur avant de lancer un entraînement long multi-split.
Étendre à tous les splits consiste simplement à boucler sur
`walk_forward_splits(...)` et répéter le bloc "un split" ci-dessous.

RÈGLE ANTI-LOOK-AHEAD : les statistiques de normalisation (moyenne/écart-type)
sont calculées UNIQUEMENT sur le train du split courant
(`features.pipeline.normalize_features(train_df, ..., train_stats=None)`) puis
réutilisées telles quelles sur le test du même split
(`normalize_features(test_df, ..., train_stats=train_stats)`) — jamais
recalculées sur le test. Le HMM de régime est également fit uniquement sur le
train (`train_df=train_split` passé à `build_feature_dataframe`), jamais sur
le test. Le jeu de test gelé (`get_frozen_test_set`) n'est jamais consulté ici.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import json

import pandas as pd
import yaml

from stable_baselines3.common.callbacks import BaseCallback

from backtest.engine import run_backtest
from backtest.validation import walk_forward_splits
from features.pipeline import build_feature_dataframe, normalize_features
from models.rl_agent import RLAgentStrategy, create_ppo_agent, load_agent, save_agent, train_agent
from models.training_progress import TRAIN_PROGRESS_PATH, get_training_progress, write_training_progress
from env.trading_env import TradingEnv

_DEFAULT_TIMEFRAME = "1h"
_DEFAULT_WINDOW_SIZE = 20
_DEFAULT_TOTAL_TIMESTEPS = 50_000  # premier run de validation, pas un entraînement final

# Réexporté pour compatibilité (ancien nom utilisé ailleurs dans ce fichier ci-dessous).
_TRAIN_PROGRESS_PATH = TRAIN_PROGRESS_PATH


class TrainingProgressCallback(BaseCallback):
    """Écrit la progression de l'entraînement (step courant, récompense
    moyenne du buffer PPO) dans un fichier JSON à chaque rollout, pour que le
    dashboard (piloté depuis un processus séparé, cf. app/process_manager.py)
    puisse afficher une courbe de récompense qui se met à jour en direct
    pendant l'entraînement — sans dépendance entre les deux processus au-delà
    de ce fichier partagé, même pattern que storage/runs/live_state.py.

    La lecture (`get_training_progress`) vit dans `models/training_progress.py`
    (module léger, sans dépendance torch/stable-baselines3) pour que
    `app/dashboard.py` puisse l'importer sans charger l'intégralité de la
    stack ML — seule l'ÉCRITURE (ce callback) a besoin de `BaseCallback`.
    """

    def __init__(self, progress_path: Path = TRAIN_PROGRESS_PATH, total_timesteps: int = 0) -> None:
        super().__init__(verbose=0)
        self._progress_path = progress_path
        self._total_timesteps = total_timesteps
        self._points: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self._points = []
        write_training_progress(
            {"status": "running", "total_timesteps": self._total_timesteps, "points": []},
            self._progress_path,
        )

    def _on_rollout_end(self) -> None:
        # `ep_info_buffer` contient les épisodes terminés récemment (rempli
        # nativement par Stable-Baselines3 via son Monitor wrapper interne) ;
        # peut être vide en tout début d'entraînement (aucun épisode terminé
        # encore) — on ignore silencieusement ce cas plutôt que de planter.
        if self.model.ep_info_buffer:
            mean_reward = float(
                sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            )
            self._points.append({"step": int(self.num_timesteps), "mean_reward": mean_reward})
            write_training_progress(
                {
                    "status": "running",
                    "total_timesteps": self._total_timesteps,
                    "points": self._points,
                },
                self._progress_path,
            )

    def _on_training_end(self) -> None:
        write_training_progress(
            {
                "status": "finished",
                "total_timesteps": self._total_timesteps,
                "points": self._points,
            },
            self._progress_path,
        )

    def _on_step(self) -> bool:
        return True  # continue toujours l'entraînement — ce callback n'observe que, n'interrompt jamais


def load_config(config_path: str = "config.yaml") -> dict[str, Any]:
    """Charge config.yaml — même convention que data/fetch_historical.py::load_config."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _slugify(text: str) -> str:
    """Nom lisible -> fragment de nom de fichier sûr (minuscules, tirets, alphanumérique)."""
    slug = text.strip().lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    return slug.strip("-")


def _build_model_filename(symbol: str, timeframe: str, display_name: str | None) -> str:
    """Construit un nom de fichier unique pour un modèle entraîné.

    `ppo_{SYMBOL}_{timeframe}_{slug(display_name)}_{horodatage}` si un nom
    est fourni, sinon `ppo_{SYMBOL}_{timeframe}_{horodatage}`. L'horodatage
    (précision à la seconde) garantit l'absence de collision même si le même
    nom est réutilisé pour deux entraînements successifs — chaque
    entraînement produit systématiquement un fichier distinct, jamais un
    écrasement silencieux (contrairement à l'ancienne convention fixe
    `ppo_{SYMBOL}_{timeframe}` sans suffixe).
    """
    base = f"ppo_{symbol.replace('/', '-')}_{timeframe}"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if display_name:
        slug = _slugify(display_name)
        if slug:
            return f"{base}_{slug}_{timestamp}"
    return f"{base}_{timestamp}"


def _symbol_to_filename(symbol: str, timeframe: str) -> str:
    """"BTC/USDT", "1h" -> "BTC-USDT_1h.parquet" — même convention que data/fetch_historical.py."""
    return f"{symbol.replace('/', '-')}_{timeframe}.parquet"


def load_ohlcv(symbol: str, timeframe: str, storage_dir: str) -> pd.DataFrame:
    """Charge un Parquet OHLCV déjà téléchargé par data/fetch_historical.py.

    Lève une erreur explicite (et actionnable) si le fichier est absent —
    frontière réelle (E/S disque), donc gestion d'erreur justifiée ici.
    """
    path = Path(storage_dir) / _symbol_to_filename(symbol, timeframe)
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable: {path}. Lance d'abord `python data/fetch_historical.py` "
            f"pour peupler data/storage/ avant d'entraîner l'agent."
        )
    return pd.read_parquet(path)


def load_funding(symbol: str, storage_dir: str) -> pd.DataFrame | None:
    """Charge le funding rate optionnel écrit par data/fetch_derivatives.py, si présent."""
    path = Path(storage_dir) / f"{symbol.replace('/', '-')}_funding.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path)


def build_train_test_features(
    train_df_raw: pd.DataFrame,
    test_df_raw: pd.DataFrame,
    df_funding: pd.DataFrame | None,
    df_eth_ohlcv_train: pd.DataFrame | None,
    df_eth_ohlcv_test: pd.DataFrame | None,
    config: dict[str, Any],
    timeframe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, tuple[float, float]]]:
    """Construit les features train/test d'un split walk-forward, en respectant
    la règle anti-look-ahead : HMM fit sur train uniquement, normalisation
    fit sur train uniquement puis appliquée telle quelle au test.

    Retourne aussi `train_stats` (moyenne/écart-type par colonne, calculées
    sur le train) — l'appelant doit les persister à côté du modèle pour que
    `paper_trading/live_runner.py` puisse normaliser les features live avec
    exactement les mêmes statistiques qu'à l'entraînement (cf.
    `live_runner._load_train_stats`).
    """
    train_features, feature_columns = build_feature_dataframe(
        train_df_raw,
        df_funding=df_funding,
        df_eth_ohlcv=df_eth_ohlcv_train,
        config=config,
        timeframe=timeframe,
        train_df=train_df_raw,  # le HMM ne voit que le train de ce split
    )
    test_features, _ = build_feature_dataframe(
        test_df_raw,
        df_funding=df_funding,
        df_eth_ohlcv=df_eth_ohlcv_test,
        config=config,
        timeframe=timeframe,
        train_df=train_df_raw,  # même train pour fitter le HMM appliqué au test — jamais le test lui-même
    )

    train_features = train_features.dropna(subset=[*feature_columns, "atr"])
    test_features = test_features.dropna(subset=[*feature_columns, "atr"])

    train_norm, train_stats = normalize_features(train_features, feature_columns, train_stats=None)
    test_norm, _ = normalize_features(test_features, feature_columns, train_stats=train_stats)

    return train_norm, test_norm, feature_columns, train_stats


def run_training_pipeline(
    config_path: str = "config.yaml",
    symbol: str = "BTC/USDT",
    timeframe: str = _DEFAULT_TIMEFRAME,
    window_size: int = _DEFAULT_WINDOW_SIZE,
    total_timesteps: int = _DEFAULT_TOTAL_TIMESTEPS,
    continue_from_model: str | Path | None = None,
    model_display_name: str | None = None,
) -> dict[str, Any]:
    """Pipeline complet : charge les données -> construit les features (1er split
    walk-forward uniquement, cf. simplification documentée en tête de fichier) ->
    entraîne un agent PPO sur le train -> évalue via run_backtest sur le test.

    `continue_from_model` (optionnel) : chemin (avec ou sans extension) d'un
    modèle PPO déjà sauvegardé (`storage/models/*.zip`) — au lieu de repartir
    d'un agent vierge, charge CE modèle précis et poursuit son entraînement
    dessus. Choisi explicitement (via `models.rl_agent.list_saved_models` côté
    appelant, ex. le dashboard) plutôt que "le dernier modèle du dossier" —
    permet de reprendre n'importe quel modèle nommé, pas seulement le plus
    récent. Si le fichier n'existe pas ou que ses features sont incompatibles
    avec ce nouvel entraînement, retombe explicitement sur un nouveau départ
    (jamais d'erreur bloquante pour ce choix). LIMITE CONNUE : continuer
    indéfiniment sur les MÊMES données d'entraînement (même split walk-forward)
    risque du surapprentissage — l'agent peut mémoriser ce split au lieu de
    généraliser. Comparer régulièrement via l'onglet Comparaison stratégies
    (vs buy-and-hold et politique aléatoire) pour vérifier que continuer aide
    réellement plutôt que de le supposer.

    `model_display_name` (optionnel) : nom donné par l'utilisateur, intégré
    au nom de fichier du modèle sauvegardé et à ses métadonnées — permet de
    garder plusieurs modèles distincts pour le même (symbol, timeframe) sans
    qu'ils s'écrasent (cf. `_build_model_filename`).

    Retourne un dict avec les métriques de backtest sur le test et le chemin
    du modèle sauvegardé.
    """
    config = load_config(config_path)
    storage_dir = config["history"]["storage_dir"]

    df_raw = load_ohlcv(symbol, timeframe, storage_dir)
    df_funding = load_funding(symbol, storage_dir)

    # ETH/BTC ratio : pertinent seulement quand le symbole principal est BTC/USDT
    # (cf. asymétrie documentée dans features/pipeline.py::_merge_eth_btc_ratio).
    df_eth_raw: pd.DataFrame | None = None
    if symbol == "BTC/USDT" and "ETH/USDT" in config.get("symbols", []):
        eth_path = Path(storage_dir) / _symbol_to_filename("ETH/USDT", timeframe)
        if eth_path.exists():
            df_eth_raw = pd.read_parquet(eth_path)

    validation_cfg = config["validation"]
    splits = list(
        walk_forward_splits(
            df_raw,
            train_days=validation_cfg["walk_forward_train_days"],
            test_days=validation_cfg["walk_forward_test_days"],
            step_days=validation_cfg["walk_forward_step_days"],
        )
    )
    if not splits:
        raise ValueError(
            "Aucun split walk-forward généré à partir des données chargées — "
            "vérifier que data/storage/ contient assez d'historique par rapport à "
            "walk_forward_train_days + walk_forward_test_days de config.yaml."
        )

    # Simplification V1 documentée en tête de fichier : seulement le premier split.
    train_df_raw, test_df_raw = splits[0]

    df_eth_train = df_eth_raw.loc[df_eth_raw.index.isin(train_df_raw.index)] if df_eth_raw is not None else None
    df_eth_test = df_eth_raw.loc[df_eth_raw.index.isin(test_df_raw.index)] if df_eth_raw is not None else None

    train_features, test_features, feature_columns, train_stats = build_train_test_features(
        train_df_raw,
        test_df_raw,
        df_funding=df_funding,
        df_eth_ohlcv_train=df_eth_train,
        df_eth_ohlcv_test=df_eth_test,
        config=config,
        timeframe=timeframe,
    )

    print(f"[train_pipeline] {len(feature_columns)} features: {feature_columns}")
    print(f"[train_pipeline] train: {len(train_features)} lignes, test: {len(test_features)} lignes")

    train_env = TradingEnv(
        train_features,
        feature_columns=feature_columns,
        config=config,
        symbol=symbol,
        timeframe=timeframe,
        window_size=window_size,
    )

    model_name = _build_model_filename(symbol, timeframe, model_display_name)

    agent = None
    if continue_from_model is not None:
        from models.rl_agent import load_agent_metadata

        continue_path = Path(continue_from_model)
        if not continue_path.is_absolute() and continue_path.parent == Path("."):
            continue_path = Path("storage/models") / continue_path

        if not continue_path.with_suffix(".zip").exists():
            print(f"[train_pipeline] modèle de reprise introuvable ({continue_path}.zip) — nouveau départ.")
        else:
            existing_metadata = load_agent_metadata(continue_path)
            if existing_metadata is not None and existing_metadata.get("feature_columns") == feature_columns:
                print(f"[train_pipeline] reprise de l'entraînement depuis {continue_path}.zip")
                agent = load_agent(continue_path, env=train_env)
            else:
                # Colonnes de features différentes (ex. funding rate absent cette
                # fois, ou HMM avec un nombre d'états différent) : l'agent existant
                # attend un vecteur d'observation d'une autre forme, incompatible
                # avec ce nouvel environnement. Continuer serait silencieusement
                # incorrect (mismatch de dimension à la première prédiction) —
                # on retombe explicitement sur un nouveau départ plutôt que de
                # planter ou de deviner.
                print(
                    f"[train_pipeline] modèle {continue_path.name} trouvé mais features incompatibles "
                    f"(colonnes différentes de l'entraînement précédent) — nouveau départ."
                )

    if agent is None:
        agent = create_ppo_agent(train_env, config=config)

    progress_callback = TrainingProgressCallback(total_timesteps=total_timesteps)
    agent = train_agent(agent, total_timesteps=total_timesteps, callback=progress_callback)

    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    saved_path = save_agent(
        agent,
        model_name,
        feature_columns=feature_columns,
        window_size=window_size,
        display_name=model_display_name,
        trained_at=trained_at,
    )
    print(f"[train_pipeline] agent sauvegardé: {saved_path}")

    # Sauvegarde des stats de normalisation du train (moyenne/écart-type par
    # colonne) à côté du modèle — même convention de nom que celle attendue
    # par paper_trading/live_runner.py::_load_train_stats
    # ("{modèle}_train_stats.json"). Indispensable pour que le live runner
    # normalise les features live avec exactement les stats vues à
    # l'entraînement (anti-look-ahead / anti-dérive de distribution), au lieu
    # de retomber sur son fallback (stats recalculées sur la fenêtre live).
    train_stats_path = saved_path.with_name(saved_path.stem + "_train_stats.json")
    train_stats_serializable = {col: [mean, std] for col, (mean, std) in train_stats.items()}
    train_stats_path.write_text(json.dumps(train_stats_serializable, indent=2), encoding="utf-8")
    print(f"[train_pipeline] train_stats sauvegardés: {train_stats_path}")

    strategy = RLAgentStrategy(agent, feature_columns=feature_columns, window_size=window_size)
    result = run_backtest(
        test_features,
        strategy,
        initial_cash=config["portfolio"]["initial_cash_usdt"],
        config=config,
        symbol=symbol,
        timeframe=timeframe,
    )

    return {
        "metrics": result.metrics,
        "num_trades": len(result.trades),
        "model_path": str(saved_path),
        "feature_columns": feature_columns,
    }


def _parse_args(argv: list[str] | None = None) -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Entraîne un agent PPO de bout en bout (voir docstring du module).")
    parser.add_argument("--symbol", default="BTC/USDT", help="Symbole à entraîner (défaut: BTC/USDT).")
    parser.add_argument("--timeframe", default=_DEFAULT_TIMEFRAME, help=f"Timeframe (défaut: {_DEFAULT_TIMEFRAME}).")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=_DEFAULT_TOTAL_TIMESTEPS,
        help=f"Nombre de pas d'entraînement PPO (défaut: {_DEFAULT_TOTAL_TIMESTEPS}).",
    )
    parser.add_argument("--config", default="config.yaml", help="Chemin de config.yaml.")
    parser.add_argument(
        "--model-name",
        default=None,
        help=(
            "Nom personnalisé du modèle (optionnel) — intégré au nom de fichier "
            "sauvegardé pour distinguer plusieurs modèles entraînés sur le même "
            "symbole/timeframe (ex. 'essai-a' -> ppo_BTC-USDT_1h_essai-a_<horodatage>.zip)."
        ),
    )
    parser.add_argument(
        "--continue-from-model",
        default=None,
        help=(
            "Chemin (nom de fichier, avec ou sans .zip) d'un modèle déjà "
            "sauvegardé dans storage/models/ dont reprendre l'entraînement, "
            "au lieu de repartir d'un agent vierge. Retombe automatiquement "
            "sur un nouveau départ si le fichier est introuvable ou incompatible."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    outcome = run_training_pipeline(
        config_path=args.config,
        symbol=args.symbol,
        timeframe=args.timeframe,
        total_timesteps=args.total_timesteps,
        continue_from_model=args.continue_from_model,
        model_display_name=args.model_name,
    )

    print("\n=== Résultat backtest test (premier split walk-forward) ===")
    for key, value in outcome["metrics"].items():
        print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
    print(f"model_path: {outcome['model_path']}")
