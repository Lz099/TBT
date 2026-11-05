"""Agent RL (PPO / Stable-Baselines3) sur `env.trading_env.TradingEnv`.

Décisions actées (cf. contexte projet) :
- Action space discret (Hold/Buy/Sell) — aucun sizing continu dans l'action
  de l'agent, le dimensionnement de position reste dans risk/position_sizing.py.
- Encodeur MLP simple par défaut (baseline obligatoire). L'interface reste
  interchangeable : `create_ppo_agent(..., policy=...)` accepte n'importe
  quelle policy Stable-Baselines3 (nom enregistré ou classe custom), pour
  brancher un encodeur TCN en option plus tard sans réécrire cette fonction.

Ce module ne fait QUE instancier/entraîner/sauvegarder/charger l'agent et
l'adapter à l'interface commune `decide()`/`reset()` des autres stratégies
de `models/` — aucune logique de trading ici (déjà dans env/trading_env.py,
risk/, paper_trading/).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback

# Dossier de sauvegarde par convention (créé à la demande dans save_agent).
_DEFAULT_MODELS_DIR = Path("storage/models")


def create_ppo_agent(
    env: Any,
    config: dict[str, Any],
    policy: str | type = "MlpPolicy",
    **ppo_kwargs: Any,
) -> PPO:
    """Instancie un agent PPO sur `env` avec des hyperparamètres par défaut
    raisonnables pour un problème de RL financier discret (Hold/Buy/Sell) sur
    GPU laptop.

    `policy` est délibérément exposé (par défaut "MlpPolicy", l'encodeur MLP
    simple retenu comme baseline) pour permettre de brancher plus tard une
    policy custom (ex. un encodeur TCN via un `features_extractor_class`
    personnalisé passé dans `policy_kwargs`, ou une classe de policy dédiée)
    sans jamais devoir réécrire cette fonction — seul l'appelant change.

    `config` (dict de config.yaml) n'est pas directement consommé par PPO
    aujourd'hui mais est accepté en signature pour que l'appelant puisse
    toujours le transmettre uniformément (ex. de futurs hyperparamètres
    dérivés de config.yaml, cohérent avec le reste du projet où `config` est
    systématiquement passé en paramètre plutôt que lu en global).

    `**ppo_kwargs` permet de surcharger n'importe quel défaut (ex.
    `learning_rate=1e-4`) sans modifier cette fonction.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    default_kwargs: dict[str, Any] = {
        "learning_rate": 3e-4,
        "n_steps": 2048,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,  # encourage un minimum d'exploration (3 actions discrètes seulement)
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "device": device,
        "verbose": 1,
    }
    default_kwargs.update(ppo_kwargs)

    return PPO(policy, env, **default_kwargs)


def train_agent(
    agent: BaseAlgorithm,
    total_timesteps: int,
    callback: BaseCallback | None = None,
) -> BaseAlgorithm:
    """Entraîne `agent` pendant `total_timesteps` pas et le retourne (même objet, muté en place)."""
    agent.learn(total_timesteps=total_timesteps, callback=callback)
    return agent


def save_agent(
    agent: BaseAlgorithm,
    path: str | Path,
    feature_columns: list[str] | None = None,
    window_size: int | None = None,
    display_name: str | None = None,
    trained_at: str | None = None,
) -> Path:
    """Sauvegarde `agent` au format Stable-Baselines3 (.zip).

    Convention : `storage/models/{nom}.zip`. Si `path` n'a pas de dossier
    parent explicite (juste un nom de fichier/modèle), il est résolu sous
    `storage/models/`. Crée le(s) dossier(s) manquant(s).

    Si `feature_columns`/`window_size` sont fournis, un fichier sidecar
    `{nom}.json` est écrit à côté du `.zip` (même nom, extension différente)
    avec ces métadonnées. Nécessaire car `RLAgentStrategy` a besoin de
    connaître exactement les colonnes de features (et leur ordre) et la
    taille de fenêtre utilisées à l'entraînement pour reconstruire des
    observations compatibles — cette information n'est pas stockée par
    `PPO.save` lui-même. Optionnel pour rester rétrocompatible avec du code
    qui n'a pas encore ces valeurs sous la main.

    `display_name`/`trained_at` (optionnels) sont écrits dans le même
    sidecar — utilisés par `list_saved_models` pour afficher un libellé
    humain dans les sélecteurs du dashboard (multi-modèles) sans avoir à
    re-parser le nom de fichier. `trained_at` doit être fourni par
    l'appelant (horodatage ISO) plutôt que généré ici, pour rester une
    fonction pure sans horloge cachée.
    """
    path = Path(path)
    if path.parent == Path("."):
        path = _DEFAULT_MODELS_DIR / path
    if path.suffix != ".zip":
        path = path.with_suffix(".zip")

    path.parent.mkdir(parents=True, exist_ok=True)
    agent.save(str(path.with_suffix("")))  # PPO.save ajoute lui-même l'extension .zip

    if feature_columns is not None or window_size is not None or display_name is not None or trained_at is not None:
        metadata = {
            "feature_columns": feature_columns,
            "window_size": window_size,
            "display_name": display_name,
            "trained_at": trained_at,
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return path


def load_agent(path: str | Path, env: Any | None = None) -> PPO:
    """Charge un agent PPO sauvegardé par `save_agent`/`PPO.save`.

    `env` est optionnel (nécessaire seulement si l'agent va continuer à être
    entraîné ou utilisé via `.predict()` avec un env vectorisé attaché ;
    `RLAgentStrategy` ci-dessous n'a pas besoin d'un env attaché, il appelle
    `agent.predict()` directement sur des observations construites à la main).
    """
    path = Path(path)
    if path.parent == Path(".") and not path.exists():
        candidate = _DEFAULT_MODELS_DIR / path
        if candidate.exists() or candidate.with_suffix(".zip").exists():
            path = candidate
    return PPO.load(str(path.with_suffix("")), env=env)


def load_agent_metadata(path: str | Path) -> dict[str, Any] | None:
    """Charge le sidecar JSON (`feature_columns`, `window_size`) écrit par
    `save_agent`, si présent. Retourne None si aucun sidecar n'existe (ex.
    modèle sauvegardé par une version antérieure de `save_agent` sans
    métadonnées) — à l'appelant de gérer ce cas plutôt que de deviner.
    """
    path = Path(path)
    if path.parent == Path(".") and not path.exists():
        candidate = _DEFAULT_MODELS_DIR / path
        if candidate.exists() or candidate.with_suffix(".json").exists():
            path = candidate
    json_path = path.with_suffix(".json")
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text(encoding="utf-8"))


def list_saved_models(models_dir: str | Path = _DEFAULT_MODELS_DIR) -> list[dict[str, Any]]:
    """Liste tous les modèles PPO sauvegardés dans `models_dir`, avec leurs métadonnées.

    Retourne une liste de dicts : `path` (Path du .zip), `filename` (str),
    `display_name` (str | None, tel que fourni à l'entraînement), `symbol`,
    `timeframe` (déduits du sidecar si présent, sinon parsés depuis le nom
    de fichier — convention `ppo_{SYMBOL}_{timeframe}[_...suffixe...]`),
    `trained_at` (str ISO | None), `feature_columns`, `window_size`.

    Rétrocompatible avec les modèles sauvegardés avant l'ajout de
    `display_name`/`trained_at` (sidecar absent ou incomplet) : ces champs
    valent alors None plutôt que de lever une erreur. Triée par
    `trained_at` décroissant (les modèles sans horodatage, les plus anciens
    au sens de cette convention, sont placés en fin de liste).
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    entries: list[dict[str, Any]] = []
    for zip_path in sorted(models_dir.glob("*.zip")):
        metadata = load_agent_metadata(zip_path) or {}

        symbol, timeframe = _parse_symbol_timeframe_from_filename(zip_path.stem)

        entries.append(
            {
                "path": zip_path,
                "filename": zip_path.name,
                "display_name": metadata.get("display_name"),
                "symbol": symbol,
                "timeframe": timeframe,
                "trained_at": metadata.get("trained_at"),
                "feature_columns": metadata.get("feature_columns"),
                "window_size": metadata.get("window_size"),
            }
        )

    entries.sort(key=lambda e: e["trained_at"] or "", reverse=True)
    return entries


def _parse_symbol_timeframe_from_filename(stem: str) -> tuple[str | None, str | None]:
    """Déduit (symbol, timeframe) du nom de fichier convention `ppo_{SYM}_{tf}[_...]`.

    Ex. "ppo_BTC-USDT_1h" -> ("BTC/USDT", "1h") ; "ppo_BTC-USDT_1h_essai-a_20260829-1420"
    -> ("BTC/USDT", "1h") aussi (le suffixe nom/horodatage est ignoré ici,
    seul le sidecar JSON fait foi pour le display_name réel). Retourne
    (None, None) si le nom ne suit pas la convention (garde défensive,
    n'empêche pas `list_saved_models` de fonctionner sur un fichier inattendu).
    """
    parts = stem.split("_")
    if len(parts) < 3 or parts[0] != "ppo":
        return None, None
    symbol = parts[1].replace("-", "/")
    timeframe = parts[2]
    return symbol, timeframe


class RLAgentStrategy:
    """Adapte un agent PPO entraîné à l'interface commune `decide()`/`reset()`
    utilisée par toutes les stratégies de `models/` (cf. buy_and_hold.py,
    random_policy.py, baseline_rules.py, momentum_naive.py) — pour que
    `backtest.engine.compare_strategies` puisse comparer l'agent RL aux
    baselines SANS modifier backtest/engine.py, qui n'attend qu'un objet
    `decide(observation) -> str` / `reset() -> None`.

    ADAPTATION DÉLICATE (le point central de cette classe) : le protocole
    backtest.engine.run_backtest appelle `strategy.decide(row)` avec `row`
    = UNE SEULE LIGNE de DataFrame (une Series pandas, ex. `row["close"]`,
    `row["rsi"]`, ...) à chaque pas. L'agent PPO, lui, a été entraîné sur
    `env.trading_env.TradingEnv`, dont l'observation Gymnasium est une
    FENÊTRE GLISSANTE de forme `(window_size, len(feature_columns))` —
    plusieurs pas de temps empilés, features déjà normalisées.

    Ces deux formats sont incompatibles terme à terme. `RLAgentStrategy`
    fait le pont en maintenant en interne un buffer glissant (deque) des
    `window_size` dernières lignes vues au fil des appels successifs à
    `decide()` : chaque appel ajoute `observation[feature_columns]` au
    buffer, puis reconstruit un tableau `(window_size, n_features)`
    équivalent à ce que `TradingEnv._get_observation()` aurait produit, et
    c'est CE tableau (pas la ligne brute) qui est passé à `agent.predict()`.

    Tant que le buffer n'est pas encore plein (tout début du backtest, moins
    de `window_size` lignes vues), la stratégie retourne "HOLD" par défaut
    plutôt que de deviner une observation incomplète ou de padder
    arbitrairement — contrairement à `TradingEnv` qui, lui, peut se permettre
    un padding par répétition parce qu'il connaît toute la série dès sa
    construction ; ici, `decide()` ne voit les lignes qu'une par une, dans
    l'ordre, sans accès au futur ni au passé au-delà de ce qu'il a déjà
    accumulé — donc pas de raccourci équivalent possible sans tricher.

    Les features doivent déjà être normalisées en amont (mêmes stats
    train que celles utilisées à l'entraînement) avant d'être passées à
    `decide()` — cette classe ne renormalise rien, elle ne fait que
    ré-empiler des lignes déjà prêtes.
    """

    def __init__(
        self,
        agent: BaseAlgorithm,
        feature_columns: list[str],
        window_size: int = 20,
        deterministic: bool = True,
    ) -> None:
        self._agent = agent
        self._feature_columns = list(feature_columns)
        self._window_size = window_size
        self._deterministic = deterministic
        self._buffer: list[np.ndarray] = []

    def decide(self, observation: Any) -> str:
        try:
            row_values = np.array(
                [self._extract(observation, col) for col in self._feature_columns],
                dtype=np.float32,
            )
        except KeyError as exc:
            raise KeyError(
                f"RLAgentStrategy attend les colonnes {self._feature_columns} dans "
                f"observation (mêmes features que celles utilisées à l'entraînement, "
                f"déjà normalisées) — colonne manquante: {exc}"
            ) from exc

        self._buffer.append(row_values)
        if len(self._buffer) > self._window_size:
            self._buffer.pop(0)

        if len(self._buffer) < self._window_size:
            # Buffer pas encore plein : pas assez d'historique accumulé pour
            # reconstruire une observation Gymnasium-compatible sans deviner
            # (voir docstring de classe) — HOLD par défaut, plus sûr.
            return "HOLD"

        window = np.stack(self._buffer, axis=0).astype(np.float32)  # (window_size, n_features)
        action, _state = self._agent.predict(window, deterministic=self._deterministic)
        action_int = int(action)

        if action_int == 1:
            return "BUY"
        if action_int == 2:
            return "SELL"
        return "HOLD"

    def reset(self) -> None:
        self._buffer = []

    @staticmethod
    def _extract(observation: Any, column: str) -> float:
        """Lit `observation[column]`, compatible Series pandas ou dict-like."""
        if isinstance(observation, pd.Series):
            if column not in observation.index:
                raise KeyError(column)
            return float(observation[column])
        return float(observation[column])
