"""Dashboard Streamlit — TradingBot.

5 onglets, tous complets : "Contrôle" (démarrer/arrêter le téléchargement de
données, l'entraînement RL et le paper trading live — TOUT le pilotage se
fait depuis l'UI, aucune commande terminal requise), "Marché live" (prix +
indicateurs), "Backtest" (walk-forward + test de résilience sur périodes de
stress), "Comparaison stratégies" (baselines + agent RL optionnel),
"Portefeuille simulé" (état du paper trading live lu depuis
storage/runs/paper_trading.db, + monitoring de dégradation). Affiche un
message d'attente explicite plutôt que de planter quand des
données/résultats ne sont pas encore disponibles.

Les tâches longues (téléchargement, entraînement, paper trading live) sont
lancées comme sous-processus OS séparés via app/process_manager.py — jamais
dans le thread Streamlit lui-même (mono-thread, bloquerait l'UI) — cf.
docstring de ce module pour le détail du mécanisme.

Lancement : `streamlit run app/dashboard.py` depuis la racine du projet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from app.process_manager import get_task_logs, get_task_status, start_task, stop_task
from backtest.engine import compare_strategies, run_backtest, run_walk_forward_backtest
from backtest.stress_tests import get_stress_period_data, list_stress_periods
from features.indicators import add_all_indicators
from features.pipeline import build_feature_dataframe, normalize_features
from models.baseline_rules import EmaCrossoverStrategy
from models.buy_and_hold import BuyAndHoldStrategy
from models.momentum_naive import MomentumNaiveStrategy
from models.random_policy import RandomWeightedPolicy
from models.training_progress import get_training_progress
from storage.runs.live_state import (
    DEFAULT_DB_PATH,
    get_latest_snapshot,
    get_recent_decisions,
    get_snapshot_history,
)

st.set_page_config(page_title="TradingBot — Dashboard", layout="wide")
st.title("TradingBot — Dashboard")
st.caption("Paper trading uniquement. Aucun ordre réel n'est jamais passé.")


@st.cache_data
def load_config(config_path: str = "config.yaml") -> dict:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@st.cache_data(show_spinner=False)
def load_market_data(symbol: str, timeframe: str, storage_dir: str) -> pd.DataFrame | None:
    """Lit un fichier Parquet OHLCV (jusqu'à 800k+ lignes pour le 5m).

    Mise en cache OBLIGATOIRE : sans elle, Streamlit relit ce fichier depuis
    le disque à CHAQUE rerun (chaque clic, chaque changement d'onglet — le
    script entier se ré-exécute à chaque interaction), ce qui provoquait une
    latence croissante et une accumulation mémoire notable en usage réel.
    Le cache est invalidé automatiquement si le fichier change (mtime).
    """
    filename = symbol.replace("/", "-") + f"_{timeframe}.parquet"
    path = Path(storage_dir) / filename
    if not path.exists():
        return None
    # mtime dans la clé de cache : un nouveau téléchargement (onglet Contrôle)
    # invalide naturellement l'entrée sans qu'on ait à gérer l'invalidation à la main.
    _ = path.stat().st_mtime
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_market_data_with_indicators(symbol: str, timeframe: str, storage_dir: str) -> pd.DataFrame | None:
    """`load_market_data` + `add_all_indicators`, mis en cache ensemble.

    Évite de recalculer RSI/MACD/Bollinger/ATR/etc. sur des centaines de
    milliers de lignes à chaque interaction dans l'onglet Marché live.
    """
    df = load_market_data(symbol, timeframe, storage_dir)
    if df is None or df.empty:
        return df
    return add_all_indicators(df)


_STRATEGY_FACTORIES: dict[str, Any] = {
    "Buy & Hold": lambda: BuyAndHoldStrategy(),
    "Aléatoire pondérée": lambda: RandomWeightedPolicy(),
    "Croisement EMA": lambda: EmaCrossoverStrategy(),
    "Momentum naïf": lambda: MomentumNaiveStrategy(),
}


@st.cache_data(show_spinner=False)
def _cached_walk_forward_backtest(
    strategy_name: str, symbol: str, timeframe: str, config_hash: int
) -> dict[str, Any]:
    """Cache par (stratégie, symbole, timeframe, config) — évite de relancer un
    backtest walk-forward identique. `config` et `df` ne sont pas hashables
    proprement par Streamlit (dict imbriqué / DataFrame volumineux) : on les
    referme sur les variables globales déjà chargées plutôt que de les passer
    en argument, et on force l'invalidation via un hash explicite de config.
    """
    df_enriched = add_all_indicators(load_market_data(symbol, timeframe, config["history"]["storage_dir"]))
    strategy_factory = _STRATEGY_FACTORIES[strategy_name]
    return run_walk_forward_backtest(df_enriched, strategy_factory, config, symbol=symbol, timeframe=timeframe)


@st.cache_data(show_spinner=False)
def _cached_compare_strategies(
    strategy_names: tuple[str, ...], symbol: str, timeframe: str, config_hash: int
) -> pd.DataFrame:
    """Même logique de cache que `_cached_walk_forward_backtest`, mais pour la
    comparaison multi-stratégies (potentiellement plus longue : N backtests
    walk-forward complets).
    """
    df_enriched = add_all_indicators(load_market_data(symbol, timeframe, config["history"]["storage_dir"]))
    strategies = {name: _STRATEGY_FACTORIES[name] for name in strategy_names if name in _STRATEGY_FACTORIES}
    return compare_strategies(df_enriched, strategies, config, symbol=symbol, timeframe=timeframe)


def compute_rolling_sharpe(value_history_df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Sharpe glissant (non annualisé) sur les `window` derniers points de
    `value_history_df["total_value"]`.

    Même formule que `SimulatedPortfolio.get_metrics` (moyenne des rendements
    / écart-type des rendements, sans facteur d'annualisation) pour rester
    directement comparable aux Sharpe walk-forward affichés dans l'onglet
    Backtest. Retourne une Series indexée comme `value_history_df`, avec NaN
    tant que la fenêtre n'est pas remplie.
    """
    if value_history_df.empty or "total_value" not in value_history_df.columns:
        return pd.Series(dtype=float)
    returns = value_history_df["total_value"].pct_change()
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    return (rolling_mean / rolling_std).where(rolling_std > 0, 0.0)


config = load_config()

with st.sidebar:
    st.header("Paramètres")
    symbol = st.selectbox("Symbole", config["symbols"])
    timeframe = st.selectbox("Timeframe", config["timeframes"])
    st.divider()
    st.info("Téléchargement des données, entraînement de l'agent et paper trading live se pilotent depuis l'onglet **Contrôle**.")
    st.divider()
    st.caption(
        "Les blocs de contrôle (statut/logs de l'onglet « Contrôle ») se "
        "rafraîchissent automatiquement toutes les 2s pendant qu'une tâche "
        "tourne — sans jamais recharger le reste de la page."
    )

tab_control, tab_market, tab_backtest, tab_compare, tab_portfolio = st.tabs(
    ["Contrôle", "Marché live", "Backtest", "Comparaison stratégies", "Portefeuille simulé"]
)

_TASK_LABELS = {
    "never_started": "Jamais lancé",
    "running": "🟢 En cours",
    "finished": "✅ Terminé",
    "error": "🔴 Erreur",
    "stopped": "⏹️ Arrêté",
}


def _render_task_controls(task_name: str, start_label: str, extra_args: list[str] | None = None) -> None:
    """Bloc Start/Stop + statut + logs réutilisé pour les 3 tâches de l'onglet Contrôle.

    Rafraîchissement MANUEL uniquement (bouton dédié) — un premier essai avec
    `st.fragment(run_every=...)` a provoqué un bug de rendu (onglets restant
    vides) sur cette installation, probablement une interaction fragile entre
    plusieurs fragments auto-rafraîchis et `st.tabs`. Revenir à un pattern
    simple et éprouvé (bouton + `st.rerun()` explicite) plutôt que de courir
    après un comportement Streamlit non reproductible ici.
    """
    status = get_task_status(task_name)
    status_label = _TASK_LABELS.get(status["status"], status["status"])

    col_status, col_start, col_stop, col_refresh = st.columns([2, 1, 1, 1])
    col_status.markdown(f"**Statut :** {status_label}")

    if col_start.button("Démarrer", key=f"start_{task_name}", disabled=(status["status"] == "running")):
        try:
            start_task(task_name, extra_args=extra_args)
            st.rerun()
        except RuntimeError as exc:
            st.warning(str(exc))

    if col_stop.button("Arrêter", key=f"stop_{task_name}", disabled=(status["status"] != "running")):
        stop_task(task_name)
        st.rerun()

    if col_refresh.button("🔄 Rafraîchir", key=f"refresh_{task_name}"):
        st.rerun()

    with st.expander("Logs", expanded=(status["status"] == "running")):
        logs = get_task_logs(task_name, tail_lines=80)
        st.code(logs if logs else "(aucun log pour l'instant)", language=None)
        if status["status"] == "running":
            st.caption("Cliquez « 🔄 Rafraîchir » ci-dessus pour voir la progression la plus récente.")

with tab_control:
    st.caption(
        "Pilotage complet du bot depuis l'UI — aucune commande terminal "
        "n'est nécessaire. Chaque tâche tourne comme un processus séparé "
        "(pour ne jamais bloquer le dashboard) ; cochez « Actualisation "
        "automatique » dans la barre latérale pour suivre la progression en direct."
    )

    st.subheader("1. Télécharger les données historiques")
    st.caption("Récupère l'historique OHLCV Binance (BTC/USDT, ETH/USDT) défini dans config.yaml.")
    _render_task_controls("fetch_data", "Télécharger")

    st.divider()
    st.subheader("2. Entraîner l'agent RL")
    st.caption(
        "Entraîne un agent PPO sur le premier split walk-forward (données -> "
        "features -> environnement -> PPO -> backtest de validation)."
    )
    train_col1, train_col2, train_col3 = st.columns(3)
    train_symbol = train_col1.selectbox("Symbole à entraîner", config["symbols"], key="train_symbol")
    train_timeframe = train_col2.selectbox("Timeframe", config["timeframes"], key="train_timeframe")
    train_timesteps = train_col3.number_input(
        "Nombre de pas d'entraînement",
        min_value=1_000,
        max_value=2_000_000,
        value=50_000,
        step=1_000,
        key="train_timesteps",
        help="50 000 = premier run de validation rapide. Augmenter pour un agent plus abouti (plus long).",
    )
    train_model_name = st.text_input(
        "Nom du modèle (optionnel)",
        key="train_model_name",
        placeholder="ex. essai-a, agent-conservateur…",
        help=(
            "Intégré au nom de fichier sauvegardé, pour garder plusieurs "
            "modèles distincts sur le même symbole/timeframe sans qu'ils "
            "s'écrasent. Laissez vide pour un identifiant basé uniquement "
            "sur la date/heure."
        ),
    )

    def _model_label(m: dict) -> str:
        name_part = m["display_name"] or "(sans nom)"
        date_part = m["trained_at"][:16].replace("T", " ") if m["trained_at"] else "date inconnue"
        return f"{name_part} — {m['symbol'] or '?'} {m['timeframe'] or '?'} — {date_part}"

    try:
        from models.rl_agent import list_saved_models

        matching_models = [
            m for m in list_saved_models() if m["symbol"] == train_symbol and m["timeframe"] == train_timeframe
        ]
    except ModuleNotFoundError:
        matching_models = []

    continue_options = ["Aucun (nouveau départ)"] + [_model_label(m) for m in matching_models]
    continue_choice = st.selectbox(
        "Repartir d'un modèle existant",
        continue_options,
        key="continue_from_model_select",
        help=(
            "Reprend l'entraînement du modèle sélectionné (même symbole/"
            "timeframe) au lieu de créer un nouveau cerveau vierge. Sans "
            "effet si les features de ce modèle sont incompatibles avec cet "
            "entraînement (nouveau départ automatique). Attention : continuer "
            "indéfiniment sur les mêmes données d'entraînement peut mener à du "
            "surapprentissage — vérifiez régulièrement via l'onglet « Comparaison "
            "stratégies » que l'agent bat toujours buy-and-hold et la politique "
            "aléatoire, pas seulement qu'il progresse sur sa propre courbe."
        ),
    )
    continue_model_path = None
    if continue_choice != "Aucun (nouveau départ)":
        idx = continue_options.index(continue_choice) - 1
        continue_model_path = str(matching_models[idx]["path"])

    train_extra_args = [
        "--symbol", train_symbol,
        "--timeframe", train_timeframe,
        "--total-timesteps", str(int(train_timesteps)),
    ]
    if train_model_name.strip():
        train_extra_args.extend(["--model-name", train_model_name.strip()])
    if continue_model_path:
        train_extra_args.extend(["--continue-from-model", continue_model_path])
    _render_task_controls("train_agent", "Entraîner", extra_args=train_extra_args)

    train_progress = get_training_progress()
    if train_progress["status"] != "never_started" and train_progress["points"]:
        progress_df = pd.DataFrame(train_progress["points"])
        progress_fig = go.Figure()
        progress_fig.add_trace(
            go.Scatter(
                x=progress_df["step"],
                y=progress_df["mean_reward"],
                mode="lines+markers",
                name="Récompense moyenne",
                line=dict(color="#1baf7a"),
            )
        )
        total_steps = train_progress.get("total_timesteps") or 0
        progress_pct = (progress_df["step"].iloc[-1] / total_steps * 100.0) if total_steps else None
        title = "Courbe d'apprentissage (récompense moyenne par rollout)"
        if progress_pct is not None:
            title += f" — {progress_pct:.0f}% ({train_progress['status']})"
        progress_fig.update_layout(height=320, title=title, xaxis_title="Pas d'entraînement", yaxis_title="Récompense moyenne", margin=dict(t=40, b=10))
        st.plotly_chart(progress_fig, use_container_width=True)
    elif get_task_status("train_agent")["status"] == "running":
        st.info("Entraînement démarré — cliquez « 🔄 Rafraîchir » ci-dessus pour voir la courbe dès la fin du premier rollout PPO (~2048 pas).")

    st.divider()
    st.subheader("3. Paper trading en temps réel")
    st.caption(
        "Lance la boucle de paper trading live (portefeuille 100% simulé, "
        "aucun ordre réel) — décisions et valeur du portefeuille visibles "
        "en direct dans l'onglet « Portefeuille simulé »."
    )
    live_col1, live_col2, live_col3 = st.columns(3)
    live_symbol = live_col1.selectbox("Symbole à trader", config["symbols"], key="live_symbol")
    live_timeframe = live_col2.selectbox("Timeframe", config["timeframes"], key="live_timeframe")
    live_strategy = live_col3.selectbox(
        "Stratégie",
        ["buy_and_hold", "ema_crossover", "momentum", "random", "rl"],
        key="live_strategy",
    )

    live_extra_args = [
        "--symbol", live_symbol,
        "--timeframe", live_timeframe,
        "--strategy", live_strategy,
    ]

    if live_strategy == "rl":
        def _live_model_label(m: dict) -> str:
            name_part = m["display_name"] or "(sans nom)"
            date_part = m["trained_at"][:16].replace("T", " ") if m["trained_at"] else "date inconnue"
            return f"{name_part} — {date_part}"

        try:
            from models.rl_agent import list_saved_models

            live_matching_models = [
                m for m in list_saved_models() if m["symbol"] == live_symbol and m["timeframe"] == live_timeframe
            ]
        except ModuleNotFoundError:
            live_matching_models = []

        if not live_matching_models:
            st.warning(
                f"Aucun modèle RL entraîné trouvé pour {live_symbol} ({live_timeframe}) — "
                "entraînez-en un dans la section 2 avant de démarrer le live avec cette stratégie."
            )
        else:
            live_model_options = [_live_model_label(m) for m in live_matching_models]
            live_model_choice = st.selectbox(
                "Modèle RL à utiliser pour le live trading",
                live_model_options,
                key="live_rl_model_select",
            )
            chosen_model = live_matching_models[live_model_options.index(live_model_choice)]
            live_extra_args.extend(["--rl-model-path", str(chosen_model["path"])])

    _render_task_controls(
        "live_trading",
        "Démarrer le live",
        extra_args=live_extra_args,
    )
    st.caption(
        "⚠️ Cette boucle tourne indéfiniment tant qu'elle n'est pas arrêtée "
        "(bouton « Arrêter » ci-dessus) — elle continue même si vous fermez "
        "cet onglet ou le navigateur."
    )

with tab_market:
    df_ind = load_market_data_with_indicators(symbol, timeframe, config["history"]["storage_dir"])

    if df_ind is None or df_ind.empty:
        st.warning(
            f"Aucune donnée trouvée pour {symbol} ({timeframe}). "
            "Téléchargez l'historique depuis l'onglet « Contrôle » (section 1)."
        )
    else:
        st.subheader(f"{symbol} — {timeframe}")

        # LE bug de latence géante du dashboard : passer les 800k+ lignes du
        # 5m à Plotly fait planter `st.plotly_chart` sur un `deepcopy()`
        # interne qui prend plusieurs MINUTES sur des structures aussi
        # grosses (confirmé via faulthandler.dump_traceback_later — la pile
        # était bloquée exactement dans plotly/basedatatypes.py::to_dict ->
        # copy.deepcopy). Aucun graphique n'est de toute façon lisible avec
        # 800k points affichés simultanément — on limite à une fenêtre
        # récente pour l'affichage ; tous les calculs (backtest, indicateurs)
        # continuent d'utiliser `df_ind` complet, non tronqué.
        max_candles = st.slider(
            "Nombre de bougies affichées (graphique uniquement)",
            min_value=200,
            max_value=5000,
            value=1000,
            step=100,
            key="market_chart_window",
            help="Limité pour la lisibilité et la performance du graphique — n'affecte aucun calcul (backtest, indicateurs), qui utilisent toujours l'historique complet.",
        )
        df_chart = df_ind.tail(max_candles)

        candle_fig = go.Figure()
        candle_fig.add_trace(
            go.Candlestick(
                x=df_chart.index,
                open=df_chart["open"],
                high=df_chart["high"],
                low=df_chart["low"],
                close=df_chart["close"],
                name=symbol,
            )
        )
        candle_fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["bb_upper"], name="BB haute", line=dict(width=1, color="gray"))
        )
        candle_fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["bb_lower"], name="BB basse", line=dict(width=1, color="gray"))
        )
        candle_fig.add_trace(
            go.Scatter(x=df_chart.index, y=df_chart["bb_middle"], name="BB moyenne", line=dict(width=1, color="orange", dash="dot"))
        )
        candle_fig.update_layout(height=500, xaxis_rangeslider_visible=False, margin=dict(t=30, b=10))
        st.plotly_chart(candle_fig, use_container_width=True)

        col_rsi, col_macd = st.columns(2)

        with col_rsi:
            rsi_fig = go.Figure()
            rsi_fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["rsi"], name="RSI", line=dict(color="purple")))
            rsi_fig.add_hline(y=70, line_dash="dot", line_color="red")
            rsi_fig.add_hline(y=30, line_dash="dot", line_color="green")
            rsi_fig.update_layout(height=280, title="RSI (14)", margin=dict(t=40, b=10))
            st.plotly_chart(rsi_fig, use_container_width=True)

        with col_macd:
            macd_fig = go.Figure()
            macd_fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["macd_line"], name="MACD", line=dict(color="blue")))
            macd_fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart["macd_signal"], name="Signal", line=dict(color="orange")))
            macd_fig.add_trace(go.Bar(x=df_chart.index, y=df_chart["macd_histogram"], name="Histogramme"))
            macd_fig.update_layout(height=280, title="MACD (12/26/9)", margin=dict(t=40, b=10))
            st.plotly_chart(macd_fig, use_container_width=True)

with tab_backtest:
    st.caption(
        "Backtest walk-forward (jamais sur le jeu de test gelé) — "
        "backtest.engine.run_walk_forward_backtest."
    )
    strategy_name = st.selectbox("Stratégie", list(_STRATEGY_FACTORIES.keys()), key="backtest_strategy")
    run_clicked = st.button("Lancer le backtest", key="run_backtest_btn")

    metric_cols = st.columns(4)
    metric_labels = [
        ("Rendement", "total_return_pct", "%"),
        ("Sharpe", "sharpe_ratio", ""),
        ("Sortino", "sortino_ratio", ""),
        ("Max Drawdown", "max_drawdown_pct", "%"),
    ]

    if run_clicked:
        df_check = load_market_data(symbol, timeframe, config["history"]["storage_dir"])
        if df_check is None or df_check.empty:
            st.warning(
                f"Aucune donnée trouvée pour {symbol} ({timeframe}). "
                "Téléchargez l'historique depuis l'onglet « Contrôle » (section 1)."
            )
        else:
            try:
                with st.spinner("Backtest walk-forward en cours…"):
                    wf_result = _cached_walk_forward_backtest(
                        strategy_name, symbol, timeframe, hash(str(config))
                    )
                st.session_state["backtest_result"] = wf_result
            except ValueError as exc:
                st.error(f"Backtest impossible : {exc}")
                st.session_state.pop("backtest_result", None)

    wf_result = st.session_state.get("backtest_result")
    if wf_result is not None:
        agg = wf_result["aggregated_metrics"]
        for col, (label, key, suffix) in zip(metric_cols, metric_labels):
            mean_val = agg.get(f"{key}_mean")
            std_val = agg.get(f"{key}_std")
            if mean_val is None:
                col.metric(label, "—")
            else:
                col.metric(label, f"{mean_val:.2f}{suffix}", f"± {std_val:.2f}{suffix}" if std_val is not None else None)

        st.caption(f"{wf_result['num_splits']} split(s) walk-forward — {strategy_name}")

        first_split = wf_result["per_split_results"][0]
        value_curve = first_split.value_curve
        if value_curve:
            curve_df = pd.DataFrame(value_curve, columns=["timestamp", "portfolio_value"])
            curve_fig = go.Figure()
            curve_fig.add_trace(
                go.Scatter(
                    x=curve_df["timestamp"],
                    y=curve_df["portfolio_value"],
                    name="Valeur du portefeuille",
                    line=dict(color="teal"),
                )
            )
            curve_fig.update_layout(
                height=400,
                title="Courbe de valeur du portefeuille — 1er split walk-forward",
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(curve_fig, use_container_width=True)
    else:
        for col, (label, _key, _suffix) in zip(metric_cols, metric_labels):
            col.metric(label, "—")
        st.info("Sélectionnez une stratégie puis cliquez sur « Lancer le backtest ».")

    st.divider()
    st.subheader("Test de résilience sur périodes de crash")
    st.caption(
        "Backtest direct (pas de walk-forward — une période de stress est trop "
        "courte pour un découpage train/test) sur une période de marché extrême "
        "connue, avec la stratégie sélectionnée ci-dessus."
    )

    stress_period_names = list_stress_periods(config)
    if not stress_period_names:
        st.info("Aucune période de stress définie dans config.yaml (section `stress_periods`).")
    else:
        stress_period_name = st.selectbox(
            "Période de stress", stress_period_names, key="stress_period_select"
        )
        stress_run_clicked = st.button("Tester sur cette période", key="run_stress_btn")

        if stress_run_clicked:
            df_check = load_market_data(symbol, timeframe, config["history"]["storage_dir"])
            if df_check is None or df_check.empty:
                st.warning(
                    f"Aucune donnée trouvée pour {symbol} ({timeframe}). "
                    "Téléchargez l'historique depuis l'onglet « Contrôle » (section 1)."
                )
            else:
                try:
                    df_stress = get_stress_period_data(df_check, stress_period_name, config)
                    if df_stress.empty:
                        st.warning(
                            f"Aucune donnée disponible pour la période « {stress_period_name} » "
                            f"sur {symbol} ({timeframe}) — hors de l'historique téléchargé."
                        )
                    else:
                        df_stress_ind = add_all_indicators(df_stress)
                        strategy = _STRATEGY_FACTORIES[strategy_name]()
                        with st.spinner("Backtest sur période de stress en cours…"):
                            stress_result = run_backtest(
                                df_stress_ind,
                                strategy,
                                initial_cash=config["portfolio"]["initial_cash_usdt"],
                                config=config,
                                symbol=symbol,
                                timeframe=timeframe,
                            )
                        st.session_state["stress_result"] = stress_result
                        st.session_state["stress_result_label"] = f"{strategy_name} — {stress_period_name}"
                except ValueError as exc:
                    st.error(f"Test de résilience impossible : {exc}")
                    st.session_state.pop("stress_result", None)

        stress_result = st.session_state.get("stress_result")
        if stress_result is not None:
            st.caption(f"Résultat : {st.session_state.get('stress_result_label', '')}")
            stress_metrics = stress_result.metrics
            max_dd = stress_metrics.get("max_drawdown_pct")

            stress_cols = st.columns(3)
            stress_cols[0].metric("Rendement", f"{stress_metrics.get('total_return_pct', 0.0):.2f}%")
            stress_cols[1].metric("Max Drawdown", f"{max_dd:.2f}%" if max_dd is not None else "—")
            stress_cols[2].metric("Nombre de trades", f"{stress_metrics.get('num_trades', 0):.0f}")

            if max_dd is not None:
                if max_dd > 30:
                    st.error(
                        f"Drawdown maximal de {max_dd:.2f}% — la stratégie « explose » "
                        "en conditions extrêmes."
                    )
                elif max_dd > 15:
                    st.warning(f"Drawdown maximal de {max_dd:.2f}% — résilience limitée.")
                else:
                    st.success(f"Drawdown maximal de {max_dd:.2f}% — résilience correcte sur cette période.")

            stress_value_curve = stress_result.value_curve
            if stress_value_curve:
                stress_curve_df = pd.DataFrame(stress_value_curve, columns=["timestamp", "portfolio_value"])
                stress_curve_fig = go.Figure()
                stress_curve_fig.add_trace(
                    go.Scatter(
                        x=stress_curve_df["timestamp"],
                        y=stress_curve_df["portfolio_value"],
                        name="Valeur du portefeuille",
                        line=dict(color="crimson"),
                    )
                )
                stress_curve_fig.update_layout(
                    height=400,
                    title=f"Courbe de valeur du portefeuille — {stress_period_name}",
                    margin=dict(t=40, b=10),
                )
                st.plotly_chart(stress_curve_fig, use_container_width=True)
        else:
            st.info("Sélectionnez une période de stress puis cliquez sur « Tester sur cette période ».")

        st.caption(
            "Les périodes de stress ne doivent jamais être utilisées pour "
            "entraîner ou ajuster une stratégie — uniquement pour vérifier sa "
            "résilience une fois les décisions de conception déjà prises."
        )

with tab_compare:
    st.caption(
        "Comparaison walk-forward des 4 stratégies baseline "
        "(backtest.engine.compare_strategies)."
    )
    def _compare_model_label(m: dict) -> str:
        name_part = m["display_name"] or "(sans nom)"
        date_part = m["trained_at"][:16].replace("T", " ") if m["trained_at"] else "date inconnue"
        return f"{name_part} — {m['symbol'] or '?'} {m['timeframe'] or '?'} — {date_part}"

    try:
        from models.rl_agent import list_saved_models

        all_saved_models = list_saved_models()
    except ModuleNotFoundError:
        all_saved_models = []

    rl_model_labels = [_compare_model_label(m) for m in all_saved_models]
    selected_rl_labels = st.multiselect(
        "Modèles RL à inclure dans la comparaison",
        rl_model_labels,
        key="compare_rl_models_select",
        help="Chaque modèle sélectionné apparaît comme une ligne distincte dans le tableau (ex. « Agent RL — essai-a »).",
    )
    if not all_saved_models:
        st.caption("Aucun modèle entraîné trouvé — entraînez-en un dans la section 2 de l'onglet Contrôle.")

    compare_clicked = st.button("Comparer toutes les stratégies", key="compare_btn")

    if compare_clicked:
        df_check = load_market_data(symbol, timeframe, config["history"]["storage_dir"])
        if df_check is None or df_check.empty:
            st.warning(
                f"Aucune donnée trouvée pour {symbol} ({timeframe}). "
                "Téléchargez l'historique depuis l'onglet « Contrôle » (section 1)."
            )
        else:
            strategy_names = list(_STRATEGY_FACTORIES.keys())

            # Une entrée (nom d'affichage, stratégie chargée) par modèle RL
            # coché — remplace l'ancien comportement "le dernier fichier du
            # dossier" par un choix explicite, potentiellement multiple.
            rl_entries: list[tuple[str, Any]] = []
            if selected_rl_labels:
                try:
                    from models.rl_agent import RLAgentStrategy, load_agent

                    for label in selected_rl_labels:
                        model_info = all_saved_models[rl_model_labels.index(label)]
                        if not model_info.get("feature_columns"):
                            st.warning(
                                f"Métadonnées introuvables pour {model_info['filename']} "
                                "(feature_columns/window_size) — modèle probablement sauvegardé "
                                "avant l'ajout du sidecar de métadonnées. Ré-entraînez pour "
                                "régénérer un modèle utilisable ici."
                            )
                            continue
                        if model_info["symbol"] != symbol or model_info["timeframe"] != timeframe:
                            # Vérification explicite AVANT de charger/exécuter le
                            # modèle : sans ça, le mismatch ne se révèle qu'au
                            # moment où RLAgentStrategy lève un KeyError sur une
                            # colonne de features absente (ex. le modèle a été
                            # entraîné en 1h — où fear_greed_value est inclus —
                            # mais la comparaison tourne en 5m — où cette colonne
                            # est volontairement exclue, cf. features/pipeline.py)
                            # — un message technique difficile à interpréter pour
                            # un mismatch qui est en réalité juste "mauvais
                            # symbole/timeframe sélectionné dans la sidebar".
                            st.warning(
                                f"« {label} » a été entraîné sur {model_info['symbol']} "
                                f"({model_info['timeframe']}) — sélectionnez ce même symbole/"
                                f"timeframe dans la barre latérale pour l'inclure dans la "
                                f"comparaison (actuellement : {symbol}, {timeframe})."
                            )
                            continue
                        agent = load_agent(model_info["path"])
                        strategy = RLAgentStrategy(
                            agent,
                            feature_columns=model_info["feature_columns"],
                            window_size=model_info.get("window_size") or 20,
                        )
                        display_label = model_info["display_name"] or model_info["filename"]
                        rl_entries.append((f"Agent RL — {display_label}", strategy))
                except ModuleNotFoundError:
                    st.warning(
                        "Module models.rl_agent introuvable — entraînez d'abord un agent RL "
                        "depuis l'onglet Contrôle."
                    )
                except Exception as exc:  # chargement modèle : frontière réelle, on ne veut pas planter le dashboard
                    st.warning(f"Impossible de charger un agent RL : {exc}")

            try:
                with st.spinner("Comparaison des stratégies en cours (peut prendre plusieurs minutes)…"):
                    comparison_df = _cached_compare_strategies(
                        tuple(strategy_names), symbol, timeframe, hash(str(config))
                    )
                    if rl_entries:
                        # Chaque agent RL attend EXACTEMENT les colonnes de
                        # feature_columns (déjà normalisées, même ORDRE) sur
                        # lesquelles il a été entraîné — un simple
                        # add_all_indicators() ne suffit pas (ni régime HMM, ni
                        # Fear & Greed/funding/ratio ETH-BTC, ni normalisation).
                        # On reconstruit donc le même pipeline de features ici,
                        # y COMPRIS le chargement ETH/USDT (même logique que
                        # models/train_pipeline.py::run_training_pipeline) —
                        # sans ça, `eth_btc_ratio` est absent des colonnes
                        # reconstruites, ce qui décale tout le vecteur de
                        # features et fait échouer l'agent RL avec un KeyError.
                        # LIMITATION CONNUE : les statistiques de normalisation
                        # train utilisées à l'entraînement (par split
                        # walk-forward) ne sont pas persistées aux côtés du
                        # modèle. On recalcule donc des stats sur `df_check`
                        # lui-même en approximation — acceptable pour une
                        # comparaison exploratoire dans le dashboard, mais ce
                        # n'est PAS rigoureusement identique aux stats
                        # d'entraînement d'origine. Pour une évaluation
                        # rigoureuse, utiliser models/train_pipeline.py.
                        # Construite UNE FOIS (même pour tous les modèles du
                        # même symbole/timeframe — le pipeline de features ne
                        # dépend pas du modèle lui-même, seulement des données).
                        df_eth_for_rl = None
                        if symbol == "BTC/USDT" and "ETH/USDT" in config.get("symbols", []):
                            df_eth_for_rl = load_market_data("ETH/USDT", timeframe, config["history"]["storage_dir"])

                        rl_features_df, rl_feature_columns = build_feature_dataframe(
                            df_check,
                            df_eth_ohlcv=df_eth_for_rl,
                            config=config,
                            timeframe=timeframe,
                        )
                        rl_features_df = rl_features_df.dropna(subset=[*rl_feature_columns, "atr"])
                        rl_features_df, _ = normalize_features(rl_features_df, rl_feature_columns)

                        rl_rows: dict[str, dict[str, float]] = {}
                        for entry_label, strategy in rl_entries:
                            try:
                                rl_result = run_walk_forward_backtest(
                                    rl_features_df,
                                    lambda s=strategy: s,
                                    config,
                                    symbol=symbol,
                                    timeframe=timeframe,
                                )
                                rl_rows[entry_label] = rl_result["aggregated_metrics"]
                            except KeyError as exc:
                                st.warning(
                                    f"« {entry_label} » incompatible avec les features "
                                    f"disponibles ({exc}) — modèle probablement entraîné avec "
                                    "des données de contexte différentes. Exclu de la comparaison."
                                )
                        if rl_rows:
                            comparison_df = pd.concat(
                                [comparison_df, pd.DataFrame.from_dict(rl_rows, orient="index")]
                            )
                st.session_state["comparison_df"] = comparison_df
            except ValueError as exc:
                st.error(f"Comparaison impossible : {exc}")
                st.session_state.pop("comparison_df", None)
            except KeyError as exc:
                # Filet de sécurité générique : chaque modèle RL gère déjà son
                # propre KeyError individuellement dans la boucle ci-dessus
                # (un modèle incompatible n'empêche pas les autres) — ce bloc
                # ne couvre plus qu'un échec inattendu ailleurs dans le flux.
                st.warning(f"Erreur inattendue liée aux features ({exc}) — comparaison des baselines affichée seule.")
                st.session_state["comparison_df"] = comparison_df

    comparison_df = st.session_state.get("comparison_df")
    if comparison_df is not None:
        st.dataframe(comparison_df, use_container_width=True)

        if "sharpe_ratio_mean" in comparison_df.columns:
            _CATEGORICAL_HUES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
            bar_colors = [_CATEGORICAL_HUES[i % len(_CATEGORICAL_HUES)] for i in range(len(comparison_df))]
            bar_fig = go.Figure()
            bar_fig.add_trace(
                go.Bar(
                    x=comparison_df.index,
                    y=comparison_df["sharpe_ratio_mean"],
                    marker_color=bar_colors,
                    text=[f"{v:.2f}" for v in comparison_df["sharpe_ratio_mean"]],
                    textposition="outside",
                    name="Sharpe moyen",
                )
            )
            bar_fig.update_layout(
                height=380,
                title="Sharpe ratio moyen par stratégie (walk-forward)",
                yaxis_title="Sharpe ratio",
                margin=dict(t=40, b=10),
                showlegend=False,
            )
            st.plotly_chart(bar_fig, use_container_width=True)
    else:
        st.dataframe(
            pd.DataFrame(columns=["Stratégie", "Rendement %", "Sharpe", "Sortino", "Max Drawdown %", "Win rate %"]),
            use_container_width=True,
        )

    st.caption(
        "Rappel méthodologique : une stratégie n'a un edge réel que si elle bat "
        "SIMULTANÉMENT Buy & Hold, la politique aléatoire pondérée, ET au moins "
        "une baseline de règles (EMA crossover ou momentum naïf) — sur le même "
        "découpage walk-forward, jamais sur le jeu de test gelé."
    )

with tab_portfolio:
    st.caption(
        "État du portefeuille simulé en paper trading live "
        "(paper_trading.portfolio.SimulatedPortfolio), tel qu'enregistré par "
        "le processus de paper trading (onglet « Contrôle », section 3) dans "
        f"`{DEFAULT_DB_PATH}`. Aucun ordre réel n'est jamais passé — la boucle "
        "live ne fait que lire des données de marché publiques et simuler des "
        "décisions dessus."
    )

    refresh_clicked = st.button("Rafraîchir", key="portfolio_refresh_btn")
    st.caption(
        "Cliquez sur « Rafraîchir » pour relire le dernier état depuis "
        "SQLite, ou cochez « Actualisation automatique » dans la barre "
        "latérale pour un suivi en direct sans action manuelle."
    )

    latest_snapshot = get_latest_snapshot(DEFAULT_DB_PATH)

    if latest_snapshot is None:
        st.info(
            "Aucun état de paper trading live trouvé. Démarrez-le depuis "
            "l'onglet « Contrôle » (section 3 — Paper trading en temps réel)."
        )
    else:
        state_cols = st.columns(3)
        state_cols[0].metric("Valeur totale", f"{latest_snapshot['total_value']:.2f} USDT")
        state_cols[1].metric("Cash", f"{latest_snapshot['cash']:.2f} USDT")
        positions = latest_snapshot["positions"] or {}
        open_positions = {sym: qty for sym, qty in positions.items() if qty}
        if open_positions:
            positions_summary = ", ".join(f"{sym}: {qty:g}" for sym, qty in open_positions.items())
            state_cols[2].metric("Positions ouvertes", str(len(open_positions)))
            st.caption(f"Détail des positions : {positions_summary}")
        else:
            state_cols[2].metric("Positions ouvertes", "0")
        st.caption(
            f"Dernier snapshot : {latest_snapshot['timestamp']} — "
            f"Drawdown courant : {latest_snapshot['drawdown_pct']:.2f}%"
        )

    st.divider()
    st.subheader("Courbe de valeur du portefeuille (live)")

    history_df = get_snapshot_history(DEFAULT_DB_PATH)
    if history_df.empty:
        st.info("Aucun historique de valeur disponible pour le moment.")
    else:
        history_fig = go.Figure()
        history_fig.add_trace(
            go.Scatter(
                x=history_df["timestamp"],
                y=history_df["total_value"],
                mode="lines",
                name="Valeur du portefeuille (live)",
                line=dict(color="teal"),
            )
        )
        history_fig.update_layout(
            height=380,
            xaxis_title="Date",
            yaxis_title="Valeur (USDT)",
            margin=dict(t=30, b=10),
        )
        st.plotly_chart(history_fig, use_container_width=True)

    st.divider()
    st.subheader("Journal des décisions récentes")

    decisions_df = get_recent_decisions(DEFAULT_DB_PATH)
    if decisions_df.empty:
        st.info("Aucune décision enregistrée pour le moment.")
    else:
        st.dataframe(
            decisions_df[["timestamp", "symbol", "action", "price", "reasoning"]],
            use_container_width=True,
        )

    st.divider()
    st.subheader("Monitoring de dégradation — Sharpe glissant")
    st.caption(
        "Sharpe glissant réalisé en paper trading live, comparé à la "
        "distribution des Sharpe obtenus lors des splits walk-forward du "
        "dernier backtest de la même stratégie (onglet Backtest)."
    )

    rolling_window = st.slider(
        "Fenêtre du Sharpe glissant (nb de points)", min_value=5, max_value=100, value=30, key="rolling_sharpe_window"
    )

    if history_df.empty or len(history_df) < rolling_window + 1:
        st.info(
            "Pas assez de points dans l'historique live pour calculer un "
            f"Sharpe glissant sur une fenêtre de {rolling_window} — "
            "laissez le live runner tourner plus longtemps."
        )
    else:
        rolling_sharpe = compute_rolling_sharpe(history_df, window=rolling_window)
        rolling_sharpe_valid = rolling_sharpe.dropna()
        current_rolling_sharpe = rolling_sharpe_valid.iloc[-1] if not rolling_sharpe_valid.empty else None

        sharpe_fig = go.Figure()
        sharpe_fig.add_trace(
            go.Scatter(
                x=history_df["timestamp"],
                y=rolling_sharpe,
                mode="lines",
                name=f"Sharpe glissant ({rolling_window} pts)",
                line=dict(color="#2a78d6"),
            )
        )

        # Référence backtest : distribution des Sharpe par split walk-forward
        # de la même stratégie, si un backtest a déjà été lancé dans l'onglet
        # Backtest de cette session (st.session_state["backtest_result"]).
        wf_reference = st.session_state.get("backtest_result")
        split_sharpes: list[float] = []
        if wf_reference is not None:
            split_sharpes = [
                split.metrics["sharpe_ratio"]
                for split in wf_reference["per_split_results"]
                if "sharpe_ratio" in split.metrics
            ]

        if not split_sharpes:
            st.info(
                "Aucune référence de backtest disponible pour comparaison — "
                "lancez d'abord un backtest walk-forward pour la stratégie "
                "correspondante dans l'onglet « Backtest »."
            )
            sharpe_fig.update_layout(
                height=350,
                title="Sharpe glissant (live) — pas de référence backtest",
                yaxis_title="Sharpe",
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(sharpe_fig, use_container_width=True)
        else:
            backtest_min_sharpe = min(split_sharpes)
            backtest_mean_sharpe = sum(split_sharpes) / len(split_sharpes)

            sharpe_fig.add_hline(
                y=backtest_min_sharpe,
                line_dash="dot",
                line_color="crimson",
                annotation_text="Min backtest walk-forward",
            )
            sharpe_fig.add_hline(
                y=backtest_mean_sharpe,
                line_dash="dash",
                line_color="gray",
                annotation_text="Moyenne backtest walk-forward",
            )
            sharpe_fig.update_layout(
                height=350,
                title=f"Sharpe glissant (live) vs référence backtest — {strategy_name}",
                yaxis_title="Sharpe",
                margin=dict(t=40, b=10),
            )
            st.plotly_chart(sharpe_fig, use_container_width=True)

            ref_cols = st.columns(3)
            ref_cols[0].metric(
                "Sharpe glissant live (dernier)",
                f"{current_rolling_sharpe:.2f}" if current_rolling_sharpe is not None else "—",
            )
            ref_cols[1].metric("Min backtest walk-forward", f"{backtest_min_sharpe:.2f}")
            ref_cols[2].metric("Moyenne backtest walk-forward", f"{backtest_mean_sharpe:.2f}")

            st.caption(
                "Simplification assumée pour cette V1 : le seuil d'alerte est "
                "le MINIMUM observé sur les splits walk-forward (pas un vrai "
                "calcul de percentile statistique) — un dépassement sous ce "
                "minimum déclenche une alerte visuelle simple."
            )

            if current_rolling_sharpe is not None:
                if current_rolling_sharpe < backtest_min_sharpe:
                    st.error(
                        f"Le Sharpe glissant live ({current_rolling_sharpe:.2f}) est "
                        f"passé sous le minimum observé en backtest walk-forward "
                        f"({backtest_min_sharpe:.2f}) — possible dégradation de la "
                        "stratégie en conditions réelles, à surveiller."
                    )
                elif current_rolling_sharpe < backtest_mean_sharpe:
                    st.warning(
                        f"Le Sharpe glissant live ({current_rolling_sharpe:.2f}) est "
                        f"sous la moyenne backtest walk-forward "
                        f"({backtest_mean_sharpe:.2f}), mais reste au-dessus du "
                        "minimum observé — à surveiller sans alerte critique."
                    )
                else:
                    st.success(
                        f"Le Sharpe glissant live ({current_rolling_sharpe:.2f}) est "
                        "conforme à ce qui était observé en backtest walk-forward."
                    )

    st.caption(
        "Pas de détecteur de drift formel (ADWIN/Page-Hinkley) dans cette V1 "
        "— un monitoring statistique simple (Sharpe glissant vs distribution "
        "backtest) est jugé suffisant pour un outil d'expérimentation "
        "personnel. Rappel : le live runner tourne en boucle infinie tant "
        "qu'il n'est pas arrêté depuis l'onglet « Contrôle » — pas de reprise "
        "automatique après un crash ou une extinction du PC."
    )
