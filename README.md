# TradingBot

> ⚠️ **PAPER TRADING UNIQUEMENT.** Ce projet est expérimental et éducatif.
> Aucune ligne de code ici ne passe, ni ne peut passer, un ordre réel sur un
> compte réel : aucune clé API de trading n'est utilisée, uniquement des
> endpoints publics de données (klines, funding, order book en lecture).
> Le portefeuille est entièrement simulé, en mémoire. Ce n'est pas un
> conseil financier ni un produit destiné à trader de l'argent réel.

Bot de trading crypto qui s'entraîne sur plusieurs années d'historique de
marché, puis tourne en paper trading (faux portefeuille) sur les données
Binance en temps réel, avec un dashboard local pour suivre les décisions et
comparer plusieurs stratégies entre elles.

## Contenu du dépôt et état des résultats

- **Pas de données ni de modèles inclus.** L'historique de marché (Parquet) et les modèles PPO entraînés ne sont pas versionnés : `python data/fetch_historical.py` retélécharge les données, `python models/train_pipeline.py` régénère un modèle.
- **Aucun résultat chiffré publié.** Le code permet de comparer l'agent PPO, des baselines et Buy & Hold en walk-forward, mais ce dépôt n'affirme pas que l'agent les bat. L'agent n'est entraîné et évalué que sur le premier split (voir « État du projet »).
- **Pas d'historique de paper trading publié** : la base SQLite est créée à l'exécution du live runner.
- **Pas de tests automatisés** pour l'instant ; `backtest/validation.py::assert_no_lookahead` est une vérification à l'exécution.
- **Sources de données** : API publiques de Binance (via `ccxt`) et Fear & Greed Index d'[alternative.me](https://alternative.me/crypto/fear-and-greed-index/). Les données restent soumises aux conditions de ces services.
- **Licence** : MIT, voir `LICENSE`.

## Prérequis

- Python 3.12+
- `pip install -r requirements.txt`
- **GPU NVIDIA (recommandé pour l'entraînement RL)** : l'étape ci-dessus installe torch en version CPU-only par défaut, même sur une machine avec GPU — voir les instructions d'installation CUDA détaillées en commentaire de tête de `requirements.txt` (section Deep learning / RL). Sans ce fix, l'entraînement tourne sur CPU (fonctionnel mais lent, et fait chauffer le CPU inutilement sur une machine équipée d'un GPU).

## Architecture

```
data/            récupération de données Binance (historique + temps réel)
features/        indicateurs techniques, signaux de contexte, détection de régime, pipeline d'assemblage
env/             environnement Gymnasium (TradingEnv) pour l'agent RL
risk/            dimensionnement de position, stops adaptatifs, contraintes d'exposition
models/          stratégies (baselines + agent RL PPO + script d'entraînement)
backtest/        moteur de backtest, validation walk-forward, coûts réalistes, stress tests
paper_trading/   portefeuille simulé + boucle temps réel (Vague 3)
app/             dashboard Streamlit
config.yaml      configuration centrale (symboles, timeframes, coûts, risque, validation)
```

Toute constante de simulation (frais, seuils, périodes, multiplicateurs)
vit dans `config.yaml`, jamais en dur dans le code — pour que les backtests
restent reproductibles et ajustables sans toucher au code Python.

## Démarrage rapide (données + dashboard)

1. Télécharger l'historique de marché :
   ```
   python data/fetch_historical.py
   ```
2. Lancer le dashboard :
   ```
   streamlit run app/dashboard.py
   ```
   L'onglet **Marché live** affiche le prix (chandelier + Bandes de
   Bollinger), le RSI et le MACD pour les symboles/timeframes définis dans
   `config.yaml`. Les onglets **Backtest** et **Comparaison stratégies**
   sont câblés sur `backtest.engine` (voir « Entraîner un agent RL »
   ci-dessous pour y inclure un agent RL). L'onglet **Backtest** inclut
   aussi un test de résilience sur périodes de crash historiques connues
   (`backtest/stress_tests.py`, jamais utilisées pour entraîner/ajuster une
   stratégie). L'onglet **Portefeuille simulé** affiche l'état du paper
   trading live (voir « Lancer le paper trading en temps réel » ci-dessous).

3. (Optionnel) Collecter les données dérivés (funding rate, open interest) —
   utilisé par le pipeline de features (`features/pipeline.py`) comme signal
   d'entrée pour l'agent RL si le fichier est présent :
   ```
   python data/fetch_derivatives.py
   ```

## Entraîner un agent RL (Vague 2)

1. S'assurer que l'historique OHLCV est téléchargé (`python data/fetch_historical.py`,
   voir étape 1 ci-dessus) — et optionnellement le funding rate
   (`python data/fetch_derivatives.py`) pour enrichir les features.

2. Lancer l'entraînement de bout en bout (données → features → `TradingEnv` →
   PPO → backtest sur le test du premier split walk-forward) :
   ```
   python models/train_pipeline.py
   ```
   Ce script :
   - charge `BTC/USDT` en `1h` par défaut (`run_training_pipeline(symbol=..., timeframe=..., ...)`
     pour changer),
   - construit les features (`features/pipeline.py::build_feature_dataframe` +
     `normalize_features`) en respectant le split walk-forward courant (HMM et
     stats de normalisation fit uniquement sur le train, jamais sur le test ni
     sur le jeu de test gelé),
   - entraîne un agent PPO (Stable-Baselines3, action discrète Hold/Buy/Sell,
     encodeur MLP par défaut) pendant `total_timesteps=50_000` pas (premier run
     de validation du pipeline, pas un entraînement final — augmenter ce
     paramètre pour un agent plus abouti),
   - sauvegarde le modèle dans `storage/models/ppo_{SYMBOL}_{timeframe}.zip`
     (+ un sidecar `.json` avec `feature_columns`/`window_size`, nécessaire
     pour recharger l'agent ailleurs, ex. dans le dashboard),
   - affiche les métriques de backtest obtenues sur le test du premier split.

   Note : ce premier run n'entraîne/n'évalue que sur **le premier split**
   walk-forward (simplification documentée dans le script), pas encore sur
   l'ensemble des splits — voir le docstring de `models/train_pipeline.py`
   pour étendre à tous les splits.

3. Voir les résultats dans le dashboard :
   ```
   streamlit run app/dashboard.py
   ```
   - Onglet **Backtest** : choisir une stratégie baseline et lancer un
     backtest walk-forward complet (métriques agrégées + courbe de valeur).
   - Onglet **Comparaison stratégies** : comparer les 4 baselines entre elles,
     et cocher **« Inclure l'agent RL entraîné »** pour ajouter le dernier
     modèle trouvé dans `storage/models/` à la comparaison (le dashboard
     reconstruit les features nécessaires et charge les métadonnées du
     sidecar `.json` automatiquement ; un avertissement explicite s'affiche
     si aucun modèle n'est trouvé plutôt qu'un plantage).
   - Rappel méthodologique affiché dans l'onglet : un agent n'a un edge réel
     que s'il bat *simultanément* Buy & Hold, la politique aléatoire pondérée,
     et au moins une baseline de règles, sur le même découpage walk-forward —
     jamais sur le jeu de test gelé.

## Lancer le paper trading en temps réel (Vague 3)

Le paper trading live tourne comme un **processus séparé** du dashboard : un
script avec une boucle infinie, à lancer manuellement dans son propre
terminal, qui lit des données de marché publiques, simule des décisions
dessus via `paper_trading.portfolio.SimulatedPortfolio` (100% en mémoire), et
écrit son état dans une base SQLite légère (`storage/runs/paper_trading.db`)
que le dashboard lit périodiquement pour l'afficher.

1. Dans un premier terminal, lancer le live runner :
   ```
   python paper_trading/live_runner.py --symbol BTC/USDT --timeframe 1h --strategy buy_and_hold
   ```
   Stratégies disponibles via `--strategy` : `buy_and_hold`, `ema_crossover`,
   `momentum`, `random`, `rl` (nécessite un modèle déjà entraîné via
   `python models/train_pipeline.py` — voir « Entraîner un agent RL »
   ci-dessus ; le chemin par défaut est
   `storage/models/ppo_{SYMBOL}_{timeframe}`, surchargeable avec
   `--rl-model-path`). Autres options : `--config`, `--db-path`,
   `--max-iterations` (utile pour un test court plutôt qu'une boucle
   infinie). Laisser tourner (Ctrl+C pour arrêter proprement).

2. Dans un second terminal, lancer (ou garder ouvert) le dashboard :
   ```
   streamlit run app/dashboard.py
   ```
   Onglet **Portefeuille simulé** : valeur totale, cash, positions ouvertes,
   courbe de valeur et journal des décisions (BUY/SELL/HOLD avec les
   indicateurs clés au moment de la décision), lus depuis
   `storage/runs/paper_trading.db`. Un bouton **« Rafraîchir »** relit
   l'état le plus récent (rafraîchissement manuel dans cette V1, plutôt
   qu'un auto-refresh continu basé sur
   `config.yaml::dashboard.refresh_interval_seconds`).

   Le même onglet inclut aussi un **monitoring de dégradation** : Sharpe
   glissant réalisé en live, comparé au minimum et à la moyenne des Sharpe
   obtenus sur les splits walk-forward du dernier backtest de la même
   stratégie (lancé dans l'onglet **Backtest** de la même session) — alerte
   visuelle simple si le Sharpe live passe sous ces repères.

**Limitations connues (documentées dans `paper_trading/live_runner.py`) :**
- Ce n'est **pas** un service de production : pas de Celery, pas de cron
  système, juste une boucle Python (`while True` + pause) lancée à la main.
- **Aucune reprise automatique après crash ou extinction du PC.** Si le
  processus meurt, le portefeuille en mémoire est perdu ; il faut relancer
  manuellement (le portefeuille recommence alors à zéro — pas de
  sérialisation entre deux lancements dans cette V1). Le dernier état connu
  reste visible dans la base SQLite, mais la simulation elle-même doit être
  redémarrée à la main.
- Aucune garantie de fonctionnement 24/7 (pas de watchdog, pas de
  supervision).
- Conçu pour un usage expérimental/personnel : lancer, observer la console
  et/ou le dashboard, arrêter avec Ctrl+C.

## Garde-fous du projet (non négociables)

- **Aucun ordre réel.** Portefeuille simulé uniquement, pas de clé API de
  trading dans le projet.
- **Validation walk-forward obligatoire.** Aucun résultat de performance
  n'est interprétable hors de `backtest/validation.py` (`walk_forward_splits`)
  — voir aussi le jeu de test gelé (`get_frozen_test_set`) qui ne doit
  jamais être consulté pendant le développement.
- **Benchmarks multiples obligatoires.** Une stratégie n'est jamais déclarée
  "gagnante" sans battre *simultanément* buy-and-hold, une politique
  aléatoire de même exposition moyenne (`models/random_policy.py` — le test
  le plus sévère), et une baseline de règles (`models/baseline_rules.py`),
  sur plusieurs fenêtres walk-forward indépendantes.
- **Coûts réalistes.** Le slippage (`backtest/slippage.py`) dépend du volume
  et de la volatilité, pas un pourcentage fixe arbitraire.
- **Anti-look-ahead.** Aucun calcul de feature n'utilise de donnée future —
  voir les docstrings de `features/indicators.py` et
  `backtest/validation.py::assert_no_lookahead`.

## Séquençage du projet

**Vague 1 — Fondation de validation (terminée)**
Données, indicateurs, baselines, risk layer, moteur de backtest walk-forward,
dashboard v0 (lecture seule).

**Vague 2 — Agent RL + nouveaux signaux (terminée)**
Pipeline de features (`features/pipeline.py` : Fear & Greed Index, funding
rate, ratio ETH/BTC, probabilités de régime HMM, normalisation train-only),
environnement Gymnasium (`env/trading_env.py`, action discrète Hold/Buy/Sell,
reward log-return + pénalité coûts/drawdown), agent PPO Stable-Baselines3
(`models/rl_agent.py`, encodeur MLP baseline, interface interchangeable pour
un futur encodeur TCN), script d'entraînement de bout en bout
(`models/train_pipeline.py`), onglets Backtest et Comparaison stratégies du
dashboard câblés sur `backtest.engine` (y compris l'agent RL entraîné). Voir
la section « Entraîner un agent RL » ci-dessus pour le mode d'emploi.

**Vague 3 — Temps réel + robustesse avancée (terminée)**
Boucle de paper trading live en processus séparé (`paper_trading/live_runner.py`
+ `storage/runs/live_state.py`, persistance SQLite), onglet **Portefeuille
simulé** du dashboard complet (état courant, courbe de valeur, journal des
décisions), stress tests sur crashs historiques connus (`backtest/stress_tests.py`,
intégrés à l'onglet Backtest), monitoring de dégradation (Sharpe glissant
live vs distribution des Sharpe walk-forward, alerte visuelle simple) —
toujours sans argent réel. Voir « Lancer le paper trading en temps réel »
ci-dessus. Pas de boucle de ré-entraînement périodique automatique dans
cette V1 (cf. « État du projet » ci-dessous).

Ce document et le plan détaillé reflètent une recherche multi-agents menée
avant l'implémentation (comparaison de plusieurs approches, critique
adversariale des propositions) pour prioriser les ajouts ayant une
justification technique réelle plutôt que de la complexité cosmétique.

## État du projet — ce qui reste à faire pour une V2

Le projet est fonctionnellement complet pour un usage expérimental/personnel
(données → features → env RL → entraînement → backtest walk-forward → stress
tests → paper trading live → dashboard), mais plusieurs simplifications ont
été assumées consciemment pour livrer une V1 cohérente plutôt que de viser
l'exhaustivité. Limitations connues, consignées au fil des 3 vagues :

- **Entraînement RL limité au premier split walk-forward.**
  `models/train_pipeline.py` n'entraîne et n'évalue aujourd'hui que sur le
  premier split walk-forward (cf. docstring du fichier), pour valider que le
  pipeline complet fonctionne de bout en bout. Un entraînement rigoureux
  demanderait de boucler sur `backtest.validation.walk_forward_splits(...)`
  et de répéter tout le bloc train/eval pour chaque split (un agent par
  split, ce qui est long — plusieurs dizaines/centaines de milliers de
  timesteps chacun).
- **Pas de reprise automatique après crash du live runner.** Si le process
  `paper_trading/live_runner.py` meurt ou si la machine s'éteint, le
  portefeuille simulé en mémoire est perdu ; il faut relancer manuellement,
  et la simulation recommence avec un portefeuille neuf (pas de
  sérialisation d'état entre deux lancements). Pas de watchdog, pas de
  supervision, pas de garantie 24/7 — c'est un simple script à boucle
  infinie lancé à la main dans un terminal.
- **Pas de détecteur de drift formel.** Le monitoring de dégradation en V1
  se limite à un Sharpe glissant comparé au minimum/à la moyenne observés en
  backtest walk-forward (alerte visuelle simple) — pas d'ADWIN, de
  Page-Hinkley, ni d'autre détecteur de drift statistique formel. Le seuil
  d'alerte est le minimum observé sur les splits, pas un vrai calcul de
  percentile configurable comme envisagé initialement.
- **Order book imbalance et open interest volontairement exclus des
  features de l'agent.** `data/fetch_derivatives.py` collecte déjà le
  funding rate (consommé par `features/pipeline.py`) et l'open interest,
  mais l'open interest n'est pas encore branché dans le pipeline de
  features — collecte en avance de phase, en attendant d'accumuler assez
  d'historique (l'open interest gratuit Binance ne couvre que ~30 jours
  glissants côté API). La profondeur de carnet (order book) au-delà du
  meilleur bid/ask n'est pas exposée non plus (`data/live_feed.py::fetch_current_spread`
  ne lit que le niveau 1) pour éviter un écart entraînement/inférence.
- **HMM de régime fit sur la fenêtre live entière en mode live**, faute de
  split train/test actif hors contexte d'entraînement (`paper_trading/live_runner.py::build_live_features`,
  avertissement explicite émis par `features/pipeline.py` dans ce cas) — un
  raccourci acceptable pour l'exploration, mais pas rigoureusement identique
  au HMM utilisé à l'entraînement.
- **Normalisation des features en mode live dépend de la présence du
  sidecar `_train_stats.json`.** Depuis cette passe d'audit,
  `models/train_pipeline.py` sauvegarde les statistiques de normalisation du
  train à côté du modèle (`storage/models/ppo_{SYMBOL}_{timeframe}_train_stats.json`),
  et `live_runner.py` les charge automatiquement si présentes. Pour un
  modèle entraîné *avant* cette correction, ce fichier sera absent : le live
  runner retombe alors sur une normalisation calculée sur sa propre fenêtre
  récente (toujours anti-look-ahead au sens strict, mais pas rigoureusement
  identique à la distribution vue à l'entraînement) — ré-entraîner via
  `python models/train_pipeline.py` pour régénérer le sidecar.
- **Dashboard « Portefeuille simulé » sans auto-refresh continu.** Le
  rafraîchissement de l'onglet est manuel (bouton), pas basé sur
  `config.yaml::dashboard.refresh_interval_seconds` comme envisagé au
  départ — choix délibéré pour éviter la complexité de gestion d'état
  Streamlit dans cette V1.
- **Comparaison de stratégies avec agent RL : normalisation approximée.**
  Dans l'onglet **Comparaison stratégies**, quand l'agent RL est inclus, les
  statistiques de normalisation utilisées sont recalculées sur les données
  chargées dans le dashboard plutôt que de relire le sidecar `_train_stats.json`
  du modèle (limitation documentée dans `app/dashboard.py`) — approximation
  acceptable pour une comparaison exploratoire, mais pas rigoureusement
  identique aux stats d'entraînement d'origine. Pourrait être corrigé en
  réutilisant `_load_train_stats`-style loading, comme fait dans
  `live_runner.py`.
- **`chandelier_exit_short` non utilisée.** Implémentée dans `risk/stops.py`
  pour la complétude de l'API des stops, mais aucune position courte n'est
  prévue dans le plan actuel — le paper trading reste long-only.
- **`sqlalchemy` listé dans `requirements.txt` mais non utilisé.**
  `storage/runs/live_state.py` utilise volontairement `sqlite3` (stdlib) au
  lieu d'un ORM, jugé suffisant pour un seul writer / lecteurs occasionnels
  et un schéma à deux tables simples. `sqlalchemy` reste dans les
  dépendances pour un usage futur éventuel mais pourrait être retiré si
  aucun module ne vient à l'utiliser.
- **Pas de boucle de ré-entraînement périodique automatique.** Envisagée
  dans le séquençage initial de la Vague 3, jugée hors scope pour cette V1 :
  ré-entraîner reste une action manuelle (`python models/train_pipeline.py`).
