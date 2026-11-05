# TBT

Un bot de trading crypto que j'ai écrit pour apprendre à backtester proprement et à entraîner un agent par renforcement. Il ne passe aucun ordre réel : paper trading uniquement, aucune clé API, que des données publiques de Binance. Ce n'est pas un conseil financier.

## Ce qu'il fait

- Télécharge l'historique de BTC/USDT et ETH/USDT (5 min, 1 h, 1 jour) avec `ccxt`.
- Calcule des indicateurs (RSI, MACD, Bollinger, ATR), un régime de marché par HMM, le Fear & Greed Index et le funding rate.
- Fait des backtests en walk-forward (365 jours d'entraînement, 90 de test) avec des frais de 0,1 % et un slippage en racine carrée du volume. La taille de position suit la volatilité (ATR) et il y a des stops de type chandelier.
- Compare plusieurs stratégies : buy and hold, politique aléatoire, croisement d'EMA, momentum.
- Entraîne un agent PPO (stable-baselines3) sur un environnement Gymnasium (`env/trading_env.py`) avec trois actions : garder, acheter, vendre.
- Fait tourner une boucle de paper trading en temps réel et un dashboard Streamlit pour suivre le tout.

Les paramètres (frais, périodes, risque) sont tous dans `config.yaml`.

## Où j'en suis vraiment

- L'agent n'est entraîné et évalué que sur le premier split du walk-forward. Je n'ai pas de résultat chiffré à montrer, donc je ne dis pas qu'il bat le buy and hold.
- Les frais et le slippage sont des valeurs que j'ai supposées, je ne les ai pas calibrés sur de vrais ordres.
- Le live runner est une simple boucle lancée à la main : si le processus s'arrête, le portefeuille simulé est perdu.
- Il n'y a pas de tests automatisés.
- `sqlalchemy` est dans `requirements.txt` mais n'est pas utilisé.

## Installation

Python 3.12 ou plus.

```bash
pip install -r requirements.txt
```

Sur une machine avec GPU NVIDIA, `pip` installe torch en version CPU. Les commentaires en haut de `requirements.txt` expliquent comment installer la version CUDA.

## Utilisation

Les données et les modèles ne sont pas dans le dépôt, il faut les générer :

```bash
python data/fetch_historical.py
python data/fetch_derivatives.py          # optionnel : funding rate
streamlit run app/dashboard.py
python -m models.train_pipeline           # entraîne un agent PPO
python -m paper_trading.live_runner --symbol BTC/USDT --timeframe 1h --strategy buy_and_hold
```

Les stratégies du live runner : `buy_and_hold`, `ema_crossover`, `momentum`, `random`, `rl` (il faut avoir entraîné un modèle avant).

## Structure

```
data/           récupération des données Binance
features/       indicateurs, signaux de contexte, régime, pipeline
env/            environnement Gymnasium
risk/           taille de position, stops, contraintes d'exposition
models/         stratégies de base, agent PPO, entraînement
backtest/       moteur, walk-forward, slippage, stress tests
paper_trading/  portefeuille simulé et boucle live
app/            dashboard Streamlit
storage/runs/   persistance SQLite du paper trading
```

## Données

Binance (via `ccxt`) et l'API Fear & Greed d'[alternative.me](https://alternative.me/crypto/fear-and-greed-index/). Je ne redistribue pas les données, elles restent soumises aux conditions de ces services.

Licence MIT.
