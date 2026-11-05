"""Indicateurs techniques purs sur OHLCV.

Toutes les fonctions prennent un DataFrame pandas avec colonnes au minimum
parmi open/high/low/close/volume (index temporel croissant, une ligne par
bougie fermée) et ne mutent jamais le DataFrame reçu.

Règle anti-look-ahead : tout rolling/ewm utilise uniquement des données
passées ou la bougie courante déjà "fermée" (aucun `center=True`, aucune
fenêtre qui regarde en avant). Par convention, une bougie à l'index t est
considérée comme terminée et son close/high/low/volume sont donc des
données connues au temps t. Quand un indicateur a explicitement besoin de
la clôture de la bougie *précédente* (ex: True Range), on utilise
`.shift(1)` et c'est documenté dans la fonction concernée.
"""

from __future__ import annotations

import pandas as pd


def rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """RSI de Wilder sur `close`, lissage par moyenne mobile exponentielle (alpha=1/period)."""
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    result = 100.0 - (100.0 / (1.0 + rs))
    # avg_loss == 0 et avg_gain > 0 => RSI=100 ; les deux nuls => pas de mouvement, RSI=50
    result = result.where(avg_loss != 0, 100.0)
    result = result.where(~((avg_loss == 0) & (avg_gain == 0)), 50.0)
    result.name = "rsi"
    return result


def macd(
    df: pd.DataFrame,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> pd.DataFrame:
    """MACD standard : renvoie un DataFrame avec colonnes macd_line/macd_signal/macd_histogram."""
    ema_fast = df["close"].ewm(span=fast_period, min_periods=fast_period, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow_period, min_periods=slow_period, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, min_periods=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame(
        {
            "macd_line": macd_line,
            "macd_signal": signal_line,
            "macd_histogram": histogram,
        },
        index=df.index,
    )


def bollinger_bands(
    df: pd.DataFrame, period: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """Bandes de Bollinger sur `close` : colonnes bb_middle/bb_upper/bb_lower (moyenne/écart-type glissants, pas de center)."""
    middle = df["close"].rolling(window=period, min_periods=period).mean()
    std = df["close"].rolling(window=period, min_periods=period).std(ddof=0)
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame(
        {"bb_middle": middle, "bb_upper": upper, "bb_lower": lower},
        index=df.index,
    )


def ema(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Moyenne mobile exponentielle sur `column`."""
    result = df[column].ewm(span=period, min_periods=period, adjust=False).mean()
    result.name = f"ema_{period}"
    return result


def sma(df: pd.DataFrame, period: int, column: str = "close") -> pd.Series:
    """Moyenne mobile simple sur `column`."""
    result = df[column].rolling(window=period, min_periods=period).mean()
    result.name = f"sma_{period}"
    return result


def vwap(df: pd.DataFrame, window: int | None = None) -> pd.Series:
    """VWAP glissant sur `window` bougies (typical price pondéré par le volume).

    Choix : VWAP "session glissante" (rolling window), pas de VWAP ancré sur
    minuit UTC. Nos données couvrent plusieurs années sans notion explicite
    de session d'échange (crypto = marché 24/7), donc un VWAP ancré sur une
    session calendaire serait arbitraire. Le rolling window donne une
    référence de prix moyen pondéré volume comparable quel que soit le
    timeframe (ex: window=20 sur 5m ~ dernière heure quarante). Par défaut
    `window=None` calcule un VWAP expanding depuis le début de la série
    (utile seulement si le DataFrame représente déjà une session unique).
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = typical_price * df["volume"]
    if window is None:
        cum_pv = pv.cumsum()
        cum_vol = df["volume"].cumsum()
    else:
        cum_pv = pv.rolling(window=window, min_periods=window).sum()
        cum_vol = df["volume"].rolling(window=window, min_periods=window).sum()
    result = cum_pv / cum_vol
    result.name = "vwap"
    return result


def momentum(df: pd.DataFrame, n_periods: int = 10, column: str = "close") -> pd.Series:
    """Rendement simple sur N périodes : close[t] / close[t-n] - 1."""
    result = df[column].pct_change(periods=n_periods)
    result.name = f"momentum_{n_periods}"
    return result


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range de Wilder.

    Signature stable requise par risk/ : atr(df, period=14) -> pd.Series
    alignée sur df.index. Le True Range utilise le close de la bougie
    précédente (`shift(1)`) pour capter les gaps, ce qui est correct au
    temps t car il ne s'agit que d'une donnée déjà connue (bougie
    précédente), jamais de la bougie courante non terminée.
    """
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - prev_close).abs()
    low_prev_close = (df["low"] - prev_close).abs()
    true_range = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    result = true_range.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    result.name = "atr"
    return result


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute toutes les colonnes d'indicateurs (paramètres par défaut) sur une copie de `df`."""
    out = df.copy()

    out["rsi"] = rsi(out)

    macd_df = macd(out)
    out["macd_line"] = macd_df["macd_line"]
    out["macd_signal"] = macd_df["macd_signal"]
    out["macd_histogram"] = macd_df["macd_histogram"]

    bb_df = bollinger_bands(out)
    out["bb_middle"] = bb_df["bb_middle"]
    out["bb_upper"] = bb_df["bb_upper"]
    out["bb_lower"] = bb_df["bb_lower"]

    out["ema_12"] = ema(out, period=12)
    out["ema_26"] = ema(out, period=26)
    out["sma_20"] = sma(out, period=20)
    out["vwap"] = vwap(out)
    out["momentum_10"] = momentum(out, n_periods=10)
    out["atr"] = atr(out)

    return out
