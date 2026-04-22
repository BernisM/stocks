"""
Calcule tous les indicateurs techniques sur un DataFrame OHLCV.
Retourne le même DataFrame enrichi.
"""
import numpy as np
import pandas as pd
import ta

from .config import ATR_STOP_MULTIPLIER


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) < 26:
        return df

    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]

    # ── Moyennes mobiles ──────────────────────────────────────────────────────
    df["EMA50"]  = close.ewm(span=50, adjust=False).mean()
    df["EMA200"] = close.ewm(span=200, adjust=False).mean()
    df["SMA50"]  = close.rolling(50,  min_periods=20).mean()
    df["SMA200"] = close.rolling(200, min_periods=50).mean()

    # ── RSI ───────────────────────────────────────────────────────────────────
    df["RSI"] = ta.momentum.RSIIndicator(close, window=14).rsi()

    # ── MACD (12, 26, 9) ──────────────────────────────────────────────────────
    macd_ind          = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    df["MACD"]        = macd_ind.macd()
    df["MACD_signal"] = macd_ind.macd_signal()
    df["MACD_hist"]   = macd_ind.macd_diff()

    # ── ATR & Stop Loss ───────────────────────────────────────────────────────
    df["ATR"]            = ta.volatility.AverageTrueRange(high, low, close, window=14).average_true_range()
    df["ATR_pct"]        = df["ATR"] / close * 100
    df["Stop_Loss"]      = close - ATR_STOP_MULTIPLIER * df["ATR"]

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
    df["BB_upper"]  = bb.bollinger_hband()
    df["BB_lower"]  = bb.bollinger_lband()
    df["BB_middle"] = bb.bollinger_mavg()
    df["BB_pct"]    = bb.bollinger_pband()    # %B : 0 = borne basse, 1 = borne haute

    # ── Volatilité historique annualisée ──────────────────────────────────────
    df["Volatility"] = close.pct_change().rolling(20).std() * np.sqrt(252) * 100

    # ── OBV ───────────────────────────────────────────────────────────────────
    df["OBV"]       = ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["OBV_slope"] = df["OBV"].diff(5)   # pente sur 5 jours

    # ── Volume relatif ────────────────────────────────────────────────────────
    df["Vol_ratio"] = volume / volume.rolling(20).mean()

    # ── ADX (force de tendance) ───────────────────────────────────────────────
    adx_ind    = ta.trend.ADXIndicator(high, low, close, window=14)
    df["ADX"]  = adx_ind.adx()

    # ── Pente SMA200 (% sur 10 jours) ─────────────────────────────────────────
    sma200_lag         = df["SMA200"].shift(10)
    df["SMA200_slope"] = (df["SMA200"] - sma200_lag) / sma200_lag.replace(0, np.nan) * 100

    # ── ATR percentile rank sur 50 jours ──────────────────────────────────────
    df["ATR_pct_rank"] = df["ATR_pct"].rolling(50).rank(pct=True) * 100

    # ── BB Z-score (distance à la moyenne en écarts-types) ────────────────────
    bb_std         = close.rolling(20).std().replace(0, np.nan)
    df["BB_zscore"] = (close - df["BB_middle"]) / bb_std

    # ── Régimes de marché ─────────────────────────────────────────────────────
    df["regime_trend"]    = (df["ADX"] > 25).astype(int)        # 1 = tendance, 0 = range
    df["regime_bull"]     = (df["SMA200_slope"] > 0).astype(int) # 1 = SMA200 haussière
    df["regime_vol_high"] = (df["ATR_pct_rank"] > 70).astype(int) # 1 = volatilité élevée

    # ── Ichimoku ──────────────────────────────────────────────────────────────
    df["Tenkan"]   = (high.rolling(9).max()  + low.rolling(9).min())  / 2
    df["Kijun"]    = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["Senkou_A"] = ((df["Tenkan"] + df["Kijun"]) / 2).shift(26)
    df["Senkou_B"] = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
    df["Chikou"]   = close.shift(-26)

    # ── Signaux dérivés ───────────────────────────────────────────────────────
    df["Above_cloud"] = (close > df["Senkou_A"].fillna(0)) & (close > df["Senkou_B"].fillna(0))
    df["Golden_cross"] = (df["SMA50"] > df["SMA200"]) & (df["SMA50"].shift(1) <= df["SMA200"].shift(1))
    df["Death_cross"]  = (df["SMA50"] < df["SMA200"]) & (df["SMA50"].shift(1) >= df["SMA200"].shift(1))

    return df


def get_last_row(df: pd.DataFrame) -> dict:
    """Retourne la dernière ligne sous forme de dict (valeurs scalaires)."""
    if df.empty:
        return {}
    row = df.iloc[-1]
    return {col: (float(val) if not isinstance(val, bool) else bool(val))
            for col, val in row.items()
            if val is not None and not (isinstance(val, float) and np.isnan(val))}
