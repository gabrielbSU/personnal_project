"""fonctions nécessaires pour créer les features et les labels dans le dataset"""

import numpy as np

def add_features(df):

    # Retours
    df["ret_1"] = df["Close"].pct_change(1)
    df["ret_5"] = df["Close"].pct_change(5)
    df["ret_10"] = df["Close"].pct_change(10)

    # Volatilité
    df["vol_5"] = df["ret_1"].rolling(5).std()
    df["vol_20"] = df["ret_1"].rolling(20).std()

    # Moyennes mobiles
    df["sma_10"] = df["Close"].rolling(10).mean()
    df["sma_20"] = df["Close"].rolling(20).mean()
    close = df["Close"].squeeze() #nécessaire pour ne pas avoir à manipuler un dataframe et une serie en même temps
    df["close_sma10"] = close / df["sma_10"]
    df["close_sma20"] = close / df["sma_20"]

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # ATR
    df["tr"] = np.maximum(df["High"] - df["Low"],
                          np.maximum(abs(df["High"] - df["Close"].shift(1)),
                                     abs(df["Low"] - df["Close"].shift(1))))
    df["atr"] = df["tr"].rolling(14).mean()

    # Bandes de Bollinger
    df["bb_mid"] = df["Close"].rolling(20).mean()
    df["bb_std"] = df["Close"].rolling(20).std()
    df["bb_high"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_low"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_pos"] = (close - df["bb_low"]) / (df["bb_high"] - df["bb_low"])

    # OBV
    df["obv"] = (np.sign(df["Close"].diff()) * df["Volume"]).fillna(0).cumsum()

    return df


def create_labels(df, horizon=5, threshold=0.03):
    # Retour futur
    df["future_return"] = (df["Close"].shift(-horizon) - df["Close"]) / df["Close"]
    
    # Labels
    conditions = [
        df["future_return"] > threshold,
        df["future_return"] < -threshold
    ]
    choices = ["acheter", "vendre"]
    
    df["label"] = np.select(conditions, choices, default="garder")
    
    return df