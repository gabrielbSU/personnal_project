import yfinance as yf #pour télécharger les market datas
import pandas as pd
import numpy as np
from dataset_functions import add_features, create_labels

df = yf.download("AAPL", start="2010-01-01", end="2024-01-01")

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
close = df["Close"].squeeze()
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

