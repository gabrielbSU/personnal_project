"""fichier dans lequel on va récuperer les données OHLCV pour ensuite calculer les features et construire ainsi le dataset de notre projet"""

import yfinance as yf #pour télécharger les market datas
import pandas as pd
import numpy as np
from dataset_functions import add_features, create_labels

df = yf.download("AAPL", start="2010-01-01", end="2024-01-01")

df = add_features(df)          # toutes les features
df = create_labels(df, h=10)   # labels basés sur le futur

df = df.dropna()               # nettoyage

X = df[[
    "ret_1", "ret_5", "ret_10",
    "vol_5", "vol_20",
    "close_sma10", "close_sma20",
    "rsi", "macd", "macd_signal",
    "atr", "obv", "bb_pos"
]]

y = df["label"]

print(df.head())
