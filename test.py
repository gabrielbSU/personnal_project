import yfinance as yf #pour télécharger les market datas
import pandas as pd
import numpy as np
from dataset_functions import add_features, create_labels

df = yf.download("AAPL", start="2010-01-01", end="2024-01-01")

print(df["Close"])
df["sma_10"] = df["Close"].rolling(10).mean()
print(df["sma_10"])
df["close_sma10"] = df["Close"] / df["sma_10"]