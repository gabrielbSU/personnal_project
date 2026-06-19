"""fichier dans lequel on va récuperer les données OHLCV pour ensuite calculer les features et construire ainsi le dataset de notre projet"""

import yfinance as yf #pour télécharger les market datas
import pandas as pd
import numpy as np
from dataset_functions import add_features, create_labels
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


df = yf.download("AAPL", start="2010-01-01", end="2024-01-01")

df = add_features(df)          # toutes les features
df = create_labels(df, horizon=10)   # labels basés sur le futur

df = df.dropna()               # nettoyage

X = df[[
    "ret_1", "ret_5", "ret_10",
    "vol_5", "vol_20",
    "close_sma10", "close_sma20",
    "rsi", "macd", "macd_signal",
    "atr", "obv", "bb_pos"
]]

y = df["label"]


### Maintenant que le dataset est construit on retire le futur_return et le label pour l'entrainenemnt du modèle

X = df.drop(columns=["future_return", "label"])
y = df["label"]

## On split train/test

train_size = int(0.8 * len(df))

X_train = X.iloc[:train_size]
X_test  = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test  = y.iloc[train_size:]

## On scale les features 

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

## On train

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

## prediction

y_pred = rf.predict(X_test_scaled)

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred, labels=["acheter", "garder", "vendre"])

sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["acheter", "garder", "vendre"],
            yticklabels=["acheter", "garder", "vendre"])
plt.title("Matrice de confusion")
plt.show()

print(df["label"].value_counts(normalize=True))


