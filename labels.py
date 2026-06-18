"""fichier dans lequel on définit les labels pour la classification"""

import pandas as pd
import numpy as np

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
