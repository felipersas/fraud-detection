"""Turning raw columns into model inputs.

Two steps with very different rules:
- make_features: stateless, row-by-row transformations. They learn nothing
  from the data, so they are safe to apply to any part of the split.
- Preprocessor: learns statistics (mean and standard deviation per feature),
  so it is fitted on train only and merely applied to cv and test.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import TARGET

SECONDS_PER_DAY = 24 * 60 * 60


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix from raw columns (the target is dropped).

    - Time counts seconds since the first transaction. Under a time-based split
      the test set holds Time values the model never saw during training, so the
      raw value cannot generalize. Hour of day can: it repeats across the two
      days in the dataset and may capture, say, fraud spiking at night.
    - Amount is heavily skewed (median ~22, max ~25k). log1p compresses the long
      tail so a few huge purchases do not dominate gradient descent.
    - V1..V28 are already PCA components and are used as they are.
    """
    X = df.drop(columns=[TARGET, "Time"])
    X["hour"] = (df["Time"] % SECONDS_PER_DAY) / 3600
    X["Amount"] = np.log1p(df["Amount"])
    return X


class Preprocessor:
    """make_features followed by standardization fitted on the training set.

    Standardizing (zero mean, unit variance) puts every feature on the same
    scale, which helps gradient descent converge for logistic regression and
    neural networks alike.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        self.scaler.fit(make_features(train_df))
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) ready for a model, using the statistics learned in fit."""
        X = self.scaler.transform(make_features(df))
        y = df[TARGET].to_numpy()
        return X, y

    @property
    def feature_names(self) -> list[str]:
        return list(self.scaler.feature_names_in_)
