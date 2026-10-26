"""Turning raw columns into model inputs.

Two steps with very different rules:
- make_features: stateless, row-by-row transformations. They learn nothing
  from the data, so they are safe to apply to any part of the split.
- Preprocessor: learns statistics (mean and standard deviation per numeric
  feature, the list of categories), so it is fitted on train only and merely
  applied to cv and test.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_core.columns import TARGET, TIME

# Raw columns that make_features reads; anything scoring a transaction must provide them.
RAW_INPUTS = [TIME, "amt", "dob", "city_pop", "lat", "long", "merch_lat", "merch_long", "category", "gender"]

NUMERIC = ["log_amt", "hour", "day_of_week", "age", "log_city_pop", "distance_km"]
CATEGORICAL = ["category", "gender"]

EARTH_RADIUS_KM = 6371


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build one row of features per transaction, using only that transaction.

    Columns that identify a person (name, street, card number, transaction id)
    are left out on purpose: they would let the model memorize who was
    defrauded in the training period instead of learning what fraud looks like.
    """
    t = df[TIME]
    return pd.DataFrame(
        {
            # Amounts are skewed (most purchases are small, a few are huge);
            # log1p compresses the long tail so it does not dominate training.
            "log_amt": np.log1p(df["amt"]),
            "hour": t.dt.hour,
            "day_of_week": t.dt.day_of_week,
            "age": (t - df["dob"]).dt.days / 365.25,
            "log_city_pop": np.log1p(df["city_pop"]),
            # How far the purchase happened from the cardholder's home.
            "distance_km": haversine_km(df["lat"], df["long"], df["merch_lat"], df["merch_long"]),
            "category": df["category"],
            "gender": df["gender"],
        },
        index=df.index,
    )


class Preprocessor:
    """make_features followed by transformations fitted on the training set.

    - Numeric features are standardized (zero mean, unit variance), which helps
      gradient descent converge for logistic regression and neural networks alike.
    - Categorical features are one-hot encoded: "category = travel" becomes a
      column that is 1 for travel purchases and 0 otherwise. Categories never
      seen in train are encoded as all zeros instead of raising an error.
    """

    def __init__(self):
        self.columns = ColumnTransformer(
            [
                ("numeric", StandardScaler(), NUMERIC),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        self.columns.fit(make_features(train_df))
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) ready for a model, using what was learned in fit."""
        return self.transform_features(df), df[TARGET].to_numpy()

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return only X, for transactions whose label is unknown (as in serving)."""
        return self.columns.transform(make_features(df))

    @property
    def feature_names(self) -> list[str]:
        return list(self.columns.get_feature_names_out())
