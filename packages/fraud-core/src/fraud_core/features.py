"""Turning raw columns into model inputs.

Two steps with very different rules:
- make_features: stateless, row-by-row transformations. They learn nothing
  from the data, so they are safe to apply to any part of the split.
- Preprocessor: learns statistics (mean and standard deviation per numeric
  feature, the list of categories), so it is fitted on train only and merely
  applied to cv and test.

Card-history features (fraud_core.history) are optional: models from 2.0.0 on
use them, earlier models score each transaction on its own.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_core.columns import TARGET, TIME
from fraud_core.history import HISTORY_COLUMNS

# Raw columns that make_features reads; anything scoring a transaction must provide them.
RAW_INPUTS = [TIME, "amt", "dob", "city_pop", "lat", "long", "merch_lat", "merch_long", "category", "gender"]

# Columns card_history adds, needed on top of RAW_INPUTS by card-history models.
HISTORY_INPUTS = HISTORY_COLUMNS

NUMERIC = ["log_amt", "hour", "day_of_week", "age", "log_city_pop", "distance_km"]
HISTORY_NUMERIC = ["tx_last_24h", "night_tx_last_24h", "large_tx_last_24h", "log_amt_last_24h", "log_hours_since_prev"]
CATEGORICAL = ["category", "gender"]

EARTH_RADIUS_KM = 6371


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points given in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def make_features(df: pd.DataFrame, card_history: bool = False) -> pd.DataFrame:
    """Build one row of features per transaction.

    Columns that identify a person (name, street, card number, transaction id)
    are never used as features: they would let the model memorize who was
    defrauded in the training period instead of learning what fraud looks like.
    The card number only serves to look up the card's own history, and only
    when `card_history` is set (the columns from fraud_core.history must be present).
    """
    t = df[TIME]
    features = pd.DataFrame(
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
    if card_history:
        # Counts of recent purchases are used as they are; money and time gaps are
        # skewed like the amount, so they are log-compressed the same way.
        features["tx_last_24h"] = df["tx_last_24h"]
        features["night_tx_last_24h"] = df["night_tx_last_24h"]
        features["large_tx_last_24h"] = df["large_tx_last_24h"]
        features["log_amt_last_24h"] = np.log1p(df["amt_last_24h"])
        features["log_hours_since_prev"] = np.log1p(df["hours_since_prev"])
    return features


class Preprocessor:
    """make_features followed by transformations fitted on the training set.

    - Numeric features are standardized (zero mean, unit variance), which helps
      gradient descent converge for logistic regression and neural networks alike.
    - Categorical features are one-hot encoded: "category = travel" becomes a
      column that is 1 for travel purchases and 0 otherwise. Categories never
      seen in train are encoded as all zeros instead of raising an error.
    """

    # Class-level default, so preprocessors saved before card history existed still load.
    card_history = False

    def __init__(self, card_history: bool = False):
        self.card_history = card_history
        numeric = NUMERIC + HISTORY_NUMERIC if card_history else NUMERIC
        self.columns = ColumnTransformer(
            [
                ("numeric", StandardScaler(), numeric),
                ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
            ]
        )

    def fit(self, train_df: pd.DataFrame) -> "Preprocessor":
        self.columns.fit(make_features(train_df, self.card_history))
        return self

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (X, y) ready for a model, using what was learned in fit."""
        return self.transform_features(df), df[TARGET].to_numpy()

    def transform_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return only X, for transactions whose label is unknown (as in serving)."""
        return self.columns.transform(make_features(df, self.card_history))

    @property
    def feature_names(self) -> list[str]:
        return list(self.columns.get_feature_names_out())

    @property
    def categories(self) -> dict[str, list[str]]:
        """Values seen in train for each categorical column; anything else is unknown to the model."""
        encoder = self.columns.named_transformers_["categorical"]
        return {col: [str(v) for v in values] for col, values in zip(CATEGORICAL, encoder.categories_)}
