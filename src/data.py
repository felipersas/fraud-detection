"""Dataset loading and train / cv / test splitting.

The split happens BEFORE any processing. Anything that learns from the data
(means, standard deviations, oversampling, feature selection) must only ever
see the training set. If it peeks at cv or test, the final metrics become
optimistic and stop reflecting how the model behaves on unseen data
(this is called data leakage).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"

TARGET = "Class"

# OpenML hosts two copies of the ULB dataset. The popular one ("creditcard",
# id 1597) drops the Time column, which makes a time-based split impossible.
# This one keeps all 31 original columns.
OPENML_ID = 42175


def load_raw(path: Path = CSV_PATH) -> pd.DataFrame:
    """Return the raw dataset, downloading it from OpenML on the first call."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        df = fetch_openml(data_id=OPENML_ID, as_frame=True, parser="auto").frame
        df[TARGET] = df[TARGET].astype(int)
        df.to_csv(path, index=False)
    return pd.read_csv(path)


@dataclass
class Split:
    train: pd.DataFrame
    cv: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Size and fraud rate of each part, to check the split at a glance."""
        rows = {
            name: {"rows": len(part), "frauds": int(part[TARGET].sum()), "fraud_rate_%": 100 * part[TARGET].mean()}
            for name, part in [("train", self.train), ("cv", self.cv), ("test", self.test)]
        }
        return pd.DataFrame(rows).T


def split_by_time(df: pd.DataFrame, train_frac: float = 0.6, cv_frac: float = 0.2) -> Split:
    """Train on the past, validate and test on the future.

    This mirrors production: a deployed model is trained on old transactions
    and has to score the ones that arrive next. It is harsher than a random
    split, because fraud patterns (and the fraud rate) drift over time.
    """
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    i = int(len(df) * train_frac)
    j = int(len(df) * (train_frac + cv_frac))
    return Split(train=df.iloc[:i], cv=df.iloc[i:j], test=df.iloc[j:])


def split_stratified(df: pd.DataFrame, cv_frac: float = 0.2, test_frac: float = 0.2, seed: int = 42) -> Split:
    """Random split that keeps the same fraud rate (~0.17%) in every part.

    Kept as a reference point: comparing it with split_by_time shows how much
    of the score comes from mixing past and future transactions.
    """
    rest, test = train_test_split(df, test_size=test_frac, stratify=df[TARGET], random_state=seed)
    # cv_frac is a fraction of the full dataset, so rescale it to what is left.
    train, cv = train_test_split(
        rest, test_size=cv_frac / (1 - test_frac), stratify=rest[TARGET], random_state=seed
    )
    return Split(train=train, cv=cv, test=test)
