"""Dataset loading and train / cv / test splitting.

The split happens BEFORE any processing. Anything that learns from the data
(means, standard deviations, category lists, oversampling) must only ever see
the training set. If it peeks at cv or test, the final metrics become
optimistic and stop reflecting how the model behaves on unseen data
(this is called data leakage).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "sparkov"

TARGET = "is_fraud"
TIME = "trans_date_trans_time"

# Kaggle ships the data as two files that are already split by time:
# fraudTrain.csv covers 2019-01-01 to 2020-06-21 and fraudTest.csv the rest of 2020.
FILES = ["fraudTrain.csv", "fraudTest.csv"]
KAGGLE_DATASET = "kartik2112/fraud-detection"

# Train on 2019, tune on the first half of 2020, and keep Kaggle's own test file
# (second half of 2020) as the final, untouched test set.
CV_START = pd.Timestamp("2020-01-01")
TEST_START = pd.Timestamp("2020-06-21 12:14:00")


def load_raw(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Return every transaction from both Kaggle files, sorted by time.

    Download the files first with:
        kaggle datasets download -d kartik2112/fraud-detection -p data/sparkov --unzip
    """
    missing = [f for f in FILES if not (data_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not found in {data_dir}. Download them with:\n"
            f"  kaggle datasets download -d {KAGGLE_DATASET} -p {data_dir} --unzip"
        )
    # index_col=0 drops the unnamed row-number column that the CSVs carry.
    df = pd.concat([pd.read_csv(data_dir / f, index_col=0) for f in FILES], ignore_index=True)
    df[TIME] = pd.to_datetime(df[TIME])
    df["dob"] = pd.to_datetime(df["dob"])
    return df.sort_values(TIME, kind="stable").reset_index(drop=True)


@dataclass
class Split:
    train: pd.DataFrame
    cv: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Period, size and fraud rate of each part, to check the split at a glance."""
        rows = {
            name: {
                "from": part[TIME].min().date(),
                "to": part[TIME].max().date(),
                "rows": len(part),
                "frauds": int(part[TARGET].sum()),
                "fraud_rate_%": round(100 * part[TARGET].mean(), 3),
            }
            for name, part in [("train", self.train), ("cv", self.cv), ("test", self.test)]
        }
        return pd.DataFrame(rows).T


def split_by_time(df: pd.DataFrame, cv_start: pd.Timestamp = CV_START, test_start: pd.Timestamp = TEST_START) -> Split:
    """Train on the past, validate and test on the future.

    This mirrors production: a deployed model is trained on old transactions
    and has to score the ones that arrive next. It is harsher than a random
    split, because fraud patterns (and the fraud rate) drift over time.
    """
    t = df[TIME]
    return Split(
        train=df[t < cv_start],
        cv=df[(t >= cv_start) & (t < test_start)],
        test=df[t >= test_start],
    )
