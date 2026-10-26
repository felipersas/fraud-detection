"""Pipeline tests on a small synthetic dataset, so they run without the real download."""

import numpy as np
import pandas as pd
import pytest

from src.data import TARGET, split_by_time, split_stratified
from src.evaluate import evaluate
from src.features import Preprocessor


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 10_000
    data = {"Time": np.sort(rng.uniform(0, 2 * 86400, n))}
    data |= {f"V{i}": rng.normal(size=n) for i in range(1, 29)}
    data["Amount"] = rng.exponential(80, n)
    data[TARGET] = (rng.random(n) < 0.02).astype(int)
    # Shuffle rows so the tests prove split_by_time does its own sorting.
    return pd.DataFrame(data).sample(frac=1, random_state=0)


def test_split_by_time_train_is_past(df):
    s = split_by_time(df)
    assert s.train["Time"].max() <= s.cv["Time"].min()
    assert s.cv["Time"].max() <= s.test["Time"].min()
    assert len(s.train) + len(s.cv) + len(s.test) == len(df)


def test_split_stratified_keeps_fraud_rate(df):
    s = split_stratified(df)
    rate = df[TARGET].mean()
    for part in (s.train, s.cv, s.test):
        assert part[TARGET].mean() == pytest.approx(rate, abs=0.003)


def test_split_stratified_does_not_overlap(df):
    s = split_stratified(df)
    train, cv, test = (set(p.index) for p in (s.train, s.cv, s.test))
    assert not (train & cv or train & test or cv & test)
    assert len(train | cv | test) == len(df)


def test_preprocessor_uses_only_train_stats(df):
    s = split_by_time(df)
    prep = Preprocessor().fit(s.train)
    X_train, _ = prep.transform(s.train)
    # Standardized train features have mean 0 and std 1 by construction.
    np.testing.assert_allclose(X_train.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(X_train.std(axis=0), 1, atol=1e-8)
    # Transforming a wildly different test set must not change what was learned.
    before = prep.scaler.mean_.copy()
    prep.transform(s.test.assign(Amount=s.test["Amount"] * 1000))
    np.testing.assert_array_equal(prep.scaler.mean_, before)


def test_features_drop_raw_time_and_target(df):
    prep = Preprocessor().fit(df)
    assert "Time" not in prep.feature_names
    assert TARGET not in prep.feature_names
    assert "hour" in prep.feature_names


def test_lazy_model_has_high_accuracy_and_zero_recall():
    y = np.array([0] * 998 + [1] * 2)
    m = evaluate(y, np.zeros(len(y)))
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0
