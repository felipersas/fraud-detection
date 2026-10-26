"""Feature engineering tests on synthetic transactions, so they run without the real download."""

import numpy as np
import pytest

from fraud_core.columns import TARGET, TIME
from fraud_core.features import NUMERIC, Preprocessor, haversine_km, make_features


@pytest.fixture
def train_and_new(transactions):
    """The earliest 70% of transactions as train, the rest as data the model has never seen."""
    ordered = transactions.sort_values(TIME)
    cut = int(len(ordered) * 0.7)
    return ordered.iloc[:cut], ordered.iloc[cut:]


def test_preprocessor_uses_only_train_stats(train_and_new):
    train, new = train_and_new
    prep = Preprocessor().fit(train)
    X_train, _ = prep.transform(train)
    # Standardized numeric train features have mean 0 and std 1 by construction.
    numeric = X_train[:, : len(NUMERIC)]
    np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(numeric.std(axis=0), 1, atol=1e-8)
    # Transforming wildly different new data must not change what was learned.
    scaler = prep.columns.named_transformers_["numeric"]
    before = scaler.mean_.copy()
    prep.transform(new.assign(amt=new["amt"] * 1000))
    np.testing.assert_array_equal(scaler.mean_, before)


def test_unseen_category_does_not_break_transform(train_and_new):
    train, new = train_and_new
    prep = Preprocessor().fit(train)
    X, _ = prep.transform(new.assign(category="brand_new_category"))
    assert X.shape[1] == len(prep.feature_names)


def test_features_exclude_identifiers_and_target(transactions):
    features = make_features(transactions.assign(cc_num=1, trans_num="x"))
    for col in ["cc_num", "trans_num", TARGET, TIME]:
        assert col not in features.columns


def test_haversine_known_distance():
    # São Paulo to Rio de Janeiro is roughly 360 km in a straight line.
    assert haversine_km(-23.55, -46.63, -22.91, -43.17) == pytest.approx(360, abs=10)


def test_categories_lists_what_train_contained(train_and_new):
    train, _ = train_and_new
    prep = Preprocessor().fit(train)
    assert prep.categories["category"] == sorted(train["category"].unique())
    assert prep.categories["gender"] == ["F", "M"]
