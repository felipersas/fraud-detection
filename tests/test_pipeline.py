"""Pipeline tests on a small synthetic dataset, so they run without the real download."""

import numpy as np
import pandas as pd
import pytest

from src.data import TARGET, TIME, split_by_time
from src.error_analysis import count_tags, recent_card_activity
from src.evaluate import evaluate
from src.features import NUMERIC, Preprocessor, haversine_km, make_features


@pytest.fixture
def df():
    """Fake transactions with the same columns as the Sparkov files, over 2019-2020."""
    rng = np.random.default_rng(0)
    n = 5_000
    start = pd.Timestamp("2019-01-01").value
    end = pd.Timestamp("2020-12-31").value
    return pd.DataFrame(
        {
            TIME: pd.to_datetime(np.sort(rng.integers(start, end, n))),
            "amt": rng.exponential(70, n),
            "dob": pd.to_datetime("1980-01-01") + pd.to_timedelta(rng.integers(0, 15_000, n), unit="D"),
            "city_pop": rng.integers(100, 1_000_000, n),
            "lat": rng.uniform(30, 45, n),
            "long": rng.uniform(-120, -75, n),
            "merch_lat": rng.uniform(30, 45, n),
            "merch_long": rng.uniform(-120, -75, n),
            "category": rng.choice(["travel", "grocery_pos", "shopping_net"], n),
            "gender": rng.choice(["F", "M"], n),
            TARGET: (rng.random(n) < 0.02).astype(int),
        }
    )


def test_split_by_time_train_is_past(df):
    s = split_by_time(df)
    assert s.train[TIME].max() < s.cv[TIME].min()
    assert s.cv[TIME].max() < s.test[TIME].min()
    assert len(s.train) + len(s.cv) + len(s.test) == len(df)


def test_preprocessor_uses_only_train_stats(df):
    s = split_by_time(df)
    prep = Preprocessor().fit(s.train)
    X_train, _ = prep.transform(s.train)
    # Standardized numeric train features have mean 0 and std 1 by construction.
    numeric = X_train[:, : len(NUMERIC)]
    np.testing.assert_allclose(numeric.mean(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(numeric.std(axis=0), 1, atol=1e-8)
    # Transforming a wildly different test set must not change what was learned.
    scaler = prep.columns.named_transformers_["numeric"]
    before = scaler.mean_.copy()
    prep.transform(s.test.assign(amt=s.test["amt"] * 1000))
    np.testing.assert_array_equal(scaler.mean_, before)


def test_unseen_category_does_not_break_transform(df):
    s = split_by_time(df)
    prep = Preprocessor().fit(s.train)
    X, _ = prep.transform(s.test.assign(category="brand_new_category"))
    assert X.shape[1] == len(prep.feature_names)


def test_features_exclude_identifiers_and_target(df):
    features = make_features(df.assign(cc_num=1, trans_num="x"))
    for col in ["cc_num", "trans_num", TARGET, TIME]:
        assert col not in features.columns


def test_haversine_known_distance():
    # São Paulo to Rio de Janeiro is roughly 360 km in a straight line.
    assert haversine_km(-23.55, -46.63, -22.91, -43.17) == pytest.approx(360, abs=10)


def test_lazy_model_has_high_accuracy_and_zero_recall():
    y = np.array([0] * 998 + [1] * 2)
    m = evaluate(y, np.zeros(len(y)))
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0


def test_recent_card_activity_counts_only_the_past():
    t = pd.to_datetime(["2020-01-01 10:00", "2020-01-01 23:00", "2020-01-02 09:00", "2020-01-03 12:00", "2020-01-01 12:00"])
    df = pd.DataFrame({TIME: t, "cc_num": [1, 1, 1, 1, 2], "amt": [500.0, 10.0, 20.0, 30.0, 999.0]})
    activity = recent_card_activity(df)
    # Card 1: the 2nd purchase sees the 1st (large); the 3rd sees both (one at night);
    # the 4th is more than 24h after all of them. Card 2 never sees card 1's purchases.
    assert activity["tx_last_24h"].tolist() == [0, 1, 2, 0, 0]
    assert activity["large_tx_last_24h"].tolist() == [0, 1, 1, 0, 0]
    assert activity["night_tx_last_24h"].tolist() == [0, 0, 1, 0, 0]


def test_count_tags_allows_overlapping_tags():
    sample = pd.DataFrame({"amt": [10, 20, 900]})
    table = count_tags(sample, {"small": sample["amt"] < 100, "any": sample["amt"] > 0})
    assert table.loc["any", "errors"] == 3
    assert table.loc["small", "share_%"] == pytest.approx(66.7)
