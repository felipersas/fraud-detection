"""Training pipeline tests on synthetic data, so they run without the real download."""

import numpy as np
import pandas as pd
import pytest

from fraud_core.columns import TIME
from fraud_training.data import split_by_time
from fraud_training.error_analysis import count_tags, recent_card_activity
from fraud_training.evaluate import cost_by_threshold, evaluate


def test_split_by_time_train_is_past(transactions):
    s = split_by_time(transactions)
    assert s.train[TIME].max() < s.cv[TIME].min()
    assert s.cv[TIME].max() < s.test[TIME].min()
    assert len(s.train) + len(s.cv) + len(s.test) == len(transactions)


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


def test_cost_by_threshold_prices_each_error_type():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.2, 0.6, 0.1])
    amounts = np.array([500.0, 30.0, 80.0, 20.0])
    costs = cost_by_threshold(y, prob, amounts, false_alarm_cost=10, thresholds=np.array([0.5]))
    # At 0.5: the $30 fraud is missed and the $80 legit purchase is flagged.
    row = costs.loc[0.5]
    assert row["missed_frauds"] == 1 and row["false_alarms"] == 1
    assert row["total_cost"] == pytest.approx(30 + 10)
