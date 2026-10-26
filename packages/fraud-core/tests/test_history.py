"""Card-history tests: the training path and the serving path must agree exactly."""

import numpy as np
import pandas as pd
import pytest

from fraud_core.columns import CARD, TIME
from fraud_core.history import HISTORY_COLUMNS, MAX_GAP, CardHistoryStore, card_history


def purchases(times, cards, amounts):
    return pd.DataFrame({TIME: pd.to_datetime(times), CARD: cards, "amt": amounts})


def test_counts_only_earlier_purchases_of_the_same_card():
    df = purchases(
        ["2020-01-01 10:00", "2020-01-01 23:00", "2020-01-02 09:00", "2020-01-03 12:00", "2020-01-01 12:00"],
        [1, 1, 1, 1, 2],
        [500.0, 10.0, 20.0, 30.0, 999.0],
    )
    history = card_history(df)
    # Card 1: the 2nd purchase sees the 1st (large); the 3rd sees both (one at night);
    # the 4th comes more than 24h after all of them. Card 2 never sees card 1.
    assert history["tx_last_24h"].tolist() == [0, 1, 2, 0, 0]
    assert history["large_tx_last_24h"].tolist() == [0, 1, 1, 0, 0]
    assert history["night_tx_last_24h"].tolist() == [0, 0, 1, 0, 0]
    assert history["amt_last_24h"].tolist() == [0, 500, 510, 0, 0]
    assert history["hours_since_prev"].tolist() == pytest.approx([720, 13, 10, 27, 720])


def test_first_purchase_and_long_gaps_are_capped():
    df = purchases(["2020-01-01", "2020-03-01"], [1, 1], [10.0, 10.0])
    cap_hours = MAX_GAP.total_seconds() / 3600
    assert card_history(df)["hours_since_prev"].tolist() == [cap_hours, cap_hours]


def test_store_matches_batch_computation():
    rng = np.random.default_rng(0)
    n = 400
    start = pd.Timestamp("2020-01-01").value
    df = purchases(
        np.sort(rng.integers(start, start + 10 * 86_400 * 10**9, n)),
        rng.choice([11, 22, 33], n),
        rng.choice([5.0, 50.0, 250.0, 900.0], n),
    )
    batch = card_history(df)

    # Serving sees purchases one by one, in time order, scoring each before recording it.
    store = CardHistoryStore()
    served = []
    for row in df.itertuples(index=False):
        when = getattr(row, TIME)
        served.append(store.features(str(getattr(row, CARD)), when))
        store.add(str(getattr(row, CARD)), when, row.amt)

    pd.testing.assert_frame_equal(pd.DataFrame(served, index=df.index)[HISTORY_COLUMNS], batch[HISTORY_COLUMNS])


def test_store_ignores_purchases_recorded_after_the_scored_time():
    store = CardHistoryStore()
    store.add("card", pd.Timestamp("2020-01-02 12:00"), 900.0)
    # A late-arriving request for an earlier purchase must not see the future one.
    assert store.features("card", pd.Timestamp("2020-01-02 08:00"))["tx_last_24h"] == 0
