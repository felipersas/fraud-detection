"""Fixtures shared by every package's tests."""

import numpy as np
import pandas as pd
import pytest

from fraud_core.columns import TARGET, TIME


@pytest.fixture
def transactions():
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
