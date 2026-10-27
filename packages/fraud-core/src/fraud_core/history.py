"""Card-history features: what each card did shortly before a purchase.

Error analysis showed that most missed frauds are small purchases made while
the card is already in a fraud burst, and most false alarms are large
purchases from cards with a calm history. Both are invisible when a purchase
is scored on its own, so these features describe the card's recent past.

Only purchases strictly before the current one are counted, never the
current purchase or anything after it, which is exactly what a bank knows when
a purchase arrives. No fraud labels are used.

The same numbers are computed in two ways:
- card_history: for a whole table at once, used in training;
- CardHistoryStore: one purchase at a time, used when serving.
Tests check that both agree, so training and serving never drift apart.
"""

import threading
from collections import defaultdict, deque
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from fraud_core.columns import CARD, TIME

WINDOW = timedelta(hours=24)
# How far back to look for the previous purchase; older gaps are capped here.
MAX_GAP = timedelta(days=30)
LARGE_AMOUNT = 200
NIGHT_HOURS = [22, 23, 0, 1, 2, 3]  # the high-risk hours found in the EDA

HISTORY_COLUMNS = ["tx_last_24h", "night_tx_last_24h", "large_tx_last_24h", "amt_last_24h", "hours_since_prev"]


def card_history(df: pd.DataFrame) -> pd.DataFrame:
    """Recent activity of each purchase's card, aligned with `df.index`.

    Pass every available row (all splits together) so early purchases of a
    split still see the card's history from the period before it.

    - tx_last_24h, night_tx_last_24h, large_tx_last_24h: purchases on the card in the
      previous 24 hours, and how many of them were at night or above LARGE_AMOUNT
    - amt_last_24h: total spent on the card in the previous 24 hours
    - hours_since_prev: hours since the card's previous purchase, capped at MAX_GAP
    """
    events = pd.DataFrame(
        {
            CARD: df[CARD],
            "time": df[TIME],
            "tx": 1.0,
            "night_tx": df[TIME].dt.hour.isin(NIGHT_HOURS).astype(float),
            "large_tx": (df["amt"] > LARGE_AMOUNT).astype(float),
            "amt": df["amt"].astype(float),
            # Time as a number, so a rolling max finds the most recent earlier purchase.
            "seconds": (df[TIME] - pd.Timestamp(0)).dt.total_seconds(),
        },
        index=df.index,
    ).sort_values([CARD, "time"], kind="stable")

    by_card = events.set_index("time").groupby(CARD, sort=True)
    # closed="left" makes each window [t - window, t): earlier purchases only.
    recent = by_card[["tx", "night_tx", "large_tx", "amt"]].rolling(WINDOW, closed="left").sum()
    previous = by_card["seconds"].rolling(MAX_GAP, closed="left").max()

    out = pd.DataFrame(index=events.index)
    out["tx_last_24h"] = recent["tx"].to_numpy()
    out["night_tx_last_24h"] = recent["night_tx"].to_numpy()
    out["large_tx_last_24h"] = recent["large_tx"].to_numpy()
    out["amt_last_24h"] = recent["amt"].to_numpy()
    gap_hours = (events["seconds"].to_numpy() - previous.to_numpy()) / 3600
    out["hours_since_prev"] = np.minimum(np.nan_to_num(gap_hours, nan=np.inf), MAX_GAP / timedelta(hours=1))
    # An empty window sums to NaN; it means "no purchases".
    out = out.fillna(0)
    return out.reindex(df.index)


def history_features(purchases: Iterable[tuple[datetime, float]], time: datetime) -> dict[str, float]:
    """A card's history before `time`, from its purchases as (time, amount) pairs.

    Gives the same numbers as card_history for one purchase. Every serving
    store computes its features through this function, whatever it keeps the
    purchases in, so the arithmetic exists in one place only.
    """
    earlier = [(t, amt) for t, amt in purchases if time - MAX_GAP <= t < time]
    last_day = [(t, amt) for t, amt in earlier if t >= time - WINDOW]
    gap = (time - max(t for t, _ in earlier)) if earlier else MAX_GAP
    return {
        "tx_last_24h": float(len(last_day)),
        "night_tx_last_24h": float(sum(t.hour in NIGHT_HOURS for t, _ in last_day)),
        "large_tx_last_24h": float(sum(amt > LARGE_AMOUNT for _, amt in last_day)),
        "amt_last_24h": float(sum(amt for _, amt in last_day)),
        "hours_since_prev": min(gap, MAX_GAP) / timedelta(hours=1),
    }


class CardHistoryStore:
    """In-memory record of recent purchases per card, for scoring one purchase at a time.

    It keeps each card's purchases from the last MAX_GAP only. Being in memory,
    it starts empty and is lost on restart, and every API replica holds its own
    copy; the API's Redis store fixes both.
    """

    def __init__(self):
        self._purchases: dict[str, deque[tuple[datetime, float]]] = defaultdict(deque)
        self._lock = threading.Lock()

    def features(self, card: str, time: datetime) -> dict[str, float]:
        """The card's history before `time`, with the same meaning as card_history."""
        return history_features(self._purchases.get(card, ()), time)

    def add(self, card: str, time: datetime, amount: float) -> None:
        purchases = self._purchases[card]
        purchases.append((time, amount))
        # Drop purchases too old to affect any future feature.
        newest = max(t for t, _ in purchases)
        while purchases and purchases[0][0] < newest - MAX_GAP:
            purchases.popleft()

    @contextmanager
    def lock(self, cards: Iterable[str]) -> Iterator[None]:
        """Hold while reading a card's history and recording its new purchase.

        Without it, two concurrent requests for the same card could each miss
        the other. One lock for the whole store is enough in a single process.
        """
        with self._lock:
            yield
