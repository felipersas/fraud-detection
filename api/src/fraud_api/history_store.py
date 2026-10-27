"""Card history kept in Redis, shared by every API replica and kept across restarts.

Each card is a Redis sorted set: one member per purchase, scored by its time,
so "this card's purchases in the last 30 days, before this one" is a single
range query. The features themselves are computed by
fraud_core.history.history_features, the same function the in-memory store
uses, so both stores always agree with training.
"""

from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from datetime import datetime
from uuid import uuid4

import redis

from fraud_core.history import MAX_GAP, history_features

EPOCH = datetime(1970, 1, 1)
# How long a lock may be held before Redis frees it, in case a replica dies holding it.
LOCK_TIMEOUT_SECONDS = 5


def seconds(time: datetime) -> float:
    """Purchase times are local wall-clock times without a timezone; count seconds from a fixed epoch.

    datetime.timestamp() would instead read them as the server's local time,
    which shifts with the server's timezone and daylight saving changes.
    """
    return (time - EPOCH).total_seconds()


class RedisCardHistoryStore:
    def __init__(self, client: redis.Redis, prefix: str = "card-history:"):
        self.client = client
        self.prefix = prefix

    def _key(self, card: str) -> str:
        return f"{self.prefix}{card}"

    def features(self, card: str, time: datetime) -> dict[str, float]:
        """The card's history before `time`, with the same meaning as card_history."""
        # "(" makes the upper bound exclusive: purchases at exactly `time` are not "before" it.
        members = self.client.zrangebyscore(self._key(card), seconds(time - MAX_GAP), f"({seconds(time)}")
        purchases = []
        for member in members:
            when, amount, _ = member.decode().split("|")
            purchases.append((datetime.fromisoformat(when), float(amount)))
        return history_features(purchases, time)

    def add(self, card: str, time: datetime, amount: float) -> None:
        key = self._key(card)
        # The random suffix keeps two identical purchases (same time and amount) as two members.
        member = f"{time.isoformat()}|{amount!r}|{uuid4().hex}"
        pipe = self.client.pipeline()
        pipe.zadd(key, {member: seconds(time)})
        # Purchases older than MAX_GAP before this one can no longer affect any feature.
        pipe.zremrangebyscore(key, "-inf", f"({seconds(time - MAX_GAP)}")
        # Cards that stop buying are forgotten after MAX_GAP instead of being kept forever.
        pipe.expire(key, int(MAX_GAP.total_seconds()))
        pipe.execute()

    @contextmanager
    def lock(self, cards: Iterable[str]) -> Iterator[None]:
        """Lock the given cards across every replica while their history is read and updated.

        Cards are locked in sorted order, so two requests sharing cards can never
        wait on each other in a cycle.
        """
        with ExitStack() as stack:
            for card in sorted(set(cards)):
                lock = self.client.lock(
                    f"lock:{self._key(card)}", timeout=LOCK_TIMEOUT_SECONDS, blocking_timeout=LOCK_TIMEOUT_SECONDS
                )
                stack.enter_context(lock)
            yield

    def ping(self) -> bool:
        try:
            return bool(self.client.ping())
        except redis.RedisError:
            return False
