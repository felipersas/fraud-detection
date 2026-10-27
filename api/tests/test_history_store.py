"""Redis card-history tests, against an in-process fake Redis server."""

from contextlib import contextmanager
from pathlib import Path

import fakeredis
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_api import main
from fraud_api.history_store import RedisCardHistoryStore
from fraud_api.main import EXAMPLE_TRANSACTION, app
from fraud_core.columns import CARD, TIME
from fraud_core.history import HISTORY_COLUMNS, card_history

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SMALL_NIGHT_PURCHASE = {
    **EXAMPLE_TRANSACTION,
    "transaction_time": "2020-06-17T02:10:00",
    "amount": 18.0,
    "category": "personal_care",
}
BURST = [("2020-06-17T00:33:41", 1167.06), ("2020-06-17T00:56:36", 369.32)]


@pytest.fixture
def server():
    return fakeredis.FakeServer()


def test_redis_store_matches_batch_computation(server):
    rng = np.random.default_rng(1)
    n = 300
    # Whole seconds, like the real data (Python datetimes cannot hold nanoseconds).
    start = pd.Timestamp("2020-01-01")
    df = pd.DataFrame(
        {
            TIME: start + pd.to_timedelta(np.sort(rng.integers(0, 10 * 86_400, n)), unit="s"),
            CARD: rng.choice([11, 22, 33], n),
            "amt": rng.choice([5.0, 50.0, 250.0, 900.0], n),
        }
    )
    store = RedisCardHistoryStore(fakeredis.FakeRedis(server=server))
    served = []
    for row in df.itertuples(index=False):
        when = getattr(row, TIME).to_pydatetime()
        served.append(store.features(str(getattr(row, CARD)), when))
        store.add(str(getattr(row, CARD)), when, row.amt)
    pd.testing.assert_frame_equal(pd.DataFrame(served)[HISTORY_COLUMNS], card_history(df)[HISTORY_COLUMNS])


@contextmanager
def serving_with_redis(monkeypatch, server):
    """An API instance keeping card history in the given (fake) Redis server."""
    monkeypatch.setenv("FRAUD_MODELS_DIR", str(MODELS_DIR))
    monkeypatch.delenv("FRAUD_MODEL_VERSION", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://fake")
    monkeypatch.setattr(main.redis.Redis, "from_url", lambda url: fakeredis.FakeRedis(server=server))
    with TestClient(app) as client:
        yield client


def test_history_survives_an_api_restart(monkeypatch, server):
    with serving_with_redis(monkeypatch, server) as client:
        assert client.get("/health").json()["card_history_store"] == "redis"
        fresh = client.post("/predict", json={**SMALL_NIGHT_PURCHASE, "card_id": "quiet"}).json()
        for time, amount in BURST:
            client.post(
                "/predict", json={**EXAMPLE_TRANSACTION, "transaction_time": time, "amount": amount, "card_id": "burst"}
            )
    # A brand-new instance, as after a restart or on another replica, sharing the same Redis.
    with serving_with_redis(monkeypatch, server) as client:
        after_restart = client.post("/predict", json={**SMALL_NIGHT_PURCHASE, "card_id": "burst"}).json()
    assert after_restart["fraud_probability"] > 10 * fresh["fraud_probability"]


def test_health_reports_unavailable_when_redis_is_down(monkeypatch, server):
    with serving_with_redis(monkeypatch, server) as client:
        server.connected = False
        response = client.get("/health")
    assert response.status_code == 503
    assert response.json()["status"] == "unavailable"
