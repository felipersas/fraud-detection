"""API tests against the model bundles committed in models/."""

from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud_api.main import EXAMPLE_TRANSACTION, Transaction, app, find_bundle, to_frame
from fraud_core.history import HISTORY_COLUMNS
from fraud_core.model import FraudModel

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
V1, V2 = "1.0.1", "2.0.1"

# Same customer as the example, but a small daytime purchase in a low-risk store.
CALM_TRANSACTION = {**EXAMPLE_TRANSACTION, "transaction_time": "2020-06-17T14:10:00", "amount": 25.0, "category": "food_dining"}
# A small night purchase, the kind fraudsters mix into a burst.
SMALL_NIGHT_PURCHASE = {**EXAMPLE_TRANSACTION, "transaction_time": "2020-06-17T02:10:00", "amount": 18.0, "category": "personal_care"}


@contextmanager
def serving(monkeypatch, version=None):
    """A fresh API instance (empty card history) serving `version`, or the highest one."""
    monkeypatch.setenv("FRAUD_MODELS_DIR", str(MODELS_DIR))
    if version:
        monkeypatch.setenv("FRAUD_MODEL_VERSION", version)
    else:
        monkeypatch.delenv("FRAUD_MODEL_VERSION", raising=False)
    with TestClient(app) as client:  # the context manager runs the startup that loads the model
        yield client


@pytest.fixture
def client(monkeypatch):
    with serving(monkeypatch) as client:
        yield client


@pytest.fixture
def client_v1(monkeypatch):
    with serving(monkeypatch, V1) as client:
        yield client


@pytest.fixture
def client_v2(monkeypatch):
    with serving(monkeypatch, V2) as client:
        yield client


def without_card(transaction: dict) -> dict:
    return {k: v for k, v in transaction.items() if k != "card_id"}


# Behaviour shared by every model version


def test_legacy_bundles_are_not_served(monkeypatch):
    # 2.0.0 pickles scikit-learn objects; the API points to its repackaged version instead.
    monkeypatch.setenv("FRAUD_MODELS_DIR", str(MODELS_DIR))
    monkeypatch.setenv("FRAUD_MODEL_VERSION", "2.0.0")
    with pytest.raises(RuntimeError, match="legacy format"):
        with TestClient(app):
            pass


def test_serves_the_highest_version_by_default(client):
    highest = find_bundle(MODELS_DIR, version=None).name
    assert client.get("/health").json() == {"status": "ok", "model_version": highest, "card_history_store": "memory"}


@pytest.mark.parametrize("version", [V1, V2])
def test_serves_the_requested_version(monkeypatch, version):
    with serving(monkeypatch, version) as client:
        assert client.get("/model").json()["version"] == version


def test_unknown_category_is_rejected(client):
    response = client.post("/predict", json={**EXAMPLE_TRANSACTION, "category": "casino"})
    assert response.status_code == 422
    assert "unknown category" in response.json()["detail"]


@pytest.mark.parametrize(
    "field, value",
    [("amount", -5), ("gender", "X"), ("customer_latitude", 120), ("transaction_time", "yesterday"), ("card_id", "")],
)
def test_invalid_input_is_rejected(client, field, value):
    assert client.post("/predict", json={**EXAMPLE_TRANSACTION, field: value}).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422


def test_timezone_offset_does_not_shift_the_hour(monkeypatch):
    with_offset = {**EXAMPLE_TRANSACTION, "transaction_time": EXAMPLE_TRANSACTION["transaction_time"] + "+05:00"}
    # Fresh instances, so the second request does not see the first one in the card's history.
    with serving(monkeypatch) as client:
        plain = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
    with serving(monkeypatch) as client:
        assert client.post("/predict", json=with_offset).json() == plain


@pytest.mark.parametrize("version", [V1, V2])
def test_night_online_purchase_is_riskier_than_daytime_meal(monkeypatch, version):
    # The main pattern learned from the data must survive in every version.
    with serving(monkeypatch, version) as client:
        risky = client.post("/predict", json={**EXAMPLE_TRANSACTION, "card_id": "a"}).json()
        calm = client.post("/predict", json={**CALM_TRANSACTION, "card_id": "b"}).json()
    assert risky["is_fraud"] and not calm["is_fraud"]
    assert risky["fraud_probability"] > calm["fraud_probability"]


# Model 1.x: each purchase is scored on its own


def test_v1_prediction_matches_the_bundle(client_v1):
    body = client_v1.post("/predict", json=EXAMPLE_TRANSACTION).json()
    bundle = FraudModel.load(MODELS_DIR / V1)
    expected = bundle.predict_proba(to_frame([Transaction(**EXAMPLE_TRANSACTION)]))[0]
    assert body["fraud_probability"] == pytest.approx(expected, abs=1e-6)
    assert body["threshold"] == bundle.threshold
    assert body["model_version"] == V1


def test_v1_does_not_need_a_card(client_v1):
    assert client_v1.post("/predict", json=without_card(EXAMPLE_TRANSACTION)).status_code == 200


def test_v1_ignores_earlier_purchases(client_v1):
    first = client_v1.post("/predict", json=SMALL_NIGHT_PURCHASE).json()
    client_v1.post("/predict", json=EXAMPLE_TRANSACTION)
    assert client_v1.post("/predict", json=SMALL_NIGHT_PURCHASE).json() == first


# Model 2.x: the card's recent history is part of the input


def test_v2_requires_a_card(client_v2):
    response = client_v2.post("/predict", json=without_card(EXAMPLE_TRANSACTION))
    assert response.status_code == 422
    assert "card_id is required" in response.json()["detail"]


def test_v2_first_purchase_matches_the_bundle_with_empty_history(client_v2):
    body = client_v2.post("/predict", json=EXAMPLE_TRANSACTION).json()
    bundle = FraudModel.load(MODELS_DIR / V2)
    frame = to_frame([Transaction(**EXAMPLE_TRANSACTION)])
    # A card seen for the first time has no purchases in the window and the capped gap.
    frame = frame.join(pd.DataFrame([{c: 0.0 for c in HISTORY_COLUMNS} | {"hours_since_prev": 720.0}]))
    assert body["fraud_probability"] == pytest.approx(bundle.predict_proba(frame)[0], abs=1e-6)
    assert body["model_version"] == V2


def test_v2_burst_raises_the_risk_of_a_small_purchase(client_v2):
    fresh = client_v2.post("/predict", json={**SMALL_NIGHT_PURCHASE, "card_id": "quiet"}).json()
    for time, amount in [("2020-06-17T00:33:41", 1167.06), ("2020-06-17T00:56:36", 369.32)]:
        burst = {**EXAMPLE_TRANSACTION, "transaction_time": time, "amount": amount, "card_id": "burst"}
        client_v2.post("/predict", json=burst)
    after_burst = client_v2.post("/predict", json={**SMALL_NIGHT_PURCHASE, "card_id": "burst"}).json()
    assert after_burst["fraud_probability"] > 10 * fresh["fraud_probability"]


def test_v2_other_cards_do_not_leak_into_history(client_v2):
    alone = client_v2.post("/predict", json={**SMALL_NIGHT_PURCHASE, "card_id": "first-card"}).json()
    # A burst on another card, just before the same purchase on a third, fresh card.
    for time in ["2020-06-17T00:33:41", "2020-06-17T00:56:36"]:
        client_v2.post("/predict", json={**EXAMPLE_TRANSACTION, "transaction_time": time, "card_id": "busy-card"})
    fresh = client_v2.post("/predict", json={**SMALL_NIGHT_PURCHASE, "card_id": "fresh-card"}).json()
    assert fresh == alone


def test_v2_batch_scores_in_time_order(monkeypatch):
    burst = {**EXAMPLE_TRANSACTION, "card_id": "card"}
    small = {**SMALL_NIGHT_PURCHASE, "card_id": "card"}
    # Sent out of order: the batch must still score the earlier purchase first.
    with serving(monkeypatch, V2) as client:
        batch = client.post("/predict/batch", json={"transactions": [small, burst]}).json()["predictions"]
    with serving(monkeypatch, V2) as client:
        one_by_one = [client.post("/predict", json=burst).json(), client.post("/predict", json=small).json()]
    assert batch == [one_by_one[1], one_by_one[0]]
