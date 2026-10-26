"""API tests against the model bundle committed in models/."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from fraud_api.main import EXAMPLE_TRANSACTION, Transaction, app, find_bundle, to_frame
from fraud_core.model import FraudModel

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

# Same customer as the example, but a small daytime purchase in a low-risk store.
CALM_TRANSACTION = {**EXAMPLE_TRANSACTION, "transaction_time": "2020-06-17T14:10:00", "amount": 25.0, "category": "food_dining"}


@pytest.fixture(scope="module")
def client(monkeypatch_module):
    monkeypatch_module.setenv("FRAUD_MODELS_DIR", str(MODELS_DIR))
    with TestClient(app) as client:  # the context manager runs the startup that loads the model
        yield client


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp


@pytest.fixture(scope="module")
def bundle_model():
    return FraudModel.load(find_bundle(MODELS_DIR, version=None))


def test_health_reports_the_served_version(client, bundle_model):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": bundle_model.version}


def test_serves_the_highest_version_by_default(client):
    versions = [p.name for p in MODELS_DIR.iterdir() if p.is_dir() and not p.name.startswith(".")]
    highest = max(versions, key=lambda v: tuple(int(x) for x in v.split(".")))
    assert client.get("/model").json()["version"] == highest


def test_prediction_matches_the_bundle(client, bundle_model):
    body = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
    expected = bundle_model.predict_proba(to_frame([Transaction(**EXAMPLE_TRANSACTION)]))[0]
    assert body["fraud_probability"] == pytest.approx(expected, abs=1e-6)
    assert body["threshold"] == bundle_model.threshold
    assert body["is_fraud"] == (expected >= bundle_model.threshold)


def test_night_online_purchase_is_riskier_than_daytime_meal(client):
    # A behaviour check: the model must keep the main pattern it learned from the data.
    risky = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
    calm = client.post("/predict", json=CALM_TRANSACTION).json()
    assert risky["is_fraud"] and not calm["is_fraud"]
    assert risky["fraud_probability"] > calm["fraud_probability"]


def test_batch_returns_one_prediction_per_transaction(client):
    response = client.post("/predict/batch", json={"transactions": [EXAMPLE_TRANSACTION, CALM_TRANSACTION]})
    assert response.status_code == 200
    assert [p["is_fraud"] for p in response.json()["predictions"]] == [True, False]


def test_timezone_offset_does_not_shift_the_hour(client):
    with_offset = {**EXAMPLE_TRANSACTION, "transaction_time": EXAMPLE_TRANSACTION["transaction_time"] + "+05:00"}
    plain = client.post("/predict", json=EXAMPLE_TRANSACTION).json()
    assert client.post("/predict", json=with_offset).json() == plain


def test_unknown_category_is_rejected(client):
    response = client.post("/predict", json={**EXAMPLE_TRANSACTION, "category": "casino"})
    assert response.status_code == 422
    assert "unknown category" in response.json()["detail"]


@pytest.mark.parametrize(
    "field, value",
    [("amount", -5), ("gender", "X"), ("customer_latitude", 120), ("transaction_time", "yesterday")],
)
def test_invalid_input_is_rejected(client, field, value):
    assert client.post("/predict", json={**EXAMPLE_TRANSACTION, field: value}).status_code == 422


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"transactions": []}).status_code == 422
