"""HTTP API that scores card transactions with a published fraud model bundle.

    uv run fraud-api                     # from the repository root

Configuration (environment variables):
- FRAUD_MODELS_DIR: folder holding one subfolder per model version (default: models)
- FRAUD_MODEL_VERSION: version to serve (default: the highest version found)
- HOST / PORT: where to listen (default: 0.0.0.0:8000)
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_core.columns import TIME
from fraud_core.model import FraudModel

logger = logging.getLogger("fraud_api")

MAX_BATCH_SIZE = 1000

EXAMPLE_TRANSACTION = {
    "transaction_time": "2020-06-17T00:33:41",
    "amount": 1167.06,
    "category": "shopping_net",
    "gender": "F",
    "date_of_birth": "1985-03-02",
    "city_population": 2500,
    "customer_latitude": 40.71,
    "customer_longitude": -74.0,
    "merchant_latitude": 40.95,
    "merchant_longitude": -73.6,
}


class Transaction(BaseModel):
    """One card purchase, described with the same information the model was trained on."""

    model_config = ConfigDict(json_schema_extra={"examples": [EXAMPLE_TRANSACTION]})

    transaction_time: datetime = Field(description="local wall-clock time of the purchase")
    amount: float = Field(gt=0, description="purchase amount in USD")
    category: str = Field(description="store category, e.g. grocery_pos or shopping_net (see GET /model)")
    gender: Literal["F", "M"] = Field(description="cardholder gender")
    date_of_birth: date = Field(description="cardholder date of birth")
    city_population: int = Field(ge=0, description="population of the cardholder's home city")
    customer_latitude: float = Field(ge=-90, le=90)
    customer_longitude: float = Field(ge=-180, le=180)
    merchant_latitude: float = Field(ge=-90, le=90)
    merchant_longitude: float = Field(ge=-180, le=180)
    card_id: str | None = Field(
        default=None,
        description="card identifier; unused by 1.x models, which score each purchase on its own",
    )

    @field_validator("transaction_time")
    @classmethod
    def drop_timezone(cls, value: datetime) -> datetime:
        # The model learned from local wall-clock hours (fraud peaks at night), so the
        # hour must stay the customer's local hour even if a UTC offset is attached.
        return value.replace(tzinfo=None)


class Batch(BaseModel):
    transactions: list[Transaction] = Field(min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    fraud_probability: float = Field(description="probability that the purchase is fraud, from 0 to 1")
    is_fraud: bool = Field(description="decision: true when the probability reaches the model's threshold")
    threshold: float
    model_version: str


class BatchPrediction(BaseModel):
    predictions: list[Prediction]


def find_bundle(models_dir: Path, version: str | None) -> Path:
    """Folder of the requested version, or of the highest version when none is given."""
    if version:
        return models_dir / version
    # Folders like ".1.1.0.partial" are unfinished exports and must never be served.
    versions = [p for p in models_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
    if not versions:
        raise RuntimeError(f"no model bundle found in {models_dir}")
    return max(versions, key=lambda p: tuple(int(part) for part in p.name.split(".")))


def to_frame(transactions: list[Transaction]) -> pd.DataFrame:
    """Rename API fields to the raw column names the feature pipeline expects."""
    return pd.DataFrame(
        {
            TIME: pd.to_datetime([t.transaction_time for t in transactions]),
            "amt": [t.amount for t in transactions],
            "category": [t.category for t in transactions],
            "gender": [t.gender for t in transactions],
            "dob": pd.to_datetime([t.date_of_birth for t in transactions]),
            "city_pop": [t.city_population for t in transactions],
            "lat": [t.customer_latitude for t in transactions],
            "long": [t.customer_longitude for t in transactions],
            "merch_lat": [t.merchant_latitude for t in transactions],
            "merch_long": [t.merchant_longitude for t in transactions],
        }
    )


def score(model: FraudModel, transactions: list[Transaction]) -> list[Prediction]:
    known = model.preprocessor.categories["category"]
    unknown = sorted({t.category for t in transactions} - set(known))
    if unknown:
        # The encoder would silently map an unseen category to "none of them";
        # refusing is safer than returning a score the model was never trained for.
        raise HTTPException(status_code=422, detail=f"unknown category {unknown}; expected one of {known}")

    start = time.perf_counter()
    probabilities = model.predict_proba(to_frame(transactions))
    predictions = [
        Prediction(
            fraud_probability=round(float(p), 6),
            is_fraud=bool(p >= model.threshold),
            threshold=model.threshold,
            model_version=model.version,
        )
        for p in probabilities
    ]
    # One line per request, enough to monitor volume, flag rate and latency per model version.
    logger.info(
        "scored transactions=%d flagged=%d model_version=%s latency_ms=%.1f",
        len(predictions),
        sum(p.is_fraud for p in predictions),
        model.version,
        1000 * (time.perf_counter() - start),
    )
    return predictions


@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = find_bundle(Path(os.environ.get("FRAUD_MODELS_DIR", "models")), os.environ.get("FRAUD_MODEL_VERSION"))
    app.state.model = FraudModel.load(bundle)
    logger.info("loaded model version=%s from %s", app.state.model.version, bundle)
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Scores card transactions with a neural network trained on the Sparkov dataset.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.model.version}


@app.get("/model")
def model_info(request: Request) -> dict:
    """What the served model is, how it was trained and which inputs it accepts."""
    model: FraudModel = request.app.state.model
    keys = ["version", "created_at", "git_commit", "train_data", "threshold", "decision_cost", "cv_metrics"]
    return {**{k: model.metadata[k] for k in keys}, "categories": model.preprocessor.categories}


@app.post("/predict")
def predict(transaction: Transaction, request: Request) -> Prediction:
    return score(request.app.state.model, [transaction])[0]


@app.post("/predict/batch")
def predict_batch(batch: Batch, request: Request) -> BatchPrediction:
    return BatchPrediction(predictions=score(request.app.state.model, batch.transactions))


def run():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "8000")))
