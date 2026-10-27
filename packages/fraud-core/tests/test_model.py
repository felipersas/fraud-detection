"""Tests for loading a model bundle and running its ONNX network."""

from pathlib import Path

import numpy as np
import pytest

from fraud_core.model import FraudModel

MODELS_DIR = Path(__file__).resolve().parents[3] / "models"


@pytest.fixture(scope="module")
def model():
    return FraudModel.load(MODELS_DIR / "2.0.2")


def test_bundle_describes_itself(model):
    assert model.version == "2.0.2"
    assert model.uses_card_history
    assert 0 < model.threshold < 1
    # 22 per-transaction features plus 5 card-history features.
    assert model.n_features == len(model.preprocessor.feature_names) == 27


def test_network_returns_one_probability_per_row(model):
    X = np.random.default_rng(0).normal(size=(100, model.n_features))
    p = model.forward(X)
    assert p.shape == (100,)
    assert ((p >= 0) & (p <= 1)).all()
