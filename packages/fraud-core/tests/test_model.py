"""Tests for loading a model bundle and running the network with NumPy."""

import numpy as np
import pytest

from fraud_core.model import DenseLayer, FraudModel, load_weights, save_weights


@pytest.fixture
def layers():
    rng = np.random.default_rng(0)
    return [
        DenseLayer(rng.normal(size=(3, 4)), rng.normal(size=4), "relu"),
        DenseLayer(rng.normal(size=(4, 1)), rng.normal(size=1), "sigmoid"),
    ]


def test_forward_pass_matches_the_math(layers):
    X = np.array([[0.5, -1.0, 2.0], [0.0, 0.0, 0.0]])
    hidden = np.maximum(X @ layers[0].W + layers[0].b, 0)
    expected = 1 / (1 + np.exp(-(hidden @ layers[1].W + layers[1].b)))
    model = FraudModel(layers, preprocessor=None, metadata={})
    np.testing.assert_allclose(model.forward(X), expected.ravel())


def test_weights_survive_a_save_and_load(layers, tmp_path):
    save_weights(tmp_path / "weights.npz", layers)
    loaded = load_weights(tmp_path / "weights.npz")
    assert [layer.activation for layer in loaded] == ["relu", "sigmoid"]
    for original, restored in zip(layers, loaded, strict=True):
        np.testing.assert_array_equal(original.W, restored.W)
        np.testing.assert_array_equal(original.b, restored.b)
