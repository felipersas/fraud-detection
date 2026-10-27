"""Load a trained model bundle and score transactions with it.

A bundle is a folder written by the training package (`fraud-train`):

- weights.npz: the weights and bias of every Dense layer, plus each layer's activation
- preprocessor.json: what the Preprocessor learned on the training set (means, scales, categories)
- metadata.json: version, training data, cv metrics and the decision threshold

Bundles 1.0.0 and 2.0.0 predate the JSON format and store a pickled,
scikit-learn based preprocessor.joblib instead. They still load wherever
scikit-learn and joblib are installed (the training environment); 1.0.1 and
2.0.1 are the same models repackaged, and they are what the API serves.

The network is evaluated with NumPy instead of TensorFlow. A Dense layer only
computes activation(X @ W + b), so a small stack of them needs a few lines of
code, and serving does not have to install TensorFlow at all.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_core.features import Preprocessor

WEIGHTS_FILE = "weights.npz"
PREPROCESSOR_FILE = "preprocessor.json"
LEGACY_PREPROCESSOR_FILE = "preprocessor.joblib"
METADATA_FILE = "metadata.json"

ACTIVATIONS = {
    "relu": lambda z: np.maximum(z, 0),
    "sigmoid": lambda z: 1 / (1 + np.exp(-z)),
    "linear": lambda z: z,
}


@dataclass
class DenseLayer:
    W: np.ndarray  # shape (inputs, units)
    b: np.ndarray  # shape (units,)
    activation: str

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return ACTIVATIONS[self.activation](X @ self.W + self.b)


def save_weights(path: Path, layers: list[DenseLayer]) -> None:
    """Write the layers to a single .npz file: W0, b0, W1, b1, ... and their activations."""
    arrays = {}
    for i, layer in enumerate(layers):
        arrays[f"W{i}"] = layer.W
        arrays[f"b{i}"] = layer.b
    arrays["activations"] = np.array([layer.activation for layer in layers])
    np.savez(path, **arrays)


def load_weights(path: Path) -> list[DenseLayer]:
    with np.load(path) as data:
        activations = [str(a) for a in data["activations"]]
        return [DenseLayer(data[f"W{i}"], data[f"b{i}"], act) for i, act in enumerate(activations)]


def save_preprocessor(path: Path, preprocessor: Preprocessor) -> None:
    path.write_text(json.dumps(preprocessor.to_dict(), indent=2) + "\n")


def load_preprocessor(bundle_dir: Path) -> Preprocessor:
    if (bundle_dir / PREPROCESSOR_FILE).exists():
        return Preprocessor.from_dict(json.loads((bundle_dir / PREPROCESSOR_FILE).read_text()))
    # Legacy bundles: unpickling can run arbitrary code, so only load bundles you produced.
    import joblib

    return joblib.load(bundle_dir / LEGACY_PREPROCESSOR_FILE)


class FraudModel:
    """A trained model ready to score raw transactions.

    It applies the same feature engineering as training (through the fitted
    Preprocessor), runs the network forward, and turns the fraud probability
    into a decision with the threshold chosen by cost during training.
    """

    def __init__(self, layers: list[DenseLayer], preprocessor: Preprocessor, metadata: dict):
        self.layers = layers
        self.preprocessor = preprocessor
        self.metadata = metadata

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "FraudModel":
        """Load a bundle written by training."""
        bundle_dir = Path(bundle_dir)
        return cls(
            layers=load_weights(bundle_dir / WEIGHTS_FILE),
            preprocessor=load_preprocessor(bundle_dir),
            metadata=json.loads((bundle_dir / METADATA_FILE).read_text()),
        )

    @property
    def version(self) -> str:
        return self.metadata["version"]

    @property
    def threshold(self) -> float:
        return self.metadata["threshold"]

    @property
    def uses_card_history(self) -> bool:
        """True when the model needs the card-history columns (fraud_core.history) as input."""
        return self.metadata.get("card_history", False)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run already-transformed features through the network; returns P(fraud) per row."""
        for layer in self.layers:
            X = layer(X)
        return X.ravel()

    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each raw transaction."""
        X = self.preprocessor.transform_features(transactions)
        return self.forward(X)

    def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        """True where a transaction should be flagged as fraud."""
        return self.predict_proba(transactions) >= self.threshold
