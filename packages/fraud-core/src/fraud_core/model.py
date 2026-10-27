"""Load a trained model bundle and score transactions with it.

A bundle is a folder written by the training package (`fraud-train`):

- model.onnx: the network in ONNX, the standard exchange format for trained models
- preprocessor.json: what the Preprocessor learned on the training set (means, scales, categories)
- metadata.json: version, training data, cv metrics and the decision threshold

The network runs on ONNX Runtime, a small inference engine that executes an
exported model without the framework it was trained with, so serving does not
need TensorFlow. The graph is the network's forward propagation:
MatMul -> Add -> activation, once per Dense layer.
"""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

from fraud_core.features import Preprocessor

ONNX_FILE = "model.onnx"
PREPROCESSOR_FILE = "preprocessor.json"
METADATA_FILE = "metadata.json"


def onnx_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    # The network is tiny; one thread per request avoids competing with the web
    # server's own worker threads for CPU cores.
    options.intra_op_num_threads = 1
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def save_preprocessor(path: Path, preprocessor: Preprocessor) -> None:
    path.write_text(json.dumps(preprocessor.to_dict(), indent=2) + "\n")


def load_preprocessor(path: Path) -> Preprocessor:
    return Preprocessor.from_dict(json.loads(path.read_text()))


class FraudModel:
    """A trained model ready to score raw transactions.

    It applies the same feature engineering as training (through the fitted
    Preprocessor), runs the ONNX network, and turns the fraud probability into
    a decision with the threshold chosen by cost during training.
    """

    def __init__(self, session: ort.InferenceSession, preprocessor: Preprocessor, metadata: dict):
        self.session = session
        self.preprocessor = preprocessor
        self.metadata = metadata

    @classmethod
    def load(cls, bundle_dir: str | Path) -> "FraudModel":
        """Load a bundle written by training."""
        bundle_dir = Path(bundle_dir)
        return cls(
            session=onnx_session(bundle_dir / ONNX_FILE),
            preprocessor=load_preprocessor(bundle_dir / PREPROCESSOR_FILE),
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

    @property
    def n_features(self) -> int:
        return self.session.get_inputs()[0].shape[1]

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Run already-transformed features through the network; returns P(fraud) per row."""
        # The graph was exported with float32 inputs, the precision the network was trained in.
        inputs = {self.session.get_inputs()[0].name: X.astype(np.float32)}
        return self.session.run(None, inputs)[0].ravel().astype(np.float64)

    def predict_proba(self, transactions: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each raw transaction."""
        return self.forward(self.preprocessor.transform_features(transactions))

    def predict(self, transactions: pd.DataFrame) -> np.ndarray:
        """True where a transaction should be flagged as fraud."""
        return self.predict_proba(transactions) >= self.threshold
