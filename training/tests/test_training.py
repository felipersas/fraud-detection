"""Training pipeline tests on synthetic data, so they run without the real download."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraud_core.columns import TIME
from fraud_core.features import CATEGORICAL, NUMERIC, Preprocessor, make_features
from fraud_core.model import FraudModel
from fraud_training.data import split_by_time
from fraud_training.error_analysis import count_tags
from fraud_training.evaluate import cost_by_threshold, evaluate
from fraud_training.export import keras_to_layers
from fraud_training.train import build_neural_net


def test_split_by_time_train_is_past(transactions):
    s = split_by_time(transactions)
    assert s.train[TIME].max() < s.cv[TIME].min()
    assert s.cv[TIME].max() < s.test[TIME].min()
    assert len(s.train) + len(s.cv) + len(s.test) == len(transactions)


def test_lazy_model_has_high_accuracy_and_zero_recall():
    y = np.array([0] * 998 + [1] * 2)
    m = evaluate(y, np.zeros(len(y)))
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0


def test_count_tags_allows_overlapping_tags():
    sample = pd.DataFrame({"amt": [10, 20, 900]})
    table = count_tags(sample, {"small": sample["amt"] < 100, "any": sample["amt"] > 0})
    assert table.loc["any", "errors"] == 3
    assert table.loc["small", "share_%"] == pytest.approx(66.7)


def test_cost_by_threshold_prices_each_error_type():
    y = np.array([1, 1, 0, 0])
    prob = np.array([0.9, 0.2, 0.6, 0.1])
    amounts = np.array([500.0, 30.0, 80.0, 20.0])
    costs = cost_by_threshold(y, prob, amounts, false_alarm_cost=10, thresholds=np.array([0.5]))
    # At 0.5: the $30 fraud is missed and the $80 legit purchase is flagged.
    row = costs.loc[0.5]
    assert row["missed_frauds"] == 1 and row["false_alarms"] == 1
    assert row["total_cost"] == pytest.approx(30 + 10)


def test_exported_layers_reproduce_keras():
    # An untrained network is enough: the point is that NumPy and Keras compute the same thing.
    model = build_neural_net(n_features=5)
    X = np.random.default_rng(0).normal(size=(50, 5)).astype("float32")
    numpy_model = FraudModel(keras_to_layers(model), preprocessor=None, metadata={})
    np.testing.assert_allclose(numpy_model.forward(X), model.predict(X, verbose=0).ravel(), atol=1e-6)


def test_numpy_preprocessor_matches_scikit_learn(transactions):
    # The Preprocessor reimplements StandardScaler and OneHotEncoder so serving needs no
    # scikit-learn; both must produce exactly the same features.
    prep = Preprocessor().fit(transactions)
    features = make_features(transactions)
    reference = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    ).fit(features)
    np.testing.assert_array_equal(prep.transform_features(transactions), reference.transform(features))
    assert prep.feature_names == list(reference.get_feature_names_out())
