"""Search for the fraud class weight of the neural network on the cv set.

Each weight is trained with several seeds, because a single run of the same
network can land noticeably higher or lower just from its random starting
weights. Comparing averages tells a real effect apart from luck.

Everything except the weight (architecture, learning rate, epochs) is kept
fixed, so any difference in the results comes from the weight alone.

    fraud-tune-weight
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from fraud_training.data import ROOT, load_raw, split_by_time
from fraud_training.evaluate import evaluate
from fraud_core.features import Preprocessor
from fraud_training.train import build_neural_net

# 176 is the class ratio in train (legit / fraud); 1 means no weighting at all.
WEIGHTS = [1, 10, 50, 176, 400]
SEEDS = [0, 1, 2]
EPOCHS = 100
BATCH_SIZE = 512
RESULTS_PATH = ROOT / "reports" / "fraud_weight_search.csv"


def run_one(X_train, y_train, X_cv, y_cv, fraud_weight: float, seed: int) -> dict:
    """Train one network and return its cv metrics at the default 0.5 threshold."""
    tf.keras.utils.set_random_seed(seed)  # seeds Python, NumPy and TensorFlow together
    model = build_neural_net(n_features=X_train.shape[1])
    model.fit(
        X_train, y_train,
        class_weight={0: 1, 1: fraud_weight},
        epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
    )
    y_prob = model.predict(X_cv, batch_size=4096, verbose=0).ravel()
    y_pred = y_prob >= 0.5
    metrics = evaluate(y_cv, y_prob)
    return {
        "fraud_weight": fraud_weight,
        "seed": seed,
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "missed_frauds": int(((y_cv == 1) & ~y_pred).sum()),
        "false_alarms": int(((y_cv == 0) & y_pred).sum()),
    }


def main():
    split = split_by_time(load_raw())
    prep = Preprocessor().fit(split.train)
    X_train, y_train = prep.transform(split.train)
    X_cv, y_cv = prep.transform(split.cv)

    rows = []
    for weight in WEIGHTS:
        for seed in SEEDS:
            rows.append(run_one(X_train, y_train, X_cv, y_cv, weight, seed))
            print(rows[-1], flush=True)
            # Save after every run so a long search can be inspected (or resumed) midway.
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    summary = pd.DataFrame(rows).groupby("fraud_weight").agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        missed_frauds=("missed_frauds", "mean"),
        false_alarms=("false_alarms", "mean"),
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()
