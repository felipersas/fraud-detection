"""Compare learning rates and stopping rules for the neural network on the cv set.

The learning curves of the first network were still going down at epoch 100,
so training stopped before the network had finished learning. This script
compares three ways of training it, with the fraud weight fixed at the value
picked by src/tune_fraud_weight.py:

- the original setup: small learning rate, a fixed 100 epochs;
- a 10x larger learning rate, with early stopping;
- the small learning rate, with early stopping and room for up to 300 epochs.

Each setup runs with several seeds, so averages can be compared instead of luck.

    fraud-tune-training
"""

import time

import numpy as np
import pandas as pd

from fraud_core.features import Preprocessor
from fraud_training.data import ROOT, load_raw, split_by_time
from fraud_training.evaluate import evaluate
from fraud_training.train import train_neural_net

FRAUD_WEIGHT = 10
SEEDS = [0, 1, 2]
PATIENCE = 10  # epochs without J_cv improvement before stopping
SETUPS = {
    "lr 1e-4, 100 epochs": {"learning_rate": 1e-4, "max_epochs": 100, "patience": None},
    "lr 1e-3, early stopping": {"learning_rate": 1e-3, "max_epochs": 300, "patience": PATIENCE},
    "lr 1e-4, early stopping": {"learning_rate": 1e-4, "max_epochs": 300, "patience": PATIENCE},
}
RESULTS_PATH = ROOT / "reports" / "training_search.csv"


def main():
    split = split_by_time(load_raw())
    prep = Preprocessor().fit(split.train)
    X_train, y_train = prep.transform(split.train)
    X_cv, y_cv = prep.transform(split.cv)

    rows = []
    for name, setup in SETUPS.items():
        for seed in SEEDS:
            start = time.time()
            model, history = train_neural_net(X_train, y_train, X_cv, y_cv, FRAUD_WEIGHT, seed=seed, **setup)
            y_prob = model.predict(X_cv, batch_size=4096, verbose=0).ravel()
            val_loss = history.history["val_loss"]
            rows.append(
                {
                    "setup": name,
                    "seed": seed,
                    "pr_auc": evaluate(y_cv, y_prob)["pr_auc"],
                    "epochs_run": len(val_loss),
                    # With early stopping, the weights kept are those of the best epoch.
                    "best_epoch": int(np.argmin(val_loss)) + 1,
                    "minutes": (time.time() - start) / 60,
                }
            )
            print(rows[-1], flush=True)
            # Save after every run so a long search can be inspected midway.
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    summary = (
        pd.DataFrame(rows)
        .groupby("setup", sort=False)
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            best_epoch=("best_epoch", "mean"),
            minutes=("minutes", "mean"),
        )
    )
    print(summary.round(3))


if __name__ == "__main__":
    main()
