"""Metrics for a heavily imbalanced problem.

Accuracy is useless here: always predicting "not fraud" scores 99.83%.
The metrics below focus on the rare class instead:
- precision: of the transactions flagged as fraud, how many really were.
- recall: of the frauds that happened, how many the model caught.
- F1: harmonic mean of precision and recall at a single threshold.
- PR-AUC (average precision): summarizes the precision-recall curve across
  every threshold. It is the headline metric of this project.
- ROC-AUC is reported for reference only. With 0.17% positives the false
  positive rate stays tiny even when the model raises many false alarms,
  so ROC-AUC looks great while precision is poor.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    PrecisionRecallDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Assumed cost of flagging a legit purchase (support call, blocked card, annoyed
# customer). It is a business assumption, not something learned from data;
# notebooks/04_threshold.ipynb shows how the chosen threshold moves with it.
FALSE_ALARM_COST = 10


def evaluate(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute every metric from the predicted fraud probabilities.

    Threshold-based metrics (precision, recall, F1) use `threshold`;
    PR-AUC and ROC-AUC are threshold-free and use the raw probabilities.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
    }


def confusion(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Confusion matrix with readable labels (rows = truth, columns = prediction)."""
    cm = confusion_matrix(y_true, (y_prob >= threshold).astype(int), labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=["actual: legit", "actual: fraud"],
        columns=["pred: legit", "pred: fraud"],
    )


def results_table(results: dict[str, dict]) -> pd.DataFrame:
    """Comparison table with one row per model, best PR-AUC first."""
    return pd.DataFrame(results).T.sort_values("pr_auc", ascending=False).round(4)


def plot_pr_curves(y_true: np.ndarray, probs: dict[str, np.ndarray], ax=None):
    """Plot the precision-recall curves of several models on the same axes.

    The dashed line is a random classifier: its precision equals the fraud
    rate at every recall level.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    for name, y_prob in probs.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_prob, name=name, ax=ax, curve_kwargs={"lw": 2})
    ax.axhline(y_true.mean(), ls="--", c="gray", label=f"random ({y_true.mean():.4f})")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall curve (cv set)")
    ax.legend()
    return ax


def cost_by_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    false_alarm_cost: float,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    """Total cost of acting on the model's decisions, for each threshold.

    Cost model:
    - a missed fraud costs its purchase amount, since the bank refunds the customer;
    - a false alarm costs a fixed `false_alarm_cost` (support call, blocked card,
      annoyed customer);
    - correct decisions cost nothing.

    Pricing missed frauds by their amount means letting a $20 fraud through is
    much cheaper than letting a $900 one through, which is how a bank sees it.
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.01, 1.0, 0.01), 2)
    rows = []
    for t in thresholds:
        flagged = y_prob >= t
        missed = (y_true == 1) & ~flagged
        false_alarms = (y_true == 0) & flagged
        missed_cost = amounts[missed].sum()
        false_alarm_total = false_alarm_cost * false_alarms.sum()
        rows.append(
            {
                "threshold": t,
                "missed_frauds": int(missed.sum()),
                "false_alarms": int(false_alarms.sum()),
                "missed_fraud_cost": missed_cost,
                "false_alarm_cost": false_alarm_total,
                "total_cost": missed_cost + false_alarm_total,
            }
        )
    return pd.DataFrame(rows).set_index("threshold")
