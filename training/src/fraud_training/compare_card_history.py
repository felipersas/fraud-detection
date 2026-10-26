"""Compare the network with and without card-history features on the cv set.

Both feature sets use the final training setup (FINAL_PARAMS) and the same
seeds, so the only difference is the information the network receives. Each
run picks its own cost-based threshold, the way a released model would.

    fraud-compare-card-history
"""

import pandas as pd

from fraud_core.features import Preprocessor
from fraud_core.history import card_history
from fraud_training.data import ROOT, load_raw, split_by_time
from fraud_training.evaluate import FALSE_ALARM_COST, cost_by_threshold, evaluate
from fraud_training.train import FINAL_PARAMS, train_neural_net

SEEDS = [0, 1, 2]
RESULTS_PATH = ROOT / "reports" / "card_history_search.csv"


def main():
    df = load_raw()
    # Computed on the whole timeline at once; each purchase only sees earlier ones.
    df = df.join(card_history(df))
    split = split_by_time(df)
    amounts = split.cv["amt"].to_numpy()

    rows = []
    for use_history in [False, True]:
        prep = Preprocessor(card_history=use_history).fit(split.train)
        X_train, y_train = prep.transform(split.train)
        X_cv, y_cv = prep.transform(split.cv)
        for seed in SEEDS:
            model, history = train_neural_net(X_train, y_train, X_cv, y_cv, **{**FINAL_PARAMS, "seed": seed})
            y_prob = model.predict(X_cv, batch_size=4096, verbose=0).ravel()
            costs = cost_by_threshold(y_cv, y_prob, amounts, false_alarm_cost=FALSE_ALARM_COST)
            threshold = costs["total_cost"].idxmin()
            rows.append({
                "features": "transaction + card history" if use_history else "transaction only",
                "seed": seed,
                "pr_auc": evaluate(y_cv, y_prob)["pr_auc"],
                "epochs_run": len(history.history["val_loss"]),
                "threshold": threshold,
                "missed_frauds": int(costs.loc[threshold, "missed_frauds"]),
                "false_alarms": int(costs.loc[threshold, "false_alarms"]),
                "total_cost": float(costs.loc[threshold, "total_cost"]),
            })
            print(rows[-1], flush=True)
            # Save after every run so a long comparison can be inspected midway.
            pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)

    summary = pd.DataFrame(rows).groupby("features", sort=False)[
        ["pr_auc", "missed_frauds", "false_alarms", "total_cost"]
    ].agg(["mean", "std"])
    print(summary.round(3))


if __name__ == "__main__":
    main()
