"""Train the final model and export it as a versioned bundle for serving.

    fraud-train --version 1.0.0

The bundle (see fraud_core.model) holds the network weights, the fitted
preprocessor and a metadata file describing how the model was built. A
version is never overwritten: retraining means publishing a new version, so
any prediction can be traced back to the exact model that produced it.

The decision threshold is chosen here, on cv, with the same cost model as
notebooks/04_threshold.ipynb. The test set is still not used.
"""

import argparse
import json
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path

import joblib
import numpy as np

from fraud_core.columns import TARGET, TIME
from fraud_core.features import RAW_INPUTS, Preprocessor
from fraud_core.model import METADATA_FILE, PREPROCESSOR_FILE, WEIGHTS_FILE, DenseLayer, FraudModel, save_weights
from fraud_training.data import KAGGLE_DATASET, ROOT, load_raw, split_by_time
from fraud_training.evaluate import FALSE_ALARM_COST, cost_by_threshold, evaluate
from fraud_training.train import FINAL_PARAMS, train_neural_net

MODELS_DIR = ROOT.parent / "models"

# The NumPy forward pass must reproduce Keras up to float32 rounding.
MAX_PREDICTION_DIFF = 1e-5


def keras_to_layers(model) -> list[DenseLayer]:
    """Extract the weights and activation of every Dense layer of a Keras model."""
    layers = []
    for layer in model.layers:
        W, b = layer.get_weights()
        layers.append(DenseLayer(W=W, b=b, activation=layer.get_config()["activation"]))
    return layers


def git_commit() -> str | None:
    """Commit the code was trained from, with "-dirty" if there were uncommitted changes."""
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def describe(part) -> dict:
    return {
        "from": str(part[TIME].min()),
        "to": str(part[TIME].max()),
        "rows": len(part),
        "frauds": int(part[TARGET].sum()),
    }


def export_bundle(out_dir: Path, layers: list[DenseLayer], preprocessor: Preprocessor, metadata: dict) -> None:
    out_dir.mkdir(parents=True)
    save_weights(out_dir / WEIGHTS_FILE, layers)
    joblib.dump(preprocessor, out_dir / PREPROCESSOR_FILE)
    (out_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--version", required=True, help="model version, e.g. 1.0.0")
    parser.add_argument("--output", type=Path, default=MODELS_DIR, help="folder that holds every model version")
    args = parser.parse_args()

    out_dir = args.output / args.version
    if out_dir.exists():
        parser.error(f"{out_dir} already exists; publish a new version instead of overwriting one")

    split = split_by_time(load_raw())
    prep = Preprocessor().fit(split.train)
    X_train, y_train = prep.transform(split.train)
    X_cv, y_cv = prep.transform(split.cv)

    print("Training the network...", flush=True)
    model, history = train_neural_net(X_train, y_train, X_cv, y_cv, **FINAL_PARAMS)
    cv_prob = model.predict(X_cv, batch_size=4096, verbose=0).ravel()

    costs = cost_by_threshold(y_cv, cv_prob, split.cv["amt"].to_numpy(), false_alarm_cost=FALSE_ALARM_COST)
    threshold = float(costs["total_cost"].idxmin())
    at_threshold = evaluate(y_cv, cv_prob, threshold=threshold)

    metadata = {
        "version": args.version,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "dataset": f"kaggle:{KAGGLE_DATASET}",
        "train_data": describe(split.train),
        "cv_data": describe(split.cv),
        "training": {
            **FINAL_PARAMS,
            "epochs_run": len(history.history["val_loss"]),
            "best_epoch": int(np.argmin(history.history["val_loss"])) + 1,
        },
        "threshold": threshold,
        "decision_cost": {
            "false_alarm_cost": FALSE_ALARM_COST,
            "missed_fraud_cost": "purchase amount",
            "cv_total_cost": round(float(costs.loc[threshold, "total_cost"]), 2),
        },
        "cv_metrics": {
            "pr_auc": round(at_threshold["pr_auc"], 4),
            "precision": round(at_threshold["precision"], 4),
            "recall": round(at_threshold["recall"], 4),
        },
        "inputs": RAW_INPUTS,
        "features": prep.feature_names,
        "trained_with": {
            "python": platform.python_version(),
            **{lib: package_version(lib) for lib in ["numpy", "pandas", "scikit-learn", "tensorflow"]},
        },
    }

    # Write to a temporary folder first, so a failed check never leaves a half-published version.
    staging = out_dir.with_name(f".{args.version}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    export_bundle(staging, keras_to_layers(model), prep, metadata)

    # Reload the bundle exactly as serving will, and check it reproduces training.
    served = FraudModel.load(staging).predict_proba(split.cv)
    diff = float(np.abs(served - cv_prob).max())
    if diff > MAX_PREDICTION_DIFF:
        raise RuntimeError(f"bundle predictions differ from Keras by {diff:.2e}; left unpublished in {staging}")
    staging.rename(out_dir)

    print(f"Exported model {args.version} to {out_dir}")
    print(f"  threshold {threshold} | cv PR-AUC {metadata['cv_metrics']['pr_auc']} | "
          f"precision {metadata['cv_metrics']['precision']} | recall {metadata['cv_metrics']['recall']}")
    print(f"  bundle matches Keras on cv (max difference {diff:.1e})")


if __name__ == "__main__":
    main()
