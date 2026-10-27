"""Repackage a published model bundle in the current bundle format, without retraining.

    fraud-repackage --from 2.0.0 --to 2.0.1

The new version holds exactly the same weights, threshold and preprocessing
statistics; only the file format changes (bundles 1.0.0 and 2.0.0 pickled
scikit-learn objects, newer ones store the preprocessor as JSON). That is a
patch release: nothing about the model's behaviour changes, which this
command checks on the full cv set before publishing.
"""

import argparse
import json
import shutil
from datetime import UTC, datetime

import numpy as np

from fraud_core.history import card_history
from fraud_core.model import FraudModel
from fraud_training.data import load_raw, split_by_time
from fraud_training.export import BUNDLE_FORMAT, MODELS_DIR, export_bundle, git_commit

# Same model, same arithmetic: predictions must agree to floating-point rounding.
MAX_PREDICTION_DIFF = 1e-9


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--from", dest="source", required=True, help="published version to repackage, e.g. 2.0.0")
    parser.add_argument("--to", dest="target", required=True, help="new patch version, e.g. 2.0.1")
    args = parser.parse_args()

    out_dir = MODELS_DIR / args.target
    if out_dir.exists():
        parser.error(f"{out_dir} already exists; publish a new version instead of overwriting one")

    old = FraudModel.load(MODELS_DIR / args.source)
    metadata = {
        **old.metadata,
        "version": args.target,
        "bundle_format": BUNDLE_FORMAT,
        "repackaged_from": {
            "version": args.source,
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "git_commit": git_commit(),
        },
    }

    staging = out_dir.with_name(f".{args.target}.partial")
    shutil.rmtree(staging, ignore_errors=True)
    export_bundle(staging, old.layers, old.preprocessor, metadata)

    # Score the whole cv set with both bundles; the new one must behave identically.
    df = load_raw()
    if old.uses_card_history:
        df = df.join(card_history(df))
    cv = split_by_time(df).cv
    diff = float(np.abs(FraudModel.load(staging).predict_proba(cv) - old.predict_proba(cv)).max())
    if diff > MAX_PREDICTION_DIFF:
        raise RuntimeError(f"repackaged predictions differ by {diff:.2e}; left unpublished in {staging}")
    staging.rename(out_dir)

    print(f"Repackaged {args.source} as {args.target} in {out_dir}")
    print(f"  identical on cv (max difference {diff:.1e})")
    print(json.dumps({k: metadata[k] for k in ["version", "bundle_format", "threshold"]}))


if __name__ == "__main__":
    main()
