"""Cross-validate, fit, and select an Ames regression model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import KFold, cross_val_score

from .data import FEATURES, TARGET, load_data, split_data
from .modeling import candidate_models, regression_metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--report", type=Path, default=Path("reports/validation_metrics.json"))
    parser.add_argument("--seed", type=int, default=20250621)
    args = parser.parse_args(argv)

    train, validation, _ = split_data(load_data(args.data), seed=args.seed)
    folds = KFold(n_splits=3, shuffle=True, random_state=args.seed)
    results: dict[str, dict[str, float]] = {}
    fitted = {}
    for name, model in candidate_models(args.seed).items():
        cv_mae = -cross_val_score(
            model,
            train[FEATURES],
            train[TARGET],
            scoring="neg_mean_absolute_error",
            cv=folds,
            n_jobs=1,
        )
        model.fit(train[FEATURES], train[TARGET])
        metrics = regression_metrics(validation[TARGET], model.predict(validation[FEATURES]))
        metrics["cv_mae_mean"] = float(np.mean(cv_mae))
        metrics["cv_mae_std"] = float(np.std(cv_mae))
        results[name] = metrics
        fitted[name] = model
    best_name = min(results, key=lambda name: results[name]["mae"])
    price_bounds = train[TARGET].quantile([1 / 3, 2 / 3]).tolist()
    artifact = {
        "model": fitted[best_name],
        "model_name": best_name,
        "seed": args.seed,
        "features": FEATURES,
        "price_band_bounds": [float(value) for value in price_bounds],
    }
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.artifact)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps({"selected_model": best_name, "models": results}, indent=2), encoding="utf-8"
    )
    print(f"selected {best_name}; artifact written to {args.artifact}")


if __name__ == "__main__":
    main()
