"""Evaluate the frozen Ames model and analyze residuals by price band."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .data import FEATURES, TARGET, load_data, split_data
from .modeling import regression_metrics, validate_model_artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--metrics", type=Path, default=Path("reports/test_metrics.json"))
    parser.add_argument("--errors", type=Path, default=Path("reports/largest_errors.csv"))
    args = parser.parse_args(argv)

    artifact = validate_model_artifact(joblib.load(args.artifact))
    _, _, test = split_data(load_data(args.data), seed=int(artifact["seed"]))
    predictions = artifact["model"].predict(test[FEATURES])
    metrics: dict[str, object] = regression_metrics(test[TARGET], predictions)
    low, high = artifact["price_band_bounds"]
    bands = pd.cut(
        test[TARGET],
        bins=[-np.inf, low, high, np.inf],
        labels=["cheap", "middle", "expensive"],
    )
    metrics["mae_by_price_band"] = {
        str(band): float(
            mean_absolute_error(test.loc[bands == band, TARGET], predictions[bands == band])
        )
        for band in bands.cat.categories
        if (bands == band).any()
    }
    metrics["model_name"] = artifact["model_name"]
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    errors = test[["Neighborhood", "OverallQual", "GrLivArea", TARGET]].copy()
    errors["prediction"] = predictions
    errors["residual"] = test[TARGET].to_numpy() - predictions
    errors["absolute_error"] = np.abs(errors["residual"])
    errors["price_band"] = bands.astype(str)
    errors = errors.sort_values("absolute_error", ascending=False).head(25)
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    errors.to_csv(args.errors, index=False)
    print(f"test metrics written to {args.metrics}; largest errors written to {args.errors}")


if __name__ == "__main__":
    main()
