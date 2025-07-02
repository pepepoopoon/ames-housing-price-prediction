"""Predict sale prices from a trained Ames artifact."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import FEATURES, load_data
from .modeling import validate_model_artifact


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/model.joblib"))
    parser.add_argument("--output", type=Path, default=Path("reports/predictions.csv"))
    args = parser.parse_args(argv)
    artifact = validate_model_artifact(joblib.load(args.artifact))
    frame = load_data(args.data, require_target=False)
    predictions = artifact["model"].predict(frame[FEATURES])
    output = pd.DataFrame({"row_id": frame.index, "predicted_sale_price": predictions})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    print(f"predictions written to {args.output}")


if __name__ == "__main__":
    main()
