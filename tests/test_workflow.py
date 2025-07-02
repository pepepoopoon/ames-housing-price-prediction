from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import pandas as pd

from ames_housing.data import FEATURES, SchemaError, split_data, validate_frame
from ames_housing.evaluate import main as evaluate
from ames_housing.generate_smoke_data import generate_smoke_frame
from ames_housing.modeling import ModelArtifactError
from ames_housing.predict import main as predict
from ames_housing.train import main as train


class AmesWorkflowTest(unittest.TestCase):
    def test_schema_and_split(self) -> None:
        frame = validate_frame(generate_smoke_frame(120))
        pd.testing.assert_frame_equal(generate_smoke_frame(120), generate_smoke_frame(120))
        train_frame, validation, test = split_data(frame)
        self.assertEqual(len(frame), len(train_frame) + len(validation) + len(test))
        with self.assertRaises(SchemaError):
            validate_frame(frame.assign(SalePrice=-1))

    def test_schema_rejects_infinite_feature_and_target(self) -> None:
        frame = generate_smoke_frame(120)
        for column in ["GrLivArea", "SalePrice"]:
            invalid = frame.copy()
            invalid.loc[0, column] = float("inf")
            with self.assertRaisesRegex(SchemaError, "finite"):
                validate_frame(invalid)

    def test_predict_rejects_incompatible_artifact_feature_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "inference.csv"
            artifact_path = root / "stale.joblib"
            output_path = root / "predictions.csv"
            generate_smoke_frame(80).drop(columns=["SalePrice"]).head(2).to_csv(
                data_path, index=False
            )
            joblib.dump({"model": object(), "features": list(reversed(FEATURES))}, artifact_path)

            with self.assertRaisesRegex(ModelArtifactError, "feature schema"):
                predict(
                    [
                        "--data",
                        str(data_path),
                        "--artifact",
                        str(artifact_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertFalse(output_path.exists())

    def test_end_to_end_without_downloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "smoke.csv"
            artifact = root / "model.joblib"
            validation = root / "validation.json"
            metrics = root / "metrics.json"
            errors = root / "errors.csv"
            predictions = root / "predictions.csv"
            frame = generate_smoke_frame(160)
            frame.to_csv(data_path, index=False)
            train(
                ["--data", str(data_path), "--artifact", str(artifact), "--report", str(validation)]
            )
            evaluate(
                [
                    "--data",
                    str(data_path),
                    "--artifact",
                    str(artifact),
                    "--metrics",
                    str(metrics),
                    "--errors",
                    str(errors),
                ]
            )
            inference = root / "inference.csv"
            frame.drop(columns=["SalePrice"]).head(6).to_csv(inference, index=False)
            predict(
                [
                    "--data",
                    str(inference),
                    "--artifact",
                    str(artifact),
                    "--output",
                    str(predictions),
                ]
            )
            self.assertIn("mae_by_price_band", json.loads(metrics.read_text(encoding="utf-8")))
            self.assertEqual(len(pd.read_csv(predictions)), 6)


if __name__ == "__main__":
    unittest.main()
