"""Run deterministic offline experiments on the synthetic Ames benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import FEATURES, TARGET, split_data, validate_frame
from .generate_smoke_data import generate_smoke_frame
from .modeling import candidate_models, regression_metrics


@dataclass(frozen=True)
class ExperimentConfig:
    """Parameters that uniquely describe one reproducible experiment."""

    label: str
    seed: int
    rows: int
    train_fraction: float
    stress: str = "none"
    stress_level: float = 0.1
    learning_fractions: tuple[float, ...] = ()


def apply_stress(frame: pd.DataFrame, kind: str, level: float, seed: int) -> pd.DataFrame:
    """Create a deterministic inference-only perturbation without changing the target."""
    if not 0 <= level <= 0.5:
        raise ValueError("stress_level must be between 0 and 0.5")
    stressed = frame.copy()
    rng = np.random.default_rng(seed)
    if kind == "none":
        return stressed
    if kind == "numeric_missing":
        mask = rng.random(len(stressed)) < level
        stressed.loc[mask, "TotalBsmtSF"] = np.nan
    elif kind == "unseen_neighborhood":
        count = max(1, round(len(stressed) * level))
        stressed.loc[stressed.index[:count], "Neighborhood"] = "FutureDistrict"
    elif kind == "area_scale":
        stressed["GrLivArea"] *= 1 + level
        stressed["LotArea"] *= 1 + level
    elif kind == "numeric_noise":
        for column in ["GrLivArea", "TotalBsmtSF"]:
            scale = stressed[column].std() * level
            stressed[column] = stressed[column] + rng.normal(0, scale, len(stressed))
    else:
        raise ValueError(f"unknown stress scenario: {kind}")
    return stressed


def _group_error_metrics(
    truth: pd.Series, predictions: np.ndarray, groups: pd.Series
) -> dict[str, dict[str, float | int]]:
    errors = truth.to_numpy(dtype=float) - predictions
    summaries: dict[str, dict[str, float | int]] = {}
    normalized_groups = groups.astype(str)
    for group in sorted(normalized_groups.unique()):
        mask = normalized_groups.to_numpy() == group
        group_errors = errors[mask]
        summaries[group] = {
            "count": int(mask.sum()),
            "mae": float(np.mean(np.abs(group_errors))),
            "rmse": float(np.sqrt(np.mean(np.square(group_errors)))),
            "mean_residual": float(np.mean(group_errors)),
        }
    return summaries


def segment_diagnostics(
    frame: pd.DataFrame, predictions: np.ndarray
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Measure errors across location, quality, and construction-era slices."""
    quality_band = pd.cut(
        frame["OverallQual"],
        bins=[-np.inf, 5, 7, np.inf],
        labels=["basic", "standard", "premium"],
    )
    construction_era = pd.cut(
        frame["YearBuilt"],
        bins=[-np.inf, 1945, 1979, 1999, np.inf],
        labels=["pre_1946", "1946_1979", "1980_1999", "post_1999"],
    )
    return {
        "neighborhood": _group_error_metrics(frame[TARGET], predictions, frame["Neighborhood"]),
        "quality_band": _group_error_metrics(frame[TARGET], predictions, quality_band),
        "construction_era": _group_error_metrics(frame[TARGET], predictions, construction_era),
    }


def measure_learning_curve(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    seed: int,
    fractions: tuple[float, ...],
) -> list[dict[str, float | int]]:
    """Fit independent Ridge models on increasing deterministic train samples."""
    points: list[dict[str, float | int]] = []
    for fraction in sorted(set(fractions)):
        if not 0.2 <= fraction <= 1.0:
            raise ValueError("learning fractions must be between 0.2 and 1.0")
        sample = train.sample(frac=fraction, random_state=seed).reset_index(drop=True)
        model = candidate_models(seed)["ridge_log_target"]
        model.fit(sample[FEATURES], sample[TARGET])
        metrics = regression_metrics(validation[TARGET], model.predict(validation[FEATURES]))
        points.append(
            {
                "fraction": fraction,
                "train_rows": len(sample),
                "validation_mae": metrics["mae"],
                "validation_rmse": metrics["rmse"],
                "validation_r2": metrics["r2"],
            }
        )
    return points


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Compare Ridge with a median baseline on a deterministic synthetic split."""
    if config.rows < 80:
        raise ValueError("rows must be at least 80")
    if not 0.2 <= config.train_fraction <= 1.0:
        raise ValueError("train_fraction must be between 0.2 and 1.0")

    frame = validate_frame(generate_smoke_frame(config.rows, config.seed))
    full_train, validation, test = split_data(frame, seed=config.seed)
    learning_curve = measure_learning_curve(
        full_train, validation, config.seed, config.learning_fractions
    )
    train = full_train.sample(frac=config.train_fraction, random_state=config.seed).reset_index(
        drop=True
    )
    models = candidate_models(config.seed)
    measured: dict[str, dict[str, dict[str, float]]] = {}
    fitted = {}
    for name in ["dummy_median", "ridge_log_target"]:
        model = models[name]
        model.fit(train[FEATURES], train[TARGET])
        measured[name] = {
            "validation": regression_metrics(
                validation[TARGET], model.predict(validation[FEATURES])
            ),
            "test": regression_metrics(test[TARGET], model.predict(test[FEATURES])),
        }
        fitted[name] = model

    baseline_mae = measured["dummy_median"]["test"]["mae"]
    candidate_mae = measured["ridge_log_target"]["test"]["mae"]
    stressed_test = apply_stress(test, config.stress, config.stress_level, config.seed + 1_000)
    stressed_metrics = regression_metrics(
        stressed_test[TARGET], fitted["ridge_log_target"].predict(stressed_test[FEATURES])
    )
    test_predictions = fitted["ridge_log_target"].predict(test[FEATURES])

    return {
        "schema_version": 1,
        "config": asdict(config),
        "sample_sizes": {
            "full": len(frame),
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "models": measured,
        "baseline_comparison": {
            "baseline": "dummy_median",
            "candidate": "ridge_log_target",
            "test_mae_reduction": baseline_mae - candidate_mae,
            "test_mae_reduction_pct": 100 * (baseline_mae - candidate_mae) / baseline_mae,
        },
        "stress_diagnostics": {
            "scenario": config.stress,
            "level": config.stress_level,
            "metrics": stressed_metrics,
            "mae_change": stressed_metrics["mae"] - candidate_mae,
            "rmse_change": stressed_metrics["rmse"] - measured["ridge_log_target"]["test"]["rmse"],
        },
        "segment_diagnostics": segment_diagnostics(test, test_predictions),
        "learning_curve": learning_curve,
    }


def write_result(result: dict[str, object], output: Path) -> None:
    """Write a stable, reviewable JSON artifact."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--seed", type=int, default=20250621)
    parser.add_argument("--rows", type=int, default=360)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument(
        "--stress",
        choices=["none", "numeric_missing", "unseen_neighborhood", "area_scale", "numeric_noise"],
        default="none",
    )
    parser.add_argument("--stress-level", type=float, default=0.1)
    parser.add_argument(
        "--learning-fractions",
        default="",
        help="Comma-separated fractions from 0.2 to 1.0",
    )
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        label=args.label,
        seed=args.seed,
        rows=args.rows,
        train_fraction=args.train_fraction,
        stress=args.stress,
        stress_level=args.stress_level,
        learning_fractions=tuple(
            float(value) for value in args.learning_fractions.split(",") if value
        ),
    )
    write_result(run_experiment(config), args.output)
    print(f"experiment {config.label} written to {args.output}")


if __name__ == "__main__":
    main()
