"""Run deterministic offline experiments on the synthetic Ames benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    split_data,
    validate_frame,
)
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
    repeat_seeds: tuple[int, ...] = ()
    model: str = "ridge_log_target"
    ridge_alpha: float = 5.0
    n_estimators: int = 80
    max_depth: int = 3
    learning_rate: float = 0.05
    min_samples_leaf: int = 2
    missing_numeric_rate: float = 0.0
    missing_categorical_rate: float = 0.0
    duplicate_rate: float = 0.0
    target_noise_scale: float = 0.0


def build_experiment_model(config: ExperimentConfig, seed: int):
    """Build a fresh candidate with the hyperparameters recorded in the config."""
    models = candidate_models(seed)
    if config.model == "ridge_log_target":
        return models[config.model].set_params(model__regressor__alpha=config.ridge_alpha)
    if config.model == "random_forest":
        return models[config.model].set_params(
            model__n_estimators=config.n_estimators,
            model__max_depth=config.max_depth,
            model__min_samples_leaf=config.min_samples_leaf,
        )
    if config.model == "gradient_boosting":
        return models[config.model].set_params(
            model__n_estimators=config.n_estimators,
            model__max_depth=config.max_depth,
            model__learning_rate=config.learning_rate,
        )
    raise ValueError(f"unsupported experiment model: {config.model}")


def degrade_training_data(
    train: pd.DataFrame, config: ExperimentConfig, seed: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Apply controlled train-only quality defects and report their realized size."""
    rates = [
        config.missing_numeric_rate,
        config.missing_categorical_rate,
        config.duplicate_rate,
        config.target_noise_scale,
    ]
    if any(not 0 <= value <= 0.5 for value in rates):
        raise ValueError("data-quality rates must be between 0 and 0.5")

    degraded = train.copy()
    rng = np.random.default_rng(seed)
    numeric_before = int(degraded[NUMERIC_FEATURES].isna().sum().sum())
    categorical_before = int(degraded[CATEGORICAL_FEATURES].isna().sum().sum())

    numeric_mask = rng.random((len(degraded), len(NUMERIC_FEATURES))) < config.missing_numeric_rate
    degraded[NUMERIC_FEATURES] = degraded[NUMERIC_FEATURES].mask(numeric_mask)
    categorical_mask = (
        rng.random((len(degraded), len(CATEGORICAL_FEATURES))) < config.missing_categorical_rate
    )
    degraded[CATEGORICAL_FEATURES] = degraded[CATEGORICAL_FEATURES].mask(categorical_mask)

    duplicate_rows = round(len(degraded) * config.duplicate_rate)
    if duplicate_rows:
        duplicated = degraded.sample(n=duplicate_rows, replace=True, random_state=seed)
        degraded = pd.concat([degraded, duplicated], ignore_index=True)

    if config.target_noise_scale:
        noise = rng.normal(0, config.target_noise_scale, len(degraded))
        degraded[TARGET] = np.maximum(1.0, degraded[TARGET].to_numpy() * (1 + noise))

    diagnostics: dict[str, float | int] = {
        "rows_before": len(train),
        "rows_after": len(degraded),
        "numeric_missing_added": int(degraded[NUMERIC_FEATURES].isna().sum().sum())
        - numeric_before,
        "categorical_missing_added": int(degraded[CATEGORICAL_FEATURES].isna().sum().sum())
        - categorical_before,
        "duplicate_rows_added": duplicate_rows,
        "target_noise_scale": config.target_noise_scale,
    }
    return degraded, diagnostics


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
    config: ExperimentConfig,
) -> list[dict[str, float | int]]:
    """Fit independent candidate models on increasing deterministic train samples."""
    points: list[dict[str, float | int]] = []
    for fraction in sorted(set(config.learning_fractions)):
        if not 0.2 <= fraction <= 1.0:
            raise ValueError("learning fractions must be between 0.2 and 1.0")
        sample = train.sample(frac=fraction, random_state=config.seed).reset_index(drop=True)
        model = build_experiment_model(config, config.seed)
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


def measure_seed_stability(config: ExperimentConfig) -> dict[str, object]:
    """Repeat data generation, splitting, and fitting for independent seeds."""
    runs: list[dict[str, float | int]] = []
    for seed in config.repeat_seeds:
        frame = validate_frame(generate_smoke_frame(config.rows, seed))
        train, _, test = split_data(frame, seed=seed)
        train = train.sample(frac=config.train_fraction, random_state=seed).reset_index(drop=True)
        train, _ = degrade_training_data(train, config, seed + 2_000)
        model = build_experiment_model(config, seed)
        model.fit(train[FEATURES], train[TARGET])
        metrics = regression_metrics(test[TARGET], model.predict(test[FEATURES]))
        runs.append({"seed": seed, **metrics})
    maes = np.array([run["mae"] for run in runs], dtype=float)
    return {
        "runs": runs,
        "summary": (
            {
                "run_count": len(runs),
                "mae_mean": float(maes.mean()),
                "mae_std": float(maes.std()),
                "mae_min": float(maes.min()),
                "mae_max": float(maes.max()),
            }
            if runs
            else {"run_count": 0}
        ),
    }


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Compare Ridge with a median baseline on a deterministic synthetic split."""
    if config.rows < 80:
        raise ValueError("rows must be at least 80")
    if not 0.2 <= config.train_fraction <= 1.0:
        raise ValueError("train_fraction must be between 0.2 and 1.0")

    frame = validate_frame(generate_smoke_frame(config.rows, config.seed))
    full_train, validation, test = split_data(frame, seed=config.seed)
    learning_curve = measure_learning_curve(full_train, validation, config)
    seed_stability = measure_seed_stability(config)
    train = full_train.sample(frac=config.train_fraction, random_state=config.seed).reset_index(
        drop=True
    )
    train_rows_before_quality = len(train)
    train, data_quality = degrade_training_data(train, config, config.seed + 2_000)
    models = {
        "dummy_median": candidate_models(config.seed)["dummy_median"],
        config.model: build_experiment_model(config, config.seed),
    }
    measured: dict[str, dict[str, dict[str, float]]] = {}
    fitted = {}
    for name in ["dummy_median", config.model]:
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
    candidate_mae = measured[config.model]["test"]["mae"]
    stressed_test = apply_stress(test, config.stress, config.stress_level, config.seed + 1_000)
    stressed_metrics = regression_metrics(
        stressed_test[TARGET], fitted[config.model].predict(stressed_test[FEATURES])
    )
    test_predictions = fitted[config.model].predict(test[FEATURES])

    return {
        "schema_version": 1,
        "config": asdict(config),
        "sample_sizes": {
            "full": len(frame),
            "train": train_rows_before_quality,
            "train_after_quality": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "models": measured,
        "baseline_comparison": {
            "baseline": "dummy_median",
            "candidate": config.model,
            "test_mae_reduction": baseline_mae - candidate_mae,
            "test_mae_reduction_pct": 100 * (baseline_mae - candidate_mae) / baseline_mae,
        },
        "stress_diagnostics": {
            "scenario": config.stress,
            "level": config.stress_level,
            "metrics": stressed_metrics,
            "mae_change": stressed_metrics["mae"] - candidate_mae,
            "rmse_change": stressed_metrics["rmse"] - measured[config.model]["test"]["rmse"],
        },
        "segment_diagnostics": segment_diagnostics(test, test_predictions),
        "learning_curve": learning_curve,
        "seed_stability": seed_stability,
        "data_quality_diagnostics": data_quality,
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
        "--model",
        choices=["ridge_log_target", "random_forest", "gradient_boosting"],
        default="ridge_log_target",
    )
    parser.add_argument("--ridge-alpha", type=float, default=5.0)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--missing-numeric-rate", type=float, default=0.0)
    parser.add_argument("--missing-categorical-rate", type=float, default=0.0)
    parser.add_argument("--duplicate-rate", type=float, default=0.0)
    parser.add_argument("--target-noise-scale", type=float, default=0.0)
    parser.add_argument(
        "--learning-fractions",
        default="",
        help="Comma-separated fractions from 0.2 to 1.0",
    )
    parser.add_argument(
        "--repeat-seeds",
        default="",
        help="Comma-separated seeds for full pipeline stability checks",
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
        repeat_seeds=tuple(int(value) for value in args.repeat_seeds.split(",") if value),
        model=args.model,
        ridge_alpha=args.ridge_alpha,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        missing_numeric_rate=args.missing_numeric_rate,
        missing_categorical_rate=args.missing_categorical_rate,
        duplicate_rate=args.duplicate_rate,
        target_noise_scale=args.target_noise_scale,
    )
    write_result(run_experiment(config), args.output)
    print(f"experiment {config.label} written to {args.output}")


if __name__ == "__main__":
    main()
