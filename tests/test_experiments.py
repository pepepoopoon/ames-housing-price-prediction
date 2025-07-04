from __future__ import annotations

import json

from ames_housing.data import split_data
from ames_housing.experiments import (
    ExperimentConfig,
    apply_stress,
    degrade_training_data,
    main,
    measure_learning_curve,
    measure_seed_stability,
    run_experiment,
    segment_diagnostics,
)
from ames_housing.generate_smoke_data import generate_smoke_frame


def test_experiment_is_reproducible() -> None:
    config = ExperimentConfig(label="deterministic", seed=17, rows=100, train_fraction=0.8)

    first = run_experiment(config)
    second = run_experiment(config)

    assert first == second
    assert first["sample_sizes"] == {
        "full": 100,
        "train": 48,
        "train_after_quality": 48,
        "validation": 20,
        "test": 20,
    }
    assert first["models"]["ridge_log_target"]["test"]["mae"] > 0


def test_experiment_compares_candidate_with_median_baseline() -> None:
    result = run_experiment(
        ExperimentConfig(label="baseline", seed=31, rows=120, train_fraction=1.0)
    )

    assert set(result["models"]) == {"dummy_median", "ridge_log_target"}
    comparison = result["baseline_comparison"]
    assert comparison["baseline"] == "dummy_median"
    assert comparison["candidate"] == "ridge_log_target"
    assert comparison["test_mae_reduction"] > 0
    assert comparison["test_mae_reduction_pct"] > 0


def test_stress_diagnostics_measure_unseen_categories() -> None:
    config = ExperimentConfig(
        label="unknown-category",
        seed=37,
        rows=120,
        train_fraction=1.0,
        stress="unseen_neighborhood",
        stress_level=0.25,
    )

    result = run_experiment(config)
    stressed = apply_stress(generate_smoke_frame(80), config.stress, config.stress_level, 1)

    assert (stressed["Neighborhood"] == "FutureDistrict").sum() == 20
    assert result["stress_diagnostics"]["scenario"] == "unseen_neighborhood"
    assert result["stress_diagnostics"]["metrics"]["mae"] > 0


def test_segment_diagnostics_cover_each_test_observation() -> None:
    config = ExperimentConfig(label="segments", seed=41, rows=150, train_fraction=1.0)

    result = run_experiment(config)
    segments = result["segment_diagnostics"]

    for dimension in ["neighborhood", "quality_band", "construction_era"]:
        assert sum(group["count"] for group in segments[dimension].values()) == 30
        assert all(group["mae"] >= 0 for group in segments[dimension].values())

    frame = generate_smoke_frame(80)
    perfect = segment_diagnostics(frame, frame["SalePrice"].to_numpy())
    assert all(group["mae"] == 0 for group in perfect["quality_band"].values())


def test_learning_curve_uses_increasing_train_samples() -> None:
    config = ExperimentConfig(
        label="curve",
        seed=43,
        rows=150,
        train_fraction=1.0,
        learning_fractions=(1.0, 0.25, 0.5),
    )

    curve = run_experiment(config)["learning_curve"]

    assert [point["fraction"] for point in curve] == [0.25, 0.5, 1.0]
    assert [point["train_rows"] for point in curve] == [22, 45, 90]
    assert all(point["validation_mae"] > 0 for point in curve)

    frame = generate_smoke_frame(100)
    train, validation, _ = split_data(frame, seed=43)
    empty_config = ExperimentConfig(label="empty-curve", seed=43, rows=100, train_fraction=1.0)
    assert measure_learning_curve(train, validation, empty_config) == []


def test_seed_stability_repeats_the_full_pipeline() -> None:
    stability_config = ExperimentConfig(
        label="stable", seed=101, rows=100, train_fraction=0.75, repeat_seeds=(101, 103, 107)
    )
    stability = measure_seed_stability(stability_config)

    assert [run["seed"] for run in stability["runs"]] == [101, 103, 107]
    assert stability["summary"]["run_count"] == 3
    assert stability["summary"]["mae_min"] <= stability["summary"]["mae_mean"]
    assert stability["summary"]["mae_mean"] <= stability["summary"]["mae_max"]

    result = run_experiment(
        ExperimentConfig(
            label="seeds",
            seed=47,
            rows=100,
            train_fraction=1.0,
            repeat_seeds=(109, 113),
        )
    )
    assert result["seed_stability"]["summary"]["run_count"] == 2


def test_hyperparameters_select_the_requested_model() -> None:
    config = ExperimentConfig(
        label="small-forest",
        seed=53,
        rows=100,
        train_fraction=1.0,
        model="random_forest",
        n_estimators=12,
        max_depth=2,
        min_samples_leaf=3,
    )

    result = run_experiment(config)

    assert set(result["models"]) == {"dummy_median", "random_forest"}
    assert result["baseline_comparison"]["candidate"] == "random_forest"
    assert result["config"]["n_estimators"] == 12
    assert result["models"]["random_forest"]["test"]["mae"] > 0


def test_data_quality_scenarios_report_realized_defects() -> None:
    frame = generate_smoke_frame(100).head(60)
    config = ExperimentConfig(
        label="quality",
        seed=59,
        rows=100,
        train_fraction=1.0,
        missing_numeric_rate=0.1,
        missing_categorical_rate=0.15,
        duplicate_rate=0.2,
        target_noise_scale=0.05,
    )

    degraded, diagnostics = degrade_training_data(frame, config, seed=2_059)

    assert len(degraded) == 72
    assert diagnostics["duplicate_rows_added"] == 12
    assert diagnostics["numeric_missing_added"] > 0
    assert diagnostics["categorical_missing_added"] > 0
    assert not degraded["SalePrice"].equals(frame["SalePrice"])

    result = run_experiment(config)
    assert result["data_quality_diagnostics"]["rows_after"] > result["sample_sizes"]["train"]
    assert result["models"]["ridge_log_target"]["test"]["mae"] > 0


def test_experiment_cli_writes_json(tmp_path) -> None:
    output = tmp_path / "result.json"

    main(
        [
            "--output",
            str(output),
            "--label",
            "cli-check",
            "--seed",
            "23",
            "--rows",
            "90",
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["config"]["label"] == "cli-check"
    assert result["config"]["seed"] == 23
