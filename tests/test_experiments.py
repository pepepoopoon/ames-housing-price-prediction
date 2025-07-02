from __future__ import annotations

import json

from ames_housing.experiments import ExperimentConfig, main, run_experiment


def test_experiment_is_reproducible() -> None:
    config = ExperimentConfig(label="deterministic", seed=17, rows=100, train_fraction=0.8)

    first = run_experiment(config)
    second = run_experiment(config)

    assert first == second
    assert first["sample_sizes"] == {"full": 100, "train": 48, "validation": 20, "test": 20}
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
