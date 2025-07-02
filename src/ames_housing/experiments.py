"""Run deterministic offline experiments on the synthetic Ames benchmark."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

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


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    """Compare Ridge with a median baseline on a deterministic synthetic split."""
    if config.rows < 80:
        raise ValueError("rows must be at least 80")
    if not 0.2 <= config.train_fraction <= 1.0:
        raise ValueError("train_fraction must be between 0.2 and 1.0")

    frame = validate_frame(generate_smoke_frame(config.rows, config.seed))
    train, validation, test = split_data(frame, seed=config.seed)
    train = train.sample(frac=config.train_fraction, random_state=config.seed).reset_index(
        drop=True
    )
    models = candidate_models(config.seed)
    measured: dict[str, dict[str, dict[str, float]]] = {}
    for name in ["dummy_median", "ridge_log_target"]:
        model = models[name]
        model.fit(train[FEATURES], train[TARGET])
        measured[name] = {
            "validation": regression_metrics(
                validation[TARGET], model.predict(validation[FEATURES])
            ),
            "test": regression_metrics(test[TARGET], model.predict(test[FEATURES])),
        }

    baseline_mae = measured["dummy_median"]["test"]["mae"]
    candidate_mae = measured["ridge_log_target"]["test"]["mae"]

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
    args = parser.parse_args(argv)
    config = ExperimentConfig(
        label=args.label,
        seed=args.seed,
        rows=args.rows,
        train_fraction=args.train_fraction,
    )
    write_result(run_experiment(config), args.output)
    print(f"experiment {config.label} written to {args.output}")


if __name__ == "__main__":
    main()
