"""Create deterministic synthetic housing data for smoke tests only."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_smoke_frame(rows: int = 360, seed: int = 20250621) -> pd.DataFrame:
    if rows < 80:
        raise ValueError("rows must be at least 80")
    rng = np.random.default_rng(seed)
    quality = rng.integers(2, 11, rows)
    area = np.maximum(450, rng.normal(1_550, 520, rows)).round()
    basement = np.maximum(0, area * rng.uniform(0.35, 0.8, rows)).round()
    garage = rng.choice([0, 1, 2, 3], rows, p=[0.08, 0.25, 0.57, 0.10])
    year = rng.integers(1900, 2011, rows)
    neighborhood = rng.choice(["NAmes", "CollgCr", "OldTown", "Edwards", "NridgHt"], rows)
    neighborhood_effect = (
        pd.Series(neighborhood)
        .map(
            {
                "NAmes": 0,
                "CollgCr": 25_000,
                "OldTown": -18_000,
                "Edwards": -10_000,
                "NridgHt": 60_000,
            }
        )
        .to_numpy()
    )
    price = (
        18_000
        + quality * 20_000
        + area * 58
        + basement * 20
        + garage * 13_000
        + (year - 1900) * 520
        + neighborhood_effect
        + rng.normal(0, 22_000, rows)
    )
    return pd.DataFrame(
        {
            "OverallQual": quality,
            "GrLivArea": area,
            "TotalBsmtSF": basement,
            "GarageCars": garage,
            "YearBuilt": year,
            "FullBath": rng.choice([1, 2, 3], rows, p=[0.35, 0.58, 0.07]),
            "LotArea": np.maximum(1_500, rng.lognormal(9.0, 0.45, rows)).round(),
            "Neighborhood": neighborhood,
            "HouseStyle": rng.choice(["1Story", "2Story", "1.5Fin", "SLvl"], rows),
            "CentralAir": rng.choice(["Y", "N"], rows, p=[0.92, 0.08]),
            "KitchenQual": rng.choice(["Ex", "Gd", "TA", "Fa"], rows, p=[0.08, 0.42, 0.45, 0.05]),
            "SaleCondition": rng.choice(
                ["Normal", "Abnorml", "Partial"], rows, p=[0.82, 0.10, 0.08]
            ),
            "SalePrice": np.maximum(35_000, price).round(2),
        }
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke.csv"))
    parser.add_argument("--rows", type=int, default=360)
    parser.add_argument("--seed", type=int, default=20250621)
    args = parser.parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_smoke_frame(args.rows, args.seed).to_csv(args.output, index=False)
    print(f"synthetic smoke data written to {args.output}")


if __name__ == "__main__":
    main()
