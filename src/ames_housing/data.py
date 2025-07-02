"""Ames input contract and deterministic splits."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TARGET = "SalePrice"
NUMERIC_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "TotalBsmtSF",
    "GarageCars",
    "YearBuilt",
    "FullBath",
    "LotArea",
]
CATEGORICAL_FEATURES = [
    "Neighborhood",
    "HouseStyle",
    "CentralAir",
    "KitchenQual",
    "SaleCondition",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class SchemaError(ValueError):
    """Raised when Ames data violates the training contract."""


def validate_frame(frame: pd.DataFrame, *, require_target: bool = True) -> pd.DataFrame:
    required = FEATURES + ([TARGET] if require_target else [])
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    if frame.empty:
        raise SchemaError("dataset is empty")
    clean = frame.copy()
    numeric = NUMERIC_FEATURES + ([TARGET] if require_target else [])
    for column in numeric:
        converted = pd.to_numeric(clean[column], errors="coerce")
        if converted.isna().sum() > clean[column].isna().sum():
            raise SchemaError(f"{column} contains non-numeric values")
        if not np.isfinite(converted.dropna().to_numpy(dtype=float)).all():
            raise SchemaError(f"{column} must contain only finite values")
        clean[column] = converted
    for column in ["OverallQual", "GrLivArea", "TotalBsmtSF", "GarageCars", "FullBath", "LotArea"]:
        if (clean[column].dropna() < 0).any():
            raise SchemaError(f"{column} cannot be negative")
    years = clean["YearBuilt"].dropna()
    if ((years < 1800) | (years > 2100)).any():
        raise SchemaError("YearBuilt is outside the supported range")
    if require_target:
        if clean[TARGET].isna().any() or (clean[TARGET] <= 0).any():
            raise SchemaError("SalePrice must be positive and non-missing")
    return clean


def load_data(path: str | Path, *, require_target: bool = True) -> pd.DataFrame:
    return validate_frame(pd.read_csv(path), require_target=require_target)


def split_data(
    frame: pd.DataFrame, *, seed: int = 20250621
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_validation, test = train_test_split(frame, test_size=0.20, random_state=seed)
    train, validation = train_test_split(train_validation, test_size=0.25, random_state=seed)
    return (
        train.reset_index(drop=True),
        validation.reset_index(drop=True),
        test.reset_index(drop=True),
    )
