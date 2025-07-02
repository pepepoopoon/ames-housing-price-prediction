"""Regression pipelines and metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES


class ModelArtifactError(ValueError):
    """Raised when a saved model is incompatible with the current feature contract."""


def validate_model_artifact(artifact: object) -> dict[str, object]:
    """Require an estimator and the exact feature order recorded during training."""
    if not isinstance(artifact, dict):
        raise ModelArtifactError("model artifact must be a dictionary")
    if artifact.get("features") != FEATURES:
        raise ModelArtifactError("model artifact feature schema is incompatible")
    model = artifact.get("model")
    if not callable(getattr(model, "predict", None)):
        raise ModelArtifactError("model artifact must contain an estimator with predict")
    return artifact


def preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("numeric", numeric, NUMERIC_FEATURES), ("categorical", categorical, CATEGORICAL_FEATURES)]
    )


def _log_target(regressor: object) -> TransformedTargetRegressor:
    return TransformedTargetRegressor(
        regressor=regressor, func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def candidate_models(seed: int) -> dict[str, Pipeline]:
    estimators = {
        "dummy_median": DummyRegressor(strategy="median"),
        "linear_regression_log_target": _log_target(LinearRegression()),
        "ridge_log_target": _log_target(Ridge(alpha=5.0)),
        "lasso_log_target": _log_target(Lasso(alpha=0.001, max_iter=5_000, random_state=seed)),
        "random_forest": RandomForestRegressor(
            n_estimators=80, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.05, max_depth=3, random_state=seed
        ),
    }
    return {
        name: Pipeline([("preprocess", preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def regression_metrics(truth: pd.Series | np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(truth, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(truth, predictions))),
        "r2": float(r2_score(truth, predictions)),
    }
