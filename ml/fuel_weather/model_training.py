"""
Fuel & Weather - Machine Learning Model Training and Inference.
Predicts trip fuel consumption based on distance, vehicle type, and road conditions.
All training and predictions visibly disclose synthetic demonstration provenance.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import joblib
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from config.settings import MODELS_DIR
from shared.model_validation import summarize_cv_results

MODEL_PATH = MODELS_DIR / "fuel_model.joblib"
METADATA_PATH = MODELS_DIR / "fuel_metadata.json"

CATEGORICAL_FEATURES = [
    "vehicle_type",
    "fuel_type",
    "road_type",
    "weather_condition",
]

NUMERIC_FEATURES = [
    "trip_distance_km",
    "load_weight_kg",
    "temperature_c",
    "wind_speed_kmh",
    "rainfall_mm",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "fuel_consumed_liters"


def build_pipeline() -> Pipeline:
    """Build a scikit-learn preprocessing and regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    reg = HistGradientBoostingRegressor(
        max_iter=50,
        max_depth=6,
        random_state=42,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg),
    ])


def train_fuel_weather_model(df: pd.DataFrame, sample_size: int | None = 25000) -> dict:
    """Train a regression model to estimate trip fuel consumption.

    Visibly documents synthetic demonstration provenance.
    """
    if df.empty:
        raise ValueError("Cannot train model on empty DataFrame.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be a positive integer or None.")

    train_df = df.copy()
    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    # Keep only finite, strictly positive target values.
    target_values = pd.to_numeric(train_df[TARGET_COLUMN], errors="coerce")
    finite_targets = target_values.notna() & np.isfinite(target_values)
    excluded_target_rows = {
        "missing": int(train_df[TARGET_COLUMN].isna().sum()),
        "non_numeric": int((train_df[TARGET_COLUMN].notna() & target_values.isna()).sum()),
        "non_finite": int((target_values.notna() & ~np.isfinite(target_values)).sum()),
        "not_strictly_positive": int((finite_targets & (target_values <= 0)).sum()),
    }
    valid_targets = finite_targets & (target_values > 0)
    train_df = train_df.loc[valid_targets].copy()
    train_df[TARGET_COLUMN] = target_values.loc[valid_targets]
    if len(train_df) < 2:
        raise ValueError("Cannot train fuel model; at least two valid target rows are required.")

    for col in NUMERIC_FEATURES:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        else:
            train_df[col] = 0

    for col in CATEGORICAL_FEATURES:
        if col in train_df.columns:
            train_df[col] = train_df[col].astype(str)
        else:
            train_df[col] = "Unknown"

    timestamp_column = "timestamp" if "timestamp" in train_df.columns else None
    if timestamp_column:
        timestamps = pd.to_datetime(train_df[timestamp_column], errors="coerce")
        if timestamps.isna().any():
            raise ValueError(
                "Cannot train fuel model; timestamp values must all be valid for "
                "chronological validation."
            )
        train_df = train_df.assign(_validation_timestamp=timestamps).sort_values(
            "_validation_timestamp",
            kind="stable",
        )

    X = train_df[ALL_FEATURES]
    y = train_df[TARGET_COLUMN]
    splitter_configuration: dict
    split_indices = None

    if timestamp_column:
        for n_splits in range(min(5, len(X) // 2), 1, -1):
            candidate_splits = list(TimeSeriesSplit(n_splits=n_splits).split(X))
            if all(
                len(validation_indices) >= 2
                and y.iloc[validation_indices].nunique() >= 2
                for _, validation_indices in candidate_splits
            ):
                split_indices = candidate_splits
                break
        validation_method = "TimeSeriesSplit"
        if split_indices is None:
            raise ValueError(
                "Cannot train fuel model; chronological validation requires at least "
                "two folds with at least two varied-target validation samples each."
            )
        splitter_configuration = {
            "n_splits": n_splits,
            "shuffle": False,
            "timestamp_column": timestamp_column,
            "random_seed": None,
        }
    else:
        group_column = "vehicle_id" if "vehicle_id" in train_df.columns else None
        if group_column and train_df[group_column].nunique(dropna=False) < len(train_df):
            groups = train_df[group_column].astype("string").fillna("__missing_vehicle__")
            n_splits = min(5, int(groups.nunique()), len(X) // 2)
            if n_splits < 2:
                raise ValueError(
                    "Cannot train fuel model; at least two vehicle groups are required "
                    "for grouped validation."
                )
            split_indices = list(GroupKFold(n_splits=n_splits).split(X, y, groups))
            validation_method = "GroupKFold"
            splitter_configuration = {
                "n_splits": n_splits,
                "shuffle": False,
                "group_column": group_column,
            }
        else:
            n_splits = min(5, len(X) // 2)
            if n_splits < 2:
                raise ValueError(
                    "Cannot train fuel model; at least four valid target rows are "
                    "required for two-fold regression validation."
                )
            split_indices = list(KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X))
            validation_method = "KFold"
            splitter_configuration = {
                "n_splits": n_splits,
                "shuffle": True,
                "random_seed": 42,
            }

        if any(
            len(train_indices) < 2
            or len(validation_indices) < 2
            or y.iloc[validation_indices].nunique() < 2
            for train_indices, validation_indices in split_indices
        ):
            raise ValueError(
                "Cannot train fuel model; validation requires at least two training "
                "and validation samples per fold and non-constant validation targets."
            )

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }
    cv_results = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=split_indices,
        scoring=scoring,
        error_score="raise",
    )
    baseline_results = cross_validate(
        DummyRegressor(strategy="mean"),
        X,
        y,
        cv=split_indices,
        scoring={"mae": "neg_mean_absolute_error"},
        error_score="raise",
    )
    cv_metrics = summarize_cv_results(cv_results, scoring, negative_metrics={"mae", "rmse"})
    baseline_metrics = summarize_cv_results(
        baseline_results,
        {"mae": "neg_mean_absolute_error"},
        negative_metrics={"mae"},
    )

    pipeline = build_pipeline()
    pipeline.fit(X, y)

    metrics = {
        "model_name": "HistGradientBoostingRegressor",
        "task": "regression",
        "target": TARGET_COLUMN,
        "validation_method": validation_method,
        "validation_score_type": "cross_validation_estimate",
        "fold_count": n_splits,
        "splitter_configuration": splitter_configuration,
        "scoring_metrics": ["mae", "rmse", "r2"],
        "cv_metrics": cv_metrics,
        "provenance": "synthetic_demonstration_data",
        "is_synthetic": True,
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "train_samples": len(X),
        "excluded_target_rows": excluded_target_rows,
        "features": ALL_FEATURES.copy(),
        "feature_definitions": {
            "categorical": CATEGORICAL_FEATURES.copy(),
            "numeric": NUMERIC_FEATURES.copy(),
        },
        "leakage_exclusions": [
            "fuel_efficiency_km_per_liter",
            "fuel_consumption_rate",
            "fuel_cost_per_trip",
            "fuel_price_per_liter",
            "timestamp",
            "trip_id",
            "vehicle_id",
        ],
        "baseline_mae": baseline_metrics["mae"]["mean"],
        "baseline_mae_std": baseline_metrics["mae"]["std"],
        "mae": cv_metrics["mae"]["mean"],
        "mae_std": cv_metrics["mae"]["std"],
        "rmse": cv_metrics["rmse"]["mean"],
        "rmse_std": cv_metrics["rmse"]["std"],
        "r2_score": cv_metrics["r2"]["mean"],
        "r2_score_std": cv_metrics["r2"]["std"],
        "training_configuration": {
            "max_sample_size": sample_size,
            "sampling_random_seed": 42,
            "estimator": {"max_iter": 50, "max_depth": 6, "random_state": 42},
            "final_fit": "All eligible rows after cross-validation.",
        },
        "data_provenance": "synthetic_demonstration_data",
        "synthetic_record_count": int(
            (train_df["added_data_status"] == "synthetic_demo_values").sum()
        )
        if "added_data_status" in train_df.columns
        else None,
        "limitations": (
            "Every source row is synthetic demonstration data, not observed fleet "
            "operations. Chronological validation uses the recorded timestamps; "
            "vehicles repeat across time periods, and temporal autocorrelation may "
            "remain. Scores are validation estimates, not an untouched final-test score."
        ),
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    _load_fuel_pipeline_cached.cache_clear()

    return metrics


def is_fuel_model_available() -> bool:
    """Check if a trained fuel model artifact is available."""
    return MODEL_PATH.exists() and METADATA_PATH.exists()


def get_fuel_metadata() -> dict:
    """Load metadata of the trained fuel model."""
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


@lru_cache(maxsize=2)
def _load_fuel_pipeline_cached(
    model_path: str,
    model_mtime_ns: int,
    model_size: int,
    metadata_path: str,
    metadata_mtime_ns: int,
    metadata_size: int,
) -> Pipeline:
    """Load one validated pipeline per version of the model and metadata files."""
    del model_mtime_ns, model_size, metadata_mtime_ns, metadata_size
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Fuel model metadata at '{metadata_path}' is unreadable.") from exc

    if (
        metadata.get("target") != TARGET_COLUMN
        or metadata.get("features") != ALL_FEATURES
        or metadata.get("is_synthetic") is not True
    ):
        raise RuntimeError("Fuel model metadata does not match the current synthetic target/features.")
    trained_version = metadata.get("scikit_learn_version")
    if trained_version and trained_version != sklearn.__version__:
        raise RuntimeError(
            f"Fuel model was trained with scikit-learn {trained_version}, "
            f"but the active environment uses {sklearn.__version__}."
        )

    try:
        pipeline = joblib.load(model_path)
    except (AttributeError, ImportError, OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Fuel model artifact at '{model_path}' is unreadable.") from exc
    if not isinstance(pipeline, Pipeline):
        raise RuntimeError("Fuel model artifact is not a scikit-learn Pipeline.")
    trained_features = getattr(pipeline, "feature_names_in_", None)
    if trained_features is None or list(trained_features) != ALL_FEATURES:
        raise RuntimeError("Fuel model feature order does not match the current inference schema.")
    return pipeline


def load_fuel_model() -> Pipeline:
    """Load and validate the current fuel pipeline, caching by artifact version."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Fuel model artifacts are missing. Train the model before inference.")
    model_stat = MODEL_PATH.stat()
    metadata_stat = METADATA_PATH.stat()
    return _load_fuel_pipeline_cached(
        str(MODEL_PATH),
        model_stat.st_mtime_ns,
        model_stat.st_size,
        str(METADATA_PATH),
        metadata_stat.st_mtime_ns,
        metadata_stat.st_size,
    )


def predict_fuel_consumption(
    input_data: dict | pd.DataFrame,
    fuel_price_per_liter: float | None = None,
) -> dict:
    """Predict trip fuel consumption in liters, with optional cost estimation.

    Visibly discloses synthetic demonstration provenance.
    """
    if isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Fuel prediction input cannot be an empty dictionary.")
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise TypeError("Fuel prediction input must be a dictionary or pandas DataFrame.")
    if len(df_input) != 1:
        raise ValueError(f"Fuel prediction expects exactly one trip; received {len(df_input)} rows.")
    if not df_input.columns.is_unique:
        raise ValueError("Fuel prediction input contains duplicate column names.")

    for col in ALL_FEATURES:
        if col not in df_input.columns:
            if col in NUMERIC_FEATURES:
                df_input[col] = 0
            else:
                df_input[col] = "Unknown"

    value_bounds = {
        "trip_distance_km": (0, None, False, True),
        "load_weight_kg": (0, None, True, False),
        "temperature_c": (None, None, True, False),
        "wind_speed_kmh": (0, None, True, False),
        "rainfall_mm": (0, None, True, False),
    }
    for column, (minimum, maximum, allow_null, strict_minimum) in value_bounds.items():
        original = df_input[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Prediction input '{column}' must contain numeric values.")
        values = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Prediction input '{column}' must contain finite numeric values.")
        if not allow_null and numeric.isna().any():
            raise ValueError(f"Prediction input '{column}' is required.")
        if (
            minimum is not None
            and (
                (numeric.dropna() <= minimum).any()
                if strict_minimum
                else (numeric.dropna() < minimum).any()
            )
        ) or (
            maximum is not None and (numeric.dropna() > maximum).any()
        ):
            bound_text = (
                f"between {minimum} and {maximum}"
                if maximum is not None
                else f"greater than {minimum}" if strict_minimum else f"at least {minimum}"
            )
            raise ValueError(f"Prediction input '{column}' must be {bound_text}.")
        df_input[column] = numeric

    # The optional price is not a model feature; validate it only for cost calculation.
    if fuel_price_per_liter is not None:
        try:
            fuel_price = float(fuel_price_per_liter)
        except (TypeError, ValueError) as exc:
            raise ValueError("Fuel price must be a finite, non-negative number.") from exc
        if not np.isfinite(fuel_price) or fuel_price < 0:
            raise ValueError("Fuel price must be a finite, non-negative number.")
        fuel_price_per_liter = fuel_price

    pipeline = load_fuel_model()
    X = df_input[ALL_FEATURES]
    pred_liters = float(pipeline.predict(X)[0])
    # Report a small positive floor instead of impossible zero/negative fuel use.
    pred_liters = max(0.1, pred_liters)

    result = {
        "predicted_fuel_consumed_liters": round(pred_liters, 2),
        "provenance": "synthetic_demonstration_data",
        "is_synthetic": True,
        "target": TARGET_COLUMN,
        "features_evaluated": ALL_FEATURES,
    }

    if fuel_price_per_liter is not None:
        result["estimated_fuel_cost"] = round(pred_liters * float(fuel_price_per_liter), 2)
        result["fuel_price_per_liter"] = float(fuel_price_per_liter)

    return result
