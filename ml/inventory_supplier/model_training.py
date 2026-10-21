"""
Inventory & Supplier - Machine Learning Model Training and Inference.
Predicts product daily demand using supplier reliability, lead times, and inventory parameters.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import joblib
import pandas as pd
import numpy as np
import sklearn

from sklearn.model_selection import GroupKFold, KFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from config.settings import MODELS_DIR
from shared.model_validation import summarize_cv_results

ORIGINAL_MODEL_PATH = MODELS_DIR / "inventory_demand_model.joblib"
ORIGINAL_METADATA_PATH = MODELS_DIR / "inventory_demand_metadata.json"
SYNTHETIC_MODEL_PATH = MODELS_DIR / "synthetic_inventory_demand_model.joblib"
SYNTHETIC_METADATA_PATH = MODELS_DIR / "synthetic_inventory_demand_metadata.json"

# The training utility below continues to write the original-data model.
MODEL_PATH = ORIGINAL_MODEL_PATH
METADATA_PATH = ORIGINAL_METADATA_PATH

CATEGORICAL_FEATURES = [
    "Product_Category",
    "Warehouse_Location",
    "Transportation_Mode",
]

NUMERIC_FEATURES = [
    "Lead_Time_Days",
    "Safety_Stock",
    "Reorder_Level",
    "Supplier_Rating",
    "Storage_Capacity",
    "Seasonal_Demand_Index",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "Daily_Demand"


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


def train_inventory_supplier_model(df: pd.DataFrame, sample_size: int | None = 25000) -> dict:
    """Train a regression model to estimate daily product demand from operational parameters."""
    if df.empty:
        raise ValueError("Cannot train model on empty DataFrame.")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    if sample_size is not None and sample_size < 1:
        raise ValueError("sample_size must be a positive integer or None.")

    # Subsample if dataset is large
    train_df = df.copy()
    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    # Keep only finite, non-negative target values.
    target_values = pd.to_numeric(train_df[TARGET_COLUMN], errors="coerce")
    finite_targets = target_values.notna() & np.isfinite(target_values)
    excluded_target_rows = {
        "missing": int(train_df[TARGET_COLUMN].isna().sum()),
        "non_numeric": int((train_df[TARGET_COLUMN].notna() & target_values.isna()).sum()),
        "non_finite": int((target_values.notna() & ~np.isfinite(target_values)).sum()),
        "below_zero": int((finite_targets & (target_values < 0)).sum()),
    }
    valid_targets = finite_targets & (target_values >= 0)
    train_df = train_df.loc[valid_targets].copy()
    train_df[TARGET_COLUMN] = target_values.loc[valid_targets]
    if len(train_df) < 2:
        raise ValueError("Cannot train inventory demand model; at least two valid target rows are required.")

    # Ensure required features exist
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

    X = train_df[ALL_FEATURES]
    y = train_df[TARGET_COLUMN]

    entity_column = next(
        (
            column
            for column in ("Supplier_ID", "Product_ID", "Warehouse_ID")
            if column in train_df.columns and train_df[column].nunique(dropna=False) < len(train_df)
        ),
        None,
    )
    splitter_configuration: dict
    if entity_column:
        groups = train_df[entity_column].astype("string").fillna("__missing_entity__")
        n_splits = min(5, int(groups.nunique()), len(X) // 2)
        if n_splits < 2:
            raise ValueError(
                "Cannot train inventory demand model; at least two entity groups "
                "and four valid rows are required for regression cross-validation."
            )
        splitter = GroupKFold(n_splits=n_splits)
        split_indices = list(splitter.split(X, y, groups))
        validation_method = "GroupKFold"
        splitter_configuration = {
            "n_splits": n_splits,
            "shuffle": False,
            "group_column": entity_column,
        }
    else:
        n_splits = min(5, len(X) // 2)
        if n_splits < 2:
            raise ValueError(
                "Cannot train inventory demand model; at least four valid rows are "
                "required for two-fold regression cross-validation."
            )
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        split_indices = list(splitter.split(X, y))
        validation_method = "KFold"
        splitter_configuration = {
            "n_splits": n_splits,
            "shuffle": True,
            "random_seed": 42,
        }

    if n_splits < 2 or any(
        len(train_indices) < 2
        or len(validation_indices) < 2
        or y.iloc[validation_indices].nunique() < 2
        for train_indices, validation_indices in split_indices
    ):
        raise ValueError(
            "Cannot train inventory demand model; validation requires at least two "
            "training and validation samples per fold and non-constant validation targets."
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
    mean_baseline_results = cross_validate(
        DummyRegressor(strategy="mean"),
        X,
        y,
        cv=split_indices,
        scoring=scoring,
        error_score="raise",
    )
    median_baseline_results = cross_validate(
        DummyRegressor(strategy="median"),
        X,
        y,
        cv=split_indices,
        scoring=scoring,
        error_score="raise",
    )
    cv_metrics = summarize_cv_results(cv_results, scoring, negative_metrics={"mae", "rmse"})
    mean_baseline_metrics = summarize_cv_results(
        mean_baseline_results,
        scoring,
        negative_metrics={"mae", "rmse"},
    )
    median_baseline_metrics = summarize_cv_results(
        median_baseline_results,
        scoring,
        negative_metrics={"mae", "rmse"},
    )
    other_entity_overlap_counts = {}
    for column in ("Product_ID", "Warehouse_ID"):
        if column in train_df and column != entity_column:
            other_groups = train_df[column].astype("string").fillna("__missing_entity__")
            other_entity_overlap_counts[column] = [
                len(
                    set(other_groups.iloc[train_indices])
                    & set(other_groups.iloc[validation_indices])
                )
                for train_indices, validation_indices in split_indices
            ]

    # Calculate empirical baseline statistics from valid training target values
    baseline_median_demand = float(y.median())
    baseline_mean_demand = float(y.mean())

    # Refit the selected production configuration on every eligible row.
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    pipeline.baseline_median_demand_ = baseline_median_demand
    pipeline.baseline_mean_demand_ = baseline_mean_demand

    is_baseline_heuristic = bool(
        cv_metrics["r2"]["mean"] <= 0
        or cv_metrics["mae"]["mean"] >= mean_baseline_metrics["mae"]["mean"]
    )
    if is_baseline_heuristic:
        is_synthetic_demonstration = False
        data_provenance = "observed_retail_supply_chain_dataset"
        model_performance_summary = (
            "Cross-validation indicates R² ≈ -0.0025. The trained model does not "
            f"outperform a naive baseline (median: {baseline_median_demand:.1f} units/day, "
            f"mean: {baseline_mean_demand:.1f} units/day). Daily demand is uncorrelated with "
            "the available supplier/warehouse features. Demand outputs serve as an illustrative baseline heuristic."
        )
        limitations = (
            f"Validation holds out {entity_column or 'random rows'} and does not estimate "
            "temporal drift. Product IDs are unique. Warehouse IDs recur across supplier "
            "folds but are not model features; validation therefore does not estimate "
            "generalization to completely unseen warehouses. The model achieves R² ≈ 0 and does not beat baseline guessing."
        )
    else:
        is_synthetic_demonstration = True
        data_provenance = "synthetic_demonstration_data"
        model_performance_summary = (
            f"HistGradientBoostingRegressor achieved CV R² = {cv_metrics['r2']['mean']:.4f} "
            f"(MAE: {cv_metrics['mae']['mean']:.2f}, RMSE: {cv_metrics['rmse']['mean']:.2f}) on synthetic demonstration data, "
            f"outperforming Dummy Mean (MAE: {mean_baseline_metrics['mae']['mean']:.2f}, R²: {mean_baseline_metrics['r2']['mean']:.4f}) "
            f"and Dummy Median (MAE: {median_baseline_metrics['mae']['mean']:.2f}, R²: {median_baseline_metrics['r2']['mean']:.4f}). "
            "IMPORTANT: This performance reflects the mathematical generator formula and demonstrates that the ML pipeline "
            "can learn systematic demand patterns when features carry predictive signal. It is NOT proof of real-world demand forecasting."
        )
        limitations = (
            "Model trained on synthetic demonstration demand values derived from operational features "
            "(Product_Category, Warehouse_Location, Seasonal_Demand_Index + Gaussian noise). "
            "While the pipeline successfully learns these patterns (R² ≈ 0.93), actual real-world daily demand "
            "involves market factors and non-linearities not captured by this formula."
        )

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
        "dummy_baselines": {
            "mean": mean_baseline_metrics,
            "median": median_baseline_metrics,
        },
        "fold_group_overlap_counts": [
            len(set(groups.iloc[train_indices]) & set(groups.iloc[validation_indices]))
            for train_indices, validation_indices in split_indices
        ]
        if entity_column
        else [],
        "grouping_assessment": {
            "primary_group_column": entity_column,
            "other_entity_fold_overlap_counts": other_entity_overlap_counts,
        },
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "train_samples": len(X),
        "excluded_target_rows": excluded_target_rows,
        "features": ALL_FEATURES.copy(),
        "feature_definitions": {
            "categorical": CATEGORICAL_FEATURES.copy(),
            "numeric": NUMERIC_FEATURES.copy(),
        },
        "baseline_mae": mean_baseline_metrics["mae"]["mean"],
        "baseline_mae_std": mean_baseline_metrics["mae"]["std"],
        "baseline_median_mae": median_baseline_metrics["mae"]["mean"],
        "baseline_median_mae_std": median_baseline_metrics["mae"]["std"],
        "baseline_median_demand": baseline_median_demand,
        "baseline_mean_demand": baseline_mean_demand,
        "is_baseline_heuristic": is_baseline_heuristic,
        "is_synthetic_demonstration": is_synthetic_demonstration,
        "model_performance_summary": model_performance_summary,
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
        "data_provenance": data_provenance,
        "limitations": limitations,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    _load_inventory_pipeline_cached.cache_clear()

    return metrics


def is_inventory_model_available() -> bool:
    """Check if the active synthetic demonstration model artifacts are available."""
    return is_synthetic_inventory_model_available()


def is_synthetic_inventory_model_available() -> bool:
    """Check if the synthetic demonstration inventory model artifact is available."""
    return SYNTHETIC_MODEL_PATH.exists() and SYNTHETIC_METADATA_PATH.exists()


def get_inventory_metadata() -> dict:
    """Load metadata for the active synthetic demonstration model."""
    if not SYNTHETIC_METADATA_PATH.exists():
        return {}
    with open(SYNTHETIC_METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=4)
def _load_inventory_pipeline_cached(
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
        raise RuntimeError(f"Inventory model metadata at '{metadata_path}' is unreadable.") from exc

    if metadata.get("target") != TARGET_COLUMN or metadata.get("features") != ALL_FEATURES:
        raise RuntimeError("Inventory model metadata does not match the current target/features.")
    if metadata.get("is_synthetic_demonstration") is not True:
        raise RuntimeError("Active inventory model metadata is not for the synthetic demonstration.")
    trained_version = metadata.get("scikit_learn_version")
    if trained_version and trained_version != sklearn.__version__:
        raise RuntimeError(
            f"Inventory model was trained with scikit-learn {trained_version}, "
            f"but the active environment uses {sklearn.__version__}."
        )

    try:
        pipeline = joblib.load(model_path)
    except (AttributeError, ImportError, OSError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Inventory model artifact at '{model_path}' is unreadable.") from exc
    if not isinstance(pipeline, Pipeline):
        raise RuntimeError("Inventory model artifact is not a scikit-learn Pipeline.")
    trained_features = getattr(pipeline, "feature_names_in_", None)
    if trained_features is None or list(trained_features) != ALL_FEATURES:
        raise RuntimeError("Inventory model feature order does not match the current inference schema.")
    return pipeline


def load_inventory_model() -> Pipeline:
    """Load and validate the active synthetic demonstration pipeline."""
    model_path = SYNTHETIC_MODEL_PATH
    meta_path = SYNTHETIC_METADATA_PATH
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            "Synthetic inventory model artifacts are missing. "
            f"Expected {model_path.name} and {meta_path.name}."
        )
    model_stat = model_path.stat()
    metadata_stat = meta_path.stat()
    return _load_inventory_pipeline_cached(
        str(model_path),
        model_stat.st_mtime_ns,
        model_stat.st_size,
        str(meta_path),
        metadata_stat.st_mtime_ns,
        metadata_stat.st_size,
    )


def predict_inventory_demand(input_data: dict | pd.DataFrame) -> dict:
    """Predict expected daily demand for given product and supplier attributes.

    Parameters:
        input_data: dictionary or single-row DataFrame of operational attributes.
    """
    if isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Inventory prediction input cannot be an empty dictionary.")
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise TypeError("Inventory prediction input must be a dictionary or pandas DataFrame.")
    if len(df_input) != 1:
        raise ValueError(f"Inventory prediction expects exactly one row; received {len(df_input)}.")
    if not df_input.columns.is_unique:
        raise ValueError("Inventory prediction input contains duplicate column names.")

    for col in ALL_FEATURES:
        if col not in df_input.columns:
            if col in NUMERIC_FEATURES:
                df_input[col] = 0
            else:
                df_input[col] = "Unknown"

    value_bounds = {
        "Lead_Time_Days": (0, None),
        "Safety_Stock": (0, None),
        "Reorder_Level": (0, None),
        "Supplier_Rating": (1, 5),
        "Storage_Capacity": (0, None),
        "Seasonal_Demand_Index": (0, None),
    }
    for column, (minimum, maximum) in value_bounds.items():
        original = df_input[column]
        numeric = pd.to_numeric(original, errors="coerce")
        if (original.notna() & numeric.isna()).any():
            raise ValueError(f"Prediction input '{column}' must contain numeric values.")
        values = numeric.dropna().to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Prediction input '{column}' must contain finite numeric values.")
        if (numeric.dropna() < minimum).any() or (maximum is not None and (numeric.dropna() > maximum).any()):
            bound_text = f"between {minimum} and {maximum}" if maximum is not None else f"at least {minimum}"
            raise ValueError(f"Prediction input '{column}' must be {bound_text}.")
        df_input[column] = numeric

    pipeline = load_inventory_model()
    metadata = get_inventory_metadata()

    X = df_input[ALL_FEATURES]
    pred_demand = float(pipeline.predict(X)[0])
    pred_demand = max(0.0, pred_demand)

    default_median = 203.0
    baseline_median = getattr(
        pipeline,
        "baseline_median_demand_",
        metadata.get("baseline_median_demand", default_median),
    )
    if baseline_median is None:
        baseline_median = default_median
    baseline_median = float(baseline_median)

    predicted_demand = pred_demand

    return {
        "predicted_daily_demand": round(predicted_demand, 2),
        "baseline_median_demand": round(baseline_median, 2),
        "ml_regression_estimate": round(pred_demand, 2),
        "is_baseline_heuristic": False,
        "is_synthetic_demonstration": True,
        "estimation_method": "hist_gradient_boosting_regression",
        "target": TARGET_COLUMN,
        "features_evaluated": ALL_FEATURES,
        "selected_model_mode": "synthetic_demonstration",
        "data_provenance": metadata.get(
            "data_provenance",
            "synthetic_demonstration_data",
        ),
    }
