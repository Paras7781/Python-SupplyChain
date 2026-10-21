"""
Regional Risk - Machine Learning Model Training and Inference.
Predicts delivery delay risk at order placement time with strict leakage prevention.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import json
import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.model_selection import (
    StratifiedGroupKFold,
    StratifiedKFold,
    TimeSeriesSplit,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    make_scorer,
)

from config.settings import MODELS_DIR
from analytics.regional_risk.cleaning import aggregate_orders_level
from ml.regional_risk.feature_engineering import create_regional_risk_features
from shared.model_validation import summarize_cv_results

MODEL_PATH = MODELS_DIR / "regional_risk_model.joblib"
METADATA_PATH = MODELS_DIR / "regional_risk_metadata.json"

CATEGORICAL_FEATURES = [
    "Shipping Mode",
    "Order Region",
    "Market",
    "Customer Segment",
]

NUMERIC_FEATURES = [
    "Days for shipment (scheduled)",
    "order_month",
    "order_dayofweek",
    "order_hour",
    "is_weekend",
    "Total_Units",
    "Total_Order_Value",
    "Line_Items",
    "value_per_unit",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET_COLUMN = "Late_delivery_risk"


def build_pipeline() -> Pipeline:
    """Build a scikit-learn preprocessing and classification pipeline."""
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

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=10,
        min_samples_split=10,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def train_regional_risk_model(df: pd.DataFrame, sample_size: int | None = 25000) -> dict:
    """Train a binary classification model for late delivery risk at order placement.

    Uses group-level order aggregation to avoid line-item duplicate leakage,
    and strictly omits actual shipping days and post-delivery statuses.
    """
    if df.empty:
        raise ValueError("Cannot train model on empty DataFrame.")

    # Aggregate item rows to orders to prevent item-level leakage.
    if "Order Id" in df.columns and len(df) > df["Order Id"].nunique():
        order_df = aggregate_orders_level(df)
    else:
        order_df = df.copy()

    # Preserve customer identity only for fold construction, never as a feature.
    customer_column = next(
        (column for column in ("Customer Id", "Order Customer Id") if column in df.columns),
        None,
    )
    if customer_column and "Order Id" in order_df.columns and customer_column not in order_df.columns:
        customer_by_order = df.groupby("Order Id", sort=False)[customer_column].first()
        order_df["_validation_customer_group"] = order_df["Order Id"].map(customer_by_order)
    elif customer_column:
        order_df["_validation_customer_group"] = order_df[customer_column]

    date_column = next(
        (column for column in ("Order_Date", "order date (DateOrders)") if column in order_df.columns),
        None,
    )
    missing_order_date_rows = 0
    if date_column:
        order_dates = pd.to_datetime(order_df[date_column], errors="coerce")
        valid_dates = order_dates.notna()
        missing_order_date_rows = int((~valid_dates).sum())
        order_df = order_df.loc[valid_dates].copy()
        order_df["_validation_order_time"] = order_dates.loc[valid_dates]

    if TARGET_COLUMN not in order_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from dataset.")

    # Subsample if dataset is very large for responsive training
    if sample_size and len(order_df) > sample_size:
        order_df = order_df.sample(n=sample_size, random_state=42)

    # Feature engineering (with strict leakage removal)
    feat_df = create_regional_risk_features(order_df, drop_leakage=True)

    target_values = pd.to_numeric(feat_df[TARGET_COLUMN], errors="coerce")
    invalid_target_mask = feat_df[TARGET_COLUMN].notna() & target_values.isna()
    if invalid_target_mask.any():
        raise ValueError(
            f"Cannot train regional risk model; target '{TARGET_COLUMN}' contains "
            "non-numeric labels."
        )
    invalid_class_mask = target_values.notna() & ~target_values.isin([0, 1])
    if invalid_class_mask.any():
        invalid_values = sorted(target_values.loc[invalid_class_mask].unique().tolist())
        raise ValueError(
            f"Cannot train regional risk model; target '{TARGET_COLUMN}' must contain "
            f"only binary labels 0 and 1 (found {invalid_values[:5]})."
        )
    missing_target_rows = int(target_values.isna().sum())
    valid_mask = target_values.notna()
    feat_df = feat_df[valid_mask]
    feat_df[TARGET_COLUMN] = target_values.loc[valid_mask]
    missing_features = [column for column in ALL_FEATURES if column not in feat_df.columns]
    if missing_features:
        raise ValueError(
            "Cannot train regional risk model; required features are missing: "
            + ", ".join(missing_features)
        )

    X = feat_df[ALL_FEATURES]
    y = feat_df[TARGET_COLUMN].astype(int)
    class_counts = y.value_counts()
    if len(class_counts) != 2 or class_counts.min() < 2:
        raise ValueError(
            f"Cannot train regional risk model; target '{TARGET_COLUMN}' must contain "
            "at least two samples from each class (0 and 1)."
        )

    groups = None
    group_column = None
    if "_validation_customer_group" in feat_df.columns:
        groups = (
            feat_df["_validation_customer_group"]
            .astype("string")
            .fillna("__missing_customer__")
            .reset_index(drop=True)
        )
        group_column = customer_column

    dates = None
    if "_validation_order_time" in feat_df.columns:
        dates = pd.to_datetime(feat_df["_validation_order_time"]).reset_index(drop=True)

    cv_X = X.reset_index(drop=True)
    cv_y = y.reset_index(drop=True)
    cv_groups = groups
    splitter_configuration: dict = {"random_seed": 42}
    splits = None

    if dates is not None:
        order = np.argsort(dates.to_numpy(), kind="stable")
        ordered_X = cv_X.iloc[order].reset_index(drop=True)
        ordered_y = cv_y.iloc[order].reset_index(drop=True)
        ordered_groups = groups.iloc[order].reset_index(drop=True) if groups is not None else None
        cv_groups = ordered_groups

        for n_splits in range(min(5, len(ordered_y) - 1), 1, -1):
            candidate_splits = []
            for train_indices, validation_indices in TimeSeriesSplit(n_splits=n_splits).split(ordered_X):
                if ordered_groups is not None:
                    validation_groups = set(ordered_groups.iloc[validation_indices])
                    train_indices = train_indices[
                        ~ordered_groups.iloc[train_indices].isin(validation_groups).to_numpy()
                    ]
                if (
                    len(train_indices) >= 2
                    and set(ordered_y.iloc[train_indices]) == {0, 1}
                    and set(ordered_y.iloc[validation_indices]) == {0, 1}
                ):
                    candidate_splits.append((train_indices, validation_indices))
                else:
                    break
            if len(candidate_splits) == n_splits:
                splits = candidate_splits
                splitter_configuration = {
                    "base_splitter": "TimeSeriesSplit",
                    "n_splits": n_splits,
                    "shuffle": False,
                    "chronological_column": date_column,
                    "group_purging": bool(ordered_groups is not None),
                    "group_column": group_column,
                    "purge_rule": "Remove earlier training rows for customers in each validation fold.",
                    "random_seed": None,
                }
                cv_X, cv_y = ordered_X, ordered_y
                break
        if splits is None:
            raise ValueError(
                "Cannot train regional risk model; chronological validation could not "
                "produce at least two folds with both target classes in training and validation."
            )
        validation_method = (
            "TimeSeriesSplit with customer-group purging"
            if groups is not None
            else "TimeSeriesSplit"
        )
    else:
        max_splits = min(5, int(class_counts.min()))
        if groups is not None:
            group_count = int(groups.nunique())
            max_splits = min(max_splits, group_count)
            for n_splits in range(max_splits, 1, -1):
                splitter = StratifiedGroupKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42,
                )
                candidate_splits = list(splitter.split(cv_X, cv_y, groups))
                if all(
                    set(cv_y.iloc[train_indices]) == {0, 1}
                    and set(cv_y.iloc[validation_indices]) == {0, 1}
                    for train_indices, validation_indices in candidate_splits
                ):
                    splits = candidate_splits
                    splitter_configuration = {
                        "n_splits": n_splits,
                        "shuffle": True,
                        "random_seed": 42,
                        "group_column": group_column,
                    }
                    break
            validation_method = "StratifiedGroupKFold"
        else:
            for n_splits in range(max_splits, 1, -1):
                splitter = StratifiedKFold(
                    n_splits=n_splits,
                    shuffle=True,
                    random_state=42,
                )
                candidate_splits = list(splitter.split(cv_X, cv_y))
                if all(
                    set(cv_y.iloc[train_indices]) == {0, 1}
                    and set(cv_y.iloc[validation_indices]) == {0, 1}
                    for train_indices, validation_indices in candidate_splits
                ):
                    splits = candidate_splits
                    splitter_configuration = {
                        "n_splits": n_splits,
                        "shuffle": True,
                        "random_seed": 42,
                    }
                    break
            validation_method = "StratifiedKFold"

        if splits is None:
            raise ValueError(
                "Cannot train regional risk model; the available groups and class counts "
                "cannot produce at least two validation folds containing both classes."
            )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(
        build_pipeline(),
        cv_X,
        cv_y,
        cv=list(splits),
        scoring=scoring,
        error_score="raise",
    )
    baseline_results = cross_validate(
        DummyClassifier(strategy="most_frequent"),
        cv_X,
        cv_y,
        cv=list(splits),
        scoring={"accuracy": "accuracy"},
        error_score="raise",
    )
    cv_metrics = summarize_cv_results(cv_results, scoring)
    baseline_metrics = summarize_cv_results(baseline_results, {"accuracy": "accuracy"})

    # The production artifact is refit on every eligible row after validation.
    pipeline = build_pipeline()
    pipeline.fit(X, y)

    metrics = {
        "model_name": "RandomForestClassifier",
        "task": "binary_classification",
        "target": TARGET_COLUMN,
        "validation_method": validation_method,
        "validation_score_type": "cross_validation_estimate",
        "fold_count": len(splits),
        "splitter_configuration": splitter_configuration,
        "scoring_metrics": list(scoring),
        "cv_metrics": cv_metrics,
        "fold_validation_class_counts": [
            {
                str(label): int(count)
                for label, count in cv_y.iloc[validation_indices].value_counts().items()
            }
            for _, validation_indices in splits
        ],
        "fold_group_overlap_counts": [
            len(set(cv_groups.iloc[train_indices]) & set(cv_groups.iloc[validation_indices]))
            for train_indices, validation_indices in splits
        ]
        if cv_groups is not None
        else [],
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "train_samples": len(X),
        "excluded_rows": {
            "missing_target": missing_target_rows,
            "missing_order_date": missing_order_date_rows,
        },
        "features": ALL_FEATURES.copy(),
        "feature_definitions": {
            "categorical": CATEGORICAL_FEATURES.copy(),
            "numeric": NUMERIC_FEATURES.copy(),
        },
        "class_balance_positive_rate": float(y.mean()),
        "baseline_accuracy": baseline_metrics["accuracy"]["mean"],
        "baseline_accuracy_std": baseline_metrics["accuracy"]["std"],
        "model_accuracy": cv_metrics["accuracy"]["mean"],
        "model_accuracy_std": cv_metrics["accuracy"]["std"],
        "precision": cv_metrics["precision"]["mean"],
        "precision_std": cv_metrics["precision"]["std"],
        "recall": cv_metrics["recall"]["mean"],
        "recall_std": cv_metrics["recall"]["std"],
        "f1_score": cv_metrics["f1"]["mean"],
        "f1_score_std": cv_metrics["f1"]["std"],
        "roc_auc": cv_metrics["roc_auc"]["mean"],
        "roc_auc_std": cv_metrics["roc_auc"]["std"],
        "training_configuration": {
            "max_sample_size": sample_size,
            "sampling_random_seed": 42,
            "estimator": {
                "n_estimators": 50,
                "max_depth": 10,
                "min_samples_split": 10,
                "random_state": 42,
            },
            "final_fit": "All eligible rows after cross-validation.",
        },
        "data_provenance": "historical_dataco_supply_chain_orders",
        "limitations": (
            "Chronological validation estimates future performance, with earlier orders "
            "for customers in each validation fold purged from that fold's training data. "
            "Temporal or operational distribution shifts may still limit generalization."
        ),
    }

    # Save artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    _load_regional_risk_model_cached.cache_clear()

    return metrics


def is_regional_risk_model_available() -> bool:
    """Check if a trained regional risk model artifact is available."""
    return MODEL_PATH.exists() and METADATA_PATH.exists()


def get_regional_risk_metadata() -> dict:
    """Load metadata of the trained regional risk model."""
    if not METADATA_PATH.exists():
        return {}
    with open(METADATA_PATH, "r") as f:
        return json.load(f)


@lru_cache(maxsize=2)
def _load_regional_risk_model_cached(
    model_path: str,
    model_mtime_ns: int,
    model_size: int,
    metadata_path: str,
    metadata_mtime_ns: int,
    metadata_size: int,
) -> Pipeline:
    """Load and validate a model once per artifact version."""
    del model_mtime_ns, model_size, metadata_mtime_ns, metadata_size

    try:
        with open(metadata_path, "r") as metadata_file:
            metadata = json.load(metadata_file)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Regional risk model metadata at '{metadata_path}' is unreadable. "
            "Retrain it with 'python scripts/train_regional_risk_model.py'."
        ) from exc

    if metadata.get("target") != TARGET_COLUMN or metadata.get("features") != ALL_FEATURES:
        raise RuntimeError(
            "Regional risk model metadata does not match the current target/features. "
            "Retrain it with 'python scripts/train_regional_risk_model.py'."
        )

    trained_version = metadata.get("scikit_learn_version")
    if trained_version and trained_version != sklearn.__version__:
        raise RuntimeError(
            "Regional risk model was trained with scikit-learn "
            f"{trained_version}, but the active environment uses {sklearn.__version__}. "
            "Retrain it in this environment with 'python scripts/train_regional_risk_model.py'."
        )

    try:
        pipeline = joblib.load(model_path)
    except (AttributeError, ImportError, ValueError, TypeError) as exc:
        raise RuntimeError(
            "The regional risk model artifact is incompatible with the installed "
            f"scikit-learn {sklearn.__version__} ({exc}). Retrain it in this environment "
            "with 'python scripts/train_regional_risk_model.py'."
        ) from exc

    if not isinstance(pipeline, Pipeline):
        raise RuntimeError(
            "The regional risk artifact is not a scikit-learn Pipeline. "
            "Retrain it with 'python scripts/train_regional_risk_model.py'."
        )

    trained_features = getattr(pipeline, "feature_names_in_", None)
    if trained_features is None or list(trained_features) != ALL_FEATURES:
        raise RuntimeError(
            "The regional risk model's feature order does not match the current "
            "inference schema. Retrain it with 'python scripts/train_regional_risk_model.py'."
        )

    classifier = pipeline.named_steps.get("classifier")
    if classifier is None or not hasattr(classifier, "classes_") or set(classifier.classes_) != {0, 1}:
        raise RuntimeError(
            "The regional risk model must be a binary classifier with classes 0 and 1. "
            "Retrain it with 'python scripts/train_regional_risk_model.py'."
        )

    return pipeline


def load_regional_risk_model() -> Pipeline:
    """Load the current regional risk pipeline or raise an actionable artifact error."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Regional risk model artifact is missing at '{MODEL_PATH}'. "
            "Train it with 'python scripts/train_regional_risk_model.py'."
        )
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Regional risk model metadata is missing at '{METADATA_PATH}'. "
            "Train it with 'python scripts/train_regional_risk_model.py'."
        )

    model_stat = MODEL_PATH.stat()
    metadata_stat = METADATA_PATH.stat()
    return _load_regional_risk_model_cached(
        str(MODEL_PATH),
        model_stat.st_mtime_ns,
        model_stat.st_size,
        str(METADATA_PATH),
        metadata_stat.st_mtime_ns,
        metadata_stat.st_size,
    )


def predict_regional_risk(input_data: dict | pd.DataFrame) -> dict:
    """Predict delivery delay risk for a new order.

    Parameters:
        input_data: dict or DataFrame with order-time attributes.
    """
    if isinstance(input_data, dict):
        if not input_data:
            raise ValueError("Regional risk prediction input cannot be an empty dictionary.")
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise TypeError("Regional risk prediction input must be a dictionary or pandas DataFrame.")

    if len(df_input) != 1:
        raise ValueError(
            "Regional risk prediction expects exactly one order; "
            f"received {len(df_input)} rows."
        )
    if not df_input.columns.is_unique:
        raise ValueError("Regional risk prediction input contains duplicate column names.")

    value_bounds = {
        "Days for shipment (scheduled)": (0, None),
        "order_month": (1, 12),
        "order_dayofweek": (0, 6),
        "order_hour": (0, 23),
        "is_weekend": (0, 1),
        "Total_Units": (1, None),
        "Total_Order_Value": (0, None),
        "Line_Items": (1, None),
        "value_per_unit": (0, None),
    }
    for col in NUMERIC_FEATURES:
        if col not in df_input.columns:
            continue
        original_values = df_input[col]
        numeric_values = pd.to_numeric(original_values, errors="coerce")
        invalid_mask = original_values.notna() & numeric_values.isna()
        if invalid_mask.any():
            raise ValueError(f"Prediction input '{col}' must contain numeric values.")
        non_null_values = numeric_values.dropna().to_numpy(dtype=float)
        if not np.isfinite(non_null_values).all():
            raise ValueError(f"Prediction input '{col}' must contain finite numeric values.")

        minimum, maximum = value_bounds[col]
        if minimum is not None and (numeric_values.dropna() < minimum).any():
            raise ValueError(f"Prediction input '{col}' must be at least {minimum}.")
        if maximum is not None and (numeric_values.dropna() > maximum).any():
            raise ValueError(f"Prediction input '{col}' must be no greater than {maximum}.")
        df_input[col] = numeric_values

    # Apply feature defaults if missing
    for col in ALL_FEATURES:
        if col not in df_input.columns:
            if col in NUMERIC_FEATURES:
                df_input[col] = 0
            else:
                df_input[col] = "Unknown"

    pipeline = load_regional_risk_model()
    X = df_input[ALL_FEATURES]
    # Find class 1 explicitly; its column is not guaranteed to be in a fixed position.
    class_labels = list(pipeline.named_steps["classifier"].classes_)
    positive_class_index = class_labels.index(1)
    proba = float(pipeline.predict_proba(X)[0, positive_class_index])
    pred = int(proba >= 0.5)

    # Keep the displayed risk bands separate from the model's binary cutoff.
    risk_level = "High" if proba >= 0.65 else ("Medium" if proba >= 0.40 else "Low")

    return {
        "late_delivery_prediction": bool(pred),
        "late_probability": round(proba, 4),
        "late_probability_pct": round(proba * 100, 1),
        "risk_level": risk_level,
        "features_evaluated": ALL_FEATURES,
        "target": TARGET_COLUMN,
    }
