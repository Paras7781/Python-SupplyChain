"""
SupplyIQ - Train and Evaluate Synthetic Inventory Demand Model.

Retrains and evaluates the Inventory Demand model using the canonical active dataset
(data/raw/Retail_Supply_Chain_Analytics_Dataset.csv via data/processed/inventory_demand_model_ready.csv).

Key Principles:
1. Strict isolation from production model:
   - Saves to models/synthetic_inventory_demand_model.joblib
   - Saves to models/synthetic_inventory_demand_metadata.json
   - Leaves models/inventory_demand_model.joblib and its metadata strictly untouched.
2. Evaluates against both Mean and Median Dummy baselines on the same held-out folds.
3. Uses GroupKFold cross-validation on Supplier_ID with reproducible sample (seed=42).
4. Fits preprocessing strictly on training folds within Pipeline (no data leakage).
5. Explicitly documents synthetic data provenance and notes that high R² is attributable
   to the documented generator formula, not real-world predictive validity.
"""
from __future__ import annotations

import sys
from pathlib import Path
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sklearn
import joblib

from sklearn.model_selection import GroupKFold, cross_validate
from sklearn.dummy import DummyRegressor

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import MODELS_DIR
from shared.model_validation import summarize_cv_results
from ml.inventory_supplier.model_training import (
    ALL_FEATURES,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
    build_pipeline,
)

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "inventory_demand_model_ready.csv"
CANONICAL_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "Retail_Supply_Chain_Analytics_Dataset.csv"
SYNTHETIC_MODEL_PATH = MODELS_DIR / "synthetic_inventory_demand_model.joblib"
SYNTHETIC_METADATA_PATH = MODELS_DIR / "synthetic_inventory_demand_metadata.json"

PRODUCTION_MODEL_PATH = MODELS_DIR / "inventory_demand_model.joblib"
PRODUCTION_METADATA_PATH = MODELS_DIR / "inventory_demand_metadata.json"


def train_and_evaluate_synthetic_inventory_model(
    sample_size: int = 30000,
    random_seed: int = 42,
) -> dict:
    """Train and benchmark HGBR vs Mean/Median dummy baselines on synthetic demand."""
    if PROCESSED_DATA_PATH.exists():
        data_source_path = PROCESSED_DATA_PATH
        print(f"Loading inventory model-ready dataset from: {data_source_path}")
    elif CANONICAL_RAW_PATH.exists():
        data_source_path = CANONICAL_RAW_PATH
        print(f"Model-ready dataset not found. Loading active raw dataset from: {data_source_path}")
    else:
        raise FileNotFoundError(
            f"Neither processed dataset ({PROCESSED_DATA_PATH}) nor canonical raw dataset ({CANONICAL_RAW_PATH}) was found."
        )

    # Record production files mtime to verify untouched status
    prod_model_mtime = PRODUCTION_MODEL_PATH.stat().st_mtime_ns if PRODUCTION_MODEL_PATH.exists() else None
    prod_meta_mtime = PRODUCTION_METADATA_PATH.stat().st_mtime_ns if PRODUCTION_METADATA_PATH.exists() else None

    df = pd.read_csv(data_source_path)
    total_records = len(df)
    print(f"Loaded {total_records:,} records.")

    # Subsample reproducibly
    if sample_size and len(df) > sample_size:
        train_df = df.sample(n=sample_size, random_state=random_seed).copy()
    else:
        train_df = df.copy()

    # Verify and clean target
    target_values = pd.to_numeric(train_df[TARGET_COLUMN], errors="coerce")
    valid_targets = target_values.notna() & np.isfinite(target_values) & (target_values >= 0)
    train_df = train_df.loc[valid_targets].copy()
    train_df[TARGET_COLUMN] = target_values.loc[valid_targets]

    # Clean features
    for col in NUMERIC_FEATURES:
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_FEATURES:
        train_df[col] = train_df[col].astype(str).fillna("Unknown")

    X = train_df[ALL_FEATURES]
    y = train_df[TARGET_COLUMN]
    groups = train_df["Supplier_ID"].astype(str)

    n_splits = 5
    splitter = GroupKFold(n_splits=n_splits)
    split_indices = list(splitter.split(X, y, groups))

    scoring = {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }

    print("\nRunning 5-fold GroupKFold cross-validation across models on same held-out splits...")

    # 1. HistGradientBoostingRegressor
    print("  Evaluating HistGradientBoostingRegressor...")
    hgbr_results = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=split_indices,
        scoring=scoring,
        error_score="raise",
    )
    hgbr_metrics = summarize_cv_results(hgbr_results, scoring, negative_metrics={"mae", "rmse"})

    # 2. Baseline: DummyRegressor (Mean)
    print("  Evaluating DummyRegressor (strategy='mean')...")
    mean_dummy_results = cross_validate(
        DummyRegressor(strategy="mean"),
        X,
        y,
        cv=split_indices,
        scoring=scoring,
        error_score="raise",
    )
    mean_dummy_metrics = summarize_cv_results(mean_dummy_results, scoring, negative_metrics={"mae", "rmse"})

    # 3. Baseline: DummyRegressor (Median)
    print("  Evaluating DummyRegressor (strategy='median')...")
    median_dummy_results = cross_validate(
        DummyRegressor(strategy="median"),
        X,
        y,
        cv=split_indices,
        scoring=scoring,
        error_score="raise",
    )
    median_dummy_metrics = summarize_cv_results(median_dummy_results, scoring, negative_metrics={"mae", "rmse"})

    # Fit final pipeline on all training records
    print("\nFitting final pipeline on full sample...")
    pipeline = build_pipeline()
    pipeline.fit(X, y)
    pipeline.baseline_median_demand_ = float(y.median())
    pipeline.baseline_mean_demand_ = float(y.mean())

    metrics = {
        "model_name": "HistGradientBoostingRegressor",
        "task": "regression",
        "target": TARGET_COLUMN,
        "validation_method": "GroupKFold",
        "validation_score_type": "cross_validation_estimate",
        "fold_count": n_splits,
        "splitter_configuration": {
            "n_splits": n_splits,
            "shuffle": False,
            "group_column": "Supplier_ID",
        },
        "scoring_metrics": ["mae", "rmse", "r2"],
        "cv_metrics": hgbr_metrics,
        "dummy_baselines": {
            "mean": mean_dummy_metrics,
            "median": median_dummy_metrics,
        },
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "scikit_learn_version": sklearn.__version__,
        "train_samples": len(X),
        "features": ALL_FEATURES.copy(),
        "feature_definitions": {
            "categorical": CATEGORICAL_FEATURES.copy(),
            "numeric": NUMERIC_FEATURES.copy(),
        },
        "mae": hgbr_metrics["mae"]["mean"],
        "mae_std": hgbr_metrics["mae"]["std"],
        "rmse": hgbr_metrics["rmse"]["mean"],
        "rmse_std": hgbr_metrics["rmse"]["std"],
        "r2_score": hgbr_metrics["r2"]["mean"],
        "r2_score_std": hgbr_metrics["r2"]["std"],
        "baseline_mean_mae": mean_dummy_metrics["mae"]["mean"],
        "baseline_median_mae": median_dummy_metrics["mae"]["mean"],
        "is_baseline_heuristic": False,
        "is_synthetic_demonstration": True,
        "data_provenance": "synthetic_demonstration_data",
        "synthetic_generator": {
            "source_csv": "data/raw/Retail_Supply_Chain_Analytics_Dataset.csv",
            "formula": "Daily_Demand = round(category_base * warehouse_factor * seasonal_factor + noise)",
            "random_seed": 42,
            "noise_std_dev": 30.0,
        },
        "model_performance_summary": (
            f"HistGradientBoostingRegressor achieved CV R2 = {hgbr_metrics['r2']['mean']:.4f} "
            f"(MAE: {hgbr_metrics['mae']['mean']:.2f}, RMSE: {hgbr_metrics['rmse']['mean']:.2f}) on synthetic data, "
            f"outperforming Dummy Mean (MAE: {mean_dummy_metrics['mae']['mean']:.2f}, R2: {mean_dummy_metrics['r2']['mean']:.4f}) "
            f"and Dummy Median (MAE: {median_dummy_metrics['mae']['mean']:.2f}, R2: {median_dummy_metrics['r2']['mean']:.4f}). "
            "IMPORTANT: This performance reflects the mathematical generator formula and demonstrates that the ML pipeline "
            "can learn systematic demand patterns when features carry predictive signal. It is NOT proof of real-world demand forecasting."
        ),
        "limitations": (
            "Model trained strictly on synthetic demonstration values. Daily demand values are generated from "
            "Product_Category, Warehouse_Location, and Seasonal_Demand_Index. Real-world demand will have complex "
            "unobserved factors, non-linearities, and temporal regime changes not captured by this formula."
        ),
    }

    # Save synthetic artifacts
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving synthetic model to: {SYNTHETIC_MODEL_PATH}")
    joblib.dump(pipeline, SYNTHETIC_MODEL_PATH)

    print(f"Saving synthetic metadata to: {SYNTHETIC_METADATA_PATH}")
    with open(SYNTHETIC_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Confirm production files were NOT touched
    if prod_model_mtime is not None:
        curr_prod_model_mtime = PRODUCTION_MODEL_PATH.stat().st_mtime_ns
        assert curr_prod_model_mtime == prod_model_mtime, "Production model was modified!"
    if prod_meta_mtime is not None:
        curr_prod_meta_mtime = PRODUCTION_METADATA_PATH.stat().st_mtime_ns
        assert curr_prod_meta_mtime == prod_meta_mtime, "Production metadata was modified!"

    print("\nVerified: Production model and metadata files were strictly preserved.")
    return metrics


def main():
    print("=" * 80)
    print(" SupplyIQ - Synthetic Inventory Demand Model Training & Evaluation")
    print(f" scikit-learn version: {sklearn.__version__}")
    print("=" * 80)

    metrics = train_and_evaluate_synthetic_inventory_model(sample_size=30000, random_seed=42)

    hgbr_mae = metrics["mae"]
    hgbr_rmse = metrics["rmse"]
    hgbr_r2 = metrics["r2_score"]
    dummy_mean_mae = metrics["baseline_mean_mae"]
    dummy_mean_rmse = metrics["dummy_baselines"]["mean"]["rmse"]["mean"]
    dummy_mean_r2 = metrics["dummy_baselines"]["mean"]["r2"]["mean"]
    dummy_med_mae = metrics["baseline_median_mae"]
    dummy_med_rmse = metrics["dummy_baselines"]["median"]["rmse"]["mean"]
    dummy_med_r2 = metrics["dummy_baselines"]["median"]["r2"]["mean"]

    print("\n" + "=" * 80)
    print(" EVALUATION BENCHMARK SUMMARY (5-Fold GroupKFold on Supplier_ID)")
    print("=" * 80)
    header = f"{'Model / Baseline':<35} | {'MAE':<10} | {'RMSE':<10} | {'R2':<10}"
    print(header)
    print("-" * len(header))
    print(f"{'HistGradientBoostingRegressor (ML)':<35} | {hgbr_mae:<10.2f} | {hgbr_rmse:<10.2f} | {hgbr_r2:<10.4f}")
    print(f"{'DummyRegressor (Strategy=Mean)':<35} | {dummy_mean_mae:<10.2f} | {dummy_mean_rmse:<10.2f} | {dummy_mean_r2:<10.4f}")
    print(f"{'DummyRegressor (Strategy=Median)':<35} | {dummy_med_mae:<10.2f} | {dummy_med_rmse:<10.2f} | {dummy_med_r2:<10.4f}")
    print("=" * 80)

    mae_improvement_over_mean = dummy_mean_mae - hgbr_mae
    mae_improvement_over_median = dummy_med_mae - hgbr_mae
    print(f"MAE improvement over Mean baseline:   {mae_improvement_over_mean:.2f} units/day (-{mae_improvement_over_mean / dummy_mean_mae * 100:.1f}%)")
    print(f"MAE improvement over Median baseline: {mae_improvement_over_median:.2f} units/day (-{mae_improvement_over_median / dummy_med_mae * 100:.1f}%)")
    print("\nModel artifact:   models/synthetic_inventory_demand_model.joblib")
    print("Metadata artifact: models/synthetic_inventory_demand_metadata.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
