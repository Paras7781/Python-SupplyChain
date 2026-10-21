"""
SupplyIQ - Data Preprocessing & Model-Ready Dataset Preparation Pipeline.

Loads raw source datasets from data/raw/, cleans and validates them in memory,
strictly prevents target leakage, preserves synthetic data provenance, and saves
ONLY the three final model-ready training datasets into data/processed/:
  1. data/processed/regional_risk_model_ready.csv
  2. data/processed/inventory_demand_model_ready.csv
  3. data/processed/fuel_model_ready.csv

Treats data/raw/ as immutable. Intermediate cleaned files are kept in memory and
are NOT saved to disk.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from shared.data_loader import load_raw_dataset
from shared.data_validation import validate_dataframe
from analytics.regional_risk.cleaning import clean_regional_risk_data, aggregate_orders_level
from ml.regional_risk.feature_engineering import create_regional_risk_features, LEAKAGE_COLUMNS
from ml.regional_risk.model_training import (
    ALL_FEATURES as REGIONAL_FEATURES,
    TARGET_COLUMN as REGIONAL_TARGET,
)
from analytics.inventory_supplier.cleaning import clean_inventory_supplier_data
from ml.inventory_supplier.model_training import (
    ALL_FEATURES as INVENTORY_FEATURES,
    TARGET_COLUMN as INVENTORY_TARGET,
)
from analytics.fuel_weather.cleaning import clean_fuel_weather_data
from ml.fuel_weather.model_training import (
    ALL_FEATURES as FUEL_FEATURES,
    TARGET_COLUMN as FUEL_TARGET,
)


def prepare_regional_risk_datasets(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean, validate, and prepare the model-ready dataset for Regional Risk.

    In-Memory Operations:
    1. Cleans DataCo records (standardizes types, strips text, parses dates).
    2. Strips customer PII and empty product descriptions.
    3. Validates contract and data quality in memory via validate_dataframe().
    4. Aggregates item-level records to unique orders to prevent duplicate leakage.
    5. Preserves customer grouping identifier ('Customer Id') and order date for validation.
    6. Generates order-time features (calendar timing, order unit values).
    7. Strictly drops post-fulfillment leakage columns (real shipping days, delivery status).
    8. Filters rows to valid binary target labels (0 or 1).
    9. Retains only confirmed model features, validation identifiers, and target.

    Parameters:
        raw_df (pd.DataFrame): Raw DataCo supply chain records.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]:
            - Cleaned in-memory DataFrame.
            - Model-ready DataFrame for ML training.
            - Summary statistics dictionary.
    """
    if raw_df.empty:
        raise ValueError("Cannot prepare Regional Risk dataset: source DataFrame is empty.")

    # 1. Clean data in memory
    cleaned_df = clean_regional_risk_data(raw_df, remove_pii=True)

    # 2. In-memory contract and quality validation
    val_report = validate_dataframe(cleaned_df, "supply_chain")
    if not val_report["contract_valid"]:
        missing = val_report.get("missing_required_columns", [])
        raise ValueError(f"Regional Risk dataset contract violation. Missing required columns: {missing}")

    # 3. Order-level aggregation to prevent item-level leakage across splits
    order_df = aggregate_orders_level(cleaned_df)

    # Retain customer grouping column if present in raw data (used for validation splitting only)
    customer_col = next(
        (col for col in ("Customer Id", "Order Customer Id") if col in raw_df.columns),
        None,
    )
    if customer_col and "Order Id" in order_df.columns and customer_col not in order_df.columns:
        customer_map = raw_df.groupby("Order Id", sort=False)[customer_col].first()
        order_df["Customer Id"] = order_df["Order Id"].map(customer_map)

    # 4. Order-time feature engineering with strict leakage prevention
    feat_df = create_regional_risk_features(order_df, drop_leakage=True)

    # 5. Target validation and row filtering
    if REGIONAL_TARGET not in feat_df.columns:
        raise KeyError(f"Required target column '{REGIONAL_TARGET}' missing from Regional Risk data.")

    target_numeric = pd.to_numeric(feat_df[REGIONAL_TARGET], errors="coerce")
    valid_binary_mask = target_numeric.isin([0, 1])
    dropped_target_rows = int((~valid_binary_mask).sum())

    model_ready = feat_df.loc[valid_binary_mask].copy()
    model_ready[REGIONAL_TARGET] = target_numeric.loc[valid_binary_mask].astype(int)

    # Keep only target, model features, and validation tracking columns
    val_cols = [c for c in ["Order Id", "Customer Id", "Order_Date"] if c in model_ready.columns]
    keep_cols = val_cols + REGIONAL_FEATURES + [REGIONAL_TARGET]
    seen = set()
    final_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    model_ready = model_ready[final_cols]

    stats = {
        "domain": "Regional Risk",
        "raw_rows": len(raw_df),
        "raw_cols": len(raw_df.columns),
        "cleaned_rows": len(cleaned_df),
        "cleaned_cols": len(cleaned_df.columns),
        "model_ready_rows": len(model_ready),
        "model_ready_cols": len(model_ready.columns),
        "dropped_target_rows": dropped_target_rows,
        "missing_values_model_ready": int(model_ready.isna().sum().sum()),
        "target": REGIONAL_TARGET,
        "features": REGIONAL_FEATURES,
        "validation_status": val_report["status"],
    }
    return cleaned_df, model_ready, stats


def prepare_inventory_supplier_datasets(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean, validate, and prepare the model-ready dataset for Inventory & Supplier.

    In-Memory Operations:
    1. Standardizes data types for dates, text strings, and numeric quantities.
    2. Validates contract and quality in memory via validate_dataframe().
    3. Confirms target: 'Daily_Demand' (numeric daily demanded units).
    4. Filters rows to finite, non-negative target values (Daily_Demand >= 0).
    5. Preserves 'Supplier_ID' and 'Warehouse_ID' for group-aware validation.
    6. Retains only confirmed model features, group tracking columns, and target.

    Parameters:
        raw_df (pd.DataFrame): Raw retail supply chain analytics records.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]:
            - Cleaned in-memory DataFrame.
            - Model-ready DataFrame for ML training.
            - Summary statistics dictionary.
    """
    if raw_df.empty:
        raise ValueError("Cannot prepare Inventory dataset: source DataFrame is empty.")

    # 1. Clean data in memory
    cleaned_df = clean_inventory_supplier_data(raw_df)

    # 2. In-memory contract and quality validation
    val_report = validate_dataframe(cleaned_df, "retail_analytics")
    if not val_report["contract_valid"]:
        missing = val_report.get("missing_required_columns", [])
        raise ValueError(f"Inventory dataset contract violation. Missing required columns: {missing}")

    # 3. Target validation and filtering
    if INVENTORY_TARGET not in cleaned_df.columns:
        raise KeyError(f"Required target column '{INVENTORY_TARGET}' missing from Inventory data.")

    target_numeric = pd.to_numeric(cleaned_df[INVENTORY_TARGET], errors="coerce")
    valid_target_mask = target_numeric.notna() & np.isfinite(target_numeric) & (target_numeric >= 0)
    dropped_target_rows = int((~valid_target_mask).sum())

    model_ready = cleaned_df.loc[valid_target_mask].copy()
    model_ready[INVENTORY_TARGET] = target_numeric.loc[valid_target_mask]

    # Keep validation entity identifiers (Supplier_ID, Warehouse_ID) plus model features + target
    val_cols = [c for c in ["Supplier_ID", "Warehouse_ID", "Product_ID"] if c in model_ready.columns]
    keep_cols = val_cols + INVENTORY_FEATURES + [INVENTORY_TARGET]
    seen = set()
    final_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    model_ready = model_ready[final_cols]

    stats = {
        "domain": "Inventory & Supplier",
        "raw_rows": len(raw_df),
        "raw_cols": len(raw_df.columns),
        "cleaned_rows": len(cleaned_df),
        "cleaned_cols": len(cleaned_df.columns),
        "model_ready_rows": len(model_ready),
        "model_ready_cols": len(model_ready.columns),
        "dropped_target_rows": dropped_target_rows,
        "missing_values_model_ready": int(model_ready.isna().sum().sum()),
        "target": INVENTORY_TARGET,
        "features": INVENTORY_FEATURES,
        "validation_status": val_report["status"],
    }
    return cleaned_df, model_ready, stats


def prepare_fuel_weather_datasets(
    raw_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Clean, validate, and prepare the model-ready dataset for Fuel & Weather.

    In-Memory Operations:
    1. Cleans timestamps, vehicle types, road types, and weather categories.
    2. Enforces physical non-negativity (masks negatives as NaN).
    3. Explicitly preserves synthetic provenance ('added_data_status').
    4. Validates contract and quality in memory via validate_dataframe().
    5. Confirms target: 'fuel_consumed_liters' (numeric trip fuel in Liters).
    6. Filters rows to strictly positive target values (> 0) and positive distances (> 0).
    7. Preserves 'timestamp' and 'vehicle_id' for chronological / group validation.
    8. Strictly excludes target leakage columns: 'fuel_efficiency_km_per_liter',
       'fuel_consumption_rate', 'fuel_cost_per_trip', 'fuel_price_per_liter'.
    9. Retains only confirmed model features, validation fields, provenance, and target.

    Parameters:
        raw_df (pd.DataFrame): Raw fuel and weather records.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, dict]:
            - Cleaned in-memory DataFrame.
            - Model-ready DataFrame for ML training.
            - Summary statistics dictionary.
    """
    if raw_df.empty:
        raise ValueError("Cannot prepare Fuel dataset: source DataFrame is empty.")

    # 1. Clean data in memory
    cleaned_df = clean_fuel_weather_data(raw_df)

    # 2. In-memory contract and quality validation
    val_report = validate_dataframe(cleaned_df, "fuel_weather")
    if not val_report["contract_valid"]:
        missing = val_report.get("missing_required_columns", [])
        raise ValueError(f"Fuel dataset contract violation. Missing required columns: {missing}")

    # 3. Target and distance validation
    if FUEL_TARGET not in cleaned_df.columns:
        raise KeyError(f"Required target column '{FUEL_TARGET}' missing from Fuel data.")

    target_numeric = pd.to_numeric(cleaned_df[FUEL_TARGET], errors="coerce")
    distance_numeric = pd.to_numeric(cleaned_df.get("trip_distance_km"), errors="coerce")

    # Strictly positive fuel consumed (> 0) and distance (> 0)
    valid_mask = (
        target_numeric.notna()
        & np.isfinite(target_numeric)
        & (target_numeric > 0)
        & distance_numeric.notna()
        & np.isfinite(distance_numeric)
        & (distance_numeric > 0)
    )
    dropped_target_rows = int((~valid_mask).sum())

    model_ready = cleaned_df.loc[valid_mask].copy()
    model_ready[FUEL_TARGET] = target_numeric.loc[valid_mask]

    # Keep validation fields (timestamp, vehicle_id, trip_id), provenance, features, and target
    val_cols = [c for c in ["trip_id", "vehicle_id", "timestamp", "added_data_status"] if c in model_ready.columns]
    keep_cols = val_cols + FUEL_FEATURES + [FUEL_TARGET]
    seen = set()
    final_cols = [c for c in keep_cols if not (c in seen or seen.add(c))]
    model_ready = model_ready[final_cols]

    stats = {
        "domain": "Fuel & Weather",
        "raw_rows": len(raw_df),
        "raw_cols": len(raw_df.columns),
        "cleaned_rows": len(cleaned_df),
        "cleaned_cols": len(cleaned_df.columns),
        "model_ready_rows": len(model_ready),
        "model_ready_cols": len(model_ready.columns),
        "dropped_target_rows": dropped_target_rows,
        "missing_values_model_ready": int(model_ready.isna().sum().sum()),
        "target": FUEL_TARGET,
        "features": FUEL_FEATURES,
        "validation_status": val_report["status"],
    }
    return cleaned_df, model_ready, stats


def save_dataset(df: pd.DataFrame, output_path: Path, description: str) -> None:
    """Save a DataFrame to CSV and verify its non-emptiness and readability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise IOError(f"Failed to save {description} to '{output_path}'. File is missing or empty.")


def cleanup_legacy_intermediate_files() -> None:
    """Remove legacy intermediate cleaned files if they exist in data/processed/."""
    legacy_files = [
        PROCESSED_DATA_DIR / "regional_risk_cleaned.csv",
        PROCESSED_DATA_DIR / "inventory_supplier_cleaned.csv",
        PROCESSED_DATA_DIR / "fuel_weather_cleaned.csv",
    ]
    for file_path in legacy_files:
        if file_path.exists():
            file_path.unlink()
            print(f"  [Cleanup] Removed intermediate file: {file_path.name}")


def main() -> None:
    """Execute the dataset preparation pipeline and print a structured summary."""
    import argparse

    parser = argparse.ArgumentParser(description="SupplyIQ - Model-Ready Dataset Preparation Pipeline")
    parser.add_argument(
        "--domain",
        choices=["all", "regional", "inventory", "fuel"],
        default="all",
        help="Select which dataset to prepare (default: all).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" SupplyIQ - Model-Ready Dataset Preparation Pipeline")
    print("=" * 80)
    print(f"Destination folder: {PROCESSED_DATA_DIR}")
    print(f"Domain selection:   {args.domain}")
    print("Note: Intermediate cleaned datasets are processed in memory and not written to disk.\n")

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_legacy_intermediate_files()

    all_stats = []
    output_files = []

    # ---------------- 1. Regional Risk ----------------
    if args.domain in ("all", "regional"):
        print("\nProcessing Regional Risk dataset (DataCoSupplyChainDataset.csv)...")
        raw_regional = load_raw_dataset("supply_chain")
        _, ready_reg, stats_reg = prepare_regional_risk_datasets(raw_regional)
        path_reg_ready = PROCESSED_DATA_DIR / "regional_risk_model_ready.csv"
        save_dataset(ready_reg, path_reg_ready, "Regional Risk Model-Ready")
        print(f"  -> Validated in memory: Status = {stats_reg['validation_status']}")
        print(f"  -> Model-ready saved:   {path_reg_ready.name} ({len(ready_reg):,} rows, {len(ready_reg.columns)} cols)")
        all_stats.append(stats_reg)
        output_files.append("regional_risk_model_ready.csv")

    # ---------------- 2. Inventory & Supplier ----------------
    if args.domain in ("all", "inventory"):
        print("\nProcessing Inventory & Supplier dataset (Retail_Supply_Chain_Analytics_Dataset.csv)...")
        raw_retail = load_raw_dataset("retail_analytics")
        _, ready_inv, stats_inv = prepare_inventory_supplier_datasets(raw_retail)
        path_inv_ready = PROCESSED_DATA_DIR / "inventory_demand_model_ready.csv"
        save_dataset(ready_inv, path_inv_ready, "Inventory Model-Ready")
        print(f"  -> Validated in memory: Status = {stats_inv['validation_status']}")
        print(f"  -> Model-ready saved:   {path_inv_ready.name} ({len(ready_inv):,} rows, {len(ready_inv.columns)} cols)")
        all_stats.append(stats_inv)
        output_files.append("inventory_demand_model_ready.csv")

    # ---------------- 3. Fuel & Weather ----------------
    if args.domain in ("all", "fuel"):
        print("\nProcessing Fuel & Weather dataset (supplyiq_final_dataset_filled_synthetic_fuel_weather.csv)...")
        raw_fuel = load_raw_dataset("fuel_weather")
        _, ready_fuel, stats_fuel = prepare_fuel_weather_datasets(raw_fuel)
        path_fuel_ready = PROCESSED_DATA_DIR / "fuel_model_ready.csv"
        save_dataset(ready_fuel, path_fuel_ready, "Fuel Model-Ready")
        print(f"  -> Validated in memory: Status = {stats_fuel['validation_status']}")
        print(f"  -> Model-ready saved:   {path_fuel_ready.name} ({len(ready_fuel):,} rows, {len(ready_fuel.columns)} cols)")
        all_stats.append(stats_fuel)
        output_files.append("fuel_model_ready.csv")

    # ---------------- Summary Table ----------------
    print("\n" + "=" * 80)
    print(" MODEL-READY DATASET GENERATION SUMMARY")
    print("=" * 80)
    header = f"{'Domain':<22} | {'Raw Rows':<10} | {'Model Rows':<10} | {'Model Cols':<10} | {'Dropped':<8} | {'Output File'}"
    print(header)
    print("-" * len(header))
    for s, filename in zip(all_stats, output_files):
        print(
            f"{s['domain']:<22} | "
            f"{s['raw_rows']:<10,d} | "
            f"{s['model_ready_rows']:<10,d} | "
            f"{s['model_ready_cols']:<10,d} | "
            f"{s['dropped_target_rows']:<8,d} | "
            f"{filename}"
        )
    print("=" * 80)


if __name__ == "__main__":
    main()
