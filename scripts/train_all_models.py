"""
Script to train and evaluate all three SupplyIQ machine learning models.

Uses cross-validation (Group-aware / Time-aware / Stratified) and refits the production
artifacts on all eligible rows. Saves model pipelines and metadata JSON files into models/.
Prefers pre-generated model-ready datasets from data/processed/ if available, or falls
back to loading from raw datasets.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import sklearn

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from shared.data_loader import load_raw_dataset
from ml.regional_risk.model_training import train_regional_risk_model
from ml.inventory_supplier.model_training import train_inventory_supplier_model
from ml.fuel_weather.model_training import train_fuel_weather_model


def load_dataset_for_training(domain: str) -> pd.DataFrame:
    """Load model-ready dataset if pre-generated, or fall back to raw dataset.

    Parameters:
        domain: 'regional_risk', 'inventory_supplier', or 'fuel_weather'

    Returns:
        pd.DataFrame: DataFrame ready for model training.
    """
    ready_files = {
        "regional_risk": PROCESSED_DATA_DIR / "regional_risk_model_ready.csv",
        "inventory_supplier": PROCESSED_DATA_DIR / "inventory_demand_model_ready.csv",
        "fuel_weather": PROCESSED_DATA_DIR / "fuel_model_ready.csv",
    }
    raw_keys = {
        "regional_risk": "supply_chain",
        "inventory_supplier": "retail_analytics",
        "fuel_weather": "fuel_weather",
    }

    ready_path = ready_files[domain]
    if ready_path.exists():
        print(f"  [+] Loading model-ready data from: {ready_path.name}")
        return pd.read_csv(ready_path)

    print(f"  [!] Model-ready file not found. Loading raw source key '{raw_keys[domain]}'...")
    return load_raw_dataset(raw_keys[domain])


def main():
    parser = argparse.ArgumentParser(description="SupplyIQ ML Model Training & Validation")
    parser.add_argument(
        "--model",
        "-m",
        choices=["all", "regional_risk", "inventory_supplier", "fuel_weather"],
        default="all",
        help="Specify which model to train (default: all)",
    )
    args = parser.parse_args()

    print("=" * 68)
    print(" SupplyIQ - Machine Learning Model Training & CV Validation")
    print(f" Environment scikit-learn version: {sklearn.__version__}")
    print(f" Target model(s): {args.model}")
    print("=" * 68)

    # 1. Regional Risk Model (Classification)
    if args.model in ("all", "regional_risk"):
        print("\n[1/3] Regional Delivery Risk Model (Binary Classification)...")
        df_regional = load_dataset_for_training("regional_risk")
        print(f"  Training on up to 30,000 samples (from {len(df_regional):,} available records)...")
        reg_metrics = train_regional_risk_model(df_regional, sample_size=30000)
        print("  Regional Risk Model Trained Successfully:")
        print(f"    - Validation Method: {reg_metrics['validation_method']} ({reg_metrics['fold_count']} folds)")
        print(f"    - Baseline Accuracy: {reg_metrics['baseline_accuracy']:.4f} +/- {reg_metrics['baseline_accuracy_std']:.4f}")
        print(f"    - CV Accuracy:       {reg_metrics['model_accuracy']:.4f} +/- {reg_metrics['model_accuracy_std']:.4f}")
        print(f"    - CV Precision:      {reg_metrics['precision']:.4f} +/- {reg_metrics['precision_std']:.4f}")
        print(f"    - CV Recall:         {reg_metrics['recall']:.4f} +/- {reg_metrics['recall_std']:.4f}")
        print(f"    - CV F1 Score:       {reg_metrics['f1_score']:.4f} +/- {reg_metrics['f1_score_std']:.4f}")
        print(f"    - CV ROC-AUC:        {reg_metrics['roc_auc']:.4f} +/- {reg_metrics['roc_auc_std']:.4f}")

    # 2. Inventory Demand Model (Regression)
    if args.model in ("all", "inventory_supplier"):
        print("\n[2/3] Inventory Daily Demand Model (Regression)...")
        df_retail = load_dataset_for_training("inventory_supplier")
        print(f"  Training on up to 30,000 samples (from {len(df_retail):,} available records)...")
        inv_metrics = train_inventory_supplier_model(df_retail, sample_size=30000)
        print("  Inventory Demand Model Trained Successfully:")
        print(f"    - Validation Method: {inv_metrics['validation_method']} ({inv_metrics['fold_count']} folds)")
        print(f"    - Baseline MAE:      {inv_metrics['baseline_mae']:.4f} +/- {inv_metrics['baseline_mae_std']:.4f}")
        print(f"    - CV MAE:            {inv_metrics['mae']:.4f} +/- {inv_metrics['mae_std']:.4f}")
        print(f"    - CV RMSE:           {inv_metrics['rmse']:.4f} +/- {inv_metrics['rmse_std']:.4f}")
        print(f"    - CV R2 Score:       {inv_metrics['r2_score']:.4f} +/- {inv_metrics['r2_score_std']:.4f}")

    # 3. Fuel Consumption Model (Regression)
    if args.model in ("all", "fuel_weather"):
        print("\n[3/3] Fuel Consumption Model (Regression · Synthetic Provenance)...")
        df_fuel = load_dataset_for_training("fuel_weather")
        print(f"  Training on up to 30,000 samples (from {len(df_fuel):,} available records)...")
        fuel_metrics = train_fuel_weather_model(df_fuel, sample_size=30000)
        print("  Fuel Consumption Model Trained Successfully (Synthetic Data Provenance):")
        print(f"    - Validation Method: {fuel_metrics['validation_method']} ({fuel_metrics['fold_count']} folds)")
        print(f"    - Baseline MAE:      {fuel_metrics['baseline_mae']:.4f} +/- {fuel_metrics['baseline_mae_std']:.4f}")
        print(f"    - CV MAE:            {fuel_metrics['mae']:.4f} +/- {fuel_metrics['mae_std']:.4f}")
        print(f"    - CV RMSE:           {fuel_metrics['rmse']:.4f} +/- {fuel_metrics['rmse_std']:.4f}")
        print(f"    - CV R2 Score:       {fuel_metrics['r2_score']:.4f} +/- {fuel_metrics['r2_score_std']:.4f}")

    print("\n" + "=" * 68)
    print("Specified model(s) successfully trained and saved to models/.")
    print("=" * 68)


if __name__ == "__main__":
    main()
