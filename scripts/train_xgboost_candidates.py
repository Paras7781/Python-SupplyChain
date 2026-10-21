"""
Script to train and benchmark XGBoost candidate models for SupplyIQ.
Saves candidate artifacts and metadata under models/candidates/
without modifying existing production models.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import xgboost as xgb
import sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from ml.candidate_models import (
    train_regional_risk_xgboost,
    train_inventory_demand_xgboost,
    train_fuel_xgboost,
)


def main():
    print("=" * 70)
    print(" SupplyIQ - XGBoost Candidate Training & Benchmarking Pipeline")
    print(f" Environment scikit-learn version: {sklearn.__version__}")
    print(f" Environment XGBoost version:      {xgb.__version__}")
    print("=" * 70)

    # 1. Regional Risk
    print("\n[1/3] Training Regional Risk XGBoost Candidate...")
    rr_path = PROCESSED_DATA_DIR / "regional_risk_model_ready.csv"
    if not rr_path.exists():
        raise FileNotFoundError(f"Missing processed file: {rr_path}")
    df_rr = pd.read_csv(rr_path)
    rr_meta = train_regional_risk_xgboost(df_rr, sample_size=30000)
    print(f"  - Accuracy: {rr_meta['model_accuracy']:.4f} +/- {rr_meta['model_accuracy_std']:.4f}")
    print(f"  - ROC-AUC:  {rr_meta['roc_auc']:.4f} +/- {rr_meta['roc_auc_std']:.4f}")
    print(f"  - Precision:{rr_meta['precision']:.4f} +/- {rr_meta['precision_std']:.4f}")
    print(f"  - Recall:   {rr_meta['recall']:.4f} +/- {rr_meta['recall_std']:.4f}")
    print(f"  - F1-Score: {rr_meta['f1_score']:.4f} +/- {rr_meta['f1_score_std']:.4f}")

    # 2. Inventory Demand
    print("\n[2/3] Training Inventory Demand XGBoost Candidate...")
    inv_path = PROCESSED_DATA_DIR / "inventory_demand_model_ready.csv"
    if not inv_path.exists():
        raise FileNotFoundError(f"Missing processed file: {inv_path}")
    df_inv = pd.read_csv(inv_path)
    inv_meta = train_inventory_demand_xgboost(df_inv, sample_size=30000)
    print(f"  - MAE:      {inv_meta['mae']:.4f} +/- {inv_meta['mae_std']:.4f} units/day")
    print(f"  - RMSE:     {inv_meta['rmse']:.4f} +/- {inv_meta['rmse_std']:.4f} units/day")
    print(f"  - R2 Score: {inv_meta['r2_score']:.4f} +/- {inv_meta['r2_score_std']:.4f}")
    print(f"  - Baseline: Median = {inv_meta['baseline_median_demand']:.1f} units/day")

    # 3. Fuel Consumption
    print("\n[3/3] Training Fuel Consumption XGBoost Candidate...")
    fuel_path = PROCESSED_DATA_DIR / "fuel_model_ready.csv"
    if not fuel_path.exists():
        raise FileNotFoundError(f"Missing processed file: {fuel_path}")
    df_fuel = pd.read_csv(fuel_path)
    fuel_meta = train_fuel_xgboost(df_fuel, sample_size=30000)
    print(f"  - MAE:      {fuel_meta['mae']:.4f} +/- {fuel_meta['mae_std']:.4f} L")
    print(f"  - RMSE:     {fuel_meta['rmse']:.4f} +/- {fuel_meta['rmse_std']:.4f} L")
    print(f"  - R2 Score: {fuel_meta['r2_score']:.4f} +/- {fuel_meta['r2_score_std']:.4f}")

    print("\n" + "=" * 70)
    print(" All XGBoost candidate models successfully trained and serialized.")
    print(" Artifacts and metadata saved in models/candidates/")
    print("=" * 70)


if __name__ == "__main__":
    main()
