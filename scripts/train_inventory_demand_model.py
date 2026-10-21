"""
Train the inventory daily demand pipeline using the Retail Analytics dataset.

Loads model-ready dataset if available, or falls back to raw dataset.
Evaluates model using GroupKFold cross-validation on Supplier_ID,
refits on eligible rows, and writes inventory_demand_model.joblib and
inventory_demand_metadata.json.
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd
import sklearn

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import PROCESSED_DATA_DIR
from shared.data_loader import load_raw_dataset
from ml.inventory_supplier.model_training import train_inventory_supplier_model


def load_dataset() -> pd.DataFrame:
    """Load inventory dataset from processed model-ready CSV or raw Retail Analytics data."""
    model_ready_path = PROCESSED_DATA_DIR / "inventory_demand_model_ready.csv"
    if model_ready_path.exists():
        print(f"Loading model-ready dataset from: {model_ready_path}")
        return pd.read_csv(model_ready_path)

    print("Model-ready file not found. Loading raw Retail Analytics dataset...")
    df = load_raw_dataset("retail_analytics")
    if df.empty:
        raise FileNotFoundError(
            "The Retail Analytics dataset is missing or empty. "
            "Expected it at data/raw/Retail_Supply_Chain_Analytics_Dataset.csv."
        )
    return df


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def main():
    print("=" * 68)
    print(" SupplyIQ - Inventory Daily Demand Model Training")
    print(f" Environment scikit-learn version: {sklearn.__version__}")
    print("=" * 68)

    dataset = load_dataset()
    print(f"Dataset loaded ({len(dataset):,} records). Starting training on up to 30,000 samples...")
    metrics = train_inventory_supplier_model(dataset, sample_size=30000)

    print("\nInventory demand model trained and saved successfully.")
    print(f"  - scikit-learn version: {metrics['scikit_learn_version']}")
    print(f"  - training samples:     {metrics['train_samples']}")
    print(f"  - validation method:    {metrics['validation_method']} ({metrics['fold_count']} folds)")
    print(f"  - baseline Mean MAE:    {metrics['baseline_mae']:.4f} +/- {metrics['baseline_mae_std']:.4f}")
    if "baseline_median_mae" in metrics:
        print(f"  - baseline Median MAE:  {metrics['baseline_median_mae']:.4f} +/- {metrics['baseline_median_mae_std']:.4f}")
    print(f"  - CV MAE:               {metrics['mae']:.4f} +/- {metrics['mae_std']:.4f}")
    print(f"  - CV RMSE:              {metrics['rmse']:.4f} +/- {metrics['rmse_std']:.4f}")
    print(f"  - CV R2 score:          {metrics['r2_score']:.4f} +/- {metrics['r2_score_std']:.4f}")

    mae_delta = metrics['mae'] - metrics['baseline_mae']
    print("\nBaseline Comparison:")
    if mae_delta < 0:
        print(f"  Model improves on baseline Dummy Mean MAE by {-mae_delta:.4f} units/day.")
    else:
        print(f"  Model does NOT improve on baseline Dummy MAE (difference: +{mae_delta:.4f} units/day).")
    if metrics['r2_score'] > 0:
        print(f"  R2 score is {metrics['r2_score']:.4f} (captures systematic operational variance in demand).")
    else:
        print(f"  R2 score is {metrics['r2_score']:.4f} (indicates negligible or negative explanatory power).")


if __name__ == "__main__":
    main()
