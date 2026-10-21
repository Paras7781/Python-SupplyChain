"""
Train the trip fuel consumption pipeline using the synthetic fuel/weather dataset.

Loads model-ready dataset if available, or falls back to raw dataset.
Evaluates model using TimeSeriesSplit cross-validation on timestamp,
refits on eligible rows, preserves synthetic provenance metadata, and writes
fuel_model.joblib and fuel_metadata.json.
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
from ml.fuel_weather.model_training import train_fuel_weather_model


def load_dataset() -> pd.DataFrame:
    """Load fuel dataset from processed model-ready CSV or raw synthetic dataset."""
    model_ready_path = PROCESSED_DATA_DIR / "fuel_model_ready.csv"
    if model_ready_path.exists():
        print(f"Loading model-ready dataset from: {model_ready_path}")
        return pd.read_csv(model_ready_path)

    print("Model-ready file not found. Loading raw synthetic fuel/weather dataset...")
    df = load_raw_dataset("fuel_weather")
    if df.empty:
        raise FileNotFoundError(
            "The synthetic fuel/weather dataset is missing or empty. "
            "Expected it at data/raw/supplyiq_final_dataset_filled_synthetic_fuel_weather.csv."
        )
    return df


def main():
    print("=" * 68)
    print(" SupplyIQ - Fuel Consumption Model Training (Synthetic Data)")
    print(f" Environment scikit-learn version: {sklearn.__version__}")
    print("=" * 68)

    dataset = load_dataset()
    print(f"Dataset loaded ({len(dataset):,} records). Starting training on up to 30,000 samples...")
    metrics = train_fuel_weather_model(dataset, sample_size=30000)

    print("\nFuel consumption model trained and saved successfully.")
    print(f"  - scikit-learn version: {metrics['scikit_learn_version']}")
    print(f"  - data provenance:      {metrics['data_provenance']}")
    print(f"  - is_synthetic:         {metrics['is_synthetic']}")
    print(f"  - training samples:     {metrics['train_samples']}")
    print(f"  - validation method:    {metrics['validation_method']} ({metrics['fold_count']} folds)")
    print(f"  - baseline MAE:         {metrics['baseline_mae']:.4f} +/- {metrics['baseline_mae_std']:.4f} L")
    print(f"  - CV MAE:               {metrics['mae']:.4f} +/- {metrics['mae_std']:.4f} L")
    print(f"  - CV RMSE:              {metrics['rmse']:.4f} +/- {metrics['rmse_std']:.4f} L")
    print(f"  - CV R2 score:          {metrics['r2_score']:.4f} +/- {metrics['r2_score_std']:.4f}")


if __name__ == "__main__":
    main()
