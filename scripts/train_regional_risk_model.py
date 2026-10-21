"""
Train the regional risk pipeline using the project's DataCo dataset.

Loads model-ready dataset if available, or falls back to raw dataset.
Evaluates model using chronological validation with customer purging,
refits on eligible rows, and writes regional_risk_model.joblib and
regional_risk_metadata.json.
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
from ml.regional_risk.model_training import train_regional_risk_model


def load_dataset() -> pd.DataFrame:
    """Load regional risk dataset from processed model-ready CSV or raw DataCo data."""
    model_ready_path = PROCESSED_DATA_DIR / "regional_risk_model_ready.csv"
    if model_ready_path.exists():
        print(f"Loading model-ready dataset from: {model_ready_path}")
        return pd.read_csv(model_ready_path)

    print("Model-ready file not found. Loading raw DataCo supply chain dataset...")
    df = load_raw_dataset("supply_chain")
    if df.empty:
        raise FileNotFoundError(
            "The DataCo supply-chain dataset is missing or empty. "
            "Expected it at data/raw/DataCoSupplyChainDataset.csv."
        )
    return df


def main():
    print("=" * 68)
    print(" SupplyIQ - Regional Delivery Risk Model Training")
    print(f" Environment scikit-learn version: {sklearn.__version__}")
    print("=" * 68)

    dataset = load_dataset()
    print(f"Dataset loaded ({len(dataset):,} records). Starting training...")
    metrics = train_regional_risk_model(dataset, sample_size=30000)

    print("\nRegional risk model trained and saved successfully.")
    print(f"  - scikit-learn version: {metrics['scikit_learn_version']}")
    print(f"  - training samples:     {metrics['train_samples']}")
    print(f"  - validation method:    {metrics['validation_method']} ({metrics['fold_count']} folds)")
    print(f"  - baseline accuracy:    {metrics['baseline_accuracy']:.4f} +/- {metrics['baseline_accuracy_std']:.4f}")
    print(f"  - CV accuracy:          {metrics['model_accuracy']:.4f} +/- {metrics['model_accuracy_std']:.4f}")
    print(f"  - CV precision:         {metrics['precision']:.4f} +/- {metrics['precision_std']:.4f}")
    print(f"  - CV recall:            {metrics['recall']:.4f} +/- {metrics['recall_std']:.4f}")
    print(f"  - CV F1 score:          {metrics['f1_score']:.4f} +/- {metrics['f1_score_std']:.4f}")
    print(f"  - CV ROC-AUC:           {metrics['roc_auc']:.4f} +/- {metrics['roc_auc_std']:.4f}")


if __name__ == "__main__":
    main()
