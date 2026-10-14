"""
SupplyIQ Configuration Settings.
Provides central relative paths, app configuration settings, and dataset definitions.
"""
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Actual Raw Dataset Paths
DATASETS = {
    "supply_chain": RAW_DATA_DIR / "DataCoSupplyChainDataset.csv",
    "retail_analytics": RAW_DATA_DIR / "Retail_Supply_Chain_Analytics_Dataset.csv",
    "fuel_weather": RAW_DATA_DIR / "supplyiq_final_dataset_filled_synthetic_fuel_weather.csv",
}

# App Configuration
APP_NAME = "SupplyIQ"
APP_TITLE = "SupplyIQ - Supply Chain Analytics & Risk Intelligence"
APP_ICON = "📦"
