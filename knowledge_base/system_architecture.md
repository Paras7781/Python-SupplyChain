# SupplyIQ System Architecture

## Overview
SupplyIQ is a modular supply chain risk management platform featuring:
1. **Interactive Frontend**: A Streamlit application delivering executive overviews, regional order exploration, inventory monitoring, fuel planning, predictive inference, scenario simulation, report exports, and model governance.
2. **Machine Learning Pipeline**: Leakage-free Scikit-Learn pipelines serialized as joblib artifacts accompanied by JSON metadata for auditability. Predictions run in-process with the Streamlit application.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend                              │
│ 01 Exec | 02 Regional | 03 Inventory | 04 Fuel | 05 Predict            │
│ 06 What-If | 07 Reports | 09 Model Trust                               │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ (In-process model and data services)
┌─────────────────────────────────▼──────────────────────────────────────┐
│  Domain Analytics & ML                                                 │
│  - Regional Risk (RF) | Inventory Demand (HGBR) | Fuel (HGBR)         │
│  - Leakage Safeguards                                                  │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │
┌─────────────────────────────────▼──────────────────────────────────────┐
│                     Immutable Gold Datasets (data/raw/)                │
│  - DataCoSupplyChainDataset.csv (180k rows, item-level)                │
│  - Retail_Supply_Chain_Analytics_Dataset.csv (200k rows)               │
│  - supplyiq_final_dataset_filled_synthetic_fuel_weather.csv (32k rows) │
└────────────────────────────────────────────────────────────────────────┘
```
