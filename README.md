# SupplyIQ - Supply Chain Risk & Decision Intelligence Platform

SupplyIQ is a supply chain intelligence platform combining an interactive **Streamlit frontend**, in-process **Scikit-Learn machine learning pipelines**, and local data services.

The platform covers three primary supply chain domains:
1. **Regional Delivery and Order Risk**: Order-level exploration, geographical delay drill-downs, carrier comparisons, and predictive delay risk scoring.
2. **Supplier, Warehouse, and Inventory Intelligence**: Multi-warehouse stock health monitoring, reorder recommendations, and supplier performance ranking.
3. **Fuel Consumption and Transport Costing**: Trip fuel consumption estimation, vehicle class comparisons, and fuel price sensitivity simulation.

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend (UI)                         │
│ 01 Exec | 02 Regional | 03 Inventory | 04 Fuel | 05 Predict            │
│ 06 What-If | 07 Reports | 09 Model Trust                               │
└─────────────────────────────────┬──────────────────────────────────────┘
                                  │ (In-process data and model services)
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

The application uses `database/` to store explicitly saved regional
assessments (`data/supplyiq.sqlite3`). This store does not replace or modify
the raw CSVs.

---

## Data Provenance & Integrity

The three original CSV files in `data/raw/` serve as the immutable single source of truth:
- **`DataCoSupplyChainDataset.csv`** (95.9 MB, 180,519 rows, 53 columns): Historical e-commerce transactions at item-level. Grouped by `Order Id` (65,752 distinct orders) for order-level KPIs. Customer PII (passwords, emails, street addresses) is strictly sanitized and never exposed.
- **`Retail_Supply_Chain_Analytics_Dataset.csv`** (48.5 MB, 200,000 rows, 41 columns): Clean retail supply chain transactions across warehouses and suppliers. 0 missing cells, 0 duplicate rows. Note: `Daily_Demand` contains synthetic demonstration values derived from operational features (`Product_Category`, `Warehouse_Location`, `Seasonal_Demand_Index` + noise) for learnable ML modeling; the original unlearnable uniform random dataset is preserved at `data/backup/Retail_Supply_Chain_Analytics_Dataset_original.csv`.
- **`supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`** (19.2 MB, 32,065 rows, 43 columns): Marked `synthetic_demo_values` in source field `added_data_status`. All derived analytics and models explicitly carry synthetic provenance badges. No fictitious routes or origin-destinations are invented.

---

## Machine Learning Models & Evaluation

Models are serialized in `models/` with associated `metadata.json` audit records.
All reported scores below are cross-validation validation estimates (mean +/- fold
standard deviation), not untouched final-test scores. Preprocessing is fitted
inside each fold, and the production artifact is refitted on all eligible rows
after validation.

| Model | Validation | Baseline CV Metric | Model CV Estimates (mean +/- SD) | Interpretation |
|---|---|---|---|---|
| **Regional Delivery Risk** (`Late_delivery_risk`) | 5-fold chronological `TimeSeriesSplit`, with earlier rows for validation customers purged | Accuracy: 0.5487 +/- 0.0036 | Accuracy: 0.7156 +/- 0.0046; Precision: 0.8442 +/- 0.0125; Recall: 0.5909 +/- 0.0083; F1: 0.6952 +/- 0.0060; ROC-AUC: 0.7541 +/- 0.0064 | Post-order shipping fields excluded; temporal and operational drift may remain. |
| **Inventory Daily Demand** (`Daily_Demand`) | 5-fold `GroupKFold` by `Supplier_ID` | MAE: 125.3528 +/- 0.6149 | MAE: 125.4426 +/- 0.6217; RMSE: 144.9202 +/- 0.4583; R²: -0.0025 +/- 0.0020 | Negative R² and slightly worse MAE than baseline are retained. Warehouse IDs overlap across supplier folds but are not model inputs; this does not assess temporal drift or unseen-warehouse generalization. |
| **Fuel Consumption** (`fuel_consumed_liters`) | 5-fold chronological `TimeSeriesSplit` | MAE: 15.9061 +/- 0.2460 L | MAE: 2.5412 +/- 0.1005 L; RMSE: 5.6064 +/- 0.7970 L; R²: 0.9478 +/- 0.0171 | All records are synthetic demonstrations; recurring vehicles and temporal autocorrelation limit generalization claims. |

To retrain all models:
```bash
python scripts/train_all_models.py
```

---

---

## Setup & Running on Windows PowerShell

### 1. Prerequisites
- Python 3.11 installed.

### 2. Environment Configuration
Copy `.env.example` to `.env`:
```powershell
Copy-Item .env.example .env
```

### 3. Install Dependencies
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Train Machine Learning Models
```powershell
.\.venv\Scripts\python.exe scripts/train_all_models.py
```

### 5. Launch Streamlit Dashboard
Run this from the repository root so the project Streamlit configuration is loaded:
```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```
Open `http://localhost:8501` in your browser. Predictions run in-process with the dashboard.

---

## Running Automated Tests

Run the full pytest suite using the project virtual environment (data loaders, validation, cleaning, EDA, feature engineering, ML models, XGBoost candidates, and risk assessment flows):
```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

> **Important:** Always use `.venv\Scripts\python.exe`, not the bare `python` command. The system Python interpreter may have a different scikit-learn version and will not have xgboost installed, causing false import errors and model compatibility warnings.

---

## Project Structure

```
supply_iq/
├── app.py                      # Streamlit application entry point & launchpad
├── requirements.txt            # Python dependencies
├── README.md                   # System documentation
├── AUDIT_REPORT.md             # Change audit and integrity report
├── .env.example                # Environment variable template
├── .gitignore                  # Git exclusions
├── config/
│   ├── __init__.py
│   └── settings.py             # Paths and application configurations
├── shared/                     # Cross-cutting utilities used by pages and ML
│   ├── api_client.py           # In-process model and assessment client
│   ├── data_loader.py          # Data ingestion with mtime caching & PII sanitization
│   ├── data_validation.py      # Non-destructive quality audit & provenance checks
│   └── model_validation.py     # Sklearn version guard and pipeline sanity checks
├── ml/                         # Machine learning training, inference, and explainability
│   ├── candidate_models.py     # XGBoost candidate pipeline definitions
│   ├── explainability.py       # SHAP value computation and feature importance helpers
│   ├── regional_risk/
│   │   ├── feature_engineering.py
│   │   └── model_training.py   # Regional delivery risk RF classifier & prediction
│   ├── inventory_supplier/
│   │   ├── feature_engineering.py
│   │   └── model_training.py   # Inventory demand HGBR regressor & prediction
│   └── fuel_weather/
│       ├── feature_engineering.py
│       └── model_training.py   # Fuel consumption HGBR regressor & prediction
├── analytics/                  # Domain-specific data cleaning and EDA helpers
│   ├── regional_risk/
│   │   ├── cleaning.py
│   │   └── eda.py
│   ├── inventory_supplier/
│   │   ├── cleaning.py
│   │   └── eda.py
│   └── fuel_weather/
│       ├── cleaning.py
│       └── eda.py
├── ui/                         # Reusable Streamlit UI components
│   ├── cards.py
│   ├── charts.py
│   └── navigation.py
├── rag/                        # Retrieval-augmented generation (placeholder scaffolds)
│   ├── config.py
│   ├── engine.py
│   ├── ingest.py
│   ├── semantic_assistant.py
│   └── vector_store.py
├── backend/                    # FastAPI scaffolding (reserved for future REST API)
│   ├── config.py
│   ├── dependencies.py
│   ├── main.py
│   └── schemas.py
├── database/
│   ├── connection.py           # SQLite connection and transaction helpers
│   ├── schema.py               # Database table and index definitions
│   ├── repository.py           # Saved assessment persistence operations
│   └── __main__.py             # Database initialization command entry point
├── data/
│   ├── raw/                    # Immutable source CSV files
│   │   ├── DataCoSupplyChainDataset.csv
│   │   ├── Retail_Supply_Chain_Analytics_Dataset.csv
│   │   └── supplyiq_final_dataset_filled_synthetic_fuel_weather.csv
│   └── processed/              # Model-ready CSV outputs (generated by scripts/)
│       ├── regional_risk_model_ready.csv
│       ├── inventory_demand_model_ready.csv
│       └── fuel_model_ready.csv
├── models/                     # Serialized ML pipelines (.joblib) and metadata (.json)
│   └── candidates/             # XGBoost candidate artifacts
├── knowledge_base/             # Project and domain documentation
│   ├── data_dictionary.md
│   ├── business_rules.md
│   ├── cleaning_methodology.md
│   ├── model_cards.md
│   └── system_architecture.md
├── docs/                       # Extended project documentation and audit artefacts
│   ├── DATA_ML_INTEGRATION_HANDOFF.md
│   ├── Project_Change_Audit_and_File_Documentation.md
│   ├── build_documentation.py
│   └── generate_report_pdf.py
├── pages/                      # All nine Streamlit pages
│   ├── 01_Executive_Overview.py
│   ├── 02_Regional_Risk.py
│   ├── 03_Inventory_Supplier.py
│   ├── 04_Fuel_Weather_Analytics.py
│   ├── 05_Risk_Prediction.py
│   ├── 06_What_If_Simulator.py
│   ├── 07_Reports_Exports.py
│   ├── 08_AI_Assistant.py
│   └── 09_Project_Info_Model_Trust.py
├── scripts/
│   ├── prepare_datasets.py         # Data preparation and processed CSV generation
│   ├── train_all_models.py         # Reproducible full model training pipeline
│   ├── train_regional_risk_model.py
│   ├── train_inventory_demand_model.py
│   ├── train_fuel_model.py
│   └── train_xgboost_candidates.py
└── tests/                      # Automated unit and integration test suite
    ├── test_api.py
    ├── test_api_client.py
    ├── test_backend_scaffolds.py
    ├── test_candidate_models.py
    ├── test_cleaning_eda_features.py
    ├── test_data_loader.py
    ├── test_data_preparation.py
    ├── test_data_validation.py
    ├── test_database.py
    ├── test_explainability.py
    ├── test_ml_models.py
    ├── test_rag.py
    ├── test_rag_semantic.py
    ├── test_risk_assessment.py
    └── test_ui_planning.py
```
