# SupplyIQ — Data & ML Integration Handoff for FastAPI & RAG Backend

**Author:** Data & ML Engineering  
**Date:** 2026-10-17  
**Scope:** Public Data/ML module contracts, model artifacts, inference APIs, data provenance, and integration guidelines for FastAPI routes and RAG AI assistant.

---

## 1. Executive Summary

All data processing, exploratory analysis, feature engineering, and machine learning pipelines for the three SupplyIQ operational domains are fully implemented, verified, and backed by a comprehensive unit test suite (44 passed tests).

This document provides complete interface contracts for the backend engineer attaching FastAPI endpoints and the RAG assistant.

### Key Architecture & Design Highlights
- **Decoupled from Frontend:** None of the data or ML modules depend on Streamlit. All inference functions accept standard Python dictionaries or Pandas DataFrames and return standard Python dictionaries with JSON-serializable primitives.
- **Strict Leakage Prevention:** Regional risk classification strictly removes post-fulfillment fields (`Days for shipping (real)`, `Delivery Status`, `Shipping_Difference_Days`, etc.) so only order-intake attributes are evaluated.
- **Model Caching:** Model pipelines are cached in memory using `@lru_cache` keyed on artifact file path, modification timestamp (`st_mtime_ns`), and file size (`st_size`). No repeated disk I/O on repeated requests.
- **Honest Provenance & Metrics:** All fuel consumption predictions explicitly return synthetic data provenance indicators (`is_synthetic: true`). Inventory demand regression honestly documents weak explanatory power ($R^2 \approx -0.0015$) rather than concealing real data limitations.

---

## 2. Model Artifacts & Evaluation Summary

| Domain | Task | Model Algorithm | Target Column | Units / Label | Artifact Path | Test Performance | Provenance |
|---|---|---|---|---|---|---|---|
| **Regional Risk** | Binary Classification | `RandomForestClassifier` | `Late_delivery_risk` | 0: On-time / Advance<br>1: Late delivery | `models/regional_risk_model.joblib`<br>`models/regional_risk_metadata.json` | Accuracy: **72.0%** (Baseline: 54.8%)<br>ROC-AUC: **0.757**<br>Precision: **85.4%**<br>Recall: **59.1%** | Real historical observed orders (`DataCoSupplyChainDataset.csv`) |
| **Inventory & Supplier** | Regression | `HistGradientBoostingRegressor` | `Daily_Demand` | Units demanded per day | `models/inventory_demand_model.joblib`<br>`models/inventory_demand_metadata.json` | MAE: **124.84** (Baseline MAE: 124.76)<br>RMSE: **144.23**<br>$R^2$: **-0.0015** | Real observed retail transactions (`Retail_Supply_Chain_Analytics_Dataset.csv`). Demand is uncorrelated with warehouse lead-time features. |
| **Fuel & Weather** | Regression | `HistGradientBoostingRegressor` | `fuel_consumed_liters` | Liters (L) | `models/fuel_model.joblib`<br>`models/fuel_metadata.json` | MAE: **2.51 L** (Baseline MAE: 15.81 L)<br>RMSE: **5.94 L**<br>$R^2$: **0.9417** | **Synthetic demonstration data** (`supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`). Explicitly flagged in metadata and inference. |

---

## 3. Public Inference Functions (Ready for FastAPI Routes)

### 3.1. Regional Delivery Risk Inference

- **Module:** `ml.regional_risk.model_training`
- **Function:** `predict_regional_risk(input_data: dict | pd.DataFrame) -> dict`
- **Availability Check:** `is_regional_risk_model_available() -> bool`
- **Metadata Inspection:** `get_regional_risk_metadata() -> dict`

#### Required / Accepted Input Fields:
```json
{
  "Shipping Mode": "Standard Class",       // str: 'Standard Class', 'First Class', 'Second Class', 'Same Day'
  "Order Region": "Central America",       // str: e.g., 'Central America', 'Western Europe', etc.
  "Market": "LATAM",                       // str: 'LATAM', 'Europe', 'Pacific Asia', 'USCA', 'Africa'
  "Customer Segment": "Consumer",          // str: 'Consumer', 'Corporate', 'Home Office'
  "Days for shipment (scheduled)": 4,      // int/float: >= 0 (planned days)
  "order_month": 5,                        // int: 1 to 12
  "order_dayofweek": 2,                    // int: 0 (Monday) to 6 (Sunday)
  "order_hour": 14,                        // int: 0 to 23
  "is_weekend": 0,                         // int: 0 or 1
  "Total_Units": 2,                        // int/float: >= 1
  "Total_Order_Value": 100.0,              // float: >= 0.0
  "Line_Items": 1,                         // int: >= 1
  "value_per_unit": 50.0                   // float: >= 0.0
}
```
*Note: Any missing numeric features automatically default to 0; missing categoricals default to `"Unknown"`.*

#### Return Structure:
```json
{
  "late_delivery_prediction": false,
  "late_probability": 0.3245,
  "late_probability_pct": 32.5,
  "risk_level": "Low",                     // "High" (>=0.65), "Medium" (0.40-0.65), "Low" (<0.40)
  "features_evaluated": [ ... ],
  "target": "Late_delivery_risk"
}
```

#### Proposed FastAPI Route (Teammate to Confirm):
- `POST /api/v1/predict/regional-risk`
- Request Body: JSON matching input fields above.
- Response Body: JSON matching return structure above.

---

### 3.2. Inventory Daily Demand Inference

- **Module:** `ml.inventory_supplier.model_training`
- **Function:** `predict_inventory_demand(input_data: dict | pd.DataFrame) -> dict`
- **Availability Check:** `is_inventory_model_available() -> bool`
- **Metadata Inspection:** `get_inventory_metadata() -> dict`

#### Required / Accepted Input Fields:
```json
{
  "Product_Category": "Electronics",       // str: merchandise category
  "Warehouse_Location": "North",           // str: facility geographic zone
  "Transportation_Mode": "Air",            // str: 'Air', 'Road', 'Rail', 'Sea'
  "Lead_Time_Days": 7,                     // int/float: >= 0
  "Safety_Stock": 150,                     // int/float: >= 0
  "Reorder_Level": 300,                    // int/float: >= 0
  "Supplier_Rating": 4.5,                  // float: 1.0 to 5.0
  "Storage_Capacity": 50000,               // int/float: >= 0
  "Seasonal_Demand_Index": 1.15            // float: >= 0.0
}
```

#### Return Structure:
```json
{
  "predicted_daily_demand": 142.3,
  "unit": "units_per_day",
  "features_evaluated": [ ... ],
  "model_performance_note": "Weak predictability (R2 ~ 0); interpret as rough heuristic."
}
```

#### Proposed FastAPI Route (Teammate to Confirm):
- `POST /api/v1/predict/inventory-demand`

---

### 3.3. Trip Fuel Consumption Inference

- **Module:** `ml.fuel_weather.model_training`
- **Function:** `predict_fuel_consumption(input_data: dict | pd.DataFrame, fuel_price_per_liter: float | None = None) -> dict`
- **Availability Check:** `is_fuel_model_available() -> bool`
- **Metadata Inspection:** `get_fuel_metadata() -> dict`

#### Required / Accepted Input Fields:
```json
{
  "vehicle_type": "Heavy Truck",           // str: 'Heavy Truck', 'Medium Truck', 'Light Van'
  "fuel_type": "Diesel",                   // str: 'Diesel', 'Gasoline', 'Electric'
  "road_type": "Highway",                  // str: 'Highway', 'Urban', 'Rural'
  "weather_condition": "Rain",             // str: 'Clear', 'Rain', 'Fog', 'Snow', etc.
  "trip_distance_km": 240.5,               // float: >= 0.0
  "load_weight_kg": 8500.0,                // float: >= 0.0
  "temperature_c": 18.5,                   // float: finite
  "wind_speed_kmh": 22.0,                  // float: >= 0.0
  "rainfall_mm": 5.2                       // float: >= 0.0
}
```

#### Return Structure:
```json
{
  "predicted_fuel_liters": 72.85,
  "fuel_price_per_liter": 1.45,
  "estimated_fuel_cost_usd": 105.63,
  "provenance": "synthetic_demonstration_data",
  "is_synthetic": true,
  "features_evaluated": [ ... ]
}
```

#### Proposed FastAPI Route (Teammate to Confirm):
- `POST /api/v1/predict/fuel-consumption`

---

## 4. Analytical & Data Summary Contracts

For endpoints returning high-level summary cards, charts, or tabular exports:

### 4.1. Regional Delivery Risk Summary
- **Function:** `analytics.regional_risk.eda.get_regional_summary(df: pd.DataFrame) -> dict`
- **Returns:**
  - `total_records`: int
  - `total_orders`: int
  - `late_orders`: int
  - `late_delivery_rate`: float (0.0 to 1.0)
  - `late_delivery_pct`: float (percentage)
  - `delivery_status_breakdown`: dict
  - `shipping_mode_breakdown`: dict
  - `regional_performance`: list of dicts with region, orders, late_orders, late_percentage, avg_real_days, avg_scheduled_days.

### 4.2. Inventory & Supplier Health Summary
- **Function:** `analytics.inventory_supplier.eda.get_inventory_supplier_summary(df: pd.DataFrame) -> dict`
- **Returns:**
  - `total_records`: int
  - `distinct_products`: int
  - `distinct_suppliers`: int
  - `distinct_warehouses`: int
  - `stock_health`: `{"Healthy": int, "Reorder required": int, "Critical": int, "Unknown": int}`
  - `categories`: list of category averages (stock, daily demand, lead time)
  - `top_suppliers`: top 15 suppliers sorted by mean rating.

### 4.3. Fuel & Trip Analytics Summary
- **Function:** `analytics.fuel_weather.eda.get_fuel_weather_summary(df: pd.DataFrame) -> dict`
- **Returns:**
  - `total_records`: int
  - `provenance`: `"synthetic_demonstration_data"`
  - `is_synthetic`: `True`
  - `mean_fuel_consumed_liters`: float
  - `mean_fuel_cost`: float
  - `mean_distance_km`: float
  - `vehicle_types`: breakdown by vehicle
  - `road_types`: breakdown by road condition
  - `weather_conditions`: breakdown by weather

### 4.4. Raw Dataset Loader & Validation
- **Loader:** `shared.data_loader.load_raw_dataset(dataset_key: str, sanitize_pii: bool = False) -> pd.DataFrame`
  - Keys: `"supply_chain"`, `"retail_analytics"`, `"fuel_weather"`
- **Validation:** `shared.data_validation.validate_dataframe(df: pd.DataFrame, dataset_key: str | None = None) -> dict`
  - Returns: `status` (`'valid'`, `'warning'`, `'contract_violation'`, `'empty'`), `missing_values`, `duplicate_rows`, `sensitive_columns_detected`, `type_violations`. Non-mutating.

---

## 5. RAG AI Assistant Integration Grounding

Knowledge documents have been prepared in `knowledge_base/` for vector indexing:
- `knowledge_base/business_rules.md`: Supply chain operational definitions, late delivery criteria, safety stock thresholds.
- `knowledge_base/cleaning_methodology.md`: Handling of nulls, duplicates, PII exclusions, order aggregation.
- `knowledge_base/data_dictionary.md`: Comprehensive field-by-field definitions for all 3 datasets.
- `knowledge_base/model_cards.md`: Machine learning model specifications, algorithms, features, baseline comparisons, and caveats.
- `knowledge_base/system_architecture.md`: Architectural layout across data, ML, backend API, and presentation layers.

### RAG Guidelines:
1. Always ground answers to fuel-related queries with the explicit caveat that the fuel dataset is **synthetic demonstration data**.
2. Never invent routes, origins, destinations, or real-time traffic conditions; state clearly when a feature is unobserved.
3. For inventory predictions, disclose that model $R^2 \approx 0$ due to lack of linear/non-linear correlation in the source retail dataset; recommendations use deterministic replenishment arithmetic.

---

## 6. Processed Datasets & Reproducible Commands

### 6.1. Model-Ready Datasets in `data/processed/`
The preprocessing pipeline (`scripts/prepare_datasets.py`) cleans and validates each source dataset in memory and outputs strictly the three model-ready CSV files required for ML training:

| Dataset File | Rows | Columns | Purpose | Target / Exclusions |
|---|---|---|---|---|
| `data/processed/regional_risk_model_ready.csv` | 65,752 | 17 | Classification model training | Target: `Late_delivery_risk`. Aggregated to order level; fulfillment leakage strictly dropped. |
| `data/processed/inventory_demand_model_ready.csv` | 200,000 | 13 | Regression model training | Target: `Daily_Demand`. Valid non-negative targets; group IDs preserved. |
| `data/processed/fuel_model_ready.csv` | 32,065 | 14 | Regression model training | Target: `fuel_consumed_liters`. Efficiency/cost leakage strictly excluded; synthetic provenance retained. |

*Note: Intermediate cleaned CSVs are processed and validated in memory to keep repository size lean and avoid redundant disk writes.*

### 6.2. Repeatable CLI Commands for Teammate
```bash
# 1. Regenerate the three model-ready datasets into data/processed/
python scripts/prepare_datasets.py

# 2. Cross-validate and retrain all 3 models in your active environment
python scripts/train_all_models.py

# 3. Run the full unit and regression test suite (55 tests)
python -m pytest
```

---

## 7. Open Decisions for Backend Engineer

1. **API Port and Prefix:** Recommended prefix `/api/v1/`.
2. **Schema Validation:** Define Pydantic models in `backend/schemas.py` using the input specifications in Section 3.
3. **Database & Storage:** Determine if prediction logs or simulation histories should be persisted to SQLite / PostgreSQL.
4. **Vector Store Ingestion:** Select embedding model (e.g., OpenAI `text-embedding-3-small` or HuggingFace local) for indexing documents in `knowledge_base/`.

