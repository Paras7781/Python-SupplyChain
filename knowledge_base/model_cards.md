# SupplyIQ Model Cards & AI Governance

## Model Card 1: Regional Delivery Risk Classifier

- **Model Architecture**: Random Forest Classifier within Scikit-Learn Pipeline (`SimpleImputer` + `StandardScaler` + `OneHotEncoder` + `RandomForestClassifier(n_estimators=50, max_depth=10)`).
- **Task**: Binary Classification (`0` = On-Time / Advance, `1` = Late Delivery Risk).
- **Target Column**: `Late_delivery_risk`.
- **Intended Use**: Predict likelihood of delivery delay at the time of customer order placement.
- **Order-Time Features Used**:
  - `Days for shipment (scheduled)`
  - `Shipping Mode`
  - `Order Region`
  - `Market`
  - `Customer Segment`
  - `order_month`, `order_dayofweek`, `order_hour`, `is_weekend`
  - `Total_Units`, `Total_Order_Value`, `Line_Items`, `value_per_unit`
- **Leakage Prevention**:
  - `Days for shipping (real)` and `Delivery Status` are strictly excluded from training and inference.
  - Item rows are aggregated to `Order Id` before validation. Chronological folds use `TimeSeriesSplit`; earlier rows for a customer in a validation fold are purged from that fold's training data.
  - Preprocessing is fitted as part of the pipeline independently within each fold.
- **Validation Limitations**: Metrics are cross-validation validation estimates, not an untouched final-test score. Customer purging and chronological ordering reduce these specific leakage risks but do not guarantee unbiased performance under future operational drift.
- **Evaluation Metrics**:
  - 5-fold CV estimates (mean +/- SD): accuracy 0.7156 +/- 0.0046, precision 0.8442 +/- 0.0125, recall 0.5909 +/- 0.0083, F1 0.6952 +/- 0.0060, ROC-AUC 0.7541 +/- 0.0064.
  - Most-frequent baseline accuracy: 0.5487 +/- 0.0036.

---

## Model Card 2: Inventory Daily Demand Regressor

- **Model Architecture**: Histogram-based Gradient Boosting Regressor (`HistGradientBoostingRegressor(max_iter=50, max_depth=6)`).
- **Task**: Continuous Demand Estimation.
- **Target Column**: `Daily_Demand`.
- **Features Used**:
  - `Product_Category`, `Warehouse_Location`, `Transportation_Mode`
  - `Lead_Time_Days`, `Safety_Stock`, `Reorder_Level`, `Supplier_Rating`, `Storage_Capacity`, `Seasonal_Demand_Index`
- **Validation Strategy**: `GroupKFold` by `Supplier_ID` because suppliers repeat across rows. `Product_ID` is unique in the source; `Warehouse_ID` repeats across supplier folds but is not a model feature. The supplier-warehouse entity graph is connected, so simultaneously excluding both entity types would not yield multiple useful folds. The available transaction date is not a model feature, so this validation does not estimate future temporal drift.
- **Data Reality & Honest Metric Reporting**:
  - 5-fold CV estimates (mean +/- SD): baseline MAE 125.3528 +/- 0.6149; model MAE 125.4426 +/- 0.6217; RMSE 144.9202 +/- 0.4583; R² -0.0025 +/- 0.0020.
  - Negative or near-zero R² is reported as-is; cross-validation does not resolve weak predictive signal. Validation scores are estimates, not a final-test result.

---

## Model Card 3: Fuel Consumption Regressor

- **Model Architecture**: Histogram-based Gradient Boosting Regressor (`HistGradientBoostingRegressor(max_iter=50, max_depth=6)`).
- **Task**: Trip Fuel Consumption Estimation in Liters.
- **Target Column**: `fuel_consumed_liters`.
- **Features Used**:
  - `trip_distance_km`, `load_weight_kg`, `temperature_c`, `wind_speed_kmh`, `rainfall_mm`
  - `vehicle_type`, `fuel_type`, `road_type`, `weather_condition`
- **Provenance Limitation**:
  - **SYNTHETIC DEMONSTRATION DATA**: The source CSV marks all records `synthetic_demo_values`.
  - Predictions must be interpreted as simulated engineering estimates, not operational field measurements.
- **Validation Strategy**: `TimeSeriesSplit` ordered by `timestamp`. Repeated vehicle observations occur across periods; vehicle identity is not a model feature, and temporal autocorrelation may remain.
- **Validation Limitations**: Cross-validation scores are estimates on synthetic demonstration data only, not independent evidence of real-world fleet performance or an untouched final-test score.
- **Evaluation Metrics**:
  - 5-fold CV estimates (mean +/- SD): baseline MAE 15.9061 +/- 0.2460 L; model MAE 2.5412 +/- 0.1005 L; RMSE 5.6064 +/- 0.7970 L; R² 0.9478 +/- 0.0171.
