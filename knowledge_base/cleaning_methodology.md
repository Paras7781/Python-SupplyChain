# SupplyIQ Data Cleaning & Preprocessing Methodology

## 1. Data Cleaning Philosophy
- **Raw Data Immutability**: The three original CSV datasets in `data/raw/` are treated as read-only gold sources. They are never overwritten, renamed, mutated, or deleted.
- **No Fabricated Facts**: Missing values are never silently replaced with invented business records or dummy values.
- **Audit Trails**: All data transformations, type casting, filtering, and sanitization are logged with complete traceability.

---

## 2. Dataset-Specific Cleaning Strategies

### DataCo Regional Supply Chain (`DataCoSupplyChainDataset.csv`)
1. **PII Sanitization**:
   - Dropped columns: `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, `Customer Street`, `Customer Zipcode`.
   - Protects customer confidentiality in accordance with GDPR and privacy best practices.
2. **Missing Value Resolution**:
   - `Product Description`: 100% missing (all 180,519 rows null). Column dropped entirely as it carries zero variance.
   - `Order Zipcode`: 86.2% missing. Retained as missing/unobserved; no zipcodes are fabricated.
   - `Customer Lname`: 8 missing cells. Filled with 'Unknown' during order grouping.
   - `Customer Zipcode`: 3 missing cells.
3. **Type Normalization**:
   - `order date (DateOrders)` and `shipping date (DateOrders)` converted to ISO datetime.
   - Categorical fields stripped of leading/trailing whitespace.
   - Financial and quantity fields cast to float64 / int64.
4. **Order Aggregation**:
   - Line items collapsed to order-level via `Order Id`.
   - Line item count, units sum, and order revenue accurately aggregated.

### Retail Supply Chain Analytics (`Retail_Supply_Chain_Analytics_Dataset.csv`)
1. **Data Completeness**:
   - 200,000 records with 0 missing cells and 0 duplicate rows.
2. **Type Enforcement**:
   - Date field converted to datetime.
   - Strings stripped of whitespace.
   - Categorical identifiers (`Product_ID`, `Supplier_ID`, `Warehouse_ID`) formatted as clean strings.
3. **Range Validation**:
   - Verified that stock levels, daily demand, safety stock, and lead time are non-negative.
   - Zero values in demand or sales are retained as valid physical zeroes (no activity).

### Fuel & Weather Transport Dataset (`supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`)
1. **Provenance Tagging**:
   - All 32,065 rows tagged with `provenance: synthetic_demonstration_data`.
2. **Physical Constraints Validation**:
   - `trip_distance_km` > 0
   - `load_weight_kg` >= 0
   - `fuel_efficiency_km_per_liter` > 0
   - `fuel_consumed_liters` >= 0
   - Negative or non-finite records are rejected or flagged during pipeline ingestion.
