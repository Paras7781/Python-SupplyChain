# SupplyIQ Data Dictionary

## 1. Regional Delivery and Order Risk (DataCo Supply Chain)
- **Source File**: `data/raw/DataCoSupplyChainDataset.csv`
- **Granularity**: Line-item level (180,519 records representing 65,752 distinct orders).
- **Provenance**: Historical observed e-commerce and delivery operations.
- **Key Columns**:
  - `Order Id` (int64): Unique identifier for each customer purchase order.
  - `Order Region` (string): Geographic destination region (e.g., Central America, Western Europe, South Asia).
  - `Order State` (string): State or province of delivery destination.
  - `Order City` (string): City of delivery destination.
  - `order date (DateOrders)` (datetime): Timestamp when the customer placed the order.
  - `Delivery Status` (string): Categorical shipment outcome (`Late delivery`, `Advance shipping`, `Shipping on time`, `Shipping canceled`).
  - `Late_delivery_risk` (int64): Binary flag (1 if late delivery occurred, 0 if on time or advance).
  - `Days for shipping (real)` (int64): Actual observed delivery transit duration in days.
  - `Days for shipment (scheduled)` (int64): Planned delivery transit duration in days.
  - `Shipping Mode` (string): Carrier shipping service level (`Standard Class`, `First Class`, `Second Class`, `Same Day`).
  - `Order Item Quantity` (int64): Number of product units in the line item.
  - `Order Item Total` (float64): Net revenue amount for the line item.
  - `Sales` (float64): Gross sale amount.
  - `Customer Segment` (string): Customer classification (`Consumer`, `Corporate`, `Home Office`).
  - `Market` (string): High-level market region (e.g., LATAM, Europe, Pacific Asia, USCA, Africa).
- **Privacy & PII Exclusions**:
  Customer Email, Customer Password, Customer Fname, Customer Lname, Customer Street, and Customer Zipcode are classified as sensitive PII. They are strictly excluded from UI tables, logs, and analytics exports.

---

## 2. Supplier, Warehouse, and Inventory (Retail Supply Chain)
- **Source File**: `data/raw/Retail_Supply_Chain_Analytics_Dataset.csv`
- **Granularity**: SKU and product transaction records (200,000 records).
- **Provenance**: Observed retail chain supply chain dataset across warehouses and vendors.
- **Key Columns**:
  - `Product_ID` (string): Unique identifier for each product catalog entry.
  - `Product_Category` (string): High-level merchandise category (e.g., Electronics, Clothing, Home, Beauty, Sports).
  - `Brand` (string): Product brand name.
  - `SKU` (string): Stock keeping unit identifier.
  - `Current_Stock` (int64): Available physical units currently held in warehouse inventory.
  - `Reorder_Level` (int64): Inventory threshold below which a replenishment purchase order must be issued.
  - `Safety_Stock` (int64): Buffer inventory maintained to protect against demand spikes and lead-time delays.
  - `Daily_Demand` (int64): Observed average product units demanded per day.
  - `Units_Sold` (int64): Total sales volume recorded for the reporting period.
  - `Lead_Time_Days` (int64): Supplier replenishment lead time from order placement to warehouse receipt.
  - `Supplier_ID` (string): Vendor identifier.
  - `Supplier_Rating` (float64): Vendor performance evaluation score (1.0 to 5.0 scale).
  - `On_Time_Delivery_Rate` (float64): Fraction of vendor shipments delivered on or before the agreed date (0.0 to 1.0).
  - `Warehouse_ID` (string): Identifier of fulfillment facility storing the SKU.
  - `Warehouse_Location` (string): Geographic location of the warehouse.
  - `Storage_Capacity` (int64): Maximum physical capacity of the warehouse facility.
  - `Utilization_Rate` (float64): Percentage of warehouse capacity currently occupied.

---

## 3. Fuel Consumption and Transport (Fuel & Weather)
- **Source File**: `data/raw/supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`
- **Granularity**: Individual transportation trip records (32,065 records).
- **Provenance**: Marked `synthetic_demo_values` in source field `added_data_status`. Not validated live operations.
- **Key Columns**:
  - `trip_id` (string): Unique trip identifier.
  - `vehicle_id` (string): Fleet vehicle identifier.
  - `vehicle_type` (string): Vehicle class (`Light Truck`, `Van`, `Heavy Truck`).
  - `fuel_type` (string): Fuel chemistry (`Diesel`, `Petrol`, `Electric`, `Hybrid`).
  - `road_type` (string): Route road infrastructure (`Highway`, `Urban`, `Rural`).
  - `weather_condition` (string): Atmospheric conditions (`Clear`, `Rain`, `Fog`, `Snow`, `Storm`).
  - `temperature_c` (float64): Ambient temperature in Celsius.
  - `trip_distance_km` (float64): Total distance traveled in kilometers.
  - `load_weight_kg` (float64): Cargo payload mass in kilograms.
  - `fuel_efficiency_km_per_liter` (float64): Vehicle fuel efficiency in kilometers per liter.
  - `fuel_consumed_liters` (float64): Recorded fuel consumed in liters.
  - `fuel_price_per_liter` (float64): Unit fuel cost in USD per liter.
  - `fuel_cost_per_trip` (float64): Total fuel expenditure for the trip (`fuel_consumed_liters * fuel_price_per_liter`).
  - `added_data_status` (string): Source provenance indicator (`synthetic_demo_values`).
- **Data Limitations**:
  This dataset does NOT contain real route alternatives, GPS waypoints, origin/destination cities, or live travel times. All route comparisons must reflect these schema limits.
