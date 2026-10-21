# SupplyIQ Business Rules & Operational Logic

## 1. Regional Delivery & Order Risk Business Rules

### Order Definition vs Line-Item Rows
- The raw `DataCoSupplyChainDataset.csv` contains line-item level rows. A single customer order can consist of multiple distinct line items.
- Line items belonging to the same order share the same `Order Id`.
- In all executive reporting and regional drilldowns, distinct orders are computed using `Order Id.nunique()`, whereas item lines are represented by row count.
- Order total values are the sum of `Order Item Total` across the order's items. Order units are the sum of `Order Item Quantity`.

### Late Delivery Classification
- An order is defined as **Late** if `Delivery Status == 'Late delivery'` or `Late_delivery_risk == 1`.
- `Days for shipping (real)` is the actual recorded shipping time.
- `Days for shipment (scheduled)` is the carrier promised delivery window.
- `Shipping difference (days)` = `Days for shipping (real) - Days for shipment (scheduled)`.
  - Value > 0 indicates a delayed delivery.
  - Value == 0 indicates exact on-time delivery.
  - Value < 0 indicates advance delivery.

### Leakage-Free Prediction Rules
- At the time an order is placed, only the following attributes are known:
  - Destination: Region, State, City, Market.
  - Carrier: Shipping Mode, Scheduled Days.
  - Order Characteristics: Order Date/Time, Customer Segment, Total Units, Total Order Value.
- The actual shipping days (`Days for shipping (real)`), delivery status text, and actual shipping dates are strictly prohibited from entering any prediction model at order intake time because they constitute future-information leakage.

---

## 2. Supplier, Warehouse, and Inventory Business Rules

### Stock Health Classification
Given `Current_Stock` and `Reorder_Level`:
- **Critical (Out of Stock)**: `Current_Stock <= 0`. Immediate stockout hazard requiring expedite replenishment.
- **Reorder Required**: `0 < Current_Stock <= Reorder_Level`. Inventory has dipped below the reorder point; reorder purchase order must be placed.
- **Healthy**: `Current_Stock > Reorder_Level`. Sufficient operational buffer.
- **Unknown**: Either `Current_Stock` or `Reorder_Level` is null or invalid.

### Replenishment Recommendation Logic
For items requiring replenishment:
- `Target Stock = (Daily_Demand * Lead_Time_Days) + Safety_Stock`
- `Suggested Reorder Quantity = max(0, Target Stock - Current_Stock)`
- Days of Inventory Coverage:
  `Days of Cover = Current_Stock / Daily_Demand` (where `Daily_Demand > 0`).

### Supplier Reliability Thresholds
- **High Risk Vendor**: `On_Time_Delivery_Rate < 0.85` or `Supplier_Rating < 3.0`.
- **Benchmark Vendor**: `On_Time_Delivery_Rate >= 0.95` and `Supplier_Rating >= 4.0`.

---

## 3. Fuel Consumption & Transport Business Rules

### Data Provenance Constraint
- All records in `supplyiq_final_dataset_filled_synthetic_fuel_weather.csv` bear the field `added_data_status == 'synthetic_demo_values'`.
- All outputs, metrics, and models derived from this file MUST carry visible synthetic demonstration badges.
- There are no real route alternatives, origins, or destinations in the source dataset. The application must not fabricate fictitious GPS routes or travel times.

### Deterministic Fuel Arithmetic
- Estimated fuel consumption:
  `Fuel Consumed (L) = Trip Distance (km) / Fuel Efficiency (km/L)`
- Estimated fuel expenditure:
  `Trip Fuel Cost ($) = Fuel Consumed (L) * Fuel Price ($/L)`
- Constraints: Distance and efficiency must be strictly positive (> 0) and finite numbers. Zero, negative, infinite, or NaN inputs are strictly rejected.
