<<<<<<< HEAD
# SupplyIQ - Supply Chain Analytics & Risk Intelligence Platform

SupplyIQ is a Streamlit supply-chain intelligence frontend. Its primary workflows are a regional order explorer, supplier/inventory intelligence, and a fuel-consumption records viewer. Prediction, scenario simulation, and AI pages display an unavailable state until their backend services are connected.

## Current UI modules and source support

- **Regional order explorer** reads `DataCoSupplyChainDataset.csv`: `Order Region`, `Order State`, `Order City`, `Order Id`, `order date (DateOrders)`, `Delivery Status`, `Late_delivery_risk`, actual/scheduled shipping days, and `Shipping Mode`. Source rows are item-level and are grouped to one order per ID. Exports omit customer names, contact details, passwords, and street addresses.
- **Supplier & inventory intelligence** reads `Retail_Supply_Chain_Analytics_Dataset.csv`: `Product_ID`, `Product_Category`, `Brand`, `SKU`, `Supplier_ID`, `Supplier_Rating`, `Warehouse_ID`, `Warehouse_Location`, stock and reorder/safety levels, demand fields, `Date`, lead time, and on-time delivery rate. Supplier and product names are not present. Predictive and optimization fields are not presented.
- **Fuel consumption** reads `supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`: `trip_id`, `vehicle_id`, `vehicle_type`, `fuel_type`, `road_type`, `weather_condition`, `trip_distance_km`, `fuel_efficiency_km_per_liter`, `fuel_consumed_liters`, `fuel_price_per_liter`, and `fuel_cost_per_trip`. The source marks records `synthetic_demo_values`; they are not validated live operational data. The file has no route alternatives, origin/destination locations, or travel-time field. The UI calculates litres = recorded distance ÷ recorded efficiency and may calculate cost using the same row's recorded price.

The three datasets are separate; the overview does not join them. Source CSVs remain read-only. This is a frontend-only implementation and does not add data cleaning, feature engineering, model training, APIs, databases, or retrieval-augmented generation.

The Reports & Exports page downloads full source extracts with allowlisted operational columns; it does not inherit filters from other pages. The Regional Order Explorer also provides a CSV download of the orders matching its current filters.

Fuel calculation tests can be run with:

```bash
python -m unittest tests.test_ui_planning -v
```

## Project Structure

```
SupplyIQ/
├── app.py
├── requirements.txt
├── README.md
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── raw/
│   │   ├── DataCoSupplyChainDataset.csv
│   │   ├── Retail_Supply_Chain_Analytics_Dataset.csv
│   │   └── supplyiq_final_dataset_filled_synthetic_fuel_weather.csv
│   ├── processed/
│   │   └── .gitkeep
│   └── README.md
├── models/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── data_validation.py
│   ├── ui_planning.py
│   ├── regional_risk/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── eda.py
│   │   ├── feature_engineering.py
│   │   └── model_training.py
│   ├── inventory_supplier/
│   │   ├── __init__.py
│   │   ├── cleaning.py
│   │   ├── eda.py
│   │   ├── feature_engineering.py
│   │   └── model_training.py
│   └── fuel_weather/
│       ├── __init__.py
│       ├── cleaning.py
│       ├── eda.py
│       ├── feature_engineering.py
│       └── model_training.py
├── dashboard/
│   ├── __init__.py
│   └── components/
│       ├── __init__.py
│       ├── navigation.py
│       ├── cards.py
│       └── charts.py
├── pages/
│   ├── __init__.py
│   ├── 01_Executive_Overview.py
│   ├── 02_Regional_Risk.py
│   ├── 03_Inventory_Supplier.py
│   ├── 04_Fuel_Weather_Analytics.py
│   ├── 05_Risk_Prediction.py
│   ├── 06_What_If_Simulator.py
│   ├── 07_Reports_Exports.py
│   ├── 08_AI_Assistant.py
│   └── 09_Project_Info_Model_Trust.py
└── tests/
    └── test_ui_planning.py
```

## Setup and Running

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Streamlit Dashboard:
   ```bash
   streamlit run app.py
   ```
=======

>>>>>>> origin/Paras
