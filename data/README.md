# SupplyIQ Data Directory

This directory stores raw and processed datasets for SupplyIQ.

## Raw Datasets (`data/raw/`)

1. `DataCoSupplyChainDataset.csv`: Comprehensive supply chain dataset containing orders, shipments, delivery status, customer locations, sales, and logistics performance.
2. `Retail_Supply_Chain_Analytics_Dataset.csv`: Retail inventory, supplier performance, stock levels, lead times, and fulfillment metrics.
3. `supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`: Fuel-consumption dataset. Its `added_data_status` field identifies its records as `synthetic_demo_values`. It contains trip, vehicle, fuel, distance, and weather fields but does not provide named routes, origin/destination locations, or travel times.

The application reads these original files and does not write to them. Dataset schemas and record counts are read from the files at runtime. Missing fields are not populated with substitutes.

## Processed Datasets (`data/processed/`)

- Stores transformed, cleaned, and feature-engineered datasets ready for modeling and analytics.
