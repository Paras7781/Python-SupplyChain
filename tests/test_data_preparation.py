"""
Tests for the dataset preparation pipeline and processed dataset artifacts.
Uses lightweight synthetic DataFrames to keep unit tests fast and deterministic.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from scripts.prepare_datasets import (
    prepare_regional_risk_datasets,
    prepare_inventory_supplier_datasets,
    prepare_fuel_weather_datasets,
    save_dataset,
)
from config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR, DATASETS
from ml.regional_risk.model_training import (
    ALL_FEATURES as REGIONAL_FEATURES,
    TARGET_COLUMN as REGIONAL_TARGET,
)
from ml.inventory_supplier.model_training import (
    ALL_FEATURES as INVENTORY_FEATURES,
    TARGET_COLUMN as INVENTORY_TARGET,
)
from ml.fuel_weather.model_training import (
    ALL_FEATURES as FUEL_FEATURES,
    TARGET_COLUMN as FUEL_TARGET,
)


class DataPreparationTests(unittest.TestCase):
    """Unit tests for dataset cleaning and model-ready preparation pipelines."""

    def test_prepare_regional_risk_datasets_synthetic(self):
        """Verify Regional Risk cleaning, aggregation, leakage removal, and target validation."""
        synthetic_df = pd.DataFrame({
            "Order Id": [101, 101, 102, 103],
            "Customer Id": ["C1", "C1", "C2", "C3"],
            "Customer Password": ["pass1", "pass1", "pass2", "pass3"],  # PII
            "Customer Email": ["c1@a.com", "c1@a.com", "c2@a.com", "c3@a.com"],  # PII
            "Order Region": ["Central", "Central", "West", "East"],
            "Order State": ["State A", "State A", "State B", "State C"],
            "Order City": ["City A", "City A", "City B", "City C"],
            "Market": ["LATAM", "LATAM", "Europe", "USCA"],
            "Customer Segment": ["Consumer", "Consumer", "Corporate", "Home Office"],
            "Shipping Mode": ["Standard Class", "Standard Class", "First Class", "Second Class"],
            "Days for shipment (scheduled)": [4, 4, 2, 3],
            "Days for shipping (real)": [5, 5, 2, 4],  # Leakage
            "Delivery Status": ["Late", "Late", "On-time", "Late"],  # Leakage
            "Order Item Id": [1, 2, 3, 4],
            "Order Item Quantity": [2, 3, 1, 4],
            "Order Item Total": [50.0, 75.0, 30.0, 100.0],
            "order date (DateOrders)": ["2022-01-01 10:00", "2022-01-01 10:00", "2022-01-02 12:00", "2022-01-03 14:00"],
            "Late_delivery_risk": [1, 1, 0, 99],  # Row 103 has invalid class 99
        })

        cleaned, model_ready, stats = prepare_regional_risk_datasets(synthetic_df)

        # 1. Cleaned checks
        self.assertEqual(len(cleaned), 4)
        self.assertNotIn("Customer Password", cleaned.columns)
        self.assertNotIn("Customer Email", cleaned.columns)

        # 2. Aggregation and target filtering checks
        # Order 101 has 2 items -> grouped into 1. Order 103 has invalid label 99 -> dropped.
        # Retained valid orders: 101 (target 1) and 102 (target 0)
        self.assertEqual(len(model_ready), 2)
        self.assertEqual(stats["dropped_target_rows"], 1)

        # 3. Model-ready feature and leakage checks
        self.assertIn(REGIONAL_TARGET, model_ready.columns)
        self.assertNotIn(REGIONAL_TARGET, REGIONAL_FEATURES)
        for leak in ["Days for shipping (real)", "Delivery Status"]:
            self.assertNotIn(leak, model_ready.columns)
        for feat in REGIONAL_FEATURES:
            self.assertIn(feat, model_ready.columns)

        # 4. Validation identifier checks
        self.assertIn("Customer Id", model_ready.columns)
        self.assertIn("Order Id", model_ready.columns)

    def test_prepare_inventory_supplier_datasets_synthetic(self):
        """Verify Inventory cleaning, target filtering, and feature selection."""
        synthetic_df = pd.DataFrame({
            "Product_ID": ["P1", "P2", "P3", "P4"],
            "Brand": ["Brand A", "Brand B", "Brand C", "Brand D"],
            "SKU": ["SKU1", "SKU2", "SKU3", "SKU4"],
            "Current_Stock": [100, 50, 30, 200],
            "Units_Sold": [10, 5, 2, 20],
            "Supplier_ID": ["S1", "S1", "S2", "S3"],
            "Warehouse_ID": ["W1", "W1", "W2", "W2"],
            "Product_Category": ["Electronics", "Apparel", "Home", "Electronics"],
            "Warehouse_Location": ["North", "South", "East", "West"],
            "Transportation_Mode": ["Road", "Air", "Rail", "Sea"],
            "Lead_Time_Days": [7, 3, 5, 10],
            "Safety_Stock": [50, 20, 30, 40],
            "Reorder_Level": [100, 40, 60, 80],
            "Supplier_Rating": [4.5, 3.8, 4.2, 4.0],
            "Storage_Capacity": [5000, 2000, 3000, 4000],
            "Seasonal_Demand_Index": [1.1, 0.9, 1.0, 1.2],
            "Daily_Demand": [150.0, -10.0, np.nan, 200.0],  # Negative and NaN targets
        })

        cleaned, model_ready, stats = prepare_inventory_supplier_datasets(synthetic_df)

        self.assertEqual(len(cleaned), 4)
        # P2 has negative demand, P3 has NaN -> 2 rows dropped
        self.assertEqual(len(model_ready), 2)
        self.assertEqual(stats["dropped_target_rows"], 2)

        # Confirm target is present and not duplicated in features
        self.assertIn(INVENTORY_TARGET, model_ready.columns)
        self.assertNotIn(INVENTORY_TARGET, INVENTORY_FEATURES)
        for feat in INVENTORY_FEATURES:
            self.assertIn(feat, model_ready.columns)

        # Group identifier retained
        self.assertIn("Supplier_ID", model_ready.columns)

    def test_prepare_fuel_weather_datasets_synthetic(self):
        """Verify Fuel cleaning, distance/target validation, and provenance retention."""
        synthetic_df = pd.DataFrame({
            "trip_id": ["T1", "T2", "T3", "T4"],
            "vehicle_id": ["V1", "V2", "V3", "V4"],
            "vehicle_type": ["Light Truck", "Heavy Truck", "Van", "Light Truck"],
            "fuel_type": ["Diesel", "Diesel", "Gasoline", "Diesel"],
            "road_type": ["Highway", "Urban", "Rural", "Highway"],
            "weather_condition": ["Clear", "Rain", "Fog", "Snow"],
            "trip_distance_km": [100.0, -5.0, 50.0, 0.0],  # -5 and 0 are invalid distances
            "fuel_efficiency_km_per_liter": [8.5, 4.0, 10.0, 8.0],
            "fuel_price_per_liter": [1.5, 1.5, 1.5, 1.5],
            "fuel_cost_per_trip": [18.0, 37.5, 0.0, 22.5],
            "load_weight_kg": [1000.0, 2000.0, 500.0, 1200.0],
            "temperature_c": [20.0, 15.0, 10.0, -2.0],
            "wind_speed_kmh": [10.0, 25.0, 5.0, 15.0],
            "rainfall_mm": [0.0, 5.0, 0.0, 2.0],
            "fuel_consumed_liters": [12.0, 25.0, -2.0, 15.0],  # -2 is invalid fuel
            "added_data_status": ["synthetic_demo_values"] * 4,
            "timestamp": pd.date_range("2022-01-01", periods=4, freq="D"),
        })

        cleaned, model_ready, stats = prepare_fuel_weather_datasets(synthetic_df)

        self.assertEqual(len(cleaned), 4)
        # T2 has invalid distance, T3 has negative fuel, T4 has 0 distance -> only T1 valid
        self.assertEqual(len(model_ready), 1)
        self.assertEqual(stats["dropped_target_rows"], 3)

        # Provenance must be preserved
        self.assertIn("added_data_status", model_ready.columns)
        self.assertEqual(model_ready.iloc[0]["added_data_status"], "synthetic_demo_values")

        # Target not in features
        self.assertIn(FUEL_TARGET, model_ready.columns)
        self.assertNotIn(FUEL_TARGET, FUEL_FEATURES)
        for feat in FUEL_FEATURES:
            self.assertIn(feat, model_ready.columns)

    def test_save_dataset_creates_directory_and_file(self):
        """Verify save_dataset creates missing parent directories and writes readable CSV."""
        with tempfile.TemporaryDirectory() as temp_dir:
            test_path = Path(temp_dir) / "nested" / "subfolder" / "output.csv"
            df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
            save_dataset(df, test_path, "Test Dataset")
            self.assertTrue(test_path.exists())
            reloaded = pd.read_csv(test_path)
            self.assertEqual(len(reloaded), 2)
            self.assertEqual(list(reloaded.columns), ["col_a", "col_b"])

    def test_processed_directory_csv_files_exist_and_match_contract(self):
        """Verify only the 3 model-ready CSVs exist on disk and are non-empty."""
        expected_model_ready_files = [
            "regional_risk_model_ready.csv",
            "inventory_demand_model_ready.csv",
            "fuel_model_ready.csv",
        ]
        for filename in expected_model_ready_files:
            file_path = PROCESSED_DATA_DIR / filename
            self.assertTrue(file_path.exists(), f"Model-ready file missing: {filename}")
            self.assertGreater(file_path.stat().st_size, 0, f"File is empty: {filename}")

            # Verify header is readable
            header = pd.read_csv(file_path, nrows=5)
            self.assertGreater(len(header.columns), 0)

        # Verify intermediate cleaned files are NOT written to disk
        intermediate_cleaned_files = [
            "regional_risk_cleaned.csv",
            "inventory_supplier_cleaned.csv",
            "fuel_weather_cleaned.csv",
        ]
        for filename in intermediate_cleaned_files:
            file_path = PROCESSED_DATA_DIR / filename
            self.assertFalse(
                file_path.exists(),
                f"Intermediate cleaned file should not be saved: {filename}",
            )


if __name__ == "__main__":
    unittest.main()
