"""
Tests for XGBoost Candidate Models in SupplyIQ.
Verifies candidate model artifacts, metadata, loading, and prediction contracts.
"""
import unittest
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from config.settings import MODELS_DIR
from ml.candidate_models import CANDIDATES_DIR


class TestCandidateModels(unittest.TestCase):
    def test_candidate_artifacts_and_metadata_exist(self):
        """Verify that candidate artifacts and JSON metadata were generated."""
        expected_files = [
            "regional_risk_model_xgboost.joblib",
            "regional_risk_xgboost_metadata.json",
            "inventory_demand_model_xgboost.joblib",
            "inventory_demand_xgboost_metadata.json",
            "fuel_model_xgboost.joblib",
            "fuel_xgboost_metadata.json",
        ]
        for fname in expected_files:
            file_path = CANDIDATES_DIR / fname
            self.assertTrue(file_path.exists(), f"Candidate file {fname} should exist in {CANDIDATES_DIR}")

    def test_regional_risk_xgboost_reload_and_predict(self):
        """Verify that the Regional Risk XGBoost candidate reloads and predicts valid probabilities."""
        model_path = CANDIDATES_DIR / "regional_risk_model_xgboost.joblib"
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

        sample = pd.DataFrame([{
            "Shipping Mode": "Standard Class",
            "Order Region": "Western Europe",
            "Market": "Europe",
            "Customer Segment": "Consumer",
            "Days for shipment (scheduled)": 4,
            "order_month": 5,
            "order_dayofweek": 2,
            "order_hour": 14,
            "is_weekend": 0,
            "Total_Units": 2,
            "Total_Order_Value": 150.0,
            "Line_Items": 2,
            "value_per_unit": 75.0,
        }])
        proba = model.predict_proba(sample)
        self.assertEqual(proba.shape, (1, 2))
        self.assertGreaterEqual(proba[0, 1], 0.0)
        self.assertLessEqual(proba[0, 1], 1.0)

    def test_inventory_xgboost_reload_and_predict(self):
        """Verify that the Inventory Demand XGBoost candidate reloads and predicts valid demand."""
        model_path = CANDIDATES_DIR / "inventory_demand_model_xgboost.joblib"
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

        sample = pd.DataFrame([{
            "Product_Category": "Electronics",
            "Warehouse_Location": "East",
            "Transportation_Mode": "Road",
            "Lead_Time_Days": 14,
            "Safety_Stock": 100,
            "Reorder_Level": 200,
            "Supplier_Rating": 3.5,
            "Storage_Capacity": 10000,
            "Seasonal_Demand_Index": 50.0,
        }])
        pred = model.predict(sample)
        self.assertEqual(len(pred), 1)
        self.assertTrue(np.isfinite(pred[0]))

    def test_fuel_xgboost_reload_and_predict(self):
        """Verify that the Fuel XGBoost candidate reloads and predicts positive fuel consumption."""
        model_path = CANDIDATES_DIR / "fuel_model_xgboost.joblib"
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

        sample = pd.DataFrame([{
            "vehicle_type": "Light Truck",
            "fuel_type": "Diesel",
            "road_type": "Highway",
            "weather_condition": "Clear",
            "trip_distance_km": 100.0,
            "load_weight_kg": 2000.0,
            "temperature_c": 22.0,
            "wind_speed_kmh": 10.0,
            "rainfall_mm": 0.0,
        }])
        pred = model.predict(sample)
        self.assertEqual(len(pred), 1)
        self.assertGreater(pred[0], 0.0)

    def test_candidate_metadata_contracts(self):
        """Verify that candidate metadata contains required schema fields."""
        meta_path = CANDIDATES_DIR / "regional_risk_xgboost_metadata.json"
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        self.assertTrue(meta.get("is_candidate"))
        self.assertIn("roc_auc", meta)
        self.assertIn("model_accuracy", meta)
        self.assertIn("xgboost_version", meta)


if __name__ == "__main__":
    unittest.main()
