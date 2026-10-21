"""
Tests for SHAP Explainability in SupplyIQ.
Verifies individual and global SHAP attributions, feature name humanization,
and non-destructive behavior (explanations do not alter predictions).
"""
import unittest
import numpy as np
import pandas as pd

from ml.regional_risk.model_training import load_regional_risk_model, predict_regional_risk
from ml.fuel_weather.model_training import load_fuel_model, predict_fuel_consumption
from ml.explainability import (
    explain_regional_risk_prediction,
    get_regional_risk_global_importance,
    explain_fuel_prediction,
    humanize_feature_name,
)


class TestExplainability(unittest.TestCase):
    def test_humanize_feature_name(self):
        """Verify that pipeline feature names are converted into clean business terminology."""
        self.assertEqual(humanize_feature_name("num__Days for shipment (scheduled)"), "Scheduled Window (Days)")
        self.assertEqual(humanize_feature_name("cat__Shipping Mode_Standard Class"), "Shipping Mode: Standard Class")
        self.assertEqual(humanize_feature_name("cat__Order Region_Western Europe"), "Destination Region: Western Europe")
        self.assertEqual(humanize_feature_name("num__trip_distance_km"), "Trip Distance (km)")
        self.assertEqual(humanize_feature_name("cat__vehicle_type_Heavy Truck"), "Vehicle Type: Heavy Truck")

    def test_regional_risk_individual_shap(self):
        """Verify that Regional Risk SHAP generates valid attribution and does not alter predictions."""
        pipeline = load_regional_risk_model()
        sample = {
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
        }

        # Baseline prediction
        pred_before = predict_regional_risk(sample)

        # Run SHAP explanation
        shap_res = explain_regional_risk_prediction(pipeline, sample)
        self.assertTrue(shap_res["success"])
        self.assertIn("base_value", shap_res)
        self.assertGreater(len(shap_res["narratives"]), 0)
        self.assertIsNotNone(shap_res["fig"])
        self.assertIn("disclaimer", shap_res)

        # Verify prediction is unaltered
        pred_after = predict_regional_risk(sample)
        self.assertEqual(pred_before["late_probability"], pred_after["late_probability"])
        self.assertEqual(pred_before["late_delivery_prediction"], pred_after["late_delivery_prediction"])

    def test_fuel_individual_shap(self):
        """Verify that Fuel SHAP generates valid attribution and does not alter predictions."""
        pipeline = load_fuel_model()
        sample = {
            "vehicle_type": "Light Truck",
            "fuel_type": "Diesel",
            "road_type": "Highway",
            "weather_condition": "Clear",
            "trip_distance_km": 100.0,
            "load_weight_kg": 2000.0,
            "temperature_c": 22.0,
            "wind_speed_kmh": 10.0,
            "rainfall_mm": 0.0,
        }

        # Baseline prediction
        pred_before = predict_fuel_consumption(sample)

        # Run SHAP explanation
        shap_res = explain_fuel_prediction(pipeline, sample)
        self.assertTrue(shap_res["success"])
        self.assertIn("base_value", shap_res)
        self.assertGreater(len(shap_res["narratives"]), 0)
        self.assertIsNotNone(shap_res["fig"])
        self.assertIn("SYNTHETIC DATA NOTICE", shap_res["disclaimer"])

        # Verify prediction is unaltered
        pred_after = predict_fuel_consumption(sample)
        self.assertEqual(pred_before["predicted_fuel_consumed_liters"], pred_after["predicted_fuel_consumed_liters"])

    def test_global_importance_calculation(self):
        """Verify that global SHAP calculation produces sorted importance rankings."""
        pipeline = load_regional_risk_model()
        sample_df = pd.DataFrame([
            {
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
            }
        ] * 5)
        res = get_regional_risk_global_importance(pipeline, sample_df, top_n=5)
        self.assertTrue(res["success"])
        self.assertIn("df_importance", res)
        self.assertLessEqual(len(res["df_importance"]), 5)


if __name__ == "__main__":
    unittest.main()
