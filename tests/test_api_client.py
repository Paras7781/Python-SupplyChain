"""Tests for the in-process Streamlit service client."""
import unittest
from unittest.mock import patch

from shared.api_client import SupplyIQClient


class APIClientTests(unittest.TestCase):
    def setUp(self):
        self.client = SupplyIQClient()

    @patch("shared.api_client.predict_regional_risk", return_value={"risk_level": "Low"})
    def test_regional_prediction_uses_local_model(self, predict):
        result = self.client.predict_regional_risk({"Shipping Mode": "Standard Class"})

        self.assertEqual(result, {"risk_level": "Low"})
        predict.assert_called_once()

    @patch(
        "shared.api_client.predict_inventory_demand",
        return_value={"is_synthetic_demonstration": True},
    )
    def test_inventory_prediction_uses_synthetic_model(self, predict):
        payload = {"Product_Category": "Electronics"}

        result = self.client.predict_inventory_demand(payload)

        self.assertEqual(result, {"is_synthetic_demonstration": True})
        predict.assert_called_once_with(payload)

    @patch(
        "shared.api_client.predict_fuel_consumption",
        return_value={"predicted_fuel_consumed_liters": 42.0},
    )
    def test_fuel_prediction_uses_local_model(self, predict):
        result = self.client.predict_fuel({"trip_distance_km": 100}, price=1.5)

        self.assertEqual(result, {"predicted_fuel_consumed_liters": 42.0})
        predict.assert_called_once_with(
            {"trip_distance_km": 100}, fuel_price_per_liter=1.5
        )


if __name__ == "__main__":
    unittest.main()
