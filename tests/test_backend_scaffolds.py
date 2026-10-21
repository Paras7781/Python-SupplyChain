"""Tests for ML training input validation on malformed source frames."""
import unittest
import pandas as pd

from ml.fuel_weather.model_training import train_fuel_weather_model
from ml.inventory_supplier.model_training import train_inventory_supplier_model
from ml.regional_risk.model_training import train_regional_risk_model


class BackendScaffoldTests(unittest.TestCase):
    def test_trainers_reject_empty_or_target_missing_frames(self):
        malformed_frame = pd.DataFrame({"source_value": [1, 2, 3]})
        for trainer in (
            train_regional_risk_model,
            train_inventory_supplier_model,
            train_fuel_weather_model,
        ):
            with self.subTest(trainer=trainer.__module__), self.assertRaises(ValueError):
                trainer(malformed_frame)

    def test_trainers_reject_empty_frames(self):
        empty_frame = pd.DataFrame()
        for trainer in (
            train_regional_risk_model,
            train_inventory_supplier_model,
            train_fuel_weather_model,
        ):
            with self.subTest(trainer=trainer.__module__), self.assertRaises(ValueError):
                trainer(empty_frame)

    def test_regression_trainers_reject_fewer_than_two_valid_targets(self):
        for trainer, target in (
            (train_inventory_supplier_model, "Daily_Demand"),
            (train_fuel_weather_model, "fuel_consumed_liters"),
        ):
            frame = pd.DataFrame({target: ["not-a-number", None]})
            with self.subTest(trainer=trainer.__module__), self.assertRaisesRegex(ValueError, "two valid target rows"):
                trainer(frame)


if __name__ == "__main__":
    unittest.main()
