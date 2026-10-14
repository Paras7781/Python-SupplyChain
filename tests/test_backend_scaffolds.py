import unittest

import pandas as pd

from src.fuel_weather.model_training import train_fuel_weather_model
from src.inventory_supplier.model_training import train_inventory_supplier_model
from src.regional_risk.model_training import train_regional_risk_model


class BackendScaffoldTests(unittest.TestCase):
    def test_unimplemented_model_trainers_fail_explicitly(self):
        frame = pd.DataFrame({"source_value": [1, 2, 3]})
        for trainer in (
            train_regional_risk_model,
            train_inventory_supplier_model,
            train_fuel_weather_model,
        ):
            with self.subTest(trainer=trainer.__module__), self.assertRaises(NotImplementedError):
                trainer(frame)


if __name__ == "__main__":
    unittest.main()
