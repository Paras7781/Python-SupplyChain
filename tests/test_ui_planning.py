import unittest

from ml.fuel_weather.feature_engineering import calculate_fuel_estimate


class FuelEstimateTests(unittest.TestCase):
    def test_calculates_litres_from_distance_and_efficiency(self):
        self.assertAlmostEqual(calculate_fuel_estimate(300, 10), 30)

    def test_distance_increase_scales_estimate(self):
        self.assertEqual(calculate_fuel_estimate(200, 5), calculate_fuel_estimate(100, 5) * 2)

    def test_rejects_zero_or_negative_inputs(self):
        for values in ((0, 5), (-1, 5), (100, 0), (100, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calculate_fuel_estimate(*values)

    def test_rejects_missing_inputs(self):
        for values in ((None, 5), (100, None)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calculate_fuel_estimate(*values)

    def test_rejects_non_finite_inputs(self):
        for values in ((float("inf"), 5), (100, float("nan"))):
            with self.subTest(values=values), self.assertRaises(ValueError):
                calculate_fuel_estimate(*values)


if __name__ == "__main__":
    unittest.main()
