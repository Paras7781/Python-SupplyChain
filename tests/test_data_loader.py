"""Tests for dataset loading, contracts, and privacy sanitization."""
import unittest
import pandas as pd

from shared.data_loader import (
    load_raw_dataset,
    sanitize_privacy_fields,
    check_dataset_contract,
    DATASET_CONTRACTS,
    SENSITIVE_CUSTOMER_FIELDS,
)
from shared.data_validation import validate_numeric_ranges, get_dataset_provenance


class DataLoaderTests(unittest.TestCase):
    def test_unknown_dataset_key_raises_value_error(self):
        with self.assertRaises(ValueError):
            load_raw_dataset("non_existent_key")

    def test_sanitize_privacy_fields_removes_sensitive_data(self):
        df = pd.DataFrame({
            "Order Id": [1, 2],
            "Customer Password": ["secret1", "secret2"],
            "Customer Email": ["a@example.com", "b@example.com"],
            "Order Region": ["West of USA", "East of USA"],
        })
        sanitized = sanitize_privacy_fields(df)
        self.assertNotIn("Customer Password", sanitized.columns)
        self.assertNotIn("Customer Email", sanitized.columns)
        self.assertIn("Order Id", sanitized.columns)
        self.assertIn("Order Region", sanitized.columns)
        # Original remains untouched
        self.assertIn("Customer Password", df.columns)

    def test_check_dataset_contract_detects_missing(self):
        df = pd.DataFrame({"Order Id": [1, 2]})
        result = check_dataset_contract(df, "supply_chain")
        self.assertFalse(result["valid"])
        self.assertIn("Delivery Status", result["missing_required_columns"])

    def test_provenance_labels_fuel_as_synthetic(self):
        prov = get_dataset_provenance("fuel_weather")
        self.assertTrue(prov["synthetic"])
        self.assertEqual(prov["provenance_type"], "synthetic_demonstration")

    def test_validate_numeric_ranges_detects_negatives_and_infinities(self):
        df = pd.DataFrame({
            "clean": [10.0, 20.0, 30.0],
            "has_neg": [10.0, -5.0, 30.0],
            "has_inf": [10.0, float("inf"), 30.0],
        })
        res = validate_numeric_ranges(df, ["clean", "has_neg", "has_inf"], allow_negative=False)
        self.assertTrue(res["clean"]["valid"])
        self.assertFalse(res["has_neg"]["valid"])
        self.assertFalse(res["has_inf"]["valid"])


if __name__ == "__main__":
    unittest.main()
