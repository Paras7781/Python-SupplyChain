import unittest

import pandas as pd

from shared.data_validation import validate_dataframe


class DataValidationTests(unittest.TestCase):
    def test_empty_dataframe_is_reported_as_empty(self):
        result = validate_dataframe(pd.DataFrame(columns=["a", "b"]))
        self.assertEqual(result["status"], "empty")
        self.assertEqual(result["rows"], 0)
        self.assertEqual(result["columns"], 2)

    def test_clean_dataframe_is_valid(self):
        result = validate_dataframe(pd.DataFrame({"id": [1, 2], "value": ["a", "b"]}))
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["missing_values"], 0)
        self.assertEqual(result["duplicate_rows"], 0)

    def test_missing_values_trigger_warning(self):
        result = validate_dataframe(pd.DataFrame({"a": [1, None], "b": ["x", "y"]}))
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["missing_values"], 1)
        self.assertEqual(result["columns_with_missing_values"], 1)

    def test_exact_duplicate_rows_trigger_warning(self):
        result = validate_dataframe(pd.DataFrame({"a": [1, 1], "b": ["x", "x"]}))
        self.assertEqual(result["status"], "warning")
        self.assertEqual(result["duplicate_rows"], 1)

    def test_validation_does_not_mutate_input(self):
        frame = pd.DataFrame({"a": [1, None]})
        original = frame.copy(deep=True)
        validate_dataframe(frame)
        pd.testing.assert_frame_equal(frame, original)


if __name__ == "__main__":
    unittest.main()
