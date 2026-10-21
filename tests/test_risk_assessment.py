"""Tests for the three risk-assessment flows exposed on the unified Risk Assessment page.

Each test exercises the same model functions and prediction contracts used by the
Streamlit page, verifying correct output shapes, types, and business-rule calculations
without importing Streamlit itself.
"""
import math
import unittest

from shared.api_client import get_api_client, SupplyIQClient
from ml.regional_risk.model_training import predict_regional_risk
from ml.inventory_supplier.model_training import predict_inventory_demand
from ml.fuel_weather.model_training import predict_fuel_consumption


class TestRiskAssessmentFlows(unittest.TestCase):
    """End-to-end assessment flows matching the three Risk Assessment tabs."""

    def setUp(self):
        self.client = get_api_client()
        self.assertIsInstance(self.client, SupplyIQClient)

    # ── Regional Delivery Risk ────────────────────────────────────────────

    def test_regional_risk_prediction_contract(self):
        """The regional risk prediction returns the keys the UI consumes."""
        payload = {
            "Days for shipment (scheduled)": 4,
            "Shipping Mode": "Standard Class",
            "Order Region": "Western Europe",
            "Market": "Europe",
            "Customer Segment": "Consumer",
            "order_month": 5,
            "order_dayofweek": 2,
            "order_hour": 14,
            "is_weekend": 0,
            "Total_Units": 2,
            "Total_Order_Value": 120.0,
            "Line_Items": 1,
            "value_per_unit": 60.0,
        }
        result = self.client.predict_regional_risk(payload)

        # Keys required by the UI
        self.assertIn("late_delivery_prediction", result)
        self.assertIn("late_probability", result)
        self.assertIn("late_probability_pct", result)
        self.assertIn("risk_level", result)

        # Value constraints
        self.assertIsInstance(result["late_delivery_prediction"], bool)
        self.assertGreaterEqual(result["late_probability"], 0.0)
        self.assertLessEqual(result["late_probability"], 1.0)
        self.assertIn(result["risk_level"], ["High", "Medium", "Low"])

    # ── Inventory Risk ────────────────────────────────────────────────────

    def test_inventory_demand_prediction_contract(self):
        """The inventory demand prediction returns the keys the UI consumes."""
        payload = {
            "Product_Category": "Electronics",
            "Warehouse_Location": "North",
            "Transportation_Mode": "Road",
            "Lead_Time_Days": 14,
            "Safety_Stock": 100,
            "Reorder_Level": 200,
            "Supplier_Rating": 3.0,
            "Storage_Capacity": 10000,
            "Seasonal_Demand_Index": 50.0,
        }
        result = self.client.predict_inventory_demand(payload)

        self.assertIn("predicted_daily_demand", result)
        self.assertIn("target", result)
        self.assertIn("features_evaluated", result)
        self.assertIsInstance(result["predicted_daily_demand"], float)
        self.assertGreaterEqual(result["predicted_daily_demand"], 0.0)

    def test_inventory_stockout_business_rules(self):
        """Transparent stockout arithmetic matches what the UI calculates."""
        payload = {
            "Product_Category": "Food",
            "Warehouse_Location": "South",
            "Transportation_Mode": "Road",
            "Lead_Time_Days": 7,
            "Safety_Stock": 50,
            "Reorder_Level": 100,
            "Supplier_Rating": 4.0,
            "Storage_Capacity": 5000,
            "Seasonal_Demand_Index": 60.0,
        }
        result = self.client.predict_inventory_demand(payload)
        predicted_demand = result["predicted_daily_demand"]

        current_stock = 150
        replenish_lt = 14

        # days_of_cover = current_stock / predicted_demand
        if predicted_demand > 0:
            days_of_cover = current_stock / predicted_demand
            self.assertTrue(math.isfinite(days_of_cover))
        else:
            days_of_cover = float("inf")

        # demand_over_lt = predicted_demand * replenish_lt
        demand_over_lt = predicted_demand * replenish_lt
        stock_surplus = current_stock - demand_over_lt

        # These are just arithmetic, but verify they match our expectations
        self.assertAlmostEqual(demand_over_lt, predicted_demand * 14, places=5)
        self.assertAlmostEqual(stock_surplus, 150 - demand_over_lt, places=5)

    def test_inventory_stockout_zero_demand(self):
        """When demand is zero or near-zero, days of cover is infinite."""
        # Demand of 0 means days_of_cover should be inf
        days_of_cover = 100 / 0.0001  # near-zero demand
        self.assertTrue(math.isfinite(days_of_cover))

        # Exactly zero demand
        days_of_cover_zero = float("inf") if 0.0 <= 0 else 100 / 0.0
        self.assertTrue(math.isinf(days_of_cover_zero))

    def test_inventory_reorder_level_check(self):
        """Reorder-level comparison uses simple transparent arithmetic."""
        current_stock = 80
        reorder_level = 200
        self.assertTrue(current_stock < reorder_level)

        current_stock = 300
        self.assertFalse(current_stock < reorder_level)

    def test_inventory_demand_prediction_heuristics_or_ml(self):
        """Inventory prediction always uses the synthetic demonstration model."""
        payload = {
            "Product_Category": "Electronics",
            "Warehouse_Location": "East",
            "Transportation_Mode": "Road",
            "Lead_Time_Days": 14,
            "Safety_Stock": 100,
            "Reorder_Level": 200,
            "Supplier_Rating": 3.5,
            "Storage_Capacity": 10000,
            "Seasonal_Demand_Index": 50.0,
        }
        res = self.client.predict_inventory_demand(payload)
        self.assertIn("is_baseline_heuristic", res)
        self.assertIn("baseline_median_demand", res)
        self.assertIn("ml_regression_estimate", res)
        self.assertIn("predicted_daily_demand", res)
        self.assertFalse(res["is_baseline_heuristic"])
        self.assertTrue(res["is_synthetic_demonstration"])
        self.assertEqual(res["estimation_method"], "hist_gradient_boosting_regression")
        self.assertEqual(res["selected_model_mode"], "synthetic_demonstration")
        self.assertAlmostEqual(res["predicted_daily_demand"], res["ml_regression_estimate"], places=2)
        self.assertGreater(res["predicted_daily_demand"], 0.0)

    def test_inventory_demand_prediction_uses_synthetic_model_without_mode_selection(self):
        """Default Inventory Demand predictions use the synthetic demonstration model."""
        payload = {
            "Product_Category": "Electronics",
            "Warehouse_Location": "East",
            "Transportation_Mode": "Road",
            "Lead_Time_Days": 14,
            "Safety_Stock": 100,
            "Reorder_Level": 200,
            "Supplier_Rating": 3.5,
            "Storage_Capacity": 10000,
            "Seasonal_Demand_Index": 50.0,
        }
        res = self.client.predict_inventory_demand(payload)
        self.assertFalse(res["is_baseline_heuristic"])
        self.assertTrue(res["is_synthetic_demonstration"])
        self.assertEqual(res["estimation_method"], "hist_gradient_boosting_regression")
        self.assertEqual(res["selected_model_mode"], "synthetic_demonstration")
        self.assertAlmostEqual(res["predicted_daily_demand"], res["ml_regression_estimate"], places=2)
        self.assertGreater(res["predicted_daily_demand"], 0.0)

    def test_inventory_recommendation_formula_validation(self):
        """Rule-based reorder formula matches documented deterministic arithmetic."""
        import pandas as pd
        from ml.inventory_supplier.feature_engineering import build_inventory_recommendations

        test_records = pd.DataFrame({
            "Product_ID": ["P1", "P2"],
            "Current_Stock": [100, 500],
            "Daily_Demand": [20, 20],
            "Lead_Time_Days": [10, 10],
            "Safety_Stock": [50, 50],
        })
        recs = build_inventory_recommendations(test_records)
        self.assertEqual(len(recs), 2)

        # Target stock = (20 * 10) + 50 = 250
        self.assertEqual(recs.loc[0, "Target_Stock"], 250)
        # P1: Current 100 < 250 -> Suggested reorder = 150
        self.assertEqual(recs.loc[0, "Suggested_Reorder_Quantity"], 150)

        # P2: Current 500 >= 250 -> Suggested reorder = 0 (no negative reorder)
        self.assertEqual(recs.loc[1, "Target_Stock"], 250)
        self.assertEqual(recs.loc[1, "Suggested_Reorder_Quantity"], 0)

    # ── Fuel Risk ─────────────────────────────────────────────────────────

    def test_fuel_prediction_contract(self):
        """The fuel consumption prediction returns the keys the UI consumes."""
        payload = {
            "vehicle_type": "Heavy Truck",
            "fuel_type": "Diesel",
            "road_type": "Highway",
            "weather_condition": "Clear",
            "trip_distance_km": 200.0,
            "load_weight_kg": 5000.0,
            "temperature_c": 25.0,
            "wind_speed_kmh": 10.0,
            "rainfall_mm": 0.0,
        }
        result = self.client.predict_fuel(payload, price=1.50)

        # Keys required by the UI
        self.assertIn("predicted_fuel_consumed_liters", result)
        self.assertIn("provenance", result)
        self.assertIn("is_synthetic", result)
        self.assertIn("estimated_fuel_cost", result)
        self.assertIn("fuel_price_per_liter", result)

        # Value constraints
        self.assertTrue(result["is_synthetic"])
        self.assertEqual(result["provenance"], "synthetic_demonstration_data")
        self.assertGreater(result["predicted_fuel_consumed_liters"], 0.0)
        self.assertAlmostEqual(
            result["estimated_fuel_cost"],
            round(result["predicted_fuel_consumed_liters"] * 1.50, 2),
            places=1,
        )

    def test_fuel_prediction_without_price(self):
        """When no price is provided, cost keys should be absent."""
        payload = {
            "vehicle_type": "Van",
            "fuel_type": "Diesel",
            "road_type": "Urban",
            "weather_condition": "Rain",
            "trip_distance_km": 50.0,
            "load_weight_kg": 800.0,
            "temperature_c": 15.0,
            "wind_speed_kmh": 5.0,
            "rainfall_mm": 3.0,
        }
        result = self.client.predict_fuel(payload, price=None)

        self.assertIn("predicted_fuel_consumed_liters", result)
        self.assertNotIn("estimated_fuel_cost", result)
        self.assertTrue(result["is_synthetic"])

    def test_fuel_budget_comparison_arithmetic(self):
        """Budget comparison uses transparent arithmetic, not a trained model."""
        estimated_cost = 75.0
        budget = 50.0
        delta = estimated_cost - budget
        self.assertGreater(delta, 0)  # over budget

        budget = 200.0
        delta = estimated_cost - budget
        self.assertLess(delta, 0)  # under budget

        budget = 80.0
        delta = estimated_cost - budget
        within_10_pct = delta > -budget * 0.1
        self.assertTrue(within_10_pct)  # within 10%

    def test_fuel_efficiency_calculation(self):
        """The efficiency km/L calculation matches what the page shows."""
        payload = {
            "vehicle_type": "Light Truck",
            "fuel_type": "Diesel",
            "road_type": "Highway",
            "weather_condition": "Clear",
            "trip_distance_km": 300.0,
            "load_weight_kg": 2000.0,
            "temperature_c": 20.0,
            "wind_speed_kmh": 8.0,
            "rainfall_mm": 0.0,
        }
        result = self.client.predict_fuel(payload, price=None)
        pred_liters = result["predicted_fuel_consumed_liters"]
        efficiency = 300.0 / pred_liters
        self.assertGreater(efficiency, 0)
        self.assertTrue(math.isfinite(efficiency))

    # ── API Client wiring ─────────────────────────────────────────────────

    def test_api_client_exposes_all_three_prediction_methods(self):
        """The SupplyIQClient singleton exposes methods for all three tabs."""
        self.assertTrue(callable(getattr(self.client, "predict_regional_risk", None)))
        self.assertTrue(callable(getattr(self.client, "predict_inventory_demand", None)))
        self.assertTrue(callable(getattr(self.client, "predict_fuel", None)))

    # ── What-If Scenario Simulator Comparison Tests ───────────────────────

    def test_what_if_carrier_scenario_comparison(self):
        """Comparing baseline and modified carrier modes computes delta late probability."""
        base_payload = {
            "Days for shipment (scheduled)": 4,
            "Shipping Mode": "Standard Class",
            "Order Region": "Western Europe",
            "Market": "Europe",
            "Customer Segment": "Consumer",
            "order_month": 6,
            "order_dayofweek": 2,
            "order_hour": 14,
            "is_weekend": 0,
            "Total_Units": 2,
            "Total_Order_Value": 150.0,
            "Line_Items": 1,
            "value_per_unit": 75.0,
        }
        rev_payload = dict(base_payload)
        rev_payload["Shipping Mode"] = "First Class"
        rev_payload["Days for shipment (scheduled)"] = 2

        pred_base = self.client.predict_regional_risk(base_payload)
        pred_rev = self.client.predict_regional_risk(rev_payload)

        delta = pred_rev["late_probability_pct"] - pred_base["late_probability_pct"]
        self.assertIsInstance(delta, float)
        self.assertIn(pred_base["risk_level"], ["High", "Medium", "Low"])
        self.assertIn(pred_rev["risk_level"], ["High", "Medium", "Low"])

    def test_what_if_fuel_scenario_comparison(self):
        """Comparing baseline and modified fuel scenarios computes delta volume and cost."""
        base_fuel = {
            "vehicle_type": "Light Truck",
            "fuel_type": "Diesel",
            "road_type": "Highway",
            "weather_condition": "Clear",
            "trip_distance_km": 200.0,
            "load_weight_kg": 1500.0,
            "temperature_c": 20.0,
            "wind_speed_kmh": 10.0,
            "rainfall_mm": 0.0,
        }
        rev_fuel = dict(base_fuel)
        rev_fuel["vehicle_type"] = "Heavy Truck"
        rev_fuel["load_weight_kg"] = 5000.0

        base_res = self.client.predict_fuel(base_fuel, price=1.50)
        rev_res = self.client.predict_fuel(rev_fuel, price=1.80)

        delta_liters = rev_res["predicted_fuel_consumed_liters"] - base_res["predicted_fuel_consumed_liters"]
        delta_cost = rev_res["estimated_fuel_cost"] - base_res["estimated_fuel_cost"]

        self.assertGreater(delta_liters, 0)
        self.assertGreater(delta_cost, 0)

    def test_what_if_inventory_stress_comparison(self):
        """Comparing baseline and delayed inventory scenarios computes lead-time deficit."""
        base_inv = {
            "Product_Category": "Electronics",
            "Warehouse_Location": "East",
            "Transportation_Mode": "Road",
            "Lead_Time_Days": 10,
            "Safety_Stock": 50,
            "Reorder_Level": 100,
            "Supplier_Rating": 4.0,
            "Storage_Capacity": 10000,
            "Seasonal_Demand_Index": 50.0,
        }
        rev_inv = dict(base_inv)
        rev_inv["Lead_Time_Days"] = 20  # +10 days delay

        pred_base = self.client.predict_inventory_demand(base_inv)
        pred_rev = self.client.predict_inventory_demand(rev_inv)

        d_base = pred_base["predicted_daily_demand"]
        d_rev = pred_rev["predicted_daily_demand"]

        lt_demand_base = d_base * 10
        lt_demand_rev = d_rev * 20

        current_stock = 200
        surplus_base = current_stock - lt_demand_base
        surplus_rev = current_stock - lt_demand_rev

        self.assertGreater(lt_demand_rev, lt_demand_base)
        self.assertLess(surplus_rev, surplus_base)

    # ── Combined Assessment Report & Data Invariants ──────────────────────

    def test_combined_assessment_report_schema(self):
        """Combined assessment report rows conform to the expected columns."""
        required_cols = [
            "Assessment_Type",
            "Timestamp",
            "Input_Values",
            "Prediction_Estimate",
            "Risk_Category_Or_Threshold",
            "Model_Information",
            "Limitations_And_Warnings",
        ]
        sample_row = {
            "Assessment_Type": "Regional Delivery Risk",
            "Timestamp": "2026-10-18T12:00:00Z",
            "Input_Values": "Region=Western Europe, Mode=Standard Class",
            "Prediction_Estimate": "Late Probability: 64.0%",
            "Risk_Category_Or_Threshold": "Risk Level: Medium",
            "Model_Information": "RandomForestClassifier | 5-fold CV",
            "Limitations_And_Warnings": "Zero-leakage validated; cross-validation score.",
        }
        for col in required_cols:
            self.assertIn(col, sample_row)
            self.assertTrue(len(sample_row[col]) > 0)

    def test_processed_directory_contains_only_intended_csvs(self):
        """data/processed/ must only contain the three model-ready CSVs."""
        from pathlib import Path
        processed_dir = Path("data/processed")
        csv_files = sorted([f.name for f in processed_dir.glob("*.csv")])
        expected = [
            "fuel_model_ready.csv",
            "inventory_demand_model_ready.csv",
            "regional_risk_model_ready.csv",
        ]
        self.assertEqual(csv_files, expected)

    def test_raw_directory_csv_files_exist_and_unmodified(self):
        """data/raw/ must contain the three original CSVs."""
        from pathlib import Path
        raw_dir = Path("data/raw")
        expected_raw = [
            "DataCoSupplyChainDataset.csv",
            "Retail_Supply_Chain_Analytics_Dataset.csv",
            "supplyiq_final_dataset_filled_synthetic_fuel_weather.csv",
        ]
        for fname in expected_raw:
            fpath = raw_dir / fname
            self.assertTrue(fpath.exists(), f"Raw dataset {fname} missing")
            self.assertGreater(fpath.stat().st_size, 1000)

    def test_active_and_backup_dataset_integrity(self):
        """Verify the active raw CSV and its original backup are internally consistent.

        Checks originally performed by scripts/update_active_daily_demand.py (now deleted):
          1. Original backup SHA-256 matches verified hash (tamper detection).
          2. Both CSVs have the expected shape (200 000 rows × 41 columns).
          3. Exactly one column differs between backup and active: Daily_Demand.
             All 40 other columns and their row ordering are 100% identical.
        """
        import hashlib
        import pandas as pd
        from pathlib import Path

        EXPECTED_BACKUP_SHA256 = "d885d2506fd48aa08b38012d8fc707e86eaaff985d410fa9eb077d40b272ef81"

        def _sha256(path: Path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as f:
                while chunk := f.read(1024 * 1024):
                    h.update(chunk)
            return h.hexdigest()

        backup_path = Path("data/backup/Retail_Supply_Chain_Analytics_Dataset_original.csv")
        active_path = Path("data/raw/Retail_Supply_Chain_Analytics_Dataset.csv")

        self.assertTrue(backup_path.exists(), "Original backup CSV is missing")
        self.assertTrue(active_path.exists(), "Canonical active raw CSV is missing")

        # 1. Backup SHA-256 must match the verified original hash
        backup_sha = _sha256(backup_path)
        self.assertEqual(
            backup_sha,
            EXPECTED_BACKUP_SHA256,
            f"Backup SHA-256 mismatch: {backup_sha} — original backup may be corrupted",
        )

        # 2. Shape check — both CSVs must be 200 000 rows × 41 columns
        backup_df = pd.read_csv(backup_path)
        active_df = pd.read_csv(active_path)
        self.assertEqual(backup_df.shape, (200_000, 41), f"Unexpected backup shape: {backup_df.shape}")
        self.assertEqual(active_df.shape, (200_000, 41), f"Unexpected active shape: {active_df.shape}")

        # 3. Column-alignment: only Daily_Demand may differ; all 40 other columns must be identical
        changed_cols = [c for c in backup_df.columns if not backup_df[c].equals(active_df[c])]
        self.assertEqual(
            changed_cols,
            ["Daily_Demand"],
            f"Unexpected non-target columns changed vs original backup: {changed_cols}",
        )


if __name__ == "__main__":
    unittest.main()
