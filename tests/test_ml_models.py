"""Tests for machine learning models, inference functions, and metadata integrity."""
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import patch
from sklearn.model_selection import KFold, cross_validate

from ml.inventory_supplier import model_training as inventory_model
from ml.fuel_weather import model_training as fuel_model

from ml.regional_risk.model_training import (
    is_regional_risk_model_available,
    get_regional_risk_metadata,
    predict_regional_risk,
)
from ml.inventory_supplier.model_training import (
    is_inventory_model_available,
    get_inventory_metadata,
    predict_inventory_demand,
)
from ml.fuel_weather.model_training import (
    is_fuel_model_available,
    get_fuel_metadata,
    predict_fuel_consumption,
)


class MLModelsTests(unittest.TestCase):
    @staticmethod
    def _regional_training_frame(row_count=40):
        return pd.DataFrame({
            "Order Id": [f"order-{index}" for index in range(row_count)],
            "Customer Id": [f"customer-{index % 10}" for index in range(row_count)],
            "order date (DateOrders)": pd.date_range("2022-01-01", periods=row_count, freq="D"),
            "Late_delivery_risk": [index % 2 for index in range(row_count)],
            "Days for shipment (scheduled)": [2 + index % 4 for index in range(row_count)],
            "Shipping Mode": ["Standard Class", "First Class"] * (row_count // 2),
            "Order Region": ["Central", "West"] * (row_count // 2),
            "Market": ["A", "B"] * (row_count // 2),
            "Customer Segment": ["Consumer", "Corporate"] * (row_count // 2),
            "Total_Units": [1 + index % 5 for index in range(row_count)],
            "Total_Order_Value": [20.0 + index * 2 for index in range(row_count)],
            "Line_Items": [1 + index % 3 for index in range(row_count)],
        })

    @staticmethod
    def _inventory_training_frame(row_count=50):
        return pd.DataFrame({
            "Product_ID": [f"product-{index}" for index in range(row_count)],
            "Supplier_ID": [f"supplier-{index // 5}" for index in range(row_count)],
            "Warehouse_ID": [f"warehouse-{index % 4}" for index in range(row_count)],
            "Product_Category": ["A", "B"] * (row_count // 2),
            "Warehouse_Location": ["North", "South"] * (row_count // 2),
            "Transportation_Mode": ["Road", "Rail"] * (row_count // 2),
            "Lead_Time_Days": [1 + index % 12 for index in range(row_count)],
            "Safety_Stock": [10 + index % 7 for index in range(row_count)],
            "Reorder_Level": [20 + index % 11 for index in range(row_count)],
            "Supplier_Rating": [1 + index % 5 for index in range(row_count)],
            "Storage_Capacity": [100 + index * 4 for index in range(row_count)],
            "Seasonal_Demand_Index": [0.5 + (index % 9) / 10 for index in range(row_count)],
            "Daily_Demand": [15 + (index % 13) * 4 + index / 10 for index in range(row_count)],
        })

    @staticmethod
    def _fuel_training_frame(row_count=50):
        return pd.DataFrame({
            "timestamp": pd.date_range("2021-01-01", periods=row_count, freq="D"),
            "vehicle_id": [f"vehicle-{index % 5}" for index in range(row_count)],
            "vehicle_type": [["Truck", "Van"][index % 2] for index in range(row_count)],
            "fuel_type": [["Diesel", "Gasoline"][index % 2] for index in range(row_count)],
            "road_type": [["Highway", "Urban"][index % 2] for index in range(row_count)],
            "weather_condition": [["Clear", "Rain"][index % 2] for index in range(row_count)],
            "trip_distance_km": [20 + index * 3 for index in range(row_count)],
            "load_weight_kg": [100 + index * 10 for index in range(row_count)],
            "temperature_c": [10 + index % 20 for index in range(row_count)],
            "wind_speed_kmh": [2 + index % 8 for index in range(row_count)],
            "rainfall_mm": [index % 4 for index in range(row_count)],
            "fuel_consumed_liters": [
                3 + (20 + index * 3) * 0.1 + (100 + index * 10) * 0.002
                for index in range(row_count)
            ],
            "added_data_status": ["synthetic_demo_values"] * row_count,
        })

    def _assert_cv_artifact(self, module, trainer, frame, expected_method, metric_names):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_dir = Path(temporary_directory)
            with (
                patch.object(module, "MODELS_DIR", artifact_dir),
                patch.object(module, "MODEL_PATH", artifact_dir / "model.joblib"),
                patch.object(module, "METADATA_PATH", artifact_dir / "metadata.json"),
            ):
                metrics = trainer(frame, sample_size=None)
                self.assertEqual(metrics["validation_method"], expected_method)
                self.assertEqual(metrics["validation_score_type"], "cross_validation_estimate")
                self.assertGreaterEqual(metrics["fold_count"], 2)
                self.assertTrue((artifact_dir / "model.joblib").is_file())
                self.assertTrue((artifact_dir / "metadata.json").is_file())
                with (artifact_dir / "metadata.json").open(encoding="utf-8") as metadata_file:
                    saved_metadata = json.load(metadata_file)
                self.assertEqual(saved_metadata["validation_method"], expected_method)
                self.assertEqual(saved_metadata["fold_count"], metrics["fold_count"])
                if module.__name__.endswith("regional_risk.model_training"):
                    self.assertTrue(metrics["splitter_configuration"]["group_purging"])
                    self.assertEqual(set(metrics["fold_group_overlap_counts"]), {0})
                    self.assertTrue(all(
                        set(fold_counts) == {"0", "1"}
                        for fold_counts in metrics["fold_validation_class_counts"]
                    ))
                if module is inventory_model:
                    self.assertEqual(metrics["splitter_configuration"]["group_column"], "Supplier_ID")
                    self.assertEqual(set(metrics["fold_group_overlap_counts"]), {0})
                    self.assertTrue(
                        all(
                            overlap_count > 0
                            for overlap_count in metrics["grouping_assessment"][
                                "other_entity_fold_overlap_counts"
                            ]["Warehouse_ID"]
                        )
                    )
                if module is fuel_model:
                    self.assertEqual(metrics["splitter_configuration"]["timestamp_column"], "timestamp")
                for metric in metric_names:
                    summary = metrics["cv_metrics"][metric]
                    self.assertEqual(len(summary["fold_scores"]), metrics["fold_count"])
                    self.assertTrue(np.isfinite(summary["fold_scores"]).all())
                    legacy_metric = {
                        "accuracy": "model_accuracy",
                        "f1": "f1_score",
                        "r2": "r2_score",
                    }.get(metric, metric)
                    self.assertEqual(metrics[legacy_metric], summary["mean"])
                    self.assertAlmostEqual(
                        summary["mean"],
                        np.mean(summary["fold_scores"]),
                        delta=0.0001,
                    )
                    self.assertAlmostEqual(
                        summary["std"],
                        np.std(summary["fold_scores"], ddof=1),
                        delta=0.0001,
                    )
                    self.assertTrue(np.isfinite(summary["std"]))

                if hasattr(module, "_load_regional_risk_model_cached"):
                    module._load_regional_risk_model_cached.cache_clear()
                if hasattr(module, "_load_inventory_pipeline_cached"):
                    module._load_inventory_pipeline_cached.cache_clear()
                if hasattr(module, "_load_fuel_pipeline_cached"):
                    module._load_fuel_pipeline_cached.cache_clear()
                if module is inventory_model:
                    loaded_pipeline = module.load_inventory_model()
                elif module is fuel_model:
                    loaded_pipeline = module.load_fuel_model()
                else:
                    loaded_pipeline = module.load_regional_risk_model()
                self.assertTrue(hasattr(loaded_pipeline, "predict"))
                if module is fuel_model:
                    self.assertTrue(metrics["is_synthetic"])
                    self.assertEqual(metrics["provenance"], "synthetic_demonstration_data")
                return metrics

    def test_regional_risk_uses_customer_purged_chronological_cv(self):
        from ml.regional_risk import model_training as regional_model

        self._assert_cv_artifact(
            regional_model,
            regional_model.train_regional_risk_model,
            self._regional_training_frame(),
            "TimeSeriesSplit with customer-group purging",
            ("accuracy", "precision", "recall", "f1", "roc_auc"),
        )

    def test_inventory_uses_supplier_grouped_cv_and_reports_all_fold_metrics(self):
        self._assert_cv_artifact(
            inventory_model,
            inventory_model.train_inventory_supplier_model,
            self._inventory_training_frame(),
            "GroupKFold",
            ("mae", "rmse", "r2"),
        )

    def test_fuel_uses_time_series_cv_and_preserves_synthetic_provenance(self):
        self._assert_cv_artifact(
            fuel_model,
            fuel_model.train_fuel_weather_model,
            self._fuel_training_frame(),
            "TimeSeriesSplit",
            ("mae", "rmse", "r2"),
        )

    def test_preprocessing_is_fitted_inside_each_validation_fold(self):
        frame = self._inventory_training_frame(row_count=8)
        frame["Lead_Time_Days"] = [0, 2, 4, 6, 8, 100, 200, 1000]
        X = frame[inventory_model.ALL_FEATURES]
        y = frame["Daily_Demand"]
        results = cross_validate(
            inventory_model.build_pipeline(),
            X,
            y,
            cv=KFold(n_splits=2, shuffle=False),
            scoring="neg_mean_absolute_error",
            return_estimator=True,
            error_score="raise",
        )
        feature_index = inventory_model.NUMERIC_FEATURES.index("Lead_Time_Days")
        fold_medians = [
            estimator.named_steps["preprocessor"]
            .named_transformers_["num"]
            .named_steps["imputer"]
            .statistics_[feature_index]
            for estimator in results["estimator"]
        ]
        self.assertEqual(fold_medians, [150.0, 3.0])
        self.assertNotEqual(fold_medians[0], float(X["Lead_Time_Days"].median()))
        self.assertNotEqual(fold_medians[1], float(X["Lead_Time_Days"].median()))

    def test_training_rejects_unsuitable_class_and_regression_fold_counts(self):
        regional = self._regional_training_frame(row_count=8)
        regional.loc[0, "Late_delivery_risk"] = 2
        with self.assertRaisesRegex(ValueError, "only binary labels 0 and 1"):
            from ml.regional_risk.model_training import train_regional_risk_model
            train_regional_risk_model(regional, sample_size=None)

        regional = self._regional_training_frame(row_count=8)
        regional["Late_delivery_risk"] = 0
        with self.assertRaisesRegex(ValueError, "at least two samples"):
            train_regional_risk_model(regional, sample_size=None)

        with self.assertRaisesRegex(ValueError, "four valid rows"):
            inventory_model.train_inventory_supplier_model(
                pd.DataFrame({"Daily_Demand": [1, 2, 3]}),
                sample_size=None,
            )

        fuel = self._fuel_training_frame(row_count=3)
        with self.assertRaisesRegex(ValueError, "chronological validation"):
            fuel_model.train_fuel_weather_model(fuel, sample_size=None)

    def test_regression_target_exclusions_are_recorded(self):
        frame = self._inventory_training_frame()
        frame["Daily_Demand"] = frame["Daily_Demand"].astype(object)
        frame.loc[0, "Daily_Demand"] = "invalid"
        frame.loc[1, "Daily_Demand"] = -1
        metrics = self._assert_cv_artifact(
            inventory_model,
            inventory_model.train_inventory_supplier_model,
            frame,
            "GroupKFold",
            ("mae", "rmse", "r2"),
        )
        self.assertEqual(metrics["excluded_target_rows"]["non_numeric"], 1)
        self.assertEqual(metrics["excluded_target_rows"]["below_zero"], 1)

    def test_regional_risk_model_availability_and_inference(self):
        self.assertTrue(is_regional_risk_model_available())
        metadata = get_regional_risk_metadata()
        self.assertIn("model_accuracy", metadata)
        self.assertIn("roc_auc", metadata)
        self.assertGreater(metadata["model_accuracy"], metadata["baseline_accuracy"])

        # Test inference
        sample_input = {
            "Days for shipment (scheduled)": 4,
            "Shipping Mode": "Standard Class",
            "Order Region": "Central America",
            "Market": "LATAM",
            "Customer Segment": "Consumer",
            "order_month": 5,
            "order_dayofweek": 2,
            "order_hour": 14,
            "is_weekend": 0,
            "Total_Units": 2,
            "Total_Order_Value": 100.0,
            "Line_Items": 1,
            "value_per_unit": 50.0,
        }
        pred = predict_regional_risk(sample_input)
        self.assertIn("late_delivery_prediction", pred)
        self.assertIn("late_probability", pred)
        self.assertIn(pred["risk_level"], ["High", "Medium", "Low"])
        self.assertGreaterEqual(pred["late_probability"], 0.0)
        self.assertLessEqual(pred["late_probability"], 1.0)

    def test_inventory_demand_model_availability_and_inference(self):
        self.assertTrue(is_inventory_model_available())
        metadata = get_inventory_metadata()
        self.assertTrue(metadata["is_synthetic_demonstration"])
        self.assertEqual(metadata["data_provenance"], "synthetic_demonstration_data")
        self.assertIn("mae", metadata)
        self.assertAlmostEqual(metadata["r2_score"], 0.9290, places=4)
        self.assertAlmostEqual(metadata["mae"], 24.26, places=2)
        self.assertAlmostEqual(metadata["rmse"], 30.47, places=2)
        for metric in ("r2", "mae", "rmse"):
            metadata_key = "r2_score" if metric == "r2" else metric
            self.assertEqual(metadata[metadata_key], metadata["cv_metrics"][metric]["mean"])

        sample_input = {
            "Product_Category": "Electronics",
            "Warehouse_Location": "North",
            "Transportation_Mode": "Road",
            "Lead_Time_Days": 7,
            "Safety_Stock": 20,
            "Reorder_Level": 50,
            "Supplier_Rating": 4.5,
            "Storage_Capacity": 1000,
            "Seasonal_Demand_Index": 1.2,
        }
        pred = predict_inventory_demand(sample_input)
        self.assertIn("predicted_daily_demand", pred)
        self.assertGreaterEqual(pred["predicted_daily_demand"], 0.0)
        self.assertTrue(pred["is_synthetic_demonstration"])
        self.assertFalse(pred["is_baseline_heuristic"])
        self.assertEqual(pred["selected_model_mode"], "synthetic_demonstration")
        self.assertEqual(pred["data_provenance"], "synthetic_demonstration_data")

    def test_inventory_prediction_rejects_invalid_or_multiple_rows(self):
        with self.assertRaisesRegex(ValueError, "exactly one row"):
            predict_inventory_demand(pd.DataFrame({"Lead_Time_Days": [2, 3]}))
        with self.assertRaisesRegex(ValueError, "Supplier_Rating"):
            predict_inventory_demand({"Supplier_Rating": 6})
        with self.assertRaisesRegex(ValueError, "numeric"):
            predict_inventory_demand({"Lead_Time_Days": "many"})

    def test_inventory_model_pipeline_is_cached(self):
        inventory_model._load_inventory_pipeline_cached.cache_clear()
        original_load = inventory_model.joblib.load
        with patch.object(inventory_model.joblib, "load", wraps=original_load) as load:
            payload = {
                "Product_Category": "Electronics",
                "Lead_Time_Days": 7,
                "Supplier_Rating": 4.0,
            }
            predict_inventory_demand(payload)
            predict_inventory_demand(payload)
            self.assertEqual(load.call_count, 1)
        inventory_model._load_inventory_pipeline_cached.cache_clear()

    def test_fuel_consumption_model_availability_and_inference(self):
        self.assertTrue(is_fuel_model_available())
        metadata = get_fuel_metadata()
        self.assertTrue(metadata.get("is_synthetic", False))
        self.assertGreater(metadata.get("r2_score", 0), 0.8)

        sample_input = {
            "vehicle_type": "Light Truck",
            "fuel_type": "Diesel",
            "road_type": "Highway",
            "weather_condition": "Clear",
            "trip_distance_km": 150.0,
            "load_weight_kg": 1500.0,
            "temperature_c": 22.0,
            "wind_speed_kmh": 15.0,
            "rainfall_mm": 0.0,
        }
        pred = predict_fuel_consumption(sample_input, fuel_price_per_liter=1.65)
        self.assertIn("predicted_fuel_consumed_liters", pred)
        self.assertTrue(pred["is_synthetic"])
        self.assertIn("estimated_fuel_cost", pred)
        self.assertAlmostEqual(
            pred["estimated_fuel_cost"],
            round(pred["predicted_fuel_consumed_liters"] * 1.65, 2),
            places=1,
        )

    def test_fuel_prediction_rejects_invalid_distance_and_price(self):
        with self.assertRaisesRegex(ValueError, "trip_distance_km"):
            predict_fuel_consumption({"trip_distance_km": 0})
        with self.assertRaisesRegex(ValueError, "Fuel price"):
            predict_fuel_consumption(
                {"trip_distance_km": 100, "load_weight_kg": 0},
                fuel_price_per_liter=float("inf"),
            )

    def test_fuel_model_pipeline_is_cached(self):
        fuel_model._load_fuel_pipeline_cached.cache_clear()
        original_load = fuel_model.joblib.load
        with patch.object(fuel_model.joblib, "load", wraps=original_load) as load:
            payload = {"trip_distance_km": 100, "vehicle_type": "Truck"}
            predict_fuel_consumption(payload)
            predict_fuel_consumption(payload)
            self.assertEqual(load.call_count, 1)
        fuel_model._load_fuel_pipeline_cached.cache_clear()


if __name__ == "__main__":
    unittest.main()
