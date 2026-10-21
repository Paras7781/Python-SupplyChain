"""Tests for data cleaning, EDA summaries, and feature engineering across the 3 use cases."""
import unittest
import pandas as pd
import numpy as np

# Regional Risk
from analytics.regional_risk.cleaning import clean_regional_risk_data, aggregate_orders_level
from analytics.regional_risk.eda import get_regional_summary, filter_orders
from ml.regional_risk.feature_engineering import create_regional_risk_features, LEAKAGE_COLUMNS

# Inventory / Supplier
from analytics.inventory_supplier.cleaning import clean_inventory_supplier_data
from analytics.inventory_supplier.eda import get_inventory_supplier_summary, classify_stock_status
from ml.inventory_supplier.feature_engineering import (
    build_inventory_recommendations,
    build_inventory_coverage,
    create_inventory_supplier_features,
)

# Fuel / Weather
from analytics.fuel_weather.cleaning import clean_fuel_weather_data
from analytics.fuel_weather.eda import get_fuel_weather_summary, filter_fuel_trips
from ml.fuel_weather.feature_engineering import (
    calculate_fuel_estimate,
    calculate_fuel_cost_estimate,
    create_fuel_weather_features,
)


class RegionalRiskPipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw_orders = pd.DataFrame({
            "Order Id": [101, 101, 102],
            "Order Region": ["Central America", "Central America", "Western Europe"],
            "Order State": ["State A", "State A", "State B"],
            "Order City": ["City X", "City X", "City Y"],
            "Late_delivery_risk": [1, 1, 0],
            "Delivery Status": ["Late delivery", "Late delivery", "Shipping on time"],
            "Days for shipping (real)": [5, 5, 2],
            "Days for shipment (scheduled)": [4, 4, 3],
            "Shipping Mode": ["Standard Class", "Standard Class", "First Class"],
            "Order Item Id": [1, 2, 3],
            "Order Item Quantity": [2, 3, 1],
            "Order Item Total": [50.0, 75.0, 30.0],
            "Customer Password": ["pw1", "pw1", "pw2"],
            "Customer Email": ["user1@mail.com", "user1@mail.com", "user2@mail.com"],
            "Product Description": [None, None, None],
        })

    def test_cleaning_drops_pii_and_empty_description(self):
        cleaned = clean_regional_risk_data(self.raw_orders, remove_pii=True)
        self.assertNotIn("Customer Password", cleaned.columns)
        self.assertNotIn("Customer Email", cleaned.columns)
        self.assertNotIn("Product Description", cleaned.columns)
        self.assertEqual(len(cleaned), 3)

    def test_order_aggregation_groups_correctly(self):
        aggregated = aggregate_orders_level(self.raw_orders)
        self.assertEqual(len(aggregated), 2)  # 2 distinct orders
        order_101 = aggregated[aggregated["Order Id"] == 101].iloc[0]
        self.assertEqual(order_101["Line_Items"], 2)
        self.assertEqual(order_101["Total_Units"], 5)
        self.assertEqual(order_101["Total_Order_Value"], 125.0)
        self.assertEqual(order_101["Shipping_Difference_Days"], 1)

    def test_eda_summary_metrics(self):
        summary = get_regional_summary(self.raw_orders)
        self.assertEqual(summary["total_orders"], 2)
        self.assertAlmostEqual(summary["late_delivery_rate"], 0.5)

    def test_feature_engineering_strictly_drops_leakage(self):
        features = create_regional_risk_features(self.raw_orders, drop_leakage=True)
        for leak in LEAKAGE_COLUMNS:
            self.assertNotIn(leak, features.columns)
        self.assertIn("Days for shipment (scheduled)", features.columns)
        self.assertIn("order_month", features.columns)


class InventorySupplierPipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw_retail = pd.DataFrame({
            "Product_ID": ["P1", "P2", "P3"],
            "Product_Category": ["Electronics", "Electronics", "Clothing"],
            "Current_Stock": [10, 0, 50],
            "Reorder_Level": [20, 15, 25],
            "Safety_Stock": [5, 5, 10],
            "Daily_Demand": [2, 3, 5],
            "Lead_Time_Days": [7, 7, 3],
            "Supplier_ID": ["S1", "S1", "S2"],
            "Supplier_Rating": [4.5, 4.5, 2.8],
            "On_Time_Delivery_Rate": [0.95, 0.95, 0.80],
        })

    def test_stock_status_classification(self):
        self.assertEqual(classify_stock_status(0, 10), "Critical")
        self.assertEqual(classify_stock_status(5, 10), "Reorder required")
        self.assertEqual(classify_stock_status(15, 10), "Healthy")
        self.assertEqual(classify_stock_status(None, 10), "Unknown")

    def test_reorder_recommendation_math(self):
        recs = build_inventory_recommendations(self.raw_retail)
        # P1: Target stock = 2*7 + 5 = 19. Current = 10. Suggested = 9.
        p1 = recs[recs["Product_ID"] == "P1"].iloc[0]
        self.assertEqual(p1["Target_Stock"], 19)
        self.assertEqual(p1["Suggested_Reorder_Quantity"], 9)

        # P3: Target stock = 5*3 + 10 = 25. Current = 50. Suggested = 0 (clip lower=0).
        p3 = recs[recs["Product_ID"] == "P3"].iloc[0]
        self.assertEqual(p3["Target_Stock"], 25)
        self.assertEqual(p3["Suggested_Reorder_Quantity"], 0)

    def test_inventory_coverage(self):
        cov = build_inventory_coverage(self.raw_retail)
        self.assertEqual(cov.iloc[0], 5.0)  # 10 / 2
        self.assertEqual(cov.iloc[1], 0.0)  # 0 / 3
        self.assertEqual(cov.iloc[2], 10.0) # 50 / 5


class FuelWeatherPipelineTests(unittest.TestCase):
    def setUp(self):
        self.raw_fuel = pd.DataFrame({
            "trip_id": ["T1", "T2"],
            "vehicle_type": ["Light Truck", "Heavy Truck"],
            "fuel_type": ["Diesel", "Diesel"],
            "road_type": ["Highway", "Urban"],
            "weather_condition": ["Clear", "Rain"],
            "trip_distance_km": [100.0, 200.0],
            "fuel_efficiency_km_per_liter": [10.0, 4.0],
            "fuel_consumed_liters": [10.0, 50.0],
            "fuel_price_per_liter": [1.5, 1.5],
            "fuel_cost_per_trip": [15.0, 75.0],
            "added_data_status": ["synthetic_demo_values", "synthetic_demo_values"],
        })

    def test_fuel_summary_provenance(self):
        summary = get_fuel_weather_summary(self.raw_fuel)
        self.assertTrue(summary["is_synthetic"])
        self.assertEqual(summary["provenance"], "synthetic_demonstration_data")
        self.assertEqual(summary["total_records"], 2)

    def test_fuel_cost_estimate(self):
        cost = calculate_fuel_cost_estimate(20.0, 1.5)
        self.assertAlmostEqual(cost, 30.0)
        with self.assertRaises(ValueError):
            calculate_fuel_cost_estimate(-5.0, 1.5)


if __name__ == "__main__":
    unittest.main()
