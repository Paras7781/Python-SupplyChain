"""
Fuel & Weather - Exploratory Data Analysis (EDA) Module.
Calculates fuel cost trends, vehicle comparisons, and route impact summaries.
"""
from __future__ import annotations

import pandas as pd


def get_fuel_weather_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for fuel and weather observations.

    visibly flags synthetic demonstration provenance.
    """
    if df.empty:
        return {
            "total_records": 0,
            "provenance": "synthetic_demonstration_data",
            "is_synthetic": True,
            "mean_fuel_consumed_liters": 0.0,
            "mean_fuel_cost": 0.0,
            "mean_distance_km": 0.0,
            "vehicle_types": [],
            "road_types": [],
            "weather_conditions": [],
        }

    total_records = len(df)
    mean_fuel = float(df["fuel_consumed_liters"].mean()) if "fuel_consumed_liters" in df.columns else 0.0
    mean_cost = float(df["fuel_cost_per_trip"].mean()) if "fuel_cost_per_trip" in df.columns else 0.0
    mean_dist = float(df["trip_distance_km"].mean()) if "trip_distance_km" in df.columns else 0.0

    # Vehicle breakdown
    vehicles = []
    if "vehicle_type" in df.columns:
        v_group = df.groupby("vehicle_type")
        for v_type, group in v_group:
            v_fuel = float(group["fuel_consumed_liters"].mean()) if "fuel_consumed_liters" in group else 0.0
            v_eff = float(group["fuel_efficiency_km_per_liter"].mean()) if "fuel_efficiency_km_per_liter" in group else 0.0
            v_dist = float(group["trip_distance_km"].mean()) if "trip_distance_km" in group else 0.0
            vehicles.append({
                "vehicle_type": str(v_type),
                "trips": len(group),
                "mean_fuel_consumed_liters": round(v_fuel, 2),
                "mean_efficiency_km_l": round(v_eff, 2),
                "mean_distance_km": round(v_dist, 1),
            })

    # Road types
    road_types = []
    if "road_type" in df.columns:
        r_group = df.groupby("road_type")
        for r_type, group in r_group:
            r_fuel = float(group["fuel_consumed_liters"].mean()) if "fuel_consumed_liters" in group else 0.0
            road_types.append({
                "road_type": str(r_type),
                "trips": len(group),
                "mean_fuel_consumed_liters": round(r_fuel, 2),
            })

    # Weather conditions
    weather_conds = []
    if "weather_condition" in df.columns:
        w_group = df.groupby("weather_condition")
        for w_type, group in w_group:
            w_fuel = float(group["fuel_consumed_liters"].mean()) if "fuel_consumed_liters" in group else 0.0
            weather_conds.append({
                "weather_condition": str(w_type),
                "trips": len(group),
                "mean_fuel_consumed_liters": round(w_fuel, 2),
            })

    return {
        "total_records": total_records,
        "provenance": "synthetic_demonstration_data",
        "is_synthetic": True,
        "mean_fuel_consumed_liters": round(mean_fuel, 2),
        "mean_fuel_cost": round(mean_cost, 2),
        "mean_distance_km": round(mean_dist, 2),
        "vehicle_types": vehicles,
        "road_types": road_types,
        "weather_conditions": weather_conds,
    }


def filter_fuel_trips(
    df: pd.DataFrame,
    vehicle_type: str | None = None,
    fuel_type: str | None = None,
    road_type: str | None = None,
    weather_condition: str | None = None,
    min_distance: float | None = None,
    max_distance: float | None = None,
) -> pd.DataFrame:
    """Filter fuel trip records based on vehicle, environmental, and trip attributes."""
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if vehicle_type and vehicle_type != "All vehicle types" and "vehicle_type" in filtered.columns:
        filtered = filtered[filtered["vehicle_type"].astype(str) == str(vehicle_type)]

    if fuel_type and fuel_type != "All fuel types" and "fuel_type" in filtered.columns:
        filtered = filtered[filtered["fuel_type"].astype(str) == str(fuel_type)]

    if road_type and road_type != "All road types" and "road_type" in filtered.columns:
        filtered = filtered[filtered["road_type"].astype(str) == str(road_type)]

    if weather_condition and weather_condition != "All weather conditions" and "weather_condition" in filtered.columns:
        filtered = filtered[filtered["weather_condition"].astype(str) == str(weather_condition)]

    if "trip_distance_km" in filtered.columns:
        if min_distance is not None:
            filtered = filtered[filtered["trip_distance_km"] >= float(min_distance)]
        if max_distance is not None:
            filtered = filtered[filtered["trip_distance_km"] <= float(max_distance)]

    return filtered
