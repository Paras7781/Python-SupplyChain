"""
Fuel & Weather - Data Cleaning Module.
Standardizes data types, validates physical ranges, and preserves synthetic provenance labels.
"""
from __future__ import annotations

import pandas as pd


def clean_fuel_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the fuel and weather dataset.

    Performs the following cleaning operations:
    1. Removes exact duplicate rows.
    2. Parses trip timestamps to standard datetime format.
    3. Strips whitespace from identifiers, vehicle types, and weather categories.
    4. Converts trip distance, load weight, and fuel volumes to numeric types.
    5. Validates physical non-negativity (masks invalid negative values as NaN).
    6. Preserves the synthetic provenance column ('added_data_status').

    Parameters:
        df (pd.DataFrame): Raw fuel and weather dataset.

    Returns:
        pd.DataFrame: Cleaned DataFrame with valid physical ranges and preserved provenance.
    """
    if df.empty:
        return df.copy()

    cleaned = df.copy()

    # Step 1: Remove exact duplicate rows if any exist.
    cleaned = cleaned.drop_duplicates()

    # Step 2: Convert trip timestamps to datetime objects.
    if "timestamp" in cleaned.columns:
        cleaned["timestamp"] = pd.to_datetime(cleaned["timestamp"], errors="coerce")

    # Step 3: Trim whitespace from categorical and identifier string columns.
    str_cols = [
        "trip_id", "vehicle_id", "vehicle_type", "fuel_type",
        "road_type", "weather_condition", "added_data_status",
    ]
    for col in str_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # Step 4: Cast environmental and transport metrics to numeric types.
    numeric_cols = [
        "temperature_c", "rainfall_mm", "wind_speed_kmh", "visibility_km",
        "trip_distance_km", "load_weight_kg", "fuel_efficiency_km_per_liter",
        "fuel_consumed_liters", "fuel_price_per_liter", "fuel_cost_per_trip",
        "fuel_consumption_rate", "traffic_congestion_level",
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Step 5: Enforce physical non-negativity (distance, weight, fuel cannot be negative).
    non_negative_cols = [
        "trip_distance_km", "load_weight_kg", "fuel_efficiency_km_per_liter",
        "fuel_consumed_liters", "fuel_price_per_liter", "fuel_cost_per_trip",
    ]
    for col in non_negative_cols:
        if col in cleaned.columns:
            # Mask physically impossible negative numbers as NaN without inventing data.
            cleaned.loc[cleaned[col] < 0, col] = None

    return cleaned
