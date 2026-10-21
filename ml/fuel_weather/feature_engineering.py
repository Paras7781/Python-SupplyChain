"""
Fuel & Weather - Feature Engineering Module.
Creates transport metrics, unit conversions, and transparent fuel estimation functions.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def calculate_fuel_estimate(
    distance_km: float | None,
    efficiency_km_per_litre: float | None,
) -> float:
    """Calculate expected trip fuel consumption in liters.

    Formula:
        Fuel Consumption (Liters) = Distance (km) / Efficiency (km/L)

    Validates that inputs are non-null, finite, and strictly greater than zero.

    Parameters:
        distance_km (float): Planned travel distance in kilometers.
        efficiency_km_per_litre (float): Rated vehicle fuel efficiency in km per liter.

    Returns:
        float: Estimated fuel volume in liters.
    """
    values = {
        "Distance (km)": distance_km,
        "Vehicle efficiency (km/L)": efficiency_km_per_litre,
    }
    # Validate each input parameter explicitly before performing the calculation.
    for label, value in values.items():
        if value is None:
            raise ValueError(f"{label} is required to calculate an estimate.")
        try:
            val = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{label} must be a valid number.")
        if not math.isfinite(val):
            raise ValueError(f"{label} must be a finite number.")
        if val <= 0:
            raise ValueError(f"{label} must be greater than zero.")

    return float(distance_km) / float(efficiency_km_per_litre)


def calculate_fuel_cost_estimate(
    liters: float | None,
    price_per_liter: float | None,
) -> float:
    """Calculate the estimated monetary fuel cost of a trip in USD.

    Formula:
        Fuel Cost ($) = Fuel Consumed (Liters) * Fuel Price ($/Liter)

    Validates that inputs are non-null, finite, and non-negative.

    Parameters:
        liters (float): Fuel volume in liters.
        price_per_liter (float): Unit fuel price in USD per liter.

    Returns:
        float: Estimated fuel cost in USD.
    """
    values = {
        "Fuel volume (litres)": liters,
        "Fuel price ($/L)": price_per_liter,
    }
    # Validate each input parameter explicitly before multiplication.
    for label, value in values.items():
        if value is None:
            raise ValueError(f"{label} is required to calculate fuel cost.")
        try:
            val = float(value)
        except (ValueError, TypeError):
            raise ValueError(f"{label} must be a valid number.")
        if not math.isfinite(val):
            raise ValueError(f"{label} must be a finite number.")
        if val < 0:
            raise ValueError(f"{label} cannot be negative.")

    return float(liters) * float(price_per_liter)


def create_fuel_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construct transport efficiency and route condition features for analytics.

    Engineers:
    - Liters per 100km: Standard European fuel consumption metric (100 / km_per_liter).
    - Ton-km: Transport work metric (Distance in km * Payload in metric tons).
    - Adverse weather flag: Binary indicator for rain, snow, storm, or fog.

    Parameters:
        df (pd.DataFrame): Cleaned fuel and weather DataFrame.

    Returns:
        pd.DataFrame: DataFrame enriched with engineered transport metrics.
    """
    if df.empty:
        return df.copy()

    features = df.copy()

    # Step 1: Compute Liters per 100km (standard automotive metric).
    if "fuel_efficiency_km_per_liter" in features.columns:
        eff = pd.to_numeric(features["fuel_efficiency_km_per_liter"], errors="coerce")
        features["liters_per_100km"] = np.where(eff > 0, 100.0 / eff, np.nan)

    # Step 2: Compute ton-km transport work: Distance (km) * Payload (metric tons).
    if "trip_distance_km" in features.columns and "load_weight_kg" in features.columns:
        dist = pd.to_numeric(features["trip_distance_km"], errors="coerce").fillna(0)
        weight_ton = (pd.to_numeric(features["load_weight_kg"], errors="coerce") / 1000.0).fillna(0)
        features["ton_km"] = dist * weight_ton

    # Step 3: Flag adverse driving weather (rain, snow, storm, fog, blizzard).
    if "weather_condition" in features.columns:
        adverse_conditions = {"rain", "snow", "storm", "fog", "heavy rain", "blizzard"}
        features["is_adverse_weather"] = (
            features["weather_condition"].astype(str).str.lower().isin(adverse_conditions)
        ).astype(int)

    return features
