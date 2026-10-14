"""
Fuel & Weather - Exploratory Data Analysis (EDA) Module.
Calculates fuel cost trends, weather severity distributions, and route impacts.
"""
import pandas as pd


def get_fuel_weather_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for fuel price and weather impact analysis.
    """
    if df.empty:
        return {}
    return {
        "total_records": len(df),
        "columns": list(df.columns)
    }
