"""
Regional Risk - Exploratory Data Analysis (EDA) Module.
Calculates risk metrics, regional aggregates, and summary statistics.
"""
import pandas as pd


def get_regional_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for regional risk analysis.
    """
    if df.empty:
        return {}
    return {
        "total_records": len(df),
        "columns": list(df.columns)
    }
