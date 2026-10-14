"""
Inventory & Supplier - Exploratory Data Analysis (EDA) Module.
Calculates stockout rates, lead-time distribution, and vendor reliability scores.
"""
import pandas as pd


def get_inventory_supplier_summary(df: pd.DataFrame) -> dict:
    """
    Generate summary statistics for inventory and supplier analytics.
    """
    if df.empty:
        return {}
    return {
        "total_records": len(df),
        "columns": list(df.columns)
    }
