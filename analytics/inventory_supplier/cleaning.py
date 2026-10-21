"""
Inventory & Supplier - Data Cleaning Module.
Standardizes data types, validates stock and supplier fields, and ensures clean analytical frames.
"""
from __future__ import annotations

import pandas as pd


def clean_inventory_supplier_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the retail supply chain analytics dataset.

    Performs the following cleaning operations:
    1. Removes exact duplicate rows.
    2. Parses the 'Date' column into standard datetime format.
    3. Strips whitespace from identifiers and categorical text columns.
    4. Casts stock, demand, rating, and financial fields to numeric values.

    Parameters:
        df (pd.DataFrame): Raw retail supply chain DataFrame.

    Returns:
        pd.DataFrame: Cleaned DataFrame with standardized types and trimmed strings.
    """
    if df.empty:
        return df.copy()

    cleaned = df.copy()

    # Step 1: Remove exact duplicate rows if any exist.
    cleaned = cleaned.drop_duplicates()

    # Step 2: Convert date strings to datetime objects (invalid dates become NaT).
    if "Date" in cleaned.columns:
        cleaned["Date"] = pd.to_datetime(cleaned["Date"], errors="coerce")

    # Step 3: Trim whitespace from text identifiers and category columns.
    str_cols = [
        "Product_ID", "Product_Category", "Brand", "SKU",
        "Supplier_ID", "Warehouse_ID", "Warehouse_Location",
        "Transportation_Mode",
    ]
    for col in str_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # Step 4: Convert operational, demand, and financial fields to numeric types.
    numeric_cols = [
        "Current_Stock", "Reorder_Level", "Safety_Stock", "Inventory_Turnover",
        "Supplier_Rating", "Lead_Time_Days", "Storage_Capacity", "Utilization_Rate",
        "Units_Sold", "Daily_Demand", "Monthly_Demand", "Seasonal_Demand_Index",
        "Shipping_Cost_USD", "Delivery_Time_Days", "On_Time_Delivery_Rate",
        "Product_Cost_USD", "Selling_Price_USD", "Revenue_USD", "Profit_USD",
        "Stockout_Risk", "Overstock_Risk", "Supply_Disruption_Risk",
        "Demand_Forecast", "Operational_Risk_Score",
    ]
    for col in numeric_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned
