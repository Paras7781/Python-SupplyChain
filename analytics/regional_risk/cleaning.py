"""
Regional Risk - Data Cleaning Module.
Standardizes data types, removes PII, handles missing values, and aggregates item-level rows to orders.
"""
from __future__ import annotations

import pandas as pd
from shared.data_loader import SENSITIVE_CUSTOMER_FIELDS


def clean_regional_risk_data(df: pd.DataFrame, remove_pii: bool = True) -> pd.DataFrame:
    """Clean the DataCo supply chain dataset.

    Parameters:
        df (pd.DataFrame): Raw DataCo supply chain records.
        remove_pii (bool): If True, strips personal customer identifiers (passwords, emails, etc.).

    Returns:
        pd.DataFrame: A cleaned DataFrame with standardized types and no PII.
    """
    if df.empty:
        return df.copy()

    cleaned = df.copy()

    # Step 1: Remove exact duplicate rows across all columns.
    cleaned = cleaned.drop_duplicates()

    # Step 2: Remove customer PII to comply with privacy best practices.
    if remove_pii:
        pii_cols = [col for col in SENSITIVE_CUSTOMER_FIELDS if col in cleaned.columns]
        if pii_cols:
            cleaned = cleaned.drop(columns=pii_cols)

    # Step 3: Drop 'Product Description' because it contains 100% missing values in raw DataCo.
    if "Product Description" in cleaned.columns:
        cleaned = cleaned.drop(columns=["Product Description"])

    # Step 4: Parse date strings into standard datetime objects (coercing bad entries to NaT).
    date_cols = ["order date (DateOrders)", "shipping date (DateOrders)"]
    for col in date_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_datetime(cleaned[col], errors="coerce")

    # Step 5: Trim leading/trailing whitespace from categorical text columns.
    str_cols = [
        "Order Region", "Order State", "Order City", "Order Country",
        "Delivery Status", "Shipping Mode", "Order Status", "Market",
        "Customer Segment", "Category Name", "Department Name",
    ]
    for col in str_cols:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].astype(str).str.strip()

    # Step 6: Convert operational and financial fields to numeric values.
    num_cols = [
        "Days for shipping (real)", "Days for shipment (scheduled)",
        "Late_delivery_risk", "Order Item Quantity", "Order Item Total",
        "Order Profit Per Order", "Sales", "Benefit per order",
        "Order Item Discount", "Order Item Discount Rate", "Order Item Product Price",
    ]
    for col in num_cols:
        if col in cleaned.columns:
            cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    return cleaned


def aggregate_orders_level(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate item-level DataCo records into one record per unique Order Id.

    The original dataset records each line-item as a separate row. Training a
    risk model at the item level would cause data leakage between items of the
    same order. This function summarizes line-items into order-level metrics:
    - Sums item quantities to compute Total_Units.
    - Sums item totals to compute Total_Order_Value.
    - Counts items to compute Line_Items.

    Parameters:
        df (pd.DataFrame): Item-level DataCo DataFrame.

    Returns:
        pd.DataFrame: Order-level aggregated DataFrame (one row per Order Id).
    """
    if df.empty or "Order Id" not in df.columns:
        return pd.DataFrame()

    # Select order-level categorical and operational fields that remain constant per order.
    safe_first_fields = [
        "Order Region", "Order State", "Order City", "Order Country",
        "Market", "Customer Segment", "Delivery Status", "Late_delivery_risk",
        "Days for shipping (real)", "Days for shipment (scheduled)",
        "Shipping Mode", "Order Status", "order date (DateOrders)", "shipping date (DateOrders)",
    ]
    available_first = [c for c in safe_first_fields if c in df.columns]

    # Build aggregation dictionary: take first entry for order metadata, sum/count line-items.
    agg_dict: dict = {c: "first" for c in available_first}
    if "Order Item Id" in df.columns:
        agg_dict["Order Item Id"] = "count"
    if "Order Item Quantity" in df.columns:
        agg_dict["Order Item Quantity"] = "sum"
    if "Order Item Total" in df.columns:
        agg_dict["Order Item Total"] = "sum"

    orders = df.groupby("Order Id", as_index=False, sort=False).agg(agg_dict)

    # Rename aggregated columns to friendly, intuitive names.
    renames = {
        "Order Item Id": "Line_Items",
        "Order Item Quantity": "Total_Units",
        "Order Item Total": "Total_Order_Value",
        "order date (DateOrders)": "Order_Date",
        "shipping date (DateOrders)": "Shipping_Date",
    }
    orders = orders.rename(columns=renames)

    # Calculate actual shipping delay if both real and scheduled shipping days are present.
    if "Days for shipping (real)" in orders and "Days for shipment (scheduled)" in orders:
        orders["Shipping_Difference_Days"] = (
            orders["Days for shipping (real)"] - orders["Days for shipment (scheduled)"]
        )

    return orders
