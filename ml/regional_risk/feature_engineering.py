"""
Regional Risk - Feature Engineering Module.
Constructs order-time features for delivery risk assessment while strictly preventing target and temporal leakage.
"""
from __future__ import annotations

import pandas as pd


# Features strictly available at the time of order placement
ALLOWED_ORDER_TIME_FEATURES = [
    "Days for shipment (scheduled)",
    "Shipping Mode",
    "Order Region",
    "Order State",
    "Market",
    "Customer Segment",
    "order_month",
    "order_dayofweek",
    "order_hour",
    "is_weekend",
    "Total_Units",
    "Total_Order_Value",
    "Line_Items",
    "value_per_unit",
]

# Leaking features that MUST NEVER be fed to a prediction model at order time
LEAKAGE_COLUMNS = [
    "Days for shipping (real)",
    "Delivery Status",
    "Shipping_Difference_Days",
    "Shipping difference (days)",
    "Shipping gap (days)",
    "shipping date (DateOrders)",
    "Shipping_Date",
    "Order Status",
]


def create_regional_risk_features(df: pd.DataFrame, drop_leakage: bool = True) -> pd.DataFrame:
    """Generate order-placement features for regional delivery risk prediction.

    Constructs calendar variables, order unit/value summaries, and scheduled shipment
    timelines. By default, it strictly drops fulfillment outcomes (such as real shipping
    days or post-delivery status) to prevent target and temporal leakage.

    Parameters:
        df (pd.DataFrame): Input DataFrame (item-level or order-level).
        drop_leakage (bool): If True, removes post-order fulfillment fields
                             (e.g., real shipping days, delivery status text)
                             so the model only sees data available at order time.

    Returns:
        pd.DataFrame: DataFrame enriched with engineered features and leakage removed.
    """
    if df.empty:
        return df.copy()

    features = df.copy()

    # Step 1: Identify the order date column to extract calendar timing features.
    date_col = None
    for candidate in ["Order_Date", "order date (DateOrders)"]:
        if candidate in features.columns:
            date_col = candidate
            break

    if date_col:
        # Derive only calendar attributes available at the moment of order placement.
        order_dates = pd.to_datetime(features[date_col], errors="coerce")
        features["order_month"] = order_dates.dt.month.fillna(1).astype(int)
        features["order_dayofweek"] = order_dates.dt.dayofweek.fillna(0).astype(int)
        features["order_hour"] = order_dates.dt.hour.fillna(12).astype(int)
        # Binary flag: 1 if Saturday (5) or Sunday (6), else 0.
        features["is_weekend"] = features["order_dayofweek"].isin([5, 6]).astype(int)
    else:
        # Default fallback values if no order date is present in the frame.
        features["order_month"] = 1
        features["order_dayofweek"] = 0
        features["order_hour"] = 12
        features["is_weekend"] = 0

    # Step 2: Ensure uniform naming for order quantities and order financial totals.
    qty_col = "Total_Units" if "Total_Units" in features.columns else ("Order Item Quantity" if "Order Item Quantity" in features.columns else None)
    val_col = "Total_Order_Value" if "Total_Order_Value" in features.columns else ("Order Item Total" if "Order Item Total" in features.columns else None)

    if qty_col and val_col:
        features["Total_Units"] = pd.to_numeric(features[qty_col], errors="coerce").fillna(1)
        features["Total_Order_Value"] = pd.to_numeric(features[val_col], errors="coerce").fillna(0)
        # Avoid division by zero when calculating per-unit order value.
        units = features["Total_Units"].replace(0, 1)
        features["value_per_unit"] = (features["Total_Order_Value"] / units).clip(lower=0)
    else:
        features["Total_Units"] = 1
        features["Total_Order_Value"] = 0.0
        features["value_per_unit"] = 0.0

    # Step 3: Ensure line item count exists.
    if "Line_Items" not in features.columns:
        features["Line_Items"] = 1

    # Step 4: Standardize scheduled shipment days (fallback to typical 4 days if missing).
    if "Days for shipment (scheduled)" in features.columns:
        features["Days for shipment (scheduled)"] = pd.to_numeric(
            features["Days for shipment (scheduled)"], errors="coerce"
        ).fillna(4)

    # Step 5: Strictly remove fulfillment outcome columns to prevent target leakage.
    if drop_leakage:
        # Dropping fields like 'Days for shipping (real)' and 'Delivery Status' ensures
        # the model cannot 'cheat' by observing future fulfillment outcomes.
        to_drop = [col for col in LEAKAGE_COLUMNS if col in features.columns]
        if to_drop:
            features = features.drop(columns=to_drop)

    return features
