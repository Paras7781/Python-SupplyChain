"""
Regional Risk - Exploratory Data Analysis (EDA) Module.
Calculates risk metrics, regional aggregates, and summary statistics.
"""
from __future__ import annotations

import pandas as pd
from analytics.regional_risk.cleaning import aggregate_orders_level


def get_regional_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for regional delivery risk.

    Accepts item-level or order-level dataframe.
    """
    if df.empty:
        return {
            "total_records": 0,
            "total_orders": 0,
            "late_delivery_rate": 0.0,
            "delivery_status_breakdown": {},
            "shipping_mode_breakdown": {},
            "regional_performance": [],
        }

    # If item-level, group to order-level for accurate order counts
    if "Order Id" in df.columns and len(df) > df["Order Id"].nunique():
        order_df = aggregate_orders_level(df)
    else:
        order_df = df

    total_records = len(df)
    total_orders = len(order_df)

    late_count = 0
    late_rate = 0.0
    if "Late_delivery_risk" in order_df.columns:
        late_count = int((order_df["Late_delivery_risk"] == 1).sum())
        late_rate = float(late_count / total_orders) if total_orders > 0 else 0.0

    status_breakdown = {}
    if "Delivery Status" in order_df.columns:
        status_breakdown = order_df["Delivery Status"].value_counts().to_dict()

    shipping_modes = {}
    if "Shipping Mode" in order_df.columns:
        shipping_modes = order_df["Shipping Mode"].value_counts().to_dict()

    regional_perf = []
    if "Order Region" in order_df.columns:
        grouped = order_df.groupby("Order Region")
        for region, group in grouped:
            reg_orders = len(group)
            reg_late = int((group["Late_delivery_risk"] == 1).sum()) if "Late_delivery_risk" in group else 0
            reg_late_pct = round((reg_late / reg_orders) * 100, 2) if reg_orders > 0 else 0.0

            avg_real = float(group["Days for shipping (real)"].mean()) if "Days for shipping (real)" in group else None
            avg_sched = float(group["Days for shipment (scheduled)"].mean()) if "Days for shipment (scheduled)" in group else None

            regional_perf.append({
                "region": str(region),
                "orders": reg_orders,
                "late_orders": reg_late,
                "late_percentage": reg_late_pct,
                "avg_real_days": round(avg_real, 2) if avg_real is not None else None,
                "avg_scheduled_days": round(avg_sched, 2) if avg_sched is not None else None,
            })
        regional_perf.sort(key=lambda x: x["late_percentage"], reverse=True)

    return {
        "total_records": total_records,
        "total_orders": total_orders,
        "late_orders": late_count,
        "late_delivery_rate": round(late_rate, 4),
        "late_delivery_pct": round(late_rate * 100, 2),
        "delivery_status_breakdown": status_breakdown,
        "shipping_mode_breakdown": shipping_modes,
        "regional_performance": regional_perf,
    }


def filter_orders(
    df: pd.DataFrame,
    region: str | None = None,
    state: str | None = None,
    city: str | None = None,
    shipping_mode: str | None = None,
    delivery_status: str | None = None,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Filter order-level or item-level DataFrame by valid source criteria."""
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if region and region != "All" and "Order Region" in filtered.columns:
        filtered = filtered[filtered["Order Region"].astype(str) == str(region)]

    if state and state != "All" and "Order State" in filtered.columns:
        filtered = filtered[filtered["Order State"].astype(str) == str(state)]

    if city and city != "All" and "Order City" in filtered.columns:
        filtered = filtered[filtered["Order City"].astype(str) == str(city)]

    if shipping_mode and shipping_mode != "All" and "Shipping Mode" in filtered.columns:
        filtered = filtered[filtered["Shipping Mode"].astype(str) == str(shipping_mode)]

    if delivery_status and delivery_status != "All" and "Delivery Status" in filtered.columns:
        filtered = filtered[filtered["Delivery Status"].astype(str) == str(delivery_status)]

    date_col = "Order_Date" if "Order_Date" in filtered.columns else "order date (DateOrders)"
    if date_col in filtered.columns:
        if start_date:
            filtered = filtered[pd.to_datetime(filtered[date_col], errors="coerce") >= pd.to_datetime(start_date)]
        if end_date:
            filtered = filtered[pd.to_datetime(filtered[date_col], errors="coerce") <= pd.to_datetime(end_date)]

    return filtered
