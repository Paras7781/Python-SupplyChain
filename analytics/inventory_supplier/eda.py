"""
Inventory & Supplier - Exploratory Data Analysis (EDA) Module.
Calculates stockout rates, lead-time distribution, and vendor reliability scores.
"""
from __future__ import annotations

import pandas as pd


def classify_stock_status(current_stock: float | int | None, reorder_level: float | int | None) -> str:
    """Classify stock level strictly using verified numerical values.

    Returns 'Critical' if <= 0, 'Reorder required' if <= reorder_level, 'Healthy' otherwise,
    or 'Unknown' if values are missing.
    """
    if pd.isna(current_stock) or pd.isna(reorder_level):
        return "Unknown"
    try:
        curr = float(current_stock)
        reorder = float(reorder_level)
    except (ValueError, TypeError):
        return "Unknown"

    if curr <= 0:
        return "Critical"
    if curr <= reorder:
        return "Reorder required"
    return "Healthy"


def get_inventory_supplier_summary(df: pd.DataFrame) -> dict:
    """Generate summary statistics for inventory and supplier analytics."""
    if df.empty:
        return {
            "total_records": 0,
            "distinct_products": 0,
            "distinct_suppliers": 0,
            "distinct_warehouses": 0,
            "stock_health": {},
            "categories": [],
            "top_suppliers": [],
        }

    total_records = len(df)
    products = df["Product_ID"].nunique() if "Product_ID" in df.columns else 0
    suppliers = df["Supplier_ID"].nunique() if "Supplier_ID" in df.columns else 0
    warehouses = df["Warehouse_ID"].nunique() if "Warehouse_ID" in df.columns else 0

    # Stock health distribution
    stock_health = {"Healthy": 0, "Reorder required": 0, "Critical": 0, "Unknown": 0}
    if {"Current_Stock", "Reorder_Level"}.issubset(df.columns):
        stock = pd.to_numeric(df["Current_Stock"], errors="coerce")
        reorder = pd.to_numeric(df["Reorder_Level"], errors="coerce")
        critical = (stock <= 0).sum()
        reorder_needed = ((stock > 0) & (stock <= reorder)).sum()
        healthy = (stock > reorder).sum()
        unknown = (stock.isna() | reorder.isna()).sum()
        stock_health = {
            "Healthy": int(healthy),
            "Reorder required": int(reorder_needed),
            "Critical": int(critical),
            "Unknown": int(unknown),
        }

    # Category performance
    categories = []
    if "Product_Category" in df.columns:
        cat_group = df.groupby("Product_Category")
        for cat, group in cat_group:
            mean_stock = float(group["Current_Stock"].mean()) if "Current_Stock" in group else 0.0
            mean_demand = float(group["Daily_Demand"].mean()) if "Daily_Demand" in group else 0.0
            mean_lead = float(group["Lead_Time_Days"].mean()) if "Lead_Time_Days" in group else 0.0
            categories.append({
                "category": str(cat),
                "count": len(group),
                "mean_stock": round(mean_stock, 1),
                "mean_daily_demand": round(mean_demand, 1),
                "mean_lead_time_days": round(mean_lead, 1),
            })

    # Supplier performance summary
    top_suppliers = []
    if "Supplier_ID" in df.columns:
        sup_group = df.groupby("Supplier_ID")
        for sup_id, group in sup_group:
            rating = float(group["Supplier_Rating"].mean()) if "Supplier_Rating" in group else 0.0
            on_time = float(group["On_Time_Delivery_Rate"].mean()) if "On_Time_Delivery_Rate" in group else 0.0
            lead_time = float(group["Lead_Time_Days"].mean()) if "Lead_Time_Days" in group else 0.0
            top_suppliers.append({
                "supplier_id": str(sup_id),
                "records_count": len(group),
                "mean_rating": round(rating, 2),
                "on_time_delivery_rate": round(on_time, 2),
                "mean_lead_time_days": round(lead_time, 1),
            })
        top_suppliers.sort(key=lambda x: x["mean_rating"], reverse=True)

    return {
        "total_records": total_records,
        "distinct_products": products,
        "distinct_suppliers": suppliers,
        "distinct_warehouses": warehouses,
        "stock_health": stock_health,
        "categories": categories,
        "top_suppliers": top_suppliers[:15],
    }


def filter_inventory_records(
    df: pd.DataFrame,
    category: str | None = None,
    supplier_id: str | None = None,
    warehouse_id: str | None = None,
    stock_status: str | None = None,
) -> pd.DataFrame:
    """Filter inventory records by category, supplier, warehouse, and stock health."""
    if df.empty:
        return df.copy()

    filtered = df.copy()

    if category and category != "All categories" and "Product_Category" in filtered.columns:
        filtered = filtered[filtered["Product_Category"].astype(str) == str(category)]

    if supplier_id and supplier_id != "All suppliers" and "Supplier_ID" in filtered.columns:
        filtered = filtered[filtered["Supplier_ID"].astype(str) == str(supplier_id)]

    if warehouse_id and warehouse_id != "All warehouses" and "Warehouse_ID" in filtered.columns:
        filtered = filtered[filtered["Warehouse_ID"].astype(str) == str(warehouse_id)]

    if stock_status and stock_status != "All statuses" and {"Current_Stock", "Reorder_Level"}.issubset(filtered.columns):
        stock = pd.to_numeric(filtered["Current_Stock"], errors="coerce")
        reorder = pd.to_numeric(filtered["Reorder_Level"], errors="coerce")
        if stock_status == "Critical":
            filtered = filtered[stock <= 0]
        elif stock_status == "Reorder required":
            filtered = filtered[(stock > 0) & (stock <= reorder)]
        elif stock_status == "Healthy":
            filtered = filtered[stock > reorder]

    return filtered
