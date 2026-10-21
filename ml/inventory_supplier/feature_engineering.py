"""
Inventory & Supplier - Feature Engineering Module.
Computes reorder points, safety stock ratios, inventory coverage, and supplier risk indices.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def build_inventory_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent reorder recommendation table from valid source quantities.

    Formula:
        Target stock = (Daily_Demand * Lead_Time_Days) + Safety_Stock
        Suggested reorder quantity = max(0, Target stock - Current_Stock)

    Assumptions & Caveats:
        - Assumes daily demand is steady with zero variance over the replenishment window.
        - Assumes zero in-transit purchase orders or backorders.
        - Assumes replenishment lead time is strictly deterministic.
        - Does not account for supplier Minimum Order Quantities (MOQs), batch sizes, or holding costs.
        - Outputs serve as explainable heuristics on source records, not automated real-world procurement orders.
    """
    required = ["Product_ID", "Current_Stock", "Daily_Demand", "Lead_Time_Days", "Safety_Stock"]
    available = [col for col in required if col in df.columns]
    if len(available) < len(required):
        # Fallback if Product_ID is missing
        core = ["Current_Stock", "Daily_Demand", "Lead_Time_Days", "Safety_Stock"]
        if not all(col in df.columns for col in core):
            return pd.DataFrame()
        available = core

    recs = df[available].dropna().copy()
    if recs.empty:
        return pd.DataFrame()

    recs["Target_Stock"] = (
        recs["Daily_Demand"] * recs["Lead_Time_Days"] + recs["Safety_Stock"]
    )
    recs["Suggested_Reorder_Quantity"] = (recs["Target_Stock"] - recs["Current_Stock"]).clip(lower=0)
    return recs


def build_inventory_coverage(df: pd.DataFrame) -> pd.Series:
    """Estimate days of inventory cover using valid stock and positive daily demand.

    Formula:
        Days of cover = Current_Stock / Daily_Demand (where Daily_Demand > 0)
    """
    if "Current_Stock" not in df.columns or "Daily_Demand" not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    stock = pd.to_numeric(df["Current_Stock"], errors="coerce")
    demand = pd.to_numeric(df["Daily_Demand"], errors="coerce")

    # Only compute for positive demand
    coverage = np.where(demand > 0, stock / demand, np.nan)
    return pd.Series(coverage, index=df.index, dtype="float64")


def create_inventory_supplier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate feature transformations for inventory and supplier analytics."""
    if df.empty:
        return df.copy()

    features = df.copy()

    # Stock-to-reorder ratio
    if "Current_Stock" in features.columns and "Reorder_Level" in features.columns:
        stock = pd.to_numeric(features["Current_Stock"], errors="coerce").fillna(0)
        reorder = pd.to_numeric(features["Reorder_Level"], errors="coerce").replace(0, 1)
        features["Stock_To_Reorder_Ratio"] = (stock / reorder).clip(upper=10.0)

    # Days of coverage
    features["Days_Of_Coverage"] = build_inventory_coverage(features)

    # Reorder status flag (target for classification if desired)
    if "Current_Stock" in features.columns and "Reorder_Level" in features.columns:
        features["Needs_Reorder"] = (
            features["Current_Stock"] <= features["Reorder_Level"]
        ).astype(int)

    # Low supplier rating flag
    if "Supplier_Rating" in features.columns:
        features["Low_Rating_Supplier"] = (
            pd.to_numeric(features["Supplier_Rating"], errors="coerce") < 3.0
        ).astype(int)

    # Supplier delivery risk flag
    if "On_Time_Delivery_Rate" in features.columns:
        features["Delivery_Risk_Supplier"] = (
            pd.to_numeric(features["On_Time_Delivery_Rate"], errors="coerce") < 0.85
        ).astype(int)

    return features
