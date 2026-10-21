"""Source-backed inventory and supplier review."""
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.cards import render_disclaimer, render_empty_state, render_metric_card, render_smart_alert
from ui.navigation import render_page_header, render_sidebar_header
from shared.data_loader import load_dataset


st.set_page_config(page_title="Supplier & Inventory Intelligence - SupplyIQ", page_icon="📦", layout="wide")
render_sidebar_header("inventory")
render_page_header(
    "Module 02 · Supplier & inventory intelligence",
    "Supplier and inventory records",
    "Filter actual retail-source rows to review stock, recorded demand, and supplier measures.",
)
render_disclaimer(
    "SOURCE DATA: Retail_Supply_Chain_Analytics_Dataset.csv. Only columns present in this source are used. "
    "Predictive, score, and optimization fields are excluded; no replenishment recommendation is generated."
)


@st.cache_data
def load_retail_records() -> pd.DataFrame:
    source = load_dataset("retail_analytics")
    if source.empty:
        return pd.DataFrame()
    fields = [
        "Product_ID", "Product_Category", "Brand", "SKU", "Current_Stock",
        "Reorder_Level", "Safety_Stock", "Inventory_Turnover", "Supplier_ID",
        "Supplier_Rating", "Lead_Time_Days", "Warehouse_ID", "Warehouse_Location",
        "Storage_Capacity", "Utilization_Rate", "Units_Sold", "Daily_Demand",
        "Monthly_Demand", "Shipping_Cost_USD", "Transportation_Mode",
        "Delivery_Time_Days", "On_Time_Delivery_Rate", "Date",
    ]
    frame = source[[field for field in fields if field in source.columns]].copy()
    numeric_fields = [
        "Current_Stock", "Reorder_Level", "Safety_Stock", "Inventory_Turnover",
        "Supplier_Rating", "Lead_Time_Days", "Storage_Capacity", "Utilization_Rate",
        "Units_Sold", "Daily_Demand", "Monthly_Demand", "Shipping_Cost_USD",
        "Delivery_Time_Days", "On_Time_Delivery_Rate",
    ]
    for field in numeric_fields:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "Date" in frame:
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    return frame

# Presentation helpers for UI tables:
# These functions format columns and categories specifically for this page's interactive displays.
def classify_stock_status(current_stock, reorder_level):
    """Classify stock from source values without treating missing values as healthy."""
    if pd.isna(current_stock) or pd.isna(reorder_level):
        return "Unknown"
    try:
        current = float(current_stock)
        reorder = float(reorder_level)
    except (TypeError, ValueError):
        return "Unknown"
    if not pd.notna(current) or not pd.notna(reorder):
        return "Unknown"
    if current <= 0:
        return "Critical"
    if current <= reorder:
        return "Reorder required"
    return "Healthy"


def build_inventory_recommendations(frame: pd.DataFrame) -> pd.DataFrame:
    """Create a transparent reorder recommendation table from valid source quantities."""
    required = ["Current_Stock", "Daily_Demand", "Lead_Time_Days", "Safety_Stock"]
    if not all(column in frame for column in required):
        return pd.DataFrame()
    recommendations = frame[required].copy()
    suggestions = recommendations.dropna(subset=required).copy()
    if suggestions.empty:
        return pd.DataFrame()
    suggestions["Target stock"] = (
        suggestions["Daily_Demand"] * suggestions["Lead_Time_Days"] + suggestions["Safety_Stock"]
    )
    suggestions["Suggested reorder quantity"] = (suggestions["Target stock"] - suggestions["Current_Stock"]).clip(lower=0)
    return suggestions



def build_inventory_coverage(frame: pd.DataFrame) -> pd.Series:
    """Estimate days of inventory cover using valid stock and positive daily demand."""
    coverage = pd.Series(
        float("nan"),
        index=frame.index,
        dtype="float64"
    )

    valid_fields = {"Current_Stock", "Daily_Demand"}.issubset(frame.columns)

    if not valid_fields:
        return coverage

    stock = pd.to_numeric(frame["Current_Stock"], errors="coerce")
    demand = pd.to_numeric(frame["Daily_Demand"], errors="coerce")

    valid = (
        stock.notna()
        & demand.notna()
        & (stock >= 0)
        & (demand > 0)
    )

    coverage.loc[valid] = stock.loc[valid] / demand.loc[valid]

    return coverage



records = load_retail_records()
if records.empty:
    st.error("The original retail analytics dataset is missing or empty. No sample records are substituted.")
    st.stop()

st.subheader("Filter source records")
first, second, third, fourth = st.columns(4)
with first:
    categories = ["All categories"] + sorted(records["Product_Category"].dropna().astype(str).unique()) if "Product_Category" in records else ["All categories"]
    category = st.selectbox("Product category", categories)
base = records.copy()
if category != "All categories" and "Product_Category" in base:
    base = base[base["Product_Category"].astype(str) == category]

with second:
    locations = ["All locations"] + sorted(base["Warehouse_Location"].dropna().astype(str).unique()) if "Warehouse_Location" in base else ["All locations"]
    location = st.selectbox("Warehouse location", locations)
if location != "All locations" and "Warehouse_Location" in base:
    base = base[base["Warehouse_Location"].astype(str) == location]

with third:
    warehouses = ["All warehouses"] + sorted(base["Warehouse_ID"].dropna().astype(str).unique()) if "Warehouse_ID" in base else ["All warehouses"]
    warehouse = st.selectbox("Warehouse ID", warehouses, help="The source has warehouse IDs but no human-readable warehouse names.")
if warehouse != "All warehouses" and "Warehouse_ID" in base:
    base = base[base["Warehouse_ID"].astype(str) == warehouse]

with fourth:
    if "Date" in base and base["Date"].notna().any():
        valid_dates = base["Date"].dropna()
        date_range = st.date_input(
            "Source record date range",
            value=(valid_dates.min().date(), valid_dates.max().date()),
            min_value=valid_dates.min().date(),
            max_value=valid_dates.max().date(),
        )
    else:
        date_range = None
        st.caption("Date filtering is unavailable; Date is missing or has no valid values.")

product_filter, supplier_filter = st.columns(2)
with product_filter:
    product_query = st.text_input("Find a product by ID, SKU, or brand", placeholder="Type a source product value")
with supplier_filter:
    supplier_query = st.text_input("Find a supplier by ID", placeholder="Type a source supplier ID")

if product_query.strip():
    # Search all available product identifiers before building the short selection list.
    query = product_query.strip()
    product_match = pd.Series(False, index=base.index)
    for field in ("Product_ID", "SKU", "Brand"):
        if field in base:
            product_match |= base[field].astype(str).str.contains(query, case=False, na=False)
    base = base[product_match]
product_values = ["All matching products"]
for field in ("Product_ID", "SKU"):
    if field in base:
        product_values.extend(base[field].dropna().astype(str).drop_duplicates().head(250).tolist())
selected_product = st.selectbox(
    "Product record",
    product_values,
    help="After typing a search above, choose an actual product ID or SKU from the matching source records. The list is capped at 500 values.",
)
if selected_product != "All matching products":
    # Product ID and SKU are both accepted because either may identify the chosen record.
    base = base[
        base.get("Product_ID", pd.Series("", index=base.index)).astype(str).eq(selected_product)
        | base.get("SKU", pd.Series("", index=base.index)).astype(str).eq(selected_product)
    ]
if supplier_query.strip() and "Supplier_ID" in base:
    base = base[base["Supplier_ID"].astype(str).str.contains(supplier_query.strip(), case=False, na=False)]
supplier_values = ["All matching suppliers"]
if "Supplier_ID" in base:
    supplier_values += base["Supplier_ID"].dropna().astype(str).drop_duplicates().head(500).tolist()
selected_supplier = st.selectbox(
    "Supplier record",
    supplier_values,
    help="After typing a search above, choose an actual supplier ID from the matching source records. The list is capped at 500 values.",
)
if selected_supplier != "All matching suppliers" and "Supplier_ID" in base:
    base = base[base["Supplier_ID"].astype(str) == selected_supplier]

filtered = base.copy()
if date_range and isinstance(date_range, tuple) and len(date_range) == 2 and "Date" in filtered:
    filtered = filtered[filtered["Date"].dt.date.between(*date_range)]

st.caption(f"{len(filtered):,} original source records match the selected filters.")
if filtered.empty:
    render_empty_state("No retail source records match these selections.")
    st.stop()

inventory_check_supported = {"Current_Stock", "Reorder_Level"}.issubset(filtered.columns)
if inventory_check_supported:
    comparable = filtered["Current_Stock"].notna() & filtered["Reorder_Level"].notna()
    below_reorder = comparable & (filtered["Current_Stock"] <= filtered["Reorder_Level"])
    stockout = filtered["Current_Stock"].notna() & filtered["Current_Stock"].le(0)
else:
    comparable = pd.Series(False, index=filtered.index)
    below_reorder = pd.Series(False, index=filtered.index)
    stockout = pd.Series(False, index=filtered.index)

filtered = filtered.copy()
if inventory_check_supported:
    # Keep records without enough stock data in the Unknown category, not a healthy one.
    filtered["Stock status"] = filtered.apply(
        lambda row: classify_stock_status(row.get("Current_Stock"), row.get("Reorder_Level")),
        axis=1,
    )
else:
    filtered["Stock status"] = "Unknown"

status_order = ["Critical", "Reorder required", "Healthy", "Unknown"]
status_counts = filtered["Stock status"].value_counts().reindex(status_order, fill_value=0)
status_summary = pd.DataFrame({
    "Stock status": status_counts.index,
    "Count": status_counts.values,
})
status_summary["Share (%)"] = (status_summary["Count"] / status_summary["Count"].sum() * 100) if status_summary["Count"].sum() else 0

metrics = st.columns(4)
with metrics[0]:
    render_metric_card("Filtered source records", f"{len(filtered):,}")
with metrics[1]:
    critical_count = int(status_summary.loc[status_summary["Stock status"] == "Critical", "Count"].sum()) if "Critical" in status_summary["Stock status"].values else 0
    critical_pct = (critical_count / len(filtered) * 100) if len(filtered) else 0
    render_metric_card("Critical stock rows", f"{critical_count:,} ({critical_pct:.1f}%)", help_text="Current stock is zero or negative.")
with metrics[2]:
    reorder_count = int(status_summary.loc[status_summary["Stock status"] == "Reorder required", "Count"].sum()) if "Reorder required" in status_summary["Stock status"].values else 0
    reorder_pct = (reorder_count / len(filtered) * 100) if len(filtered) else 0
    render_metric_card("Reorder required", f"{reorder_count:,} ({reorder_pct:.1f}%)", help_text="Current stock is above zero but at or below reorder level.")
with metrics[3]:
    mean_lead = filtered["Lead_Time_Days"].mean() if "Lead_Time_Days" in filtered else None
    render_metric_card("Mean recorded lead time", f"{mean_lead:.1f} days" if pd.notna(mean_lead) else "Unavailable")

if critical_count > 0:
    render_smart_alert(
        title="Critical Stock Depletion Detected",
        trigger=f"{critical_count:,} filtered SKU records with zero or negative stock on hand ({critical_pct:.1f}% of filtered rows)",
        threshold="Stock level <= 0 units (Critical stockout point)",
        why_it_matters="Immediate fulfillment failure for incoming customer orders; risk of contract penalties and lost revenue.",
        next_step="Initiate expedited emergency replenishment purchase orders for critical SKUs.",
        severity="error",
        is_model_prediction=False,
    )
elif reorder_count > 0:
    render_smart_alert(
        title="Inventory Reorder Point Breached",
        trigger=f"{reorder_count:,} SKU records have dipped to or below their configured reorder point ({reorder_pct:.1f}% of filtered rows)",
        threshold="Business rule: Current_Stock <= Reorder_Level",
        why_it_matters="Safety buffer is actively engaged; unexpected demand surges or supplier lead-time delays will cause stockouts.",
        next_step="Trigger standard replenishment purchase orders matching the suggested quantities below.",
        severity="warning",
        is_model_prediction=False,
    )

st.subheader("Stock status mix")
st.dataframe(status_summary, hide_index=True, width="stretch", column_config={
    "Share (%)": st.column_config.NumberColumn(format="%.1f%%"),
})

left, right = st.columns(2)
with left:
    st.subheader("Recorded demand")
    demand_field = "Units_Sold" if "Units_Sold" in filtered else ("Daily_Demand" if "Daily_Demand" in filtered else None)
    demand_label = "average units sold" if demand_field == "Units_Sold" else "average daily demand"
    if demand_field and "Product_Category" in filtered:
        demand = filtered.groupby("Product_Category", as_index=False)[demand_field].mean().rename(columns={demand_field: f"Mean {demand_field}"})
        demand = demand.sort_values(f"Mean {demand_field}", ascending=False, kind="stable")
        chart = px.bar(
            demand,
            x="Product_Category",
            y=f"Mean {demand_field}",
            title=f"Mean source {demand_field} by product category",
            labels={"Product_Category": "Source product category", f"Mean {demand_field}": f"{demand_label.title()}"},
            hover_name="Product_Category",
        )
        chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D9E4F0"})
        st.plotly_chart(chart, width="stretch")
        if "Date" in filtered and filtered["Date"].notna().any() and demand_field in filtered:
            monthly_demand = filtered[["Date", demand_field]].copy()
            monthly_demand["Date"] = pd.to_datetime(monthly_demand["Date"], errors="coerce")
            monthly_demand = monthly_demand.dropna(subset=["Date"]).copy()
            if not monthly_demand.empty:
                monthly_demand["Month"] = monthly_demand["Date"].dt.to_period("M").dt.to_timestamp()
                monthly = monthly_demand.groupby("Month", as_index=False)[demand_field].mean()
                if len(monthly) > 1:
                    monthly_chart = px.line(
                        monthly,
                        x="Month",
                        y=demand_field,
                        title=f"Monthly {demand_label} trend",
                        labels={"Month": "Month", demand_field: f"Average {demand_label}"},
                    )
                    monthly_chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D9E4F0"})
                    st.plotly_chart(monthly_chart, width="stretch")
    else:
        render_empty_state("Demand chart unavailable: Product_Category and Units_Sold or Daily_Demand are required.")

with right:
    st.subheader("Supplier performance")
    measures = [field for field in ("Supplier_Rating", "Lead_Time_Days", "On_Time_Delivery_Rate") if field in filtered]
    if "Supplier_ID" in filtered and measures:
        summary = filtered.groupby("Supplier_ID", as_index=False).agg(
            Source_records=("Supplier_ID", "size"),
            Average_supplier_rating=("Supplier_Rating", "mean") if "Supplier_Rating" in filtered else ("Supplier_ID", lambda values: pd.NA),
            Average_lead_time_days=("Lead_Time_Days", "mean") if "Lead_Time_Days" in filtered else ("Supplier_ID", lambda values: pd.NA),
            Average_on_time_delivery_rate=("On_Time_Delivery_Rate", "mean") if "On_Time_Delivery_Rate" in filtered else ("Supplier_ID", lambda values: pd.NA),
        )
        summary = summary.dropna(subset=[col for col in summary.columns if col.startswith("Average_")], how="all")
        if {"Average_supplier_rating", "Average_lead_time_days", "Average_on_time_delivery_rate"}.issubset(summary.columns):
            # Normalize rating and lead-time measures before applying the documented weights.
            max_rating = summary["Average_supplier_rating"].max()
            max_lead = summary["Average_lead_time_days"].max()
            rating_score = (summary["Average_supplier_rating"] / max_rating * 100) if pd.notna(max_rating) and max_rating > 0 else 0
            lead_score = (100 - ((summary["Average_lead_time_days"] / max_lead) * 100)).clip(lower=0, upper=100) if pd.notna(max_lead) and max_lead > 0 else 0
            summary["Weighted supplier score"] = (
                rating_score * 0.35 + summary["Average_on_time_delivery_rate"] * 0.45 + lead_score * 0.20
            )
            summary["Weighted supplier score"] = summary["Weighted supplier score"].replace([float("inf"), -float("inf")], pd.NA)
            summary = summary.sort_values("Weighted supplier score", ascending=False, kind="stable")
            st.dataframe(
                summary[["Supplier_ID", "Source_records", "Average_supplier_rating", "Average_lead_time_days", "Average_on_time_delivery_rate", "Weighted supplier score"]].head(100),
                hide_index=True,
                width="stretch",
                column_config={
                    "Average_supplier_rating": st.column_config.NumberColumn(format="%.2f"),
                    "Average_lead_time_days": st.column_config.NumberColumn(format="%.1f"),
                    "Average_on_time_delivery_rate": st.column_config.NumberColumn(format="%.1f%%"),
                    "Weighted supplier score": st.column_config.NumberColumn(format="%.1f"),
                },
            )
            st.caption(
                "Weighted supplier score uses 35% supplier rating, 45% on-time delivery rate, and 20% lead-time performance. "
                "This is an explainable comparison among source records, not a validated prediction."
            )
        else:
            summary = summary.sort_values("Source_records", ascending=False, kind="stable")
            st.dataframe(summary.head(100), hide_index=True, width="stretch")
            st.caption("Descriptive means from the currently filtered records, not a validated ranking or prediction.")
    else:
        render_empty_state("Supplier comparison requires Supplier_ID and at least one recorded supplier measure.")

st.subheader("Inventory recommendation (Rule-Based)")
recommendation = build_inventory_recommendations(filtered)
if recommendation.empty:
    st.info("Insufficient data: current stock, daily demand, lead time, and safety stock are required to estimate a reorder quantity.")
else:
    recommendation = recommendation[["Current_Stock", "Daily_Demand", "Lead_Time_Days", "Safety_Stock", "Target stock", "Suggested reorder quantity"]].copy()
    st.dataframe(recommendation.head(50), hide_index=True, width="stretch")
    st.caption(
        "**Formula & Assumptions:** `Target stock = (Recorded Daily Demand × Lead Time) + Safety Stock`; "
        "`Suggested reorder quantity = max(0, Target stock − Current Stock)`. "
        "• **Source Data:** Current Stock, Daily Demand, Lead Time, and Safety Stock originate directly from historical records. "
        "• **Rule-Based Calculation:** Target stock and Suggested reorder quantity are deterministic arithmetic outputs, not ML predictions. "
        "• **Operational Caveats:** Assumes steady daily demand with zero variance, zero in-transit purchase orders, and fixed lead times. "
        "Do not use for automated real-world purchase commitments without supplier MOQ, batch constraints, and verified sales velocity."
    )

st.subheader("Estimated inventory coverage")
coverage = build_inventory_coverage(filtered)
coverage_values = coverage.dropna()
if coverage_values.empty:
    st.info("Inventory coverage is unavailable because current stock or daily demand is missing, invalid, or daily demand is zero or negative.")
else:
    mean_days = coverage_values.mean()
    st.caption(f"Estimated days of inventory remaining = Current stock ÷ Daily demand. Based on the filtered records, the average estimate is {mean_days:.1f} days.")
    st.dataframe(
        pd.DataFrame({
            "Current_Stock": filtered.loc[coverage.index, "Current_Stock"],
            "Daily_Demand": filtered.loc[coverage.index, "Daily_Demand"],
            "Days of inventory remaining": coverage_values,
        }).dropna(),
        hide_index=True,
        width="stretch",
    )
    st.caption("This is an estimate based on recorded daily demand, not a guaranteed stockout date.")

st.subheader("Filtered source records")
if inventory_check_supported:
    displayed = filtered.copy()
    displayed["At / below recorded reorder level"] = below_reorder
    if comparable.any():
        displayed.loc[comparable, "Gap to reorder level · units"] = (
            filtered.loc[comparable, "Reorder_Level"] - filtered.loc[comparable, "Current_Stock"]
        ).clip(lower=0)
else:
    displayed = filtered
st.dataframe(displayed.head(200), hide_index=True, width="stretch")
st.caption("Showing up to 200 matching original rows. Product and supplier values are IDs because the source does not include human-readable product or supplier names.")
