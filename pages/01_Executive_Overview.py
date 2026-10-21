"""Executive overview of independently sourced SupplyIQ datasets."""
import pandas as pd
import plotly.express as px
import streamlit as st

from ui.cards import render_disclaimer, render_empty_state, render_metric_card, render_smart_alert
from ui.navigation import render_page_header, render_sidebar_header
from shared.data_loader import load_dataset


st.set_page_config(page_title="Executive Overview - SupplyIQ", page_icon="📊", layout="wide")
render_sidebar_header("executive")
render_page_header(
    "SupplyIQ command center",
    "Supply Chain Risk & Decision Intelligence",
    "Explore regional delivery records, supplier and inventory records, and recorded fuel consumption as separate data domains.",
)
render_disclaimer(
    "All metrics are calculated from original CSV rows. Independent datasets are not joined. "
    "The fuel CSV identifies its records as synthetic_demo_values; this provenance is retained and shown."
)

# Load the three source datasets separately for the overview sections below.
orders = load_dataset("supply_chain")
retail = load_dataset("retail_analytics")
fuel = load_dataset("fuel_weather")

# ── Executive Smart Alert & Risk Radar ──────────────────────────────────────
st.subheader("Executive Risk & Alert Radar")
radar_c1, radar_c2 = st.columns(2)
with radar_c1:
    if not orders.empty and "Delivery Status" in orders.columns:
        late_mask_ov = orders["Delivery Status"].astype(str).str.contains("late", case=False, na=False)
        late_pct_ov = late_mask_ov.mean() * 100
        if late_pct_ov > 50:
            render_smart_alert(
                title="Systemic Late Delivery Concentration",
                trigger=f"{late_pct_ov:.1f}% overall late delivery rate across {orders['Order Id'].nunique():,} orders",
                threshold="SLA ceiling of 50.0% late delivery rate",
                why_it_matters="Over half of fulfilled shipments incurred delays, signaling carrier routing or scheduling bottlenecks.",
                next_step="Inspect regional drilldowns below or open Regional Explorer (Module 01) to prioritize carrier changes.",
                severity="warning",
                is_model_prediction=False,
            )
with radar_c2:
    if not retail.empty and {"Current_Stock", "Reorder_Level"}.issubset(retail.columns):
        stock_s = pd.to_numeric(retail["Current_Stock"], errors="coerce")
        reorder_s = pd.to_numeric(retail["Reorder_Level"], errors="coerce")
        v = stock_s.notna() & reorder_s.notna()
        shortage_cnt = (stock_s[v] <= reorder_s[v]).sum()
        if shortage_cnt > 0:
            render_smart_alert(
                title="Immediate Inventory Reorder Demand",
                trigger=f"{int(shortage_cnt):,} SKU records currently at or below minimum reorder thresholds",
                threshold="Business rule: Current_Stock <= Reorder_Level",
                why_it_matters="Stock buffers are compromised; lead-time variability will precipitate stockouts.",
                next_step="Open Supplier & Inventory (Module 02) to generate replenishment orders.",
                severity="warning",
                is_model_prediction=False,
            )

render_smart_alert(
    title="Synthetic Demonstration Data In Use",
    trigger="Fuel consumption dataset tagged added_data_status = 'synthetic_demo_values'",
    threshold="Live operational deployment requirement: field telemetry validation",
    why_it_matters="Fuel regression estimates demonstrate decision flows but are not validated against real vehicle sensors.",
    next_step="Interpret fuel trip metrics as synthetic demo values.",
    severity="info",
    is_model_prediction=False,
)

with st.expander("Data Quality, Row Counts & Processing Status", expanded=False):
    dq_col1, dq_col2, dq_col3 = st.columns(3)
    with dq_col1:
        st.markdown(
            "**DataCo Supply Chain**\n"
            f"- Raw rows: `{len(orders):,}` item lines\n"
            f"- Unique orders: `{orders['Order Id'].nunique() if 'Order Id' in orders else 0:,}`\n"
            "- Processed status: Model-ready aggregated order dataset generated (`data/processed/regional_risk_model_ready.csv`)\n"
            "- Leakage controls: Real shipping days purged from model training."
        )
    with dq_col2:
        st.markdown(
            "**Retail Supply Chain**\n"
            f"- Raw rows: `{len(retail):,}` records\n"
            "- Processed status: Row-level structure strictly preserved (`data/processed/inventory_demand_model_ready.csv`)\n"
            "- Validation: Missing values coerced, non-numeric checks applied."
        )
    with dq_col3:
        st.markdown(
            "**Fuel & Weather Transport**\n"
            f"- Raw rows: `{len(fuel):,}` trips\n"
            "- Processed status: Row-level preserved (`data/processed/fuel_model_ready.csv`)\n"
            "- Provenance: Marked `synthetic_demo_values`; not real fleet logs."
        )
    st.caption("Data preparation preserves raw files in `data/raw/` strictly read-only. Cleaned outputs exist only in memory and the 3 designated model-ready CSVs.")

st.divider()

# Summarize regional orders and show delivery-status metrics and a chart.
st.subheader("Regional order records · DataCoSupplyChainDataset.csv")
if orders.empty:
    render_empty_state("The original regional/order dataset is missing or empty.")
elif not {"Order Id", "Order Region"}.issubset(orders.columns):
    render_empty_state("Regional order summary requires Order Id and Order Region columns.")
else:
    regional = orders.groupby("Order Region", as_index=False).agg(
        Source_order_rows=("Order Id", "size"),
        Distinct_orders=("Order Id", "nunique"),
    )
    if "Delivery Status" in orders:
        late_mask = orders["Delivery Status"].astype(str).str.contains("late", case=False, na=False)
        late = orders.assign(_late=late_mask).groupby("Order Region", as_index=False)["_late"].mean()
        regional = regional.merge(late, on="Order Region", how="left")
        regional["Rows with late status · %"] = regional["_late"] * 100
    a, b = st.columns([1, 1.6])
    with a:
        render_metric_card("Distinct source orders", f"{orders['Order Id'].nunique():,}", help_text="DataCo Order Id distinct count.")
        render_metric_card("Source order-item rows", f"{len(orders):,}", help_text="Rows as stored in the DataCo CSV.")
        if "Delivery Status" in orders:
            render_metric_card("Rows with late delivery status", f"{late_mask.mean() * 100:.1f}%", help_text="Calculated from DataCo Delivery Status text.")
    with b:
        if "Rows with late status · %" in regional:
            chart = px.bar(
                regional,
                x="Order Region",
                y="Rows with late status · %",
                hover_data=["Distinct_orders", "Source_order_rows"],
                title="Late delivery status by source order region",
                labels={"Order Region": "Source order region"},
            )
        else:
            chart = px.bar(
                regional,
                x="Order Region",
                y="Distinct_orders",
                hover_data=["Source_order_rows"],
                title="Distinct source orders by region",
                labels={"Order Region": "Source order region"},
            )
        chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D9E4F0"})
        st.plotly_chart(chart, width="stretch")

# Compare current stock with reorder levels and summarize suppliers and categories.
st.divider()
st.subheader("Supplier & inventory records · Retail_Supply_Chain_Analytics_Dataset.csv")
if retail.empty:
    render_empty_state("The original supplier/inventory dataset is missing or empty.")
elif not {"Current_Stock", "Reorder_Level"}.issubset(retail.columns):
    render_empty_state("Inventory comparison requires Current_Stock and Reorder_Level source columns.")
else:
    stock = pd.to_numeric(retail["Current_Stock"], errors="coerce")
    reorder = pd.to_numeric(retail["Reorder_Level"], errors="coerce")
    comparable = stock.notna() & reorder.notna()
    below = stock[comparable] <= reorder[comparable]
    inv_a, inv_b, inv_c = st.columns(3)
    with inv_a:
        render_metric_card("Source records", f"{len(retail):,}")
    with inv_b:
        render_metric_card("Records at / below reorder level", f"{int(below.sum()):,}", help_text="Rows with valid Current_Stock and Reorder_Level.")
    with inv_c:
        suppliers = retail["Supplier_ID"].nunique() if "Supplier_ID" in retail else None
        render_metric_card("Distinct source suppliers", f"{suppliers:,}" if suppliers is not None else "Unavailable")
    if "Product_Category" in retail:
        category_summary = retail.groupby("Product_Category", as_index=False).size().rename(columns={"size": "Source records"})
        chart = px.bar(category_summary, x="Product_Category", y="Source records", title="Retail source records by product category")
        chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D9E4F0"})
        st.plotly_chart(chart, width="stretch")

# Show fuel record counts, average recorded fuel, and vehicle-type comparisons.
st.divider()
st.subheader("Fuel consumption records · original fuel CSV")
if fuel.empty:
    render_empty_state("The original fuel dataset is missing or empty.")
else:
    if "added_data_status" in fuel and fuel["added_data_status"].notna().any():
        status = ", ".join(sorted(fuel["added_data_status"].dropna().astype(str).unique()))
    fuel_a, fuel_b, fuel_c = st.columns(3)
    with fuel_a:
        render_metric_card("Source fuel records", f"{len(fuel):,}")
    with fuel_b:
        mean_consumed = pd.to_numeric(fuel["fuel_consumed_liters"], errors="coerce").mean() if "fuel_consumed_liters" in fuel else None
        render_metric_card("Mean recorded fuel", f"{mean_consumed:.2f} L" if pd.notna(mean_consumed) else "Unavailable")
    with fuel_c:
        vehicle_count = fuel["vehicle_type"].nunique() if "vehicle_type" in fuel else None
        render_metric_card("Source vehicle types", f"{vehicle_count:,}" if vehicle_count is not None else "Unavailable")
    if {"vehicle_type", "fuel_consumed_liters"}.issubset(fuel.columns):
        summary = fuel.assign(fuel_consumed_liters=pd.to_numeric(fuel["fuel_consumed_liters"], errors="coerce"))
        summary = summary.groupby("vehicle_type", as_index=False)["fuel_consumed_liters"].mean()
        chart = px.bar(summary, x="vehicle_type", y="fuel_consumed_liters", title="Mean recorded fuel consumed by source vehicle type", labels={"vehicle_type": "Source vehicle type", "fuel_consumed_liters": "Mean fuel consumed · L"})
        chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D9E4F0"})
        st.plotly_chart(chart, width="stretch")

# Add navigation buttons that open the related business-module pages.
st.divider()
st.subheader("Business modules")
module_a, module_b, module_c = st.columns(3)
with module_a:
    st.markdown("#### Regional Order Records")
    st.caption("Filter actual orders by source geography and delivery fields.")
    if st.button("Explore Regional Risk", type="primary", width="stretch"):
        st.switch_page("pages/02_Regional_Risk.py")
with module_b:
    st.markdown("#### Supplier & Inventory")
    st.caption("Inspect actual supplier, product, and inventory fields.")
    if st.button("Open Inventory Intelligence", width="stretch"):
        st.switch_page("pages/03_Inventory_Supplier.py")
with module_c:
    st.markdown("#### Fuel Consumption")
    st.caption("Explore original fuel records; no route choices are fabricated.")
    if st.button("Review Fuel Records", width="stretch"):
        st.switch_page("pages/04_Fuel_Weather_Analytics.py")
