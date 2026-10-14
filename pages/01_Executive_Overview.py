"""Executive overview of independently sourced SupplyIQ datasets."""
import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import render_disclaimer, render_empty_state, render_metric_card
from dashboard.components.navigation import render_page_header, render_sidebar_header
from src.data_loader import load_dataset


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

orders = load_dataset("supply_chain")
retail = load_dataset("retail_analytics")
fuel = load_dataset("fuel_weather")

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
        st.plotly_chart(chart, use_container_width=True)

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
        st.plotly_chart(chart, use_container_width=True)

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
        st.plotly_chart(chart, use_container_width=True)

st.divider()
st.subheader("Business modules")
module_a, module_b, module_c = st.columns(3)
with module_a:
    st.markdown("#### Regional Order Records")
    st.caption("Filter actual orders by source geography and delivery fields.")
    if st.button("Explore Regional Risk", type="primary", use_container_width=True):
        st.switch_page("pages/02_Regional_Risk.py")
with module_b:
    st.markdown("#### Supplier & Inventory")
    st.caption("Inspect actual supplier, product, and inventory fields.")
    if st.button("Open Inventory Intelligence", use_container_width=True):
        st.switch_page("pages/03_Inventory_Supplier.py")
with module_c:
    st.markdown("#### Fuel Consumption")
    st.caption("Explore original fuel records; no route choices are fabricated.")
    if st.button("Review Fuel Records", use_container_width=True):
        st.switch_page("pages/04_Fuel_Weather_Analytics.py")
