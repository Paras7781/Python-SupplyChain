"""Privacy-conscious exports of source-backed operational fields."""
import streamlit as st

from dashboard.components.cards import render_disclaimer, render_empty_state, render_metric_card
from dashboard.components.navigation import render_page_header, render_sidebar_header
from src.data_loader import load_dataset

st.set_page_config(page_title="Reports & Exports - SupplyIQ", page_icon="📄", layout="wide")
render_sidebar_header("reports")
render_page_header(
    "Reports & exports",
    "Source-backed data exports",
    "Export available operational fields from the original datasets without adding scenario or model outputs.",
)
render_disclaimer(
    "These downloads are full source extracts with selected operational columns; this page does not apply filters "
    "from other modules. The Regional Order Explorer separately exports its currently filtered order records. "
    "No manually entered scenarios or generated model fields are included."
)

reports = (
    (
        "Regional orders",
        "supply_chain",
        "supplyiq_regional_orders.csv",
        [
            "Order Id", "Order Region", "Order State", "Order City", "Order Country",
            "order date (DateOrders)", "Delivery Status", "Late_delivery_risk",
            "Shipping Mode", "Days for shipping (real)", "Days for shipment (scheduled)",
        ],
    ),
    (
        "Supplier & inventory",
        "retail_analytics",
        "supplyiq_supplier_inventory.csv",
        [
            "Product_ID", "Product_Category", "Brand", "SKU", "Current_Stock",
            "Reorder_Level", "Safety_Stock", "Supplier_ID", "Supplier_Rating",
            "Warehouse_ID", "Warehouse_Location", "Lead_Time_Days", "Units_Sold",
            "Daily_Demand", "Monthly_Demand", "Date", "On_Time_Delivery_Rate",
        ],
    ),
    (
        "Fuel consumption",
        "fuel_weather",
        "supplyiq_fuel_consumption.csv",
        [
            "timestamp", "trip_id", "vehicle_id", "vehicle_type", "fuel_type",
            "road_type", "weather_condition", "trip_distance_km", "load_weight_kg",
            "fuel_efficiency_km_per_liter", "fuel_consumed_liters",
            "fuel_price_per_liter", "fuel_cost_per_trip", "added_data_status",
        ],
    ),
)
for title, dataset_key, filename, allowed_fields in reports:
    st.subheader(title)
    source = load_dataset(dataset_key)
    if source.empty:
        render_empty_state(f"{title}: the original dataset is missing or empty.")
        continue
    fields = [field for field in allowed_fields if field in source.columns]
    if not fields:
        render_empty_state(f"{title}: none of the supported operational columns are available.")
        continue
    export = source[fields].copy()
    render_metric_card("Original source records", f"{len(export):,}", help_text="Rows from the original CSV; selected operational fields only.")
    st.download_button(
        f"Download {title.lower()} CSV",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=filename,
        mime="text/csv",
        key=f"download_{dataset_key}",
        use_container_width=True,
    )
    st.divider()

st.info("Model results, route recommendations, and AI-generated summaries are unavailable because their backend services are not connected.")
