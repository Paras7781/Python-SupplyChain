"""SupplyIQ application entry point and module launchpad."""
import streamlit as st

from config.settings import APP_ICON, APP_TITLE
from dashboard.components.cards import render_disclaimer, render_metric_card
from dashboard.components.navigation import render_page_header, render_sidebar_header
from src.data_loader import load_dataset


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)
render_sidebar_header("workspace")
render_page_header(
    "SupplyIQ",
    "Supply Chain Risk & Decision Intelligence",
    "Monitor risk → investigate problems → compare options → make decisions → export reports.",
)
    
render_disclaimer(
    "All business-facing metrics and records come from the three original CSV datasets. "
    "Prediction services and real-world route alternatives are not connected."
)

datasets = [
    ("supply_chain", "DataCo supply chain"),
    ("retail_analytics", "Retail supply chain"),
    ("fuel_weather", "Fuel records · source marked synthetic"),
]
st.subheader("Connected source datasets")
columns = st.columns(3)
for column, (dataset_key, label) in zip(columns, datasets):
    frame = load_dataset(dataset_key)
    with column:
        render_metric_card(
            label,
            "Connected" if not frame.empty else "Unavailable",
            help_text=f"{len(frame):,} rows · {len(frame.columns)} columns" if not frame.empty else "Check the existing raw CSV path.",
        )
st.caption(
    "Fuel dataset provenance: the source field added_data_status is synthetic_demo_values. "
    "These original CSV records are not validated live operations."
)

st.subheader("Business modules")
first, second, third = st.columns(3)
with first:
    st.markdown("#### 01 · Regional order risk")
    st.caption("Drill from order region through available state/city fields to privacy-safe individual order details.")
    if st.button("Open regional order explorer", type="primary", use_container_width=True):
        st.switch_page("pages/02_Regional_Risk.py")
with second:
    st.markdown("#### 02 · Supplier & inventory")
    st.caption("Compare source supplier measures, dated demand records, and stock against recorded reorder levels.")
    if st.button("Open supplier intelligence", use_container_width=True):
        st.switch_page("pages/03_Inventory_Supplier.py")
with third:
    st.markdown("#### 03 · Fuel & transport")
    st.caption("Review source fuel-consumption records and transparently calculated trip estimates.")
    if st.button("Open fuel planner", use_container_width=True):
        st.switch_page("pages/04_Fuel_Weather_Analytics.py")

st.caption("Prediction, scenario simulation, and AI features remain unavailable until their backend services are connected.")
