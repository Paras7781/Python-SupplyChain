"""SupplyIQ application entry point and module launchpad."""
import streamlit as st
import pandas as pd

from config.settings import APP_ICON, APP_TITLE
from ui.cards import render_disclaimer, render_metric_card, render_smart_alert
from ui.navigation import render_page_header, render_sidebar_header
from shared.data_loader import load_dataset


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
    "Machine learning models and scenario simulators run directly in this application."
)

datasets = [
    ("supply_chain", "DataCo supply chain"),
    ("retail_analytics", "Retail supply chain"),
    ("fuel_weather", "Fuel records "),
]
st.subheader("Connected source datasets")
columns = st.columns(3)
loaded_frames = {}
for column, (dataset_key, label) in zip(columns, datasets):
    frame = load_dataset(dataset_key)
    loaded_frames[dataset_key] = frame
    with column:
        render_metric_card(
            label,
            "Connected" if not frame.empty else "Unavailable",
            help_text=f"{len(frame):,} rows · {len(frame.columns)} columns" if not frame.empty else "Check the existing raw CSV path.",
        )
st.caption(
    "Fuel dataset provenance. "
    "These original CSV records are not validated live operations."
)

# ── Operational Smart Alerts ────────────────────────────────────────────────
st.subheader("Active operational alerts")
with st.expander("Systemic Risk & Provenance Radar", expanded=True):
    alert_c1, alert_c2 = st.columns(2)
    
    # Alert 1: Regional Delivery Risk Alert
    with alert_c1:
        sc_frame = loaded_frames.get("supply_chain")
        if sc_frame is not None and not sc_frame.empty and "Delivery Status" in sc_frame.columns:
            late_mask = sc_frame["Delivery Status"].astype(str).str.contains("late", case=False, na=False)
            late_pct = late_mask.mean() * 100
            if late_pct > 50:
                render_smart_alert(
                    title="Systemic Delivery Delay Concentration",
                    trigger=f"{late_pct:.1f}% of historical order records flagged with Late Delivery status",
                    threshold="Operational SLA threshold of 50.0% late rate ceiling",
                    why_it_matters="Repeated transit delays risk customer churn, chargebacks, and buffer exhaustion in downstream hubs.",
                    next_step="Review Regional Explorer (Module 01) or test carrier upgrades in What-If Simulator (Module 05).",
                    severity="warning",
                    is_model_prediction=False,
                )
        else:
            render_smart_alert(
                title="Regional Order Data Unavailable",
                trigger="DataCo dataset unreadable or missing Delivery Status column",
                threshold="Dataset contract requirement",
                why_it_matters="Cannot evaluate baseline order delivery performance.",
                next_step="Inspect data/raw/DataCoSupplyChainDataset.csv path.",
                severity="error",
                is_model_prediction=False,
            )

    # Alert 2: Inventory Stock Shortage Alert
    with alert_c2:
        ret_frame = loaded_frames.get("retail_analytics")
        if ret_frame is not None and not ret_frame.empty and {"Current_Stock", "Reorder_Level"}.issubset(ret_frame.columns):
            stock_col = pd.to_numeric(ret_frame["Current_Stock"], errors="coerce")
            reorder_col = pd.to_numeric(ret_frame["Reorder_Level"], errors="coerce")
            valid = stock_col.notna() & reorder_col.notna()
            below_reorder = (stock_col[valid] <= reorder_col[valid]).sum()
            if below_reorder > 0:
                render_smart_alert(
                    title="Inventory Stock Shortage Exposure",
                    trigger=f"{int(below_reorder):,} SKU records currently at or below reorder threshold",
                    threshold="Business rule: Current_Stock <= Reorder_Level",
                    why_it_matters="Items at or below reorder level are vulnerable to immediate stockouts upon demand spikes or supplier lead-time delays.",
                    next_step="Open Supplier & Inventory Intelligence (Module 02) to evaluate replenishment orders.",
                    severity="warning",
                    is_model_prediction=False,
                )
        else:
            render_smart_alert(
                title="Inventory Analytics Data Unavailable",
                trigger="Retail analytics dataset missing or columns unreadable",
                threshold="Dataset contract requirement",
                why_it_matters="Cannot monitor stock replenishment thresholds.",
                next_step="Inspect data/raw/Retail_Supply_Chain_Analytics_Dataset.csv path.",
                severity="error",
                is_model_prediction=False,
            )

    # Alert 3: Synthetic Data Limitation Alert
    render_smart_alert(
        title="Fuel & Weather Analytics Synthetic Provenance Notice",
        trigger="Fuel dataset tagged with added_data_status = 'synthetic_demo_values'",
        threshold="Validation requirement: live operational deployment requires telemetry calibration",
        why_it_matters="Predictions and consumption curves demonstrate decision logic but must not be relied upon for live dispatch without field validation.",
        next_step="Use Fuel Consumption (Module 03) and Fuel What-If Simulator for structural sensitivity analysis only.",
        severity="info",
        is_model_prediction=False,
    )

st.subheader("Operational business & intelligence modules")
first, second, third = st.columns(3)
with first:
    st.markdown("#### 01 · Regional order risk")
    st.caption("Drill from order region through available state/city fields to privacy-safe individual order details.")
    if st.button("Open regional order explorer", type="primary", width="stretch"):
        st.switch_page("pages/02_Regional_Risk.py")
with second:
    st.markdown("#### 02 · Supplier & inventory")
    st.caption("Compare source supplier measures, dated demand records, and stock against recorded reorder levels.")
    if st.button("Open supplier intelligence", width="stretch"):
        st.switch_page("pages/03_Inventory_Supplier.py")
with third:
    st.markdown("#### 03 · Fuel & transport")
    st.caption("Review source fuel-consumption records and transparently calculated trip estimates.")
    if st.button("Open fuel planner", width="stretch"):
        st.switch_page("pages/04_Fuel_Weather_Analytics.py")

fourth, fifth, sixth = st.columns(3)
with fourth:
    st.markdown("#### 04 · Predictive intelligence")
    st.caption("Evaluate order-time delay risk using our validated, zero-leakage machine learning pipeline.")
    if st.button("Open risk predictor", width="stretch"):
        st.switch_page("pages/05_Risk_Prediction.py")
with fifth:
    st.markdown("#### 05 · Decision intelligence")
    st.caption("Simulate carrier options, fuel price shocks, and inventory lead-time stress tests.")
    if st.button("Open scenario simulator", width="stretch"):
        st.switch_page("pages/06_What_If_Simulator.py")
with sixth:
    st.markdown("#### 06 · Project information")
    st.caption("Review source datasets, model details, and data limitations.")
    if st.button("Open project information", width="stretch"):
        st.switch_page("pages/09_Project_Info_Model_Trust.py")
