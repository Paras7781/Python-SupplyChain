"""Privacy-conscious exports of source-backed operational fields and predictive assessment reports."""
from datetime import datetime, timezone
import streamlit as st
import pandas as pd

from ui.cards import render_disclaimer, render_empty_state, render_metric_card
from ui.navigation import render_page_header, render_sidebar_header
from shared.api_client import get_api_client
from shared.data_loader import load_dataset
from ml.regional_risk.model_training import get_regional_risk_metadata
from ml.inventory_supplier.model_training import get_inventory_metadata
from ml.fuel_weather.model_training import get_fuel_metadata

st.set_page_config(page_title="Reports & Exports - SupplyIQ", page_icon="📄", layout="wide")
render_sidebar_header("reports")
render_page_header(
    "Reports & exports",
    "Source-backed data and assessment exports",
    "Export operational datasets, model governance records, and point-in-time risk assessment reports as CSV.",
)
render_disclaimer(
    "REPORT TRANSPARENCY: Operational data extracts originate directly from unmodified source CSVs. "
    "Predictive assessment exports reflect verified model outputs and deterministic business-rule calculations "
    "from active sessions with full data provenance, validation methodology, and synthetic-data limitations documented."
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
        width="stretch",
    )
    st.divider()

st.subheader("Predictive Risk Assessment Reports & Combined Export")
st.markdown(
    "Download individual domain risk assessments or a consolidated Multi-Domain Assessment Report. "
    "Assessments include inputs, model predictions, rule-based threshold comparisons, cross-validation metrics, "
    "and synthetic-data provenance notices."
)


def populate_baseline_assessments():
    """Populate default baseline assessments in session state if none have been run yet."""
    client = get_api_client()
    now_iso = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # 1. Regional Delivery Risk baseline
    base_reg_payload = {
        "Days for shipment (scheduled)": 4,
        "Shipping Mode": "Standard Class",
        "Order Region": "Central America",
        "Market": "LATAM",
        "Customer Segment": "Consumer",
        "order_month": 6,
        "order_dayofweek": 2,
        "order_hour": 14,
        "is_weekend": 0,
        "Total_Units": 2,
        "Total_Order_Value": 120.0,
        "Line_Items": 1,
        "value_per_unit": 60.0,
    }
    reg_pred = client.predict_regional_risk(base_reg_payload)
    st.session_state["last_risk_assessment"] = {
        "input_data": base_reg_payload,
        "prediction": reg_pred,
        "timestamp": now_iso,
    }

    # 2. Inventory Demand & Stockout baseline
    base_inv_payload = {
        "Product_Category": "Electronics",
        "Warehouse_Location": "East",
        "Transportation_Mode": "Road",
        "Lead_Time_Days": 14,
        "Safety_Stock": 100,
        "Reorder_Level": 200,
        "Supplier_Rating": 3.5,
        "Storage_Capacity": 10000,
        "Seasonal_Demand_Index": 50.0,
    }
    inv_pred = client.predict_inventory_demand(base_inv_payload)
    st.session_state["last_inventory_assessment"] = {
        "input_data": base_inv_payload,
        "prediction": inv_pred,
        "current_stock": 250,
        "replenish_lt": 14,
        "timestamp": now_iso,
    }

    # 3. Fuel Consumption & Cost baseline
    base_fuel_payload = {
        "vehicle_type": "Light Truck",
        "fuel_type": "Diesel",
        "road_type": "Highway",
        "weather_condition": "Clear",
        "trip_distance_km": 100.0,
        "load_weight_kg": 2000.0,
        "temperature_c": 22.0,
        "wind_speed_kmh": 10.0,
        "rainfall_mm": 0.0,
    }
    fuel_pred = client.predict_fuel(base_fuel_payload, price=1.50)
    st.session_state["last_fuel_assessment"] = {
        "input_data": base_fuel_payload,
        "prediction": fuel_pred,
        "fuel_budget": 50.0,
        "timestamp": now_iso,
    }


has_assessments = any(
    key in st.session_state
    for key in ("last_risk_assessment", "last_inventory_assessment", "last_fuel_assessment")
)

if not has_assessments:
    st.info(
        "No active assessments recorded in this session yet. "
        "Run an assessment in Module 04 (Risk Assessment) or click below to populate standard baseline assessments."
    )
    if st.button("Generate Standard Baseline Assessments for Export", key="btn_gen_baseline_reports"):
        populate_baseline_assessments()
        st.rerun()

combined_records = []
now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

# 1. Regional
if "last_risk_assessment" in st.session_state:
    item = st.session_state["last_risk_assessment"]
    inp = item["input_data"]
    p = item["prediction"]
    reg_m = get_regional_risk_metadata()
    ts = item.get("timestamp", now_str)

    reg_row = {
        "Assessment_Type": "Regional Delivery Risk",
        "Timestamp": ts,
        "Input_Values": f"Region={inp.get('Order Region')}, Mode={inp.get('Shipping Mode')}, SchedDays={inp.get('Days for shipment (scheduled)')}, Value=${inp.get('Total_Order_Value', 0):.2f}, Units={inp.get('Total_Units')}",
        "Prediction_Estimate": f"Late Probability: {p.get('late_probability_pct')}%, Binary: {'Delayed' if p.get('late_delivery_prediction') else 'On-Time'}",
        "Risk_Category_Or_Threshold": f"Risk Level: {p.get('risk_level')} (Decision cutoff: 50.0% probability)",
        "Model_Information": f"{reg_m.get('model_name', 'RandomForestClassifier')} | Chronological CV ({reg_m.get('fold_count')} folds) | Acc: {reg_m.get('model_accuracy', 0)*100:.1f}%, ROC-AUC: {reg_m.get('roc_auc', 0):.3f}",
        "Limitations_And_Warnings": "Verified DataCo historical orders. Strict zero-leakage control (fulfillment metrics purged). CV score is not an untouched final-test score.",
    }
    combined_records.append(reg_row)

    with st.expander("Regional Delivery Risk Assessment Export", expanded=False):
        df_reg_export = pd.DataFrame([reg_row])
        st.dataframe(df_reg_export, width="stretch", hide_index=True)
        st.download_button(
            "Download regional delivery risk assessment CSV",
            data=df_reg_export.to_csv(index=False).encode("utf-8"),
            file_name="supplyiq_regional_risk_assessment.csv",
            mime="text/csv",
            key="dl_regional_assessment_csv",
            width="stretch",
        )

# 2. Inventory
previous_inventory_assessment = st.session_state.get("last_inventory_assessment")
if (
    previous_inventory_assessment
    and not previous_inventory_assessment["prediction"].get("is_synthetic_demonstration", False)
):
    st.session_state.pop("last_inventory_assessment")
    st.info(
        "A stored inventory assessment used the retired original-data baseline and was cleared. "
        "Run a new synthetic demand forecast before exporting its results."
    )

if "last_inventory_assessment" in st.session_state:
    item = st.session_state["last_inventory_assessment"]
    inp = item["input_data"]
    p = item["prediction"]
    cs = item.get("current_stock", 0)
    rlt = item.get("replenish_lt", 14)
    inv_m = get_inventory_metadata()
    ts = item.get("timestamp", now_str)
    d = p.get("predicted_daily_demand", 0)
    lt_d = d * rlt
    surplus = cs - lt_d
    thresh_status = f"Surplus: {surplus:+.1f} units vs {rlt}d demand ({lt_d:.1f}u)" if cs > 0 else "Stockout status unavailable (Stock=0)"

    est_label = (
        "Synthetic Demand Demonstration"
        if p.get("is_synthetic_demonstration")
        else "ML Forecast"
    )
    inv_row = {
        "Assessment_Type": "Inventory Demand & Stockout Risk",
        "Timestamp": ts,
        "Input_Values": f"Category={inp.get('Product_Category')}, WH={inp.get('Warehouse_Location')}, Transport={inp.get('Transportation_Mode')}, CurrentStock={cs}, ReorderLevel={inp.get('Reorder_Level')}, SafetyStock={inp.get('Safety_Stock')}, LeadTime={rlt}d",
        "Prediction_Estimate": f"{est_label}: {d:.2f} units/day, Demand Over {rlt}d: {lt_d:.1f} units",
        "Risk_Category_Or_Threshold": thresh_status,
        "Model_Information": f"{inv_m.get('model_name', 'HistGradientBoostingRegressor')} | {inv_m.get('validation_method', 'GroupKFold')} ({inv_m.get('fold_count')} folds) | MAE: {inv_m.get('mae', 0):.2f}, R2: {inv_m.get('r2_score', 0):.4f}",
        "Limitations_And_Warnings": "Trained on synthetic demand values; model performance does not prove real-world forecasting accuracy. Stockout evaluation uses transparent deterministic rules, not a trained classifier.",
    }
    combined_records.append(inv_row)

    with st.expander("Inventory Demand & Stockout Risk Assessment Export", expanded=False):
        df_inv_export = pd.DataFrame([inv_row])
        st.dataframe(df_inv_export, width="stretch", hide_index=True)
        st.download_button(
            "Download inventory risk assessment CSV",
            data=df_inv_export.to_csv(index=False).encode("utf-8"),
            file_name="supplyiq_inventory_risk_assessment.csv",
            mime="text/csv",
            key="dl_inventory_assessment_csv",
            width="stretch",
        )

# 3. Fuel
if "last_fuel_assessment" in st.session_state:
    item = st.session_state["last_fuel_assessment"]
    inp = item["input_data"]
    p = item["prediction"]
    budget = item.get("fuel_budget", 0)
    fuel_m = get_fuel_metadata()
    ts = item.get("timestamp", now_str)
    liters = p.get("predicted_fuel_consumed_liters", 0)
    cost = p.get("estimated_fuel_cost", 0)
    eff = inp.get("trip_distance_km", 0) / liters if liters > 0 else 0
    b_status = f"Budget ${budget:.2f} | Variance: ${cost - budget:+.2f}" if budget > 0 else "No budget specified"

    fuel_row = {
        "Assessment_Type": "Fuel Consumption & Cost",
        "Timestamp": ts,
        "Input_Values": f"Vehicle={inp.get('vehicle_type')}, Distance={inp.get('trip_distance_km')}km, Load={inp.get('load_weight_kg')}kg, Road={inp.get('road_type')}, Weather={inp.get('weather_condition')}, Price=${p.get('fuel_price_per_liter', 0):.2f}/L",
        "Prediction_Estimate": f"Fuel Consumed: {liters:.2f} L, Estimated Cost: ${cost:.2f}, Efficiency: {eff:.2f} km/L",
        "Risk_Category_Or_Threshold": b_status,
        "Model_Information": f"{fuel_m.get('model_name', 'HistGradientBoostingRegressor')} | Time-Series Split CV ({fuel_m.get('fold_count')} folds) | MAE: {fuel_m.get('mae', 0):.2f} L, R2: {fuel_m.get('r2_score', 0):.4f}",
        "Limitations_And_Warnings": "SYNTHETIC DEMO DATA WARNING: Source records marked added_data_status = synthetic_demo_values. Must validate against real fleet telemetry before live operational reliance.",
    }
    combined_records.append(fuel_row)

    with st.expander("Fuel Consumption & Cost Assessment Export", expanded=False):
        df_fuel_export = pd.DataFrame([fuel_row])
        st.dataframe(df_fuel_export, width="stretch", hide_index=True)
        st.download_button(
            "Download fuel consumption assessment CSV",
            data=df_fuel_export.to_csv(index=False).encode("utf-8"),
            file_name="supplyiq_fuel_assessment.csv",
            mime="text/csv",
            key="dl_fuel_assessment_csv",
            width="stretch",
        )

# Combined Export
if combined_records:
    df_combined_export = pd.DataFrame(combined_records)
    st.markdown("#### Combined Multi-Domain Risk Assessment Report")
    st.dataframe(df_combined_export, width="stretch", hide_index=True)
    st.download_button(
        "Download Combined Multi-Domain Assessment Report CSV",
        data=df_combined_export.to_csv(index=False).encode("utf-8"),
        file_name="supplyiq_combined_risk_assessment_report.csv",
        mime="text/csv",
        key="dl_combined_assessment_csv",
        width="stretch",
        type="primary",
    )

st.divider()
st.subheader("Model Performance & Governance Report")

model_reports = [
    {
        "Model": "Regional Delivery Risk Classifier",
        "Type": "Binary Classification",
        "Target": "Late_delivery_risk",
        "Accuracy_CV_Mean": f"{get_regional_risk_metadata().get('model_accuracy', 0)*100:.1f}%",
        "Accuracy_CV_Std": f"{get_regional_risk_metadata().get('model_accuracy_std', 0)*100:.1f}%",
        "ROC_AUC_CV_Mean": f"{get_regional_risk_metadata().get('roc_auc', 0):.3f}",
        "ROC_AUC_CV_Std": f"{get_regional_risk_metadata().get('roc_auc_std', 0):.3f}",
        "Validation_Method": get_regional_risk_metadata().get("validation_method", "unavailable"),
        "Validation_Folds": get_regional_risk_metadata().get("fold_count", "unavailable"),
        "Data_Leakage_Controlled": "Yes (Fulfillment days excluded)",
    },
    {
        "Model": "Inventory Daily Demand Regressor",
        "Type": "Regression",
        "Target": "Daily_Demand",
        "MAE_CV_Mean": f"{get_inventory_metadata().get('mae', 0):.2f}",
        "MAE_CV_Std": f"{get_inventory_metadata().get('mae_std', 0):.2f}",
        "Baseline_MAE": f"{get_inventory_metadata().get('baseline_mean_mae', 0):.2f}",
        "RMSE_CV_Mean": f"{get_inventory_metadata().get('rmse', 0):.2f}",
        "RMSE_CV_Std": f"{get_inventory_metadata().get('rmse_std', 0):.2f}",
        "R2_CV_Mean": f"{get_inventory_metadata().get('r2_score', 0):.4f}",
        "R2_CV_Std": f"{get_inventory_metadata().get('r2_score_std', 0):.4f}",
        "Validation_Method": get_inventory_metadata().get("validation_method", "unavailable"),
        "Validation_Folds": get_inventory_metadata().get("fold_count", "unavailable"),
        "Data_Leakage_Controlled": "Yes",
    },
    {
        "Model": "Fuel Consumption Regressor",
        "Type": "Regression",
        "Target": "fuel_consumed_liters",
        "MAE_CV_Mean": f"{get_fuel_metadata().get('mae', 0):.2f} L",
        "MAE_CV_Std": f"{get_fuel_metadata().get('mae_std', 0):.2f} L",
        "RMSE_CV_Mean": f"{get_fuel_metadata().get('rmse', 0):.2f} L",
        "RMSE_CV_Std": f"{get_fuel_metadata().get('rmse_std', 0):.2f} L",
        "R2_CV_Mean": f"{get_fuel_metadata().get('r2_score', 0):.4f}",
        "R2_CV_Std": f"{get_fuel_metadata().get('r2_score_std', 0):.4f}",
        "Validation_Method": get_fuel_metadata().get("validation_method", "unavailable"),
        "Validation_Folds": get_fuel_metadata().get("fold_count", "unavailable"),
        "Data_Leakage_Controlled": "Yes (Synthetic demo data disclosed)",
    },
]
df_model_report = pd.DataFrame(model_reports)
st.dataframe(df_model_report, width="stretch", hide_index=True)

st.download_button(
    "Download model governance & evaluation report CSV",
    data=df_model_report.to_csv(index=False).encode("utf-8"),
    file_name="supplyiq_model_governance_report.csv",
    mime="text/csv",
    key="download_model_report",
    width="stretch",
)
