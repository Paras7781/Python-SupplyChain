"""Unified Risk Assessment — Regional Delivery, Inventory, and Fuel."""
import streamlit as st
import pandas as pd
import math

from ui.cards import render_disclaimer, render_metric_card, render_smart_alert
from ui.navigation import render_page_header, render_sidebar_header
from shared.api_client import get_api_client
from shared.data_loader import load_dataset
from ml.regional_risk.model_training import (
    get_regional_risk_metadata,
    is_regional_risk_model_available,
    load_regional_risk_model,
)
from ml.inventory_supplier.model_training import (
    get_inventory_metadata,
    is_inventory_model_available,
    load_inventory_model,
)
from ml.fuel_weather.model_training import (
    get_fuel_metadata,
    is_fuel_model_available,
    load_fuel_model,
)
from ml.explainability import (
    explain_regional_risk_prediction,
    get_regional_risk_global_importance,
    explain_fuel_prediction,
)


@st.cache_data
def load_cached_regional_global_importance():
    raw = load_dataset("supply_chain")
    if raw.empty:
        return None
    pipe = load_regional_risk_model()
    return get_regional_risk_global_importance(pipe, raw.head(200))


st.set_page_config(page_title="Risk Assessment - SupplyIQ", page_icon="🧭", layout="wide")
render_sidebar_header("prediction")
render_page_header(
    "Module 04 · Predictive Intelligence",
    "Risk assessment",
    "Evaluate delivery, inventory, and fuel risks using validated machine-learning models.",
)

client = get_api_client()

# ── Tabs ────────────────────────────────────────────────────────────────────
tab_regional, tab_inventory, tab_fuel = st.tabs([
    "🚚 Regional Delivery Risk",
    "📦 Inventory Risk",
    "⛽ Fuel Risk",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Regional Delivery Risk  (preserves original behaviour exactly)
# ═══════════════════════════════════════════════════════════════════════════
with tab_regional:
    render_disclaimer(
        "MODEL PROVENANCE: Trained on verified DataCo historical orders. "
        "Strict zero-leakage controls: actual shipping days, shipping difference, and post-fulfillment delivery status "
        "are strictly excluded from the model so it only evaluates variables available at order placement."
    )

    if not is_regional_risk_model_available():
        st.error(
            "Regional risk model artifacts are missing. Run "
            "'python scripts/train_regional_risk_model.py' to train the model."
        )
    else:
        try:
            load_regional_risk_model()
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

        metadata = get_regional_risk_metadata()

        st.caption(
            f"Model metrics are {metadata.get('fold_count', 'cross-validation')}-fold "
            f"cross-validation estimates ({metadata.get('validation_method', 'validation method unavailable')}); "
            "they are not an untouched final-test score."
        )

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            render_metric_card("CV Model Accuracy", f"{metadata.get('model_accuracy', 0)*100:.1f}%", help_text=f"Baseline Dummy CV estimate: {metadata.get('baseline_accuracy', 0)*100:.1f}%")
        with col_m2:
            render_metric_card("Precision (Late)", f"{metadata.get('precision', 0)*100:.1f}%", help_text="Proportion of predicted delays that were truly delayed.")
        with col_m3:
            render_metric_card("Recall (Late)", f"{metadata.get('recall', 0)*100:.1f}%", help_text="Proportion of actual delays detected.")
        with col_m4:
            render_metric_card("ROC-AUC Score", f"{metadata.get('roc_auc', 0):.3f}", help_text="Area under the Receiver Operating Characteristic curve.")

        st.markdown("---")
        st.subheader("Simulate order intake parameters")

        regions = [
            "Central America", "Western Europe", "South Asia", "Oceania",
            "Southeast Asia", "Eastern Asia", "West of USA", "East of USA",
            "South America", "Southern Europe", "North Africa", "Middle East",
        ]
        shipping_modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
        markets = ["LATAM", "Europe", "Pacific Asia", "USCA", "Africa"]
        segments = ["Consumer", "Corporate", "Home Office"]

        form_col1, form_col2 = st.columns(2)
        with form_col1:
            selected_region = st.selectbox("Destination Region", regions, key="rr_region")
            selected_mode = st.selectbox("Shipping Mode", shipping_modes, key="rr_mode")
            scheduled_days = st.slider("Days for shipment (scheduled)", min_value=1, max_value=6, value=4, key="rr_sched")
            selected_market = st.selectbox("Market Area", markets, key="rr_market")

        with form_col2:
            selected_segment = st.selectbox("Customer Segment", segments, key="rr_segment")
            total_units = st.number_input("Total Order Units", min_value=1, max_value=50, value=2, key="rr_units")
            order_val = st.number_input("Total Order Value ($)", min_value=5.0, max_value=2000.0, value=120.0, step=10.0, key="rr_val")
            order_month = st.selectbox("Order Placement Month", list(range(1, 13)), index=4, format_func=lambda m: pd.to_datetime(f"2024-{m:02d}-01").strftime("%B"), key="rr_month")

        if st.button("Evaluate Delivery Delay Risk", type="primary", key="rr_predict"):
            order_payload = {
                "Days for shipment (scheduled)": int(scheduled_days),
                "Shipping Mode": selected_mode,
                "Order Region": selected_region,
                "Market": selected_market,
                "Customer Segment": selected_segment,
                "order_month": int(order_month),
                "order_dayofweek": 2,
                "order_hour": 14,
                "is_weekend": 0,
                "Total_Units": int(total_units),
                "Total_Order_Value": float(order_val),
                "Line_Items": 1,
                "value_per_unit": float(order_val) / float(total_units),
            }

            with st.spinner("Running inference pipeline..."):
                prediction = client.predict_regional_risk(order_payload)

            st.session_state["last_risk_assessment"] = {
                "input_data": order_payload,
                "prediction": prediction,
            }

        assessment = st.session_state.get("last_risk_assessment")
        if assessment:
            order_payload = assessment["input_data"]
            prediction = assessment["prediction"]
            st.markdown("### Risk Assessment Result")
            res_c1, res_c2, res_c3 = st.columns(3)
            with res_c1:
                render_metric_card(
                    "Delay Risk Level",
                    prediction["risk_level"],
                    help_text="Classified based on model late probability threshold.",
                )
            with res_c2:
                render_metric_card(
                    "Late Delivery Probability",
                    f"{prediction['late_probability_pct']}%",
                    help_text="Calculated from model predict_proba output.",
                )
            with res_c3:
                status_text = "Predicted Delayed" if prediction["late_delivery_prediction"] else "On-Time Delivery Expected"
                render_metric_card(
                    "Binary Classification",
                    status_text,
                    help_text="Positive label = Late Delivery Risk.",
                )

            st.progress(float(prediction["late_probability"]))

            if prediction["risk_level"] == "High":
                render_smart_alert(
                    title="Elevated Late Delivery Exposure",
                    trigger=f"Model predicted late probability is {prediction['late_probability_pct']}% (Risk Level: High)",
                    threshold="High risk classification threshold (probability >= 70%)",
                    why_it_matters=f"Orders to {order_payload.get('Order Region')} via {order_payload.get('Shipping Mode')} have high historical delay frequency, risking customer dissatisfaction and fulfillment SLA breach.",
                    next_step=f"Upgrade shipping mode to First Class or Same Day, or increase scheduled buffer from {order_payload.get('Days for shipment (scheduled)')} days.",
                    severity="error",
                    is_model_prediction=True,
                )
            elif prediction["risk_level"] == "Medium":
                render_smart_alert(
                    title="Moderate Delay Vulnerability",
                    trigger=f"Model predicted late probability is {prediction['late_probability_pct']}% (Risk Level: Medium)",
                    threshold="Medium risk threshold (50% <= probability < 70%)",
                    why_it_matters=f"Transit schedule for {order_payload.get('Shipping Mode')} to {order_payload.get('Order Region')} is sensitive to carrier capacity and dispatch timing.",
                    next_step="Monitor dispatch closely or select Second Class / First Class for time-sensitive cargo.",
                    severity="warning",
                    is_model_prediction=True,
                )
            else:
                render_smart_alert(
                    title="Low Delivery Delay Risk",
                    trigger=f"Model predicted late probability is {prediction['late_probability_pct']}% (Risk Level: Low)",
                    threshold="Low risk classification threshold (probability < 50%)",
                    why_it_matters=f"Scheduled transit window of {order_payload.get('Days for shipment (scheduled)')} days via {order_payload.get('Shipping Mode')} provides adequate historical buffer.",
                    next_step="Maintain standard shipping schedule and carrier allocation.",
                    severity="success",
                    is_model_prediction=True,
                )

            with st.expander("🔍 Why this prediction? (Feature Attribution & Explainability)", expanded=True):
                st.markdown(
                    "This section provides transparent machine learning interpretability via **SHAP (SHapley Additive exPlanations)**, "
                    "measuring how each order attribute shifts the predicted late delivery probability relative to the baseline average."
                )

                subtab_local, subtab_global = st.tabs(["Order Attribution (Individual)", "Global Feature Importance"])
                with subtab_local:
                    with st.spinner("Calculating SHAP feature attribution..."):
                        shap_res = explain_regional_risk_prediction(load_regional_risk_model(), order_payload)

                    if shap_res.get("success"):
                        st.plotly_chart(shap_res["fig"], use_container_width=True)
                        st.markdown("#### Key Drivers Identified by Model:")
                        for narrative in shap_res["narratives"]:
                            st.markdown(f"- {narrative}")
                        st.caption(f"ℹ️ {shap_res['disclaimer']}")
                    else:
                        st.info(f"Feature attribution notice: {shap_res.get('error', 'SHAP summary unavailable.')}")
                        st.markdown(
                            f"- **Shipping Mode**: `{order_payload.get('Shipping Mode')}` (Carriers with shorter promised windows typically require premium handling).\n"
                            f"- **Scheduled Days**: `{order_payload.get('Days for shipment (scheduled)')}` days (Tighter scheduled windows increase delay likelihood).\n"
                            f"- **Destination Region**: `{order_payload.get('Order Region')}` (Historical cross-docking and customs clearance times vary by region).\n"
                            f"- **Market Area**: `{order_payload.get('Market')}` | Customer Segment: `{order_payload.get('Customer Segment')}`\n"
                            f"- **Order Value**: `${order_payload.get('Total_Order_Value', 0):.2f}` across `{order_payload.get('Total_Units', 1)}` units (${order_payload.get('value_per_unit', 0):.2f}/unit)."
                        )

                with subtab_global:
                    st.caption("Average absolute feature impact (|SHAP|) across historical order intake records.")
                    global_shap = load_cached_regional_global_importance()
                    if global_shap and global_shap.get("success"):
                        st.plotly_chart(global_shap["fig"], use_container_width=True)
                    else:
                        st.info("Global SHAP feature summary unavailable.")

            st.caption("Provenance: Model Prediction · RandomForestClassifier · No Future-Information Leakage")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Inventory Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════
with tab_inventory:
    st.subheader("Demand Demonstration")
    render_disclaimer(
        "This model was trained on demand values. Its performance demonstrates that the model can learn "
        "patterns in those generated values; it does not prove real-world demand forecasting accuracy."
    )

    if not is_inventory_model_available():
        st.error(
            "Inventory demand model artifacts are missing. "
            "Ensure the model has been trained before running inference."
        )
    else:
        try:
            load_inventory_model()
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

        inv_meta = get_inventory_metadata()
        st.caption(
            f"Verified {inv_meta.get('fold_count', 5)}-fold "
            f"{inv_meta.get('validation_method', 'GroupKFold')} metrics: "
            f"R² = {inv_meta['r2_score']:.4f}, MAE = {inv_meta['mae']:.2f} units/day, "
            f"RMSE = {inv_meta['rmse']:.2f} units/day."
        )
        col_im1, col_im2, col_im3 = st.columns(3)
        with col_im1:
            render_metric_card(
                "CV MAE (Inventory)",
                f"{inv_meta['mae']:.2f} units/day",
                help_text="Mean absolute error across inventory demand validation folds.",
            )
        with col_im2:
            render_metric_card(
                "CV RMSE",
                f"{inv_meta['rmse']:.2f} units/day",
                help_text="Root mean squared error across inventory demand validation folds.",
            )
        with col_im3:
            render_metric_card(
                "CV R² Score",
                f"{inv_meta['r2_score']:.4f}",
                help_text="Cross-validation estimate on inventory demand values.",
            )

        st.markdown("---")
        st.subheader("Product & supplier parameters")

        # Model features
        product_categories = ["Electronics", "Fashion", "Food", "Health", "Home"]
        warehouse_locations = ["East", "North", "South", "West"]
        transport_modes = ["Air", "Rail", "Road", "Sea"]

        inv_col1, inv_col2 = st.columns(2)
        with inv_col1:
            inv_category = st.selectbox("Product Category", product_categories, key="inv_cat")
            inv_warehouse = st.selectbox("Warehouse Location", warehouse_locations, key="inv_wh")
            inv_transport = st.selectbox("Transportation Mode", transport_modes, key="inv_trans")
            inv_lead_time = st.number_input("Lead Time (days)", min_value=1, max_value=90, value=14, key="inv_lt")
            inv_supplier_rating = st.slider("Supplier Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.5, key="inv_sr")

        with inv_col2:
            inv_safety_stock = st.number_input("Safety Stock (units)", min_value=0, max_value=500, value=100, key="inv_ss")
            inv_reorder_level = st.number_input("Reorder Level (units)", min_value=0, max_value=1000, value=200, key="inv_rl")
            inv_storage_cap = st.number_input("Storage Capacity (units)", min_value=100, max_value=100000, value=10000, key="inv_sc")
            inv_seasonal_idx = st.slider("Seasonal Demand Index", min_value=0.0, max_value=100.0, value=50.0, step=1.0, key="inv_si")

        st.markdown("---")
        st.subheader("Optional: stockout risk inputs")
        st.caption(
            "Provide your current stock level to receive a stockout risk assessment. "
            "Without this, only the demand forecast is shown."
        )
        inv_stock_col1, inv_stock_col2 = st.columns(2)
        with inv_stock_col1:
            inv_current_stock = st.number_input(
                "Current Stock on Hand (units)", min_value=0, max_value=100000, value=0,
                help="Set to 0 if unknown. Stockout analysis requires a positive value.",
                key="inv_cs",
            )
        with inv_stock_col2:
            inv_replenish_lt = st.number_input(
                "Replenishment Lead Time (days)", min_value=1, max_value=120, value=14,
                help="How many days until the next replenishment arrives.",
                key="inv_rlt",
            )

        if st.button("Forecast Demand & Assess Inventory Risk", type="primary", key="inv_predict"):
            inv_payload = {
                "Product_Category": inv_category,
                "Warehouse_Location": inv_warehouse,
                "Transportation_Mode": inv_transport,
                "Lead_Time_Days": inv_lead_time,
                "Safety_Stock": inv_safety_stock,
                "Reorder_Level": inv_reorder_level,
                "Supplier_Rating": float(inv_supplier_rating),
                "Storage_Capacity": inv_storage_cap,
                "Seasonal_Demand_Index": float(inv_seasonal_idx),
            }

            with st.spinner("Running demand forecast..."):
                inv_result = client.predict_inventory_demand(inv_payload)

            st.session_state["last_inventory_assessment"] = {
                "input_data": inv_payload,
                "prediction": inv_result,
                "current_stock": inv_current_stock,
                "replenish_lt": inv_replenish_lt,
            }

        inv_assessment = st.session_state.get("last_inventory_assessment")
        if (
            inv_assessment
            and not inv_assessment["prediction"].get("is_synthetic_demonstration", False)
        ):
            st.session_state.pop("last_inventory_assessment")
            st.info(
                "A previous assessment used the retired original-data baseline and was cleared. "
                "Run a new forecast to create an inventory demonstration assessment."
            )
            inv_assessment = None
        if inv_assessment:
            inv_pred = inv_assessment["prediction"]
            predicted_demand = inv_pred["predicted_daily_demand"]
            current_stock = inv_assessment["current_stock"]
            replenish_lt = inv_assessment["replenish_lt"]

            st.markdown("### Demand Forecast Result")
            st.caption(
                "Inventory Demand Demonstration: trained on inventory demand values; this prediction does not prove "
                "real-world forecasting accuracy."
            )
            inv_r1, inv_r2, inv_r3 = st.columns(3)
            with inv_r1:
                render_metric_card(
                    "Predicted Daily Demand (Inventory ML)",
                    f"{predicted_demand:.2f} units/day",
                    help_text="HistGradientBoostingRegressor regression prediction trained on inventory demand values.",
                )
            with inv_r2:
                demand_over_lt = predicted_demand * replenish_lt
                render_metric_card(
                    f"Projected Demand ({replenish_lt} Days)",
                    f"{demand_over_lt:.1f} units",
                    help_text=f"Predicted daily demand ({predicted_demand:.1f}) × replenishment lead time ({replenish_lt} days).",
                )
            with inv_r3:
                baseline_med = inv_pred.get("baseline_median_demand", 203.0)
                render_metric_card(
                    "Inventory Median (Reference)",
                    f"{baseline_med:.2f} units/day",
                    help_text="Training-sample median for the inventory demonstration data.",
                )

            # Stockout risk assessment (transparent business rules, not a trained classifier)
            if current_stock > 0:
                st.markdown("### Stockout Risk Assessment (Deterministic Rules)")
                st.caption(
                    "This assessment compares your reported stock against projected lead-time demand using deterministic inventory arithmetic: "
                    "`Surplus = Current Stock - (Daily Demand × Lead Time)`. **Assumptions:** constant demand, zero in-transit purchase orders, "
                    "deterministic lead time. It is not an unassisted purchasing recommendation."
                )

                if predicted_demand > 0:
                    days_of_cover = current_stock / predicted_demand
                else:
                    days_of_cover = float("inf")

                stock_surplus = current_stock - demand_over_lt

                so_c1, so_c2, so_c3 = st.columns(3)
                with so_c1:
                    cover_display = f"{days_of_cover:.1f} days" if math.isfinite(days_of_cover) else "∞"
                    render_metric_card(
                        "Days of Stock Cover",
                        cover_display,
                        help_text="Current stock ÷ predicted daily demand.",
                    )
                with so_c2:
                    render_metric_card(
                        "Stock vs Lead-Time Demand",
                        f"{stock_surplus:+.1f} units",
                        help_text="Positive = surplus after replenishment window; negative = shortfall.",
                    )
                with so_c3:
                    below_reorder = current_stock < inv_assessment["input_data"]["Reorder_Level"]
                    render_metric_card(
                        "Below Reorder Level?",
                        "Yes — reorder now" if below_reorder else "No",
                        help_text=f"Reorder level: {inv_assessment['input_data']['Reorder_Level']} units.",
                    )

                if stock_surplus < 0:
                    render_smart_alert(
                        title="Stockout Risk Identified",
                        trigger=f"Current stock ({current_stock:,} units) is below projected lead-time demand ({demand_over_lt:,.1f} units)",
                        threshold=f"Zero-surplus threshold (Estimated shortfall: {abs(stock_surplus):,.1f} units over {replenish_lt} days)",
                        why_it_matters="Inventory will be depleted before replenishment arrives, resulting in stockouts and unfulfilled customer demand.",
                        next_step=f"Place an expedited replenishment order for at least {abs(stock_surplus) + inv_assessment['input_data']['Safety_Stock']:,.0f} units.",
                        severity="error",
                        is_model_prediction=False,
                    )
                elif days_of_cover < replenish_lt * 1.2:
                    render_smart_alert(
                        title="Low Inventory Buffer Warning",
                        trigger=f"Stock cover ({days_of_cover:.1f} days) is within 20% of supplier lead time ({replenish_lt} days)",
                        threshold=f"Safety buffer cutoff ({replenish_lt * 1.2:.1f} days)",
                        why_it_matters="A minor delay in supplier delivery or unexpected demand uptick will cause a stockout.",
                        next_step="Initiate standard reorder cycle now to prevent buffer depletion.",
                        severity="warning",
                        is_model_prediction=False,
                    )
                else:
                    render_smart_alert(
                        title="Sufficient Inventory Cover",
                        trigger=f"Stock cover of {days_of_cover:.1f} days comfortably exceeds replenishment window ({replenish_lt} days)",
                        threshold=f"Surplus buffer of {stock_surplus:,.1f} units above lead-time demand",
                        why_it_matters="Warehouse inventory is adequately buffered against normal supplier lead-time variations.",
                        next_step="Maintain current replenishment schedule.",
                        severity="success",
                        is_model_prediction=False,
                    )

                if below_reorder:
                    render_smart_alert(
                        title="Inventory Below Reorder Point",
                        trigger=f"Current physical stock ({current_stock:,} units) <= Reorder Level ({inv_assessment['input_data']['Reorder_Level']:,} units)",
                        threshold=f"Configured reorder point: {inv_assessment['input_data']['Reorder_Level']:,} units",
                        why_it_matters="Reorder point trigger has been breached. Immediate order placement is recommended by inventory policy.",
                        next_step="Issue purchase order to vendor.",
                        severity="warning",
                        is_model_prediction=False,
                    )
            else:
                render_smart_alert(
                    title="Stockout Risk Assessment Inactive",
                    trigger="Current Stock on Hand is set to 0 or unpopulated",
                    threshold="Positive stock quantity required for coverage calculation",
                    why_it_matters="Daily demand regression forecast is active above, but stockout deficit cannot be evaluated without current stock.",
                    next_step="Enter a positive integer in 'Current Stock on Hand' above to calculate days of cover and reorder alerts.",
                    severity="info",
                    is_model_prediction=False,
                )

            with st.expander("ℹ️ Feature Attribution & Model Explainability Notice", expanded=False):
                st.info(
                    "**Inventory Demand Demonstration**: Feature attribution illustrates patterns learned from "
                    "inventory demand values. It does not establish real-world demand drivers or forecasting accuracy."
                )

            st.caption(
                "Provenance: Demand Regression · HistGradientBoostingRegressor · Inventory Demonstration Data"
            )


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Fuel Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════
with tab_fuel:
    render_disclaimer(
        "INVENTORY DATA WARNING: The fuel & weather model was trained on inventory demonstration data "
        "(supplyiq_final_dataset_filled_inventory_fuel_weather.csv). Predictions require validation "
        "against real fleet data before any operational reliance. Cost estimates use user-supplied fuel "
        "prices and are not model outputs."
    )

    if not is_fuel_model_available():
        st.error(
            "Fuel consumption model artifacts are missing. Run "
            "'python scripts/train_all_models.py' to train all models."
        )
    else:
        try:
            load_fuel_model()
        except (FileNotFoundError, RuntimeError) as exc:
            st.error(str(exc))
            st.stop()

        fuel_meta = get_fuel_metadata()

        st.caption(
            f"Regression metrics are {fuel_meta.get('fold_count', 'cross-validation')}-fold "
            f"cross-validation estimates ({fuel_meta.get('validation_method', 'validation method unavailable')}). "
            f"R² = {fuel_meta.get('r2_score', 0):.4f}. "
            f"Data provenance: {fuel_meta.get('data_provenance', 'inventory')}."
        )

        col_fm1, col_fm2, col_fm3 = st.columns(3)
        with col_fm1:
            render_metric_card("CV MAE", f"{fuel_meta.get('mae', 0):.2f} L",
                               help_text=f"Baseline Dummy MAE: {fuel_meta.get('baseline_mae', 0):.2f} L")
        with col_fm2:
            render_metric_card("CV RMSE", f"{fuel_meta.get('rmse', 0):.2f} L",
                               help_text="Root Mean Squared Error across validation folds.")
        with col_fm3:
            render_metric_card("CV R² Score", f"{fuel_meta.get('r2_score', 0):.4f}",
                               help_text="Coefficient of determination; 1.0 is perfect.")

        st.markdown("---")
        st.subheader("Trip parameters")

        vehicle_types = ["Heavy Truck", "Light Truck", "Van"]
        fuel_types = ["Diesel"]
        road_types = ["Highway", "Rural", "Urban"]
        weather_conditions = ["Clear", "Fog", "Rain", "Storm"]

        fuel_col1, fuel_col2 = st.columns(2)
        with fuel_col1:
            fuel_vehicle = st.selectbox("Vehicle Type", vehicle_types, key="fuel_vt")
            fuel_fuel_type = st.selectbox("Fuel Type", fuel_types, key="fuel_ft")
            fuel_road = st.selectbox("Road Type", road_types, key="fuel_rt")
            fuel_weather = st.selectbox("Weather Condition", weather_conditions, key="fuel_wc")
            fuel_distance = st.number_input("Trip Distance (km)", min_value=1.0, max_value=1500.0, value=100.0, step=10.0, key="fuel_dist")

        with fuel_col2:
            fuel_load = st.number_input("Load Weight (kg)", min_value=0.0, max_value=20000.0, value=2000.0, step=100.0, key="fuel_load")
            fuel_temp = st.number_input("Temperature (°C)", min_value=-30.0, max_value=55.0, value=22.0, step=1.0, key="fuel_temp")
            fuel_wind = st.number_input("Wind Speed (km/h)", min_value=0.0, max_value=100.0, value=10.0, step=1.0, key="fuel_wind")
            fuel_rain = st.number_input("Rainfall (mm)", min_value=0.0, max_value=100.0, value=0.0, step=1.0, key="fuel_rain")
            fuel_price = st.number_input("Fuel Price (USD/litre)", min_value=0.0, max_value=10.0, value=1.50, step=0.10, key="fuel_price",
                                         help="Used only for cost estimation; not a model feature.")

        st.markdown("---")
        st.subheader("Optional: budget comparison")
        st.caption("Provide a trip fuel budget to see whether the estimated cost exceeds it.")
        fuel_budget = st.number_input(
            "Trip Fuel Budget (USD)", min_value=0.0, max_value=50000.0, value=0.0, step=10.0,
            help="Set to 0 to skip budget comparison.",
            key="fuel_budget",
        )

        if st.button("Estimate Fuel Consumption", type="primary", key="fuel_predict"):
            fuel_payload = {
                "vehicle_type": fuel_vehicle,
                "fuel_type": fuel_fuel_type,
                "road_type": fuel_road,
                "weather_condition": fuel_weather,
                "trip_distance_km": float(fuel_distance),
                "load_weight_kg": float(fuel_load),
                "temperature_c": float(fuel_temp),
                "wind_speed_kmh": float(fuel_wind),
                "rainfall_mm": float(fuel_rain),
            }
            price_arg = float(fuel_price) if fuel_price > 0 else None

            with st.spinner("Running fuel consumption model..."):
                fuel_result = client.predict_fuel(fuel_payload, price=price_arg)

            st.session_state["last_fuel_assessment"] = {
                "input_data": fuel_payload,
                "prediction": fuel_result,
                "fuel_budget": fuel_budget,
            }

        fuel_assessment = st.session_state.get("last_fuel_assessment")
        if fuel_assessment:
            fuel_pred = fuel_assessment["prediction"]
            pred_liters = fuel_pred["predicted_fuel_consumed_liters"]
            budget = fuel_assessment["fuel_budget"]

            st.markdown("### Fuel Consumption Estimate")
            fuel_r_cols = st.columns(3 if "estimated_fuel_cost" in fuel_pred else 2)
            with fuel_r_cols[0]:
                render_metric_card(
                    "Predicted Fuel Consumption",
                    f"{pred_liters:.2f} litres",
                    help_text="Regression estimate from the fuel consumption model.",
                )
            with fuel_r_cols[1]:
                efficiency = fuel_assessment["input_data"]["trip_distance_km"] / pred_liters if pred_liters > 0 else 0
                render_metric_card(
                    "Estimated Efficiency",
                    f"{efficiency:.2f} km/L",
                    help_text="Trip distance ÷ predicted fuel consumption.",
                )
            if "estimated_fuel_cost" in fuel_pred:
                with fuel_r_cols[2]:
                    render_metric_card(
                        "Estimated Fuel Cost",
                        f"${fuel_pred['estimated_fuel_cost']:.2f}",
                        help_text=f"At ${fuel_pred.get('fuel_price_per_liter', 0):.2f}/L.",
                    )

            # Budget comparison (transparent arithmetic)
            if budget > 0 and "estimated_fuel_cost" in fuel_pred:
                cost = fuel_pred["estimated_fuel_cost"]
                delta = cost - budget
                if delta > 0:
                    render_smart_alert(
                        title="Fuel Operating Budget Overrun",
                        trigger=f"Estimated fuel cost (${cost:,.2f}) exceeds trip budget (${budget:,.2f}) by ${delta:,.2f}",
                        threshold=f"User-configured budget threshold: ${budget:,.2f}",
                        why_it_matters=f"Trip represents a {delta / budget * 100:.1f}% cost overrun against allocated transport budget.",
                        next_step="Consider route optimization, lighter cargo payload, or selecting a higher-efficiency vehicle class if cargo weight permits.",
                        severity="error",
                        is_model_prediction=True,
                    )
                elif delta > -budget * 0.1:
                    render_smart_alert(
                        title="Tight Fuel Budget Margin",
                        trigger=f"Estimated fuel cost (${cost:,.2f}) is within 10% of budget (${budget:,.2f})",
                        threshold=f"Margin buffer: ${abs(delta):,.2f}",
                        why_it_matters="Adverse weather, headwind, or road congestion could push actual fuel consumption over budget.",
                        next_step="Review weather forecasts and allow operational contingency.",
                        severity="warning",
                        is_model_prediction=True,
                    )
                else:
                    render_smart_alert(
                        title="Fuel Consumption Within Budget",
                        trigger=f"Estimated fuel cost (${cost:,.2f}) is well within trip budget (${budget:,.2f})",
                        threshold=f"Budget surplus: ${abs(delta):,.2f}",
                        why_it_matters="Trip cost is economically viable within allocated operational budget.",
                        next_step="Proceed with trip planning.",
                        severity="success",
                        is_model_prediction=True,
                    )
            elif budget > 0:
                render_smart_alert(
                    title="Budget Comparison Inactive",
                    trigger="Fuel price per liter is not specified",
                    threshold="Fuel Price (USD/litre) > 0 required for financial calculation",
                    why_it_matters="Volume estimate is calculated in liters, but financial comparison against budget requires a fuel price.",
                    next_step="Set 'Fuel Price (USD/litre)' to a positive value above to view budget variance.",
                    severity="info",
                    is_model_prediction=False,
                )

            with st.expander("🔍 Why this estimate? (Fuel Consumption Drivers & SHAP Attribution)", expanded=False):
                with st.spinner("Calculating fuel feature attribution..."):
                    shap_fuel_res = explain_fuel_prediction(load_fuel_model(), fuel_assessment["input_data"])

                if shap_fuel_res.get("success"):
                    st.plotly_chart(shap_fuel_res["fig"], use_container_width=True)
                    st.markdown("#### Primary Trip Factors:")
                    for narrative in shap_fuel_res["narratives"]:
                        st.markdown(f"- {narrative}")
                    st.warning(shap_fuel_res["disclaimer"])
                else:
                    st.info(shap_fuel_res.get("error", "Fuel explainability summary unavailable."))

            st.caption(
                "Provenance: Fuel Regression · HistGradientBoostingRegressor · "
                "**Synthetic demonstration data** — validate against real fleet records before operational use."
            )
