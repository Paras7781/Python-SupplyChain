"""Interactive What-If Scenario Simulator for Supply Chain Decision Intelligence."""
import streamlit as st
import pandas as pd
import plotly.express as px

from ui.cards import render_disclaimer, render_metric_card
from ui.navigation import render_page_header, render_sidebar_header
from shared.api_client import get_api_client
from ml.fuel_weather.feature_engineering import calculate_fuel_estimate, calculate_fuel_cost_estimate

st.set_page_config(page_title="Scenario Simulator - SupplyIQ", page_icon="🧪", layout="wide")
render_sidebar_header("scenario")
render_page_header(
    "Module 05 · Decision Intelligence",
    "What-if scenario simulator",
    "Compare baseline operational scenarios against revised alternatives using validated models and deterministic arithmetic.",
)

render_disclaimer(
    "SIMULATION METHODOLOGY & NON-CAUSAL NOTICE: What-if scenarios compute outputs under user-modified feature inputs. "
    "These scenarios illustrate model sensitivity and statistical associations based on historical data; "
    "they do not imply that modifying an input parameter proves a causal guarantee in live operations. "
    "Transport simulations reflect formulas derived from synthetic demonstration records."
)

client = get_api_client()
tab1, tab2, tab3 = st.tabs([
    "🚚 Carrier & Shipping Mode Comparison",
    "⛽ Fuel Consumption & Fleet Sensitivity",
    "📦 Inventory Lead-Time & Buffer Stress Test",
])

# ----------------- TAB 1: Carrier Comparison -----------------
with tab1:
    st.subheader("Regional Delivery: Baseline vs. Modified Carrier Scenario")
    st.markdown(
        "Evaluate how altering shipping mode or scheduled transit days impacts predicted late-delivery probability."
    )

    regions = [
        "Central America", "Western Europe", "South Asia", "Southeast Asia",
        "Oceania", "West of USA", "East of USA", "South America", "Eastern Asia"
    ]
    modes = ["Standard Class", "Second Class", "First Class", "Same Day"]
    mode_scheduled_days = {"Standard Class": 4, "Second Class": 3, "First Class": 2, "Same Day": 1}

    with st.expander("Order Context (Shared across scenarios)", expanded=False):
        c_ord1, c_ord2, c_ord3 = st.columns(3)
        with c_ord1:
            sim_region = st.selectbox("Destination Region", regions, key="sim_base_reg")
        with c_ord2:
            sim_units = st.number_input("Order Units", min_value=1, max_value=50, value=2, key="sim_base_units")
        with c_ord3:
            sim_value = st.number_input("Order Value ($)", min_value=10.0, max_value=2000.0, value=150.0, step=10.0, key="sim_base_val")
        sim_market = "LATAM" if "America" in sim_region else ("Europe" if "Europe" in sim_region else "Pacific Asia")

    col_base, col_rev = st.columns(2)
    with col_base:
        st.markdown("#### Baseline Scenario")
        base_mode = st.selectbox("Baseline Shipping Mode", modes, index=0, key="sim_base_mode")
        base_sched = st.slider("Baseline Scheduled Days", min_value=1, max_value=6, value=mode_scheduled_days[base_mode], key="sim_base_sched")

    with col_rev:
        st.markdown("#### Revised / What-If Scenario")
        rev_mode = st.selectbox("Modified Shipping Mode", modes, index=2, key="sim_rev_mode")
        rev_sched = st.slider("Modified Scheduled Days", min_value=1, max_value=6, value=mode_scheduled_days[rev_mode], key="sim_rev_sched")

    if st.button("Compare Regional Delivery Scenarios", type="primary", key="btn_compare_carrier"):
        base_payload = {
            "Days for shipment (scheduled)": int(base_sched),
            "Shipping Mode": base_mode,
            "Order Region": sim_region,
            "Market": sim_market,
            "Customer Segment": "Consumer",
            "order_month": 6,
            "order_dayofweek": 2,
            "order_hour": 14,
            "is_weekend": 0,
            "Total_Units": int(sim_units),
            "Total_Order_Value": float(sim_value),
            "Line_Items": 1,
            "value_per_unit": float(sim_value) / float(sim_units),
        }
        rev_payload = {
            "Days for shipment (scheduled)": int(rev_sched),
            "Shipping Mode": rev_mode,
            "Order Region": sim_region,
            "Market": sim_market,
            "Customer Segment": "Consumer",
            "order_month": 6,
            "order_dayofweek": 2,
            "order_hour": 14,
            "is_weekend": 0,
            "Total_Units": int(sim_units),
            "Total_Order_Value": float(sim_value),
            "Line_Items": 1,
            "value_per_unit": float(sim_value) / float(sim_units),
        }

        with st.spinner("Running predictive models for both scenarios..."):
            pred_base = client.predict_regional_risk(base_payload)
            pred_rev = client.predict_regional_risk(rev_payload)

        prob_delta = pred_rev["late_probability_pct"] - pred_base["late_probability_pct"]

        st.markdown("### Scenario Comparison Results")
        kpi_b, kpi_r, kpi_d = st.columns(3)
        with kpi_b:
            render_metric_card(
                "Baseline Late Probability",
                f"{pred_base['late_probability_pct']}%",
                delta=f"Risk: {pred_base['risk_level']}",
                help_text=f"{base_mode} with {base_sched} scheduled days.",
            )
        with kpi_r:
            render_metric_card(
                "Revised Late Probability",
                f"{pred_rev['late_probability_pct']}%",
                delta=f"Risk: {pred_rev['risk_level']}",
                help_text=f"{rev_mode} with {rev_sched} scheduled days.",
            )
        with kpi_d:
            delta_label = f"{prob_delta:+.1f}% pts"
            render_metric_card(
                "Difference (Revised - Baseline)",
                delta_label,
                delta="Improved" if prob_delta < 0 else ("Unchanged" if prob_delta == 0 else "Worsened"),
                help_text="Negative indicates reduced probability of late delivery.",
            )

        if prob_delta < 0:
            st.success(
                f"Modifying shipping configuration from **{base_mode} ({base_sched} days)** to "
                f"**{rev_mode} ({rev_sched} days)** reduces predicted delay risk by **{abs(prob_delta):.1f} percentage points** "
                f"(from {pred_base['late_probability_pct']}% down to {pred_rev['late_probability_pct']}%)."
            )
        elif prob_delta > 0:
            st.warning(
                f"Modifying shipping configuration increases predicted delay risk by **{prob_delta:.1f} percentage points** "
                f"(from {pred_base['late_probability_pct']}% up to {pred_rev['late_probability_pct']}%)."
            )
        else:
            st.info("Both configurations yield equivalent predicted delay probability.")

    st.markdown("---")
    st.subheader("All Shipping Modes Sensitivity Curve")
    st.caption("Baseline multi-mode evaluation for the selected region and order value.")

    carrier_results = []
    for mode in modes:
        payload = {
            "Days for shipment (scheduled)": mode_scheduled_days[mode],
            "Shipping Mode": mode,
            "Order Region": sim_region,
            "Market": sim_market,
            "Customer Segment": "Consumer",
            "order_month": 6,
            "order_dayofweek": 2,
            "order_hour": 14,
            "is_weekend": 0,
            "Total_Units": int(sim_units),
            "Total_Order_Value": float(sim_value),
            "Line_Items": 1,
            "value_per_unit": float(sim_value) / float(sim_units),
        }
        pred = client.predict_regional_risk(payload)
        carrier_results.append({
            "Shipping Mode": mode,
            "Scheduled Days": mode_scheduled_days[mode],
            "Delay Probability (%)": pred["late_probability_pct"],
            "Risk Level": pred["risk_level"],
        })

    df_sim = pd.DataFrame(carrier_results)
    st.dataframe(df_sim, width="stretch", hide_index=True)

    chart = px.bar(
        df_sim,
        x="Shipping Mode",
        y="Delay Probability (%)",
        color="Risk Level",
        color_discrete_map={"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"},
        title=f"Late Delivery Risk by Shipping Mode for {sim_region}",
        text="Delay Probability (%)",
    )
    chart.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D9E4F0"})
    st.plotly_chart(chart, width="stretch")

# ----------------- TAB 2: Fuel Sensitivity -----------------
with tab2:
    st.subheader("Fuel Risk: Baseline vs. Modified Fleet & Route Scenario")
    st.caption("Provenance: Predictions derived from HistGradientBoostingRegressor model trained on synthetic demonstration dataset.")

    v_classes = ["Light Truck", "Van", "Heavy Truck"]
    roads = ["Highway", "Rural", "Urban"]
    weathers = ["Clear", "Fog", "Rain", "Storm"]

    with st.expander("Trip Distance & Operating Budget", expanded=False):
        sc_f1, sc_f2 = st.columns(2)
        with sc_f1:
            trip_dist = st.number_input("Trip Distance (km)", min_value=10.0, max_value=1500.0, value=250.0, step=25.0, key="sim_f_dist")
        with sc_f2:
            trip_budget = st.number_input("Trip Budget ($)", min_value=0.0, max_value=10000.0, value=120.0, step=10.0, key="sim_f_budget")

    f_base_col, f_rev_col = st.columns(2)
    with f_base_col:
        st.markdown("#### Baseline Scenario")
        base_veh = st.selectbox("Baseline Vehicle Class", v_classes, index=0, key="sim_f_base_v")
        base_load = st.number_input("Baseline Cargo Load (kg)", min_value=0.0, max_value=20000.0, value=1500.0, step=100.0, key="sim_f_base_load")
        base_road = st.selectbox("Baseline Road Type", roads, index=0, key="sim_f_base_road")
        base_weather = st.selectbox("Baseline Weather", weathers, index=0, key="sim_f_base_weather")
        base_price = st.number_input("Baseline Fuel Price ($/L)", min_value=0.5, max_value=5.0, value=1.50, step=0.05, key="sim_f_base_price")

    with f_rev_col:
        st.markdown("#### Revised / What-If Scenario")
        rev_veh = st.selectbox("Modified Vehicle Class", v_classes, index=2, key="sim_f_rev_v")
        rev_load = st.number_input("Modified Cargo Load (kg)", min_value=0.0, max_value=20000.0, value=3000.0, step=100.0, key="sim_f_rev_load")
        rev_road = st.selectbox("Modified Road Type", roads, index=0, key="sim_f_rev_road")
        rev_weather = st.selectbox("Modified Weather", weathers, index=2, key="sim_f_rev_weather")
        rev_price = st.number_input("Modified Fuel Price ($/L)", min_value=0.5, max_value=5.0, value=1.80, step=0.05, key="sim_f_rev_price")

    if st.button("Compare Fuel Scenarios", type="primary", key="btn_compare_fuel"):
        base_fuel_payload = {
            "vehicle_type": base_veh,
            "fuel_type": "Diesel",
            "road_type": base_road,
            "weather_condition": base_weather,
            "trip_distance_km": float(trip_dist),
            "load_weight_kg": float(base_load),
            "temperature_c": 20.0,
            "wind_speed_kmh": 10.0,
            "rainfall_mm": 0.0 if base_weather == "Clear" else 15.0,
        }
        rev_fuel_payload = {
            "vehicle_type": rev_veh,
            "fuel_type": "Diesel",
            "road_type": rev_road,
            "weather_condition": rev_weather,
            "trip_distance_km": float(trip_dist),
            "load_weight_kg": float(rev_load),
            "temperature_c": 20.0,
            "wind_speed_kmh": 25.0 if rev_weather in ("Storm", "Rain") else 10.0,
            "rainfall_mm": 25.0 if rev_weather in ("Storm", "Rain") else 0.0,
        }

        with st.spinner("Calculating fuel consumption estimates..."):
            pred_f_base = client.predict_fuel(base_fuel_payload, price=float(base_price))
            pred_f_rev = client.predict_fuel(rev_fuel_payload, price=float(rev_price))

        base_liters = pred_f_base["predicted_fuel_consumed_liters"]
        rev_liters = pred_f_rev["predicted_fuel_consumed_liters"]
        delta_liters = rev_liters - base_liters

        base_cost = pred_f_base.get("estimated_fuel_cost", 0.0)
        rev_cost = pred_f_rev.get("estimated_fuel_cost", 0.0)
        delta_cost = rev_cost - base_cost

        st.markdown("### Fuel Scenario Comparison Results")
        fk1, fk2, fk3 = st.columns(3)
        with fk1:
            eff_base = trip_dist / base_liters if base_liters > 0 else 0
            render_metric_card(
                "Baseline Fuel & Cost",
                f"{base_liters:.1f} L · ${base_cost:.2f}",
                delta=f"Efficiency: {eff_base:.1f} km/L",
                help_text=f"{base_veh} at ${base_price:.2f}/L",
            )
        with fk2:
            eff_rev = trip_dist / rev_liters if rev_liters > 0 else 0
            render_metric_card(
                "Revised Fuel & Cost",
                f"{rev_liters:.1f} L · ${rev_cost:.2f}",
                delta=f"Efficiency: {eff_rev:.1f} km/L",
                help_text=f"{rev_veh} at ${rev_price:.2f}/L",
            )
        with fk3:
            render_metric_card(
                "Cost Impact (Revised - Base)",
                f"{delta_cost:+.2f} USD",
                delta=f"{delta_liters:+.1f} L fuel volume",
                help_text="Positive indicates increased expenditure under revised scenario.",
            )

        if trip_budget > 0:
            if rev_cost > trip_budget:
                st.error(
                    f"⚠ **Budget Overrun**: Revised scenario cost (${rev_cost:.2f}) exceeds trip budget (${trip_budget:.2f}) "
                    f"by **${rev_cost - trip_budget:.2f}** ({((rev_cost - trip_budget) / trip_budget) * 100:.1f}% overrun)."
                )
            else:
                st.success(
                    f"Revised scenario cost (${rev_cost:.2f}) remains within trip budget (${trip_budget:.2f}) "
                    f"with **${trip_budget - rev_cost:.2f}** headroom."
                )

    st.markdown("---")
    st.subheader("Fuel Price Shock & Vehicle Class Simulator")
    st.caption("Provenance: Calculated estimates based on empirical fleet formulas.")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        f_dist_sim = st.number_input("Benchmark Trip Distance (km)", min_value=10.0, max_value=1500.0, value=250.0, step=25.0, key="bench_dist")
    with f_col2:
        bench_base_price = st.number_input("Base Fuel Price ($/L)", min_value=0.5, max_value=5.0, value=1.50, step=0.05, key="bench_price")
    with f_col3:
        price_shock_pct = st.select_slider("Simulated Fuel Price Spike (%)", options=[0, 10, 20, 30, 50, 75, 100], value=20, key="bench_shock")

    shocked_price = bench_base_price * (1.0 + price_shock_pct / 100.0)

    vehicle_specs = {
        "Light Truck": {"efficiency_km_l": 12.0, "payload_limit_kg": 2500},
        "Van": {"efficiency_km_l": 14.5, "payload_limit_kg": 1500},
        "Heavy Truck": {"efficiency_km_l": 4.5, "payload_limit_kg": 18000},
    }

    sim_fuel_rows = []
    for v_name, spec in vehicle_specs.items():
        liters = calculate_fuel_estimate(f_dist_sim, spec["efficiency_km_l"])
        cost_base = calculate_fuel_cost_estimate(liters, bench_base_price)
        cost_shock = calculate_fuel_cost_estimate(liters, shocked_price)
        diff_cost = cost_shock - cost_base

        sim_fuel_rows.append({
            "Vehicle Class": v_name,
            "Efficiency (km/L)": spec["efficiency_km_l"],
            "Fuel Consumed (L)": round(liters, 2),
            "Base Cost ($)": round(cost_base, 2),
            f"Shocked Cost (+{price_shock_pct}%) ($)": round(cost_shock, 2),
            "Cost Impact ($)": round(diff_cost, 2),
        })

    df_fuel_sim = pd.DataFrame(sim_fuel_rows)
    st.dataframe(df_fuel_sim, width="stretch", hide_index=True)

    chart_fuel = px.bar(
        df_fuel_sim,
        x="Vehicle Class",
        y=["Base Cost ($)", f"Shocked Cost (+{price_shock_pct}%) ($)"],
        barmode="group",
        title="Fuel Expenditure Impact by Vehicle Class",
    )
    chart_fuel.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font={"color": "#D9E4F0"})
    st.plotly_chart(chart_fuel, width="stretch")


# ----------------- TAB 3: Inventory Lead Time Stress -----------------
with tab3:
    st.subheader("Inventory: Baseline vs. Modified Lead-Time & Buffer Stress Test")
    st.markdown(
        "Evaluate inventory buffer resilience and stockout vulnerability under supply chain delays."
    )
    st.caption(
        "Synthetic Demand Demonstration: inventory forecasts use a model trained on synthetic demand values. "
        "This scenario comparison is illustrative and does not prove real-world forecasting accuracy."
    )

    categories = ["Electronics", "Fashion", "Food", "Health", "Home"]
    warehouses = ["East", "North", "South", "West"]
    transport_modes = ["Air", "Rail", "Road", "Sea"]

    with st.expander("Product & Facility Attributes (Shared)", expanded=False):
        ic1, ic2, ic3 = st.columns(3)
        with ic1:
            inv_cat = st.selectbox("Product Category", categories, index=0, key="sim_inv_cat")
        with ic2:
            inv_wh = st.selectbox("Warehouse Location", warehouses, index=0, key="sim_inv_wh")
        with ic3:
            inv_rating = st.slider("Supplier Rating", 1.0, 5.0, 3.5, 0.5, key="sim_inv_rating")

    inv_b_col, inv_r_col = st.columns(2)
    with inv_b_col:
        st.markdown("#### Baseline Scenario")
        base_trans = st.selectbox("Baseline Transport Mode", transport_modes, index=2, key="sim_inv_base_trans")
        base_lead = st.number_input("Baseline Lead Time (days)", min_value=1, max_value=90, value=14, key="sim_inv_base_lead")
        base_stock = st.number_input("Current Physical Stock (units)", min_value=0, max_value=10000, value=300, key="sim_inv_base_stock")
        base_safety = st.number_input("Safety Stock Buffer (units)", min_value=0, max_value=2000, value=50, key="sim_inv_base_safety")
        base_reorder = st.number_input("Reorder Level (units)", min_value=0, max_value=5000, value=200, key="sim_inv_base_reorder")

    with inv_r_col:
        st.markdown("#### Revised / Stressed Scenario")
        rev_trans = st.selectbox("Modified Transport Mode", transport_modes, index=3, key="sim_inv_rev_trans")
        lead_delay = st.slider("Simulated Supplier Delay (+ days)", min_value=0, max_value=30, value=7, key="sim_inv_delay")
        rev_lead = int(base_lead) + int(lead_delay)
        st.write(f"**Total Stressed Lead Time:** {rev_lead} days")
        rev_stock = st.number_input("Simulated Stock Level (units)", min_value=0, max_value=10000, value=int(base_stock), key="sim_inv_rev_stock")
        rev_safety = st.number_input("Modified Safety Stock (units)", min_value=0, max_value=2000, value=int(base_safety), key="sim_inv_rev_safety")

    if st.button("Compare Inventory Scenarios", type="primary", key="btn_compare_inv"):
        base_payload = {
            "Product_Category": inv_cat,
            "Warehouse_Location": inv_wh,
            "Transportation_Mode": base_trans,
            "Lead_Time_Days": int(base_lead),
            "Safety_Stock": int(base_safety),
            "Reorder_Level": int(base_reorder),
            "Supplier_Rating": float(inv_rating),
            "Storage_Capacity": 10000,
            "Seasonal_Demand_Index": 50.0,
        }
        rev_payload = {
            "Product_Category": inv_cat,
            "Warehouse_Location": inv_wh,
            "Transportation_Mode": rev_trans,
            "Lead_Time_Days": int(rev_lead),
            "Safety_Stock": int(rev_safety),
            "Reorder_Level": int(base_reorder),
            "Supplier_Rating": float(inv_rating),
            "Storage_Capacity": 10000,
            "Seasonal_Demand_Index": 50.0,
        }

        with st.spinner("Forecasting demand and calculating stock stress..."):
            pred_inv_base = client.predict_inventory_demand(base_payload)
            pred_inv_rev = client.predict_inventory_demand(rev_payload)

        d_base = pred_inv_base["predicted_daily_demand"]
        d_rev = pred_inv_rev["predicted_daily_demand"]

        lt_demand_base = d_base * base_lead
        lt_demand_rev = d_rev * rev_lead

        surplus_base = base_stock - lt_demand_base
        surplus_rev = rev_stock - lt_demand_rev

        cover_base = base_stock / d_base if d_base > 0 else float("inf")
        cover_rev = rev_stock / d_rev if d_rev > 0 else float("inf")

        st.markdown("### Inventory Stress Comparison Results")
        ik1, ik2, ik3 = st.columns(3)
        with ik1:
            render_metric_card(
                "Baseline Lead-Time Demand",
                f"{lt_demand_base:.0f} units",
                delta=f"Surplus: {surplus_base:+.0f} units",
                help_text=f"Over {base_lead} days (Daily demand: {d_base:.1f} u/d)",
            )
        with ik2:
            render_metric_card(
                "Stressed Lead-Time Demand",
                f"{lt_demand_rev:.0f} units",
                delta=f"Surplus: {surplus_rev:+.0f} units",
                help_text=f"Over {rev_lead} days (+{lead_delay}d delay, Daily: {d_rev:.1f} u/d)",
            )
        with ik3:
            delta_surplus = surplus_rev - surplus_base
            render_metric_card(
                "Buffer Impact (Revised - Base)",
                f"{delta_surplus:+.0f} units",
                delta="Buffer Depleted" if delta_surplus < 0 else "Buffer Improved",
                help_text="Change in net inventory surplus after replenishment window.",
            )

        if surplus_rev < 0:
            deficit = abs(surplus_rev)
            st.error(
                f"⚠ **Stockout Hazard under Delay**: Under a {lead_delay}-day vendor delay ({rev_lead} days total), "
                f"available stock ({rev_stock} units) falls short of forecasted demand ({lt_demand_rev:.0f} units). "
                f"**Projected shortfall: {deficit:.0f} units.** Expedited replenishment recommended: **{deficit + rev_safety:.0f} units** (including safety buffer)."
            )
        else:
            st.success(
                f"Stock buffer of {rev_stock} units is sufficient to absorb the {lead_delay}-day vendor delay "
                f"with a remaining surplus of {surplus_rev:.0f} units ({cover_rev:.1f} days of cover)."
            )
