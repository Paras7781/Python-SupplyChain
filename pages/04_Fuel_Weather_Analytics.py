"""Fuel-consumption analysis using only the original fuel/weather CSV records."""
from math import isfinite

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.cards import render_disclaimer, render_empty_state, render_metric_card
from dashboard.components.navigation import render_page_header, render_sidebar_header
from src.data_loader import load_dataset
from src.ui_planning import calculate_fuel_estimate


st.set_page_config(page_title="Fuel Consumption - SupplyIQ", page_icon="⛽", layout="wide")
render_sidebar_header("fuel")
render_page_header(
    "Module 03 · Fuel consumption",
    "Fuel & transport records",
    "Review recorded fuel usage and supported vehicle comparisons from the original fuel dataset.",
)
render_disclaimer(
    "SOURCE DATA: supplyiq_final_dataset_filled_synthetic_fuel_weather.csv. "
    "The source labels its records synthetic_demo_values. These are values present in the original CSV, "
    "not validated live operations. No real route alternatives, locations, or route recommendation are available."
)


@st.cache_data
def load_fuel_records() -> pd.DataFrame:
    source = load_dataset("fuel_weather")
    if source.empty:
        return pd.DataFrame()
    frame = source.copy()
    numeric_fields = (
        "fuel_consumption_rate",
        "trip_distance_km",
        "fuel_efficiency_km_per_liter",
        "fuel_consumed_liters",
        "fuel_price_per_liter",
        "fuel_cost_per_trip",
        "load_weight_kg",
    )
    for field in numeric_fields:
        if field in frame:
            frame[field] = pd.to_numeric(frame[field], errors="coerce")
    if "timestamp" in frame:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    return frame


fuel_data = load_fuel_records()
if fuel_data.empty:
    st.error("The original fuel dataset is missing or empty. No sample values are substituted.")
    st.stop()

st.subheader("Filter source records")
filter_a, filter_b, filter_c, filter_d = st.columns(4)
with filter_a:
    vehicle_options = (
        ["All vehicle types"] + sorted(fuel_data["vehicle_type"].dropna().astype(str).unique().tolist())
        if "vehicle_type" in fuel_data
        else ["All vehicle types"]
    )
    selected_vehicle = st.selectbox("Vehicle type", vehicle_options, help="Values are read from fuel_weather.vehicle_type.")
with filter_b:
    fuel_type_options = (
        ["All fuel types"] + sorted(fuel_data["fuel_type"].dropna().astype(str).unique().tolist())
        if "fuel_type" in fuel_data
        else ["All fuel types"]
    )
    selected_fuel_type = st.selectbox("Fuel type", fuel_type_options, help="Values are read from fuel_weather.fuel_type.")
with filter_c:
    road_options = (
        ["All road types"] + sorted(fuel_data["road_type"].dropna().astype(str).unique().tolist())
        if "road_type" in fuel_data
        else ["All road types"]
    )
    selected_road = st.selectbox("Road type", road_options, help="Values are read from fuel_weather.road_type.")
with filter_d:
    weather_options = (
        ["All weather conditions"] + sorted(fuel_data["weather_condition"].dropna().astype(str).unique().tolist())
        if "weather_condition" in fuel_data
        else ["All weather conditions"]
    )
    selected_weather = st.selectbox("Weather condition", weather_options)

filtered = fuel_data.copy()
if selected_vehicle != "All vehicle types" and "vehicle_type" in filtered:
    filtered = filtered[filtered["vehicle_type"].astype(str) == selected_vehicle]
if selected_fuel_type != "All fuel types" and "fuel_type" in filtered:
    filtered = filtered[filtered["fuel_type"].astype(str) == selected_fuel_type]
if selected_road != "All road types" and "road_type" in filtered:
    filtered = filtered[filtered["road_type"].astype(str) == selected_road]
if selected_weather != "All weather conditions" and "weather_condition" in filtered:
    filtered = filtered[filtered["weather_condition"].astype(str) == selected_weather]
if "timestamp" in filtered and filtered["timestamp"].notna().any():
    valid_dates = filtered["timestamp"].dropna()
    selected_dates = st.date_input(
        "Trip record date range",
        value=(valid_dates.min().date(), valid_dates.max().date()),
        min_value=valid_dates.min().date(),
        max_value=valid_dates.max().date(),
    )
    if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
        filtered = filtered[filtered["timestamp"].dt.date.between(*selected_dates)]
if "trip_id" in filtered:
    trip_search = st.text_input("Search source trip ID (optional)", placeholder="Type part of an actual trip ID")
    if trip_search.strip():
        filtered = filtered[filtered["trip_id"].astype(str).str.contains(trip_search.strip(), case=False, na=False)]

st.caption(f"{len(filtered):,} original fuel records match the selected filters.")
if filtered.empty:
    render_empty_state("No fuel records match these selections.")
    st.stop()

filtered = filtered.copy()
if {"fuel_cost_per_trip", "trip_distance_km"}.issubset(filtered.columns):
    valid_cost_km = (
        filtered["fuel_cost_per_trip"].notna()
        & filtered["trip_distance_km"].notna()
        & filtered["trip_distance_km"].gt(0)
        & filtered["fuel_cost_per_trip"].ge(0)
    )
    filtered["fuel_cost_per_km"] = pd.NA
    filtered.loc[valid_cost_km, "fuel_cost_per_km"] = (
        filtered.loc[valid_cost_km, "fuel_cost_per_trip"] / filtered.loc[valid_cost_km, "trip_distance_km"]
    )

metrics = st.columns(4)
with metrics[0]:
    render_metric_card("Source trip records", f"{len(filtered):,}", help_text="Filtered rows from the original fuel CSV.")
with metrics[1]:
    distance_mean = filtered["trip_distance_km"].mean() if "trip_distance_km" in filtered else None
    render_metric_card("Mean recorded distance", f"{distance_mean:.1f} km" if pd.notna(distance_mean) else "Unavailable")
with metrics[2]:
    fuel_mean = filtered["fuel_consumed_liters"].mean() if "fuel_consumed_liters" in filtered else None
    render_metric_card("Mean recorded fuel", f"{fuel_mean:.2f} L" if pd.notna(fuel_mean) else "Unavailable")
with metrics[3]:
    cost_per_km_mean = filtered["fuel_cost_per_km"].mean() if "fuel_cost_per_km" in filtered else None
    render_metric_card("Mean fuel cost / km", f"{cost_per_km_mean:.2f}" if pd.notna(cost_per_km_mean) else "Unavailable")

left, right = st.columns(2)
with left:
    st.subheader("Recorded fuel consumption")
    if {"vehicle_type", "fuel_consumed_liters"}.issubset(filtered.columns):
        chart_records = filtered.dropna(subset=["vehicle_type", "fuel_consumed_liters"])
        if not chart_records.empty:
            chart = px.box(
                chart_records,
                x="vehicle_type",
                y="fuel_consumed_liters",
                color="vehicle_type",
                points=False,
                title="Fuel consumed per original trip record",
                labels={"vehicle_type": "Source vehicle type", "fuel_consumed_liters": "Fuel consumed · L"},
            )
            chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#D9E4F0"},
                showlegend=False,
            )
            st.plotly_chart(chart, use_container_width=True)
        else:
            render_empty_state("Fuel consumption values are missing in the selected records.")
    else:
        render_empty_state("This chart requires vehicle_type and fuel_consumed_liters in the source.")

with right:
    st.subheader("Recorded distance and efficiency")
    if {"trip_distance_km", "fuel_efficiency_km_per_liter", "vehicle_type"}.issubset(filtered.columns):
        chart_records = filtered.dropna(subset=["trip_distance_km", "fuel_efficiency_km_per_liter"])
        if not chart_records.empty:
            chart = px.scatter(
                chart_records,
                x="trip_distance_km",
                y="fuel_efficiency_km_per_liter",
                color="vehicle_type",
                hover_data=[field for field in ("trip_id", "vehicle_id") if field in chart_records],
                title="Source trip distance vs. recorded efficiency",
                labels={
                    "trip_distance_km": "Trip distance · km",
                    "fuel_efficiency_km_per_liter": "Recorded efficiency · km/L",
                    "vehicle_type": "Source vehicle type",
                },
            )
            chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#D9E4F0"},
            )
            st.plotly_chart(chart, use_container_width=True)
        else:
            render_empty_state("Distance and efficiency values are missing in the selected records.")
    else:
        missing = sorted({"trip_distance_km", "fuel_efficiency_km_per_liter", "vehicle_type"} - set(filtered.columns))
        render_empty_state("Chart unavailable; missing source fields: " + ", ".join(missing))

st.subheader("Fuel cost per kilometre")
if "fuel_cost_per_km" in filtered:
    valid_cost = filtered["fuel_cost_per_km"].dropna()
    if valid_cost.empty:
        st.info("Fuel cost per kilometre is unavailable because the selected source rows are missing valid trip distance or fuel cost values.")
    else:
        st.dataframe(
            filtered[["trip_id", "vehicle_type", "fuel_cost_per_trip", "trip_distance_km", "fuel_cost_per_km"]].dropna(subset=["fuel_cost_per_km"]).head(200),
            hide_index=True,
            use_container_width=True,
            column_config={
                "fuel_cost_per_trip": st.column_config.NumberColumn(format="%.2f"),
                "trip_distance_km": st.column_config.NumberColumn(format="%.1f"),
                "fuel_cost_per_km": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.caption("Fuel cost per kilometre = fuel cost per trip ÷ trip distance in kilometres. Values are calculated only when both fields are valid and trip distance is greater than zero.")
else:
    st.info("Fuel cost per kilometre is unavailable because required source fields are missing.")

st.subheader("Trip calculator")
col_a, col_b, col_c = st.columns(3)
with col_a:
    custom_distance = st.number_input("Custom trip distance (km)", min_value=0.0, value=100.0, step=1.0, help="Distance must be greater than zero.")
with col_b:
    custom_efficiency = st.number_input("Fuel efficiency (km/L)", min_value=0.1, value=8.0, step=0.1, help="Efficiency must be greater than zero.")
with col_c:
    custom_price = st.number_input("Fuel price per litre", min_value=0.0, value=1.50, step=0.01, help="Optional price to estimate fuel cost; leave zero if unknown.")
if custom_distance <= 0 or custom_efficiency <= 0:
    st.warning("Custom trip distance and fuel efficiency must be greater than zero to calculate estimates.")
else:
    estimated_fuel = custom_distance / custom_efficiency
    estimated_cost = estimated_fuel * custom_price if custom_price > 0 else None
    cost_per_km = (estimated_cost / custom_distance) if estimated_cost is not None and custom_distance > 0 else None
    st.success(
        f"User-entered assumptions: distance = {custom_distance:.1f} km, efficiency = {custom_efficiency:.2f} km/L, fuel price = {custom_price:.2f}. "
        f"Estimated fuel = {estimated_fuel:.2f} L."
    )
    if estimated_cost is not None:
        st.caption(f"Estimated fuel cost = {estimated_cost:.2f}; estimated cost per kilometre = {cost_per_km:.2f}.")
    else:
        st.caption("Estimated fuel cost unavailable: fuel price per litre is zero or missing. The fuel-use estimate is still shown above.")

st.subheader("Fuel estimate from a recorded trip")
required = {"trip_distance_km", "fuel_efficiency_km_per_liter"}
if required.issubset(filtered.columns):
    estimate_records = filtered.dropna(subset=list(required))
    estimate_records = estimate_records[
        estimate_records["trip_distance_km"].gt(0)
        & estimate_records["fuel_efficiency_km_per_liter"].gt(0)
        & estimate_records["trip_distance_km"].map(lambda value: isfinite(float(value)))
        & estimate_records["fuel_efficiency_km_per_liter"].map(lambda value: isfinite(float(value)))
    ]
    if estimate_records.empty:
        st.info("No selected record has positive distance and efficiency values for calculation.")
    else:
        if "trip_id" in estimate_records:
            trip_values = estimate_records["trip_id"].astype(str).tolist()
            selected_trip = st.selectbox("Select a recorded trip", trip_values)
            selected_row = estimate_records.loc[estimate_records["trip_id"].astype(str) == selected_trip].iloc[0]
        else:
            st.info("The source has no trip_id column. Choose a source row by its row number.")
            row_options = estimate_records.index.tolist()
            selected_row_index = st.selectbox("Source row", row_options)
            selected_row = estimate_records.loc[selected_row_index]
        distance = float(selected_row["trip_distance_km"])
        efficiency = float(selected_row["fuel_efficiency_km_per_liter"])
        estimate_litres = calculate_fuel_estimate(distance, efficiency)
        st.success(
            f"Transparent formula estimate: {distance:.2f} km ÷ {efficiency:.2f} km/L "
            f"= {estimate_litres:.2f} L. This is calculated from the selected source record."
        )
        if "fuel_price_per_liter" in selected_row and pd.notna(selected_row["fuel_price_per_liter"]):
            price = float(selected_row["fuel_price_per_liter"])
            if isfinite(price) and price > 0:
                estimated_cost = estimate_litres * price
                st.caption(
                    f"Using the same record's source fuel_price_per_liter ({price:.3f}), "
                    f"the formula cost estimate is {estimated_cost:.2f} in the source currency unit. "
                    "This is an arithmetic estimate, not a route recommendation."
                )
            else:
                st.caption("Cost estimate unavailable: the selected source fuel price is not a finite positive value.")
        if "fuel_consumed_liters" in selected_row and pd.notna(selected_row["fuel_consumed_liters"]):
            st.caption(
                f"Original recorded fuel consumed for this trip: "
                f"{float(selected_row['fuel_consumed_liters']):.2f} L."
            )
        else:
            st.caption("The selected source record does not contain fuel_consumed_liters.")
else:
    st.info(
        "Fuel-use calculation is unavailable because the source is missing "
        + ", ".join(sorted(required - set(filtered.columns)))
        + "."
    )

st.subheader("Vehicle efficiency comparison")
if {"vehicle_type", "fuel_efficiency_km_per_liter"}.issubset(filtered.columns):
    comparison = filtered.dropna(subset=["vehicle_type", "fuel_efficiency_km_per_liter"]).copy()
    if not comparison.empty:
        vehicle_summary = comparison.groupby("vehicle_type", as_index=False).agg(
            mean_efficiency=("fuel_efficiency_km_per_liter", "mean"),
            valid_observations=("fuel_efficiency_km_per_liter", "size"),
        )
        vehicle_summary = vehicle_summary.sort_values("mean_efficiency", ascending=False, kind="stable")
        chart = px.bar(
            vehicle_summary,
            x="vehicle_type",
            y="mean_efficiency",
            color="vehicle_type",
            title="Average recorded km/L by vehicle type",
            labels={"vehicle_type": "Source vehicle type", "mean_efficiency": "Average fuel efficiency · km/L"},
            hover_data={"valid_observations": True, "mean_efficiency": ":.2f"},
        )
        chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#D9E4F0"},
            legend={"title": "Vehicle type"},
        )
        st.plotly_chart(chart, use_container_width=True)
        st.dataframe(vehicle_summary, hide_index=True, use_container_width=True)
    else:
        render_empty_state("No valid vehicle efficiency observations remain after filtering.")
else:
    render_empty_state("Vehicle efficiency comparison requires vehicle_type and fuel_efficiency_km_per_liter in the source.")

st.subheader("Original fuel records")
display_fields = [
    "trip_id", "timestamp", "vehicle_id", "vehicle_type", "fuel_type", "road_type",
    "trip_distance_km", "fuel_efficiency_km_per_liter", "fuel_consumed_liters",
    "fuel_price_per_liter", "fuel_cost_per_trip", "fuel_cost_per_km", "weather_condition",
]
visible_fields = [field for field in display_fields if field in filtered.columns]
if visible_fields:
    st.dataframe(filtered[visible_fields].head(200), hide_index=True, use_container_width=True)
    st.caption("Showing up to 200 filtered rows from the source CSV. No origin, destination, route alternatives, or travel times are present in this interface.")
else:
    render_empty_state("No recognized source fields are available to display.")
