"""Order-level regional drill-down using the original DataCo shipment records."""
import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np

from ui.cards import render_disclaimer, render_empty_state, render_metric_card
from ui.navigation import render_page_header, render_sidebar_header
from shared.data_loader import load_dataset


DELAY_SEVERITY_THRESHOLDS = {
    "minor_delay": (1, 2),
    "moderate_delay": (3, 5),
    "severe_delay": (6, float("inf")),
}


st.set_page_config(page_title="Regional Order Explorer - SupplyIQ", page_icon="🌍", layout="wide")
render_sidebar_header("regional")
render_page_header(
    "Module 01 · Regional order risk",
    "Regional order explorer",
    "Drill from region to state and city, then inspect the matching order records and source-backed delivery indicators.",
)
render_disclaimer(
    "SOURCE DATA: DataCo order records. Risk context below uses only the source late-delivery flag, "
    "delivery status, and calculated actual-versus-scheduled shipping-day difference. No model or hazard cause is inferred."
)


@st.cache_data
def load_order_records() -> pd.DataFrame:
    """Prepare one privacy-safe record per order from item-level source rows."""
    source = load_dataset("supply_chain")
    if source.empty or "Order Id" not in source.columns:
        return pd.DataFrame()

    safe_fields = [
        "Order Id",
        "Order Region",
        "Order State",
        "Order City",
        "Order Country",
        "order date (DateOrders)",
        "Delivery Status",
        "Late_delivery_risk",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Shipping Mode",
        "Order Status",
        "Order Item Id",
        "Order Item Quantity",
        "Order Item Total",
    ]
    available_fields = [field for field in safe_fields if field in source.columns]
    frame = source[available_fields].copy()
    for column in ("Order Id", "Order Region", "Order State", "Order City",
                   "Order Country", "Delivery Status", "Shipping Mode", "Order Status"):
        if column in frame:
            frame[column] = frame[column].astype("string").str.strip()
    date_column = "order date (DateOrders)"
    if date_column in frame:
        frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    for column in ("Late_delivery_risk", "Days for shipping (real)",
                   "Days for shipment (scheduled)", "Order Item Quantity", "Order Item Total"):
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    first_fields = [
        field for field in available_fields
        if field not in ("Order Id", "Order Item Id", "Order Item Quantity", "Order Item Total")
    ]
    aggregations = {field: "first" for field in first_fields}
    if "Order Item Id" in frame:
        aggregations["Order Item Id"] = "count"
    if "Order Item Quantity" in frame:
        aggregations["Order Item Quantity"] = lambda values: values.sum(min_count=1)
    if "Order Item Total" in frame:
        aggregations["Order Item Total"] = lambda values: values.sum(min_count=1)
    orders = frame.groupby("Order Id", as_index=False, sort=False).agg(aggregations)
    rename = {
        "Order Item Id": "Line items",
        "Order Item Quantity": "Units",
        "Order Item Total": "Order value",
        date_column: "Order date",
    }
    orders = orders.rename(columns=rename)
    if "Days for shipping (real)" in orders and "Days for shipment (scheduled)" in orders:
        orders["Shipping difference (days)"] = (
            orders["Days for shipping (real)"] - orders["Days for shipment (scheduled)"]
        )
        orders["Shipping gap (days)"] = orders["Shipping difference (days)"]
    else:
        orders["Shipping difference (days)"] = pd.Series(pd.NA, index=orders.index, dtype="float64")
        orders["Shipping gap (days)"] = pd.Series(pd.NA, index=orders.index, dtype="float64")
    status = orders.get("Delivery Status", pd.Series(pd.NA, index=orders.index, dtype="string")).str.lower()
    if "Late_delivery_risk" in orders:
        risk_values = orders["Late_delivery_risk"]
        orders["Source late-risk flag"] = (
            risk_values.ge(1).where(risk_values.notna(), pd.NA).astype("boolean")
        )
    else:
        orders["Source late-risk flag"] = pd.Series(pd.NA, index=orders.index, dtype="boolean")
    orders["Late delivery"] = status.str.contains("late", regex=False, na=pd.NA).astype("boolean")
    if "Shipping gap (days)" in orders:
        # Translate valid day gaps into readable bands; missing or invalid gaps stay unknown.
        def classify_delay_severity(gap_value, thresholds=None):
            if pd.isna(gap_value):
                return "Unknown"
            try:
                gap = float(gap_value)
            except (TypeError, ValueError):
                return "Unknown"
            if not pd.notna(gap) or not np.isfinite(gap):
                return "Unknown"
            if gap <= 0:
                return "On time or early"
            minor_lower, minor_upper = DELAY_SEVERITY_THRESHOLDS["minor_delay"]
            moderate_lower, moderate_upper = DELAY_SEVERITY_THRESHOLDS["moderate_delay"]
            severe_lower, severe_upper = DELAY_SEVERITY_THRESHOLDS["severe_delay"]
            if minor_lower <= gap <= minor_upper:
                return "Minor delay"
            if moderate_lower <= gap <= moderate_upper:
                return "Moderate delay"
            if severe_lower <= gap <= severe_upper:
                return "Severe delay"
            return "Unknown"
        orders["Delay severity"] = orders["Shipping gap (days)"].apply(classify_delay_severity)
    else:
        orders["Delay severity"] = "Unknown"
    return orders


def build_monthly_delivery_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarise monthly orders and late-delivery rates from current filtered records."""
    if frame.empty or "Order date" not in frame:
        return pd.DataFrame()
    monthly = frame[["Order Id", "Order date", "Late delivery"]].copy()
    monthly = monthly.dropna(subset=["Order date"]).copy()
    if monthly.empty:
        return pd.DataFrame()
    monthly["Order month"] = pd.to_datetime(monthly["Order date"], errors="coerce").dt.to_period("M").dt.to_timestamp()
    monthly = monthly[monthly["Order month"].notna()].copy()
    if monthly.empty:
        return pd.DataFrame()
    summary = monthly.groupby("Order month", as_index=False).agg(
        Orders=("Order Id", "nunique"),
        Late_deliveries=("Late delivery", lambda values: values.fillna(False).sum()),
    )
    summary["Late delivery rate (%)"] = (summary["Late_deliveries"] / summary["Orders"] * 100).replace([float("inf"), -float("inf")], pd.NA)
    summary["Order month"] = summary["Order month"].dt.strftime("%Y-%m")
    return summary


def reset_child_location():
    # Changing a region invalidates both narrower location selections.
    st.session_state["order_state"] = "All states / provinces"
    st.session_state["order_city"] = "All cities"


def reset_city():
    # Changing the state clears any city selected under the previous state.
    st.session_state["order_city"] = "All cities"


def back_one_level():
    # Clear the deepest selected location first to move up one drill-down level.
    if st.session_state.get("order_city", "All cities") != "All cities":
        st.session_state["order_city"] = "All cities"
    elif st.session_state.get("order_state", "All states / provinces") != "All states / provinces":
        st.session_state["order_state"] = "All states / provinces"
        st.session_state["order_city"] = "All cities"
    elif st.session_state.get("order_region", "All regions") != "All regions":
        st.session_state["order_region"] = "All regions"
        st.session_state["order_state"] = "All states / provinces"
        st.session_state["order_city"] = "All cities"


orders = load_order_records()
if orders.empty:
    st.error("The DataCo source file is missing or does not contain an Order Id field.")
    st.stop()

required_dimension = "Order Region"
if required_dimension not in orders:
    st.error("The source dataset does not contain Order Region, so a regional hierarchy cannot be shown.")
    st.stop()

st.subheader("Drill-down")
region_col, state_col, city_col, back_col = st.columns([1.1, 1.1, 1.1, 0.75])
regions = ["All regions"] + sorted(orders["Order Region"].dropna().unique().tolist())
with region_col:
    selected_region = st.selectbox(
        "Region",
        regions,
        key="order_region",
        on_change=reset_child_location,
        help="Uses the source Order Region field.",
    )
region_base = orders if selected_region == "All regions" else orders[orders["Order Region"] == selected_region]

state_available = "Order State" in region_base.columns
city_available = "Order City" in region_base.columns
with state_col:
    if state_available:
        states = ["All states / provinces"] + sorted(region_base["Order State"].dropna().unique().tolist())
        selected_state = st.selectbox(
            "State / province",
            states,
            key="order_state",
            on_change=reset_city,
            help="This level is shown because Order State exists in the source.",
        )
    else:
        selected_state = "All states / provinces"
        st.caption("State / province unavailable: this field is not in the source dataset.")
state_base = region_base
if state_available and selected_state != "All states / provinces":
    state_base = state_base[state_base["Order State"] == selected_state]
with city_col:
    if city_available:
        cities = ["All cities"] + sorted(state_base["Order City"].dropna().unique().tolist())
        selected_city = st.selectbox("City", cities, key="order_city")
    else:
        selected_city = "All cities"
        st.caption("City unavailable: this field is not in the source dataset.")
with back_col:
    st.write("")
    st.button(
        "← Back one level",
        disabled=(selected_city == "All cities" and selected_state == "All states / provinces"
                  and selected_region == "All regions"),
        on_click=back_one_level,
        help="Clear the most specific selected location level.",
        width="stretch",
    )

crumbs = ["All regions"]
if selected_region != "All regions":
    crumbs = [selected_region]
if selected_state != "All states / provinces":
    crumbs.append(selected_state)
if selected_city != "All cities":
    crumbs.append(selected_city)
st.caption("Location path  /  " + "  ›  ".join(crumbs))
if not state_available or not city_available:
    missing_levels = []
    if not state_available:
        missing_levels.append("state / province")
    if not city_available:
        missing_levels.append("city")
    st.info("The source does not provide " + " or ".join(missing_levels) + " fields. The drill-down stops at the available level.")

st.subheader("Filters")
filter_a, filter_b, filter_c, filter_d = st.columns([1.1, 1.1, 1.1, 1])
working = orders.copy()
date_field = "Order date"
if date_field in working:
    working[date_field] = pd.to_datetime(working[date_field], errors="coerce")
    valid_dates = working[date_field].dropna()
    if not valid_dates.empty:
        with filter_a:
            date_range = st.date_input(
                "Order date range",
                value=(valid_dates.min().date(), valid_dates.max().date()),
                min_value=valid_dates.min().date(),
                max_value=valid_dates.max().date(),
                key="order_date_range",
                help="Use this to focus on the order window the manager is reviewing.",
            )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            working = working[
                working[date_field].dt.date.between(start_date, end_date)
            ]
    else:
        with filter_a:
            st.caption("No usable order dates in this source.")
else:
    with filter_a:
        st.caption("Order date filter unavailable.")

with filter_b:
    statuses = ["All statuses"] + sorted(working["Delivery Status"].dropna().unique().tolist()) if "Delivery Status" in working else ["All statuses"]
    selected_status = st.selectbox("Delivery status", statuses, key="order_status", help="Filter to late, on-time, or other source delivery states.")
if selected_status != "All statuses" and "Delivery Status" in working:
    working = working[working["Delivery Status"] == selected_status]
with filter_c:
    modes = ["All modes"] + sorted(working["Shipping Mode"].dropna().unique().tolist()) if "Shipping Mode" in working else ["All modes"]
    selected_mode = st.selectbox("Transport / shipping mode", modes, key="order_mode", help="Use the source shipping mode to compare carrier patterns.")
if selected_mode != "All modes" and "Shipping Mode" in working:
    working = working[working["Shipping Mode"] == selected_mode]
with filter_d:
    order_query = st.text_input("Search order reference", placeholder="Type any order ID or location clue", key="order_query", help="This is meant for business users; the dropdown below can be filtered by the same text when the list is long.")

if selected_region != "All regions":
    working = working[working["Order Region"] == selected_region]
if selected_state != "All states / provinces" and "Order State" in working:
    working = working[working["Order State"] == selected_state]
if selected_city != "All cities" and "Order City" in working:
    working = working[working["Order City"] == selected_city]
if order_query.strip():
    text = order_query.strip()
    order_text = working["Order Id"].astype(str)
    location_text = working.get("Order City", pd.Series("", index=working.index)).astype(str)
    state_text = working.get("Order State", pd.Series("", index=working.index)).astype(str)
    region_text = working.get("Order Region", pd.Series("", index=working.index)).astype(str)
    working = working[
        order_text.str.contains(text, case=False, na=False)
        | location_text.str.contains(text, case=False, na=False)
        | state_text.str.contains(text, case=False, na=False)
        | region_text.str.contains(text, case=False, na=False)
    ]

if "Late delivery" in working and working["Late delivery"].notna().any():
    late_rate = float(working["Late delivery"].mean() * 100)
else:
    late_rate = None
delay_values = working.get("Shipping difference (days)", pd.Series(dtype=float)).dropna()
positive_delays = delay_values[delay_values > 0]
average_late_days = float(positive_delays.mean()) if not positive_delays.empty else None

kpi_a, kpi_b, kpi_c, kpi_d = st.columns(4)
with kpi_a:
    render_metric_card("Orders in selection", f"{len(working):,}", help_text="Deduplicated by source Order Id.")
with kpi_b:
    render_metric_card("Late delivery rate", f"{late_rate:.1f}%" if late_rate is not None else "Unavailable", help_text="Share among orders with a nonmissing source Delivery Status containing 'late'.")
with kpi_c:
    render_metric_card("Avg. positive shipping gap", f"{average_late_days:.1f} days" if average_late_days is not None else "Unavailable", help_text="Actual minus scheduled shipping days, when positive.")
with kpi_d:
    risk_share = float(working["Source late-risk flag"].mean() * 100) if "Source late-risk flag" in working and working["Source late-risk flag"].notna().any() else None
    render_metric_card("Source late-risk flag", f"{risk_share:.1f}%" if risk_share is not None else "Unavailable", help_text="Share among nonmissing source Late_delivery_risk values; not a prediction.")

st.subheader("Monthly delivery trend")
monthly_summary = build_monthly_delivery_summary(working)
if monthly_summary.empty:
    render_empty_state("Monthly delivery trend is unavailable because the filtered orders do not have valid order dates.")
else:
    import plotly.graph_objects as go

    monthly_chart = go.Figure()
    monthly_chart.add_trace(
        go.Scatter(
            x=monthly_summary["Order month"],
            y=monthly_summary["Orders"],
            mode="lines+markers",
            name="Order count",
            line={"color": "#60A5FA", "width": 3},
            marker={"size": 8},
            hovertemplate="Month: %{x}<br>Orders: %{y}<extra></extra>",
        )
    )
    monthly_chart.add_trace(
        go.Scatter(
            x=monthly_summary["Order month"],
            y=monthly_summary["Late delivery rate (%)"],
            mode="lines+markers",
            name="Late-delivery rate",
            line={"color": "#F59E0B", "width": 3},
            marker={"size": 8},
            yaxis="y2",
            hovertemplate="Month: %{x}<br>Late delivery rate: %{y:.1f}%<extra></extra>",
        )
    )
    monthly_chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#D9E4F0"},
        title="Monthly delivery performance",
        xaxis_title="Order month",
        yaxis_title="Order count",
        yaxis2={"title": "Late-delivery rate (%)", "overlaying": "y", "side": "right"},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
    )
    st.plotly_chart(monthly_chart, width="stretch")

st.subheader("Regional comparison")
comparison = working.copy()
if not comparison.empty:
    regional = comparison.groupby("Order Region", as_index=False).agg(
        Orders=("Order Id", "nunique"),
        Late_delivery_rate=("Late delivery", "mean"),
        Source_risk_rate=("Source late-risk flag", "mean"),
    )
    regional["Late delivery rate (%)"] = regional["Late_delivery_rate"].fillna(0) * 100
    regional["Source late-risk rate (%)"] = regional["Source_risk_rate"].fillna(0) * 100
    regional = regional.sort_values("Orders", ascending=False, kind="stable")
    compare_long = regional.melt(
        id_vars=["Order Region", "Orders"],
        value_vars=["Late delivery rate (%)", "Source late-risk rate (%)"],
        var_name="Rate type",
        value_name="Rate (%)",
    )
    compare_long["Rate type"] = compare_long["Rate type"].replace({
        "Late delivery rate (%)": "Observed late delivery rate",
        "Source late-risk rate (%)": "Source late-risk rate",
    })
    chart = px.bar(
        compare_long,
        x="Order Region",
        y="Rate (%)",
        color="Rate type",
        barmode="group",
        hover_data={"Orders": True, "Rate (%)": ":.1f"},
        labels={"Order Region": "Source order region", "Rate (%)": "Rate (%)", "Rate type": "Rate source"},
        title="Observed late-delivery vs. source late-risk rate by region",
    )
    chart.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#D9E4F0"},
        coloraxis_showscale=False,
        margin={"l": 8, "r": 8, "t": 48, "b": 8},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    st.plotly_chart(chart, width="stretch")
else:
    render_empty_state("No orders match the current filters.")

st.subheader("Location ranking")
sort_label = st.radio(
    "Rank by",
    ["Source late-risk rate", "Late delivery rate", "Order volume"],
    horizontal=True,
)
if selected_city != "All cities" and city_available:
    location_field, location_label = "Order City", "City"
elif selected_state != "All states / provinces" and state_available:
    location_field, location_label = "Order State", "State / province"
else:
    location_field, location_label = "Order Region", "Region"
if location_field in working and not working.empty:
    ranked = working.groupby(location_field, as_index=False).agg(
        Orders=("Order Id", "nunique"),
        Late_delivery_rate=("Late delivery", "mean"),
        Source_risk_rate=("Source late-risk flag", "mean"),
    )
    ranked["Late delivery rate (%)"] = ranked["Late_delivery_rate"].fillna(0) * 100
    ranked["Source late-risk rate (%)"] = ranked["Source_risk_rate"].fillna(0) * 100
    sort_column = {
        "Source late-risk rate": "Source late-risk rate (%)",
        "Late delivery rate": "Late delivery rate (%)",
        "Order volume": "Orders",
    }[sort_label]
    ranked = ranked.sort_values(sort_column, ascending=False, kind="stable")
    st.caption(f"Ranked at the {location_label.lower()} level; rates use observed source fields.")
    st.dataframe(
        ranked[[location_field, "Orders", "Late delivery rate (%)", "Source late-risk rate (%)"]].head(25),
        hide_index=True,
        width="stretch",
        column_config={
            "Late delivery rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
            "Source late-risk rate (%)": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )
else:
    render_empty_state("No location rows are available for the current selection.")

st.subheader("Filtered orders")
st.caption("The export is an order-level allowlist and deliberately excludes customer names, contact details, passwords, and street addresses present in the source.")
visible_fields = [
    "Order Id", "Order Region", "Order State", "Order City", "Order Country",
    "Order date", "Delivery Status", "Delay severity", "Source late-risk flag",
    "Days for shipping (real)", "Days for shipment (scheduled)", "Shipping difference (days)", "Shipping gap (days)",
    "Shipping Mode", "Order Status", "Line items", "Units", "Order value",
]
visible_fields = [field for field in visible_fields if field in working]
if working.empty:
    render_empty_state("No order records match the selected location and filters.")
else:
    st.dataframe(working[visible_fields].head(500), hide_index=True, width="stretch")
    if len(working) > 500:
        st.caption(f"Showing the first 500 of {len(working):,} matching orders; CSV export includes all matches.")
    st.download_button(
        "Download filtered order list (CSV)",
        working[visible_fields].to_csv(index=False).encode("utf-8"),
        file_name="SupplyIQ_filtered_orders.csv",
        mime="text/csv",
        width="content",
    )
    st.subheader("Individual order")
    order_ids = sorted(working["Order Id"].dropna().astype(str).unique().tolist())
    chosen_order = st.selectbox(
        "Find an order",
        order_ids,
        key="selected_order_id",
        help="Options contain only Order Id values present in the current filtered source records.",
    )
    detail = working[working["Order Id"].astype(str) == chosen_order][visible_fields].head(1)
    st.dataframe(detail, hide_index=True, width="stretch")
    if "Source late-risk flag" in detail and pd.notna(detail["Source late-risk flag"].iloc[0]):
        flag = bool(detail["Source late-risk flag"].iloc[0])
        st.caption(
            "Supported risk context: source Late_delivery_risk flag is "
            f"{'set' if flag else 'not set'}; delivery status and shipping-day gap are shown above. "
            "No causal risk reason is available in this view."
        )
