"""Unavailable state for the unimplemented risk prediction backend."""
import streamlit as st

from dashboard.components.navigation import render_page_header, render_sidebar_header

st.set_page_config(page_title="Risk Prediction - SupplyIQ", page_icon="🧭", layout="wide")
render_sidebar_header("prediction")
render_page_header(
    "Planned backend capability",
    "Regional risk prediction",
    "No prediction model is connected to this application.",
)
st.info("Not yet available — model/backend not connected.")
st.caption("This page does not generate, estimate, or display model predictions or risk scores.")
