"""Unavailable state for the unimplemented scenario simulation backend."""
import streamlit as st

from dashboard.components.navigation import render_page_header, render_sidebar_header

st.set_page_config(page_title="Scenario Simulator - SupplyIQ", page_icon="🧪", layout="wide")
render_sidebar_header("scenario")
render_page_header(
    "Planned backend capability",
    "What-if simulator",
    "No validated scenario engine is connected to this application.",
)
st.info("Not yet available — model/backend not connected.")
st.caption("This page does not display simulated baselines, sensitivity results, or recommendations.")
