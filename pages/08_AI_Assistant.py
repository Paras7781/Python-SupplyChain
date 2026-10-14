"""Unavailable state for the unimplemented AI and retrieval backend."""
import streamlit as st

from dashboard.components.navigation import render_page_header, render_sidebar_header

st.set_page_config(page_title="AI Assistant - SupplyIQ", page_icon="🤖", layout="wide")
render_sidebar_header("assistant")
render_page_header(
    "Planned backend capability",
    "Supply chain AI assistant",
    "No language model or retrieval service is connected to the source datasets.",
)
st.info("Not yet available — model/backend not connected.")
st.caption("This page does not accept prompts or generate answers or operational advice.")
