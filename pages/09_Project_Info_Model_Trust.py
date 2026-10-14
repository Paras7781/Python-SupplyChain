"""Project and data information page describing connected sources and limitations."""
from config.settings import DATASETS
from src.data_loader import load_dataset
import streamlit as st

from dashboard.components.cards import render_disclaimer
from dashboard.components.navigation import render_page_header, render_sidebar_header

st.set_page_config(page_title="Project & Data Information - SupplyIQ", page_icon="ℹ️", layout="wide")
render_sidebar_header("information")
render_page_header(
    "Project & data information",
    "SupplyIQ source data and capabilities",
    "Review the original CSV datasets, source-backed fields, and features that are not yet connected.",
)
render_disclaimer(
    "The application reads original source CSV files without writing to them. No predictions, recommendations, route alternatives, or AI answers are fabricated."
)

st.subheader("Project overview")
st.markdown(
    "SupplyIQ displays source-backed regional orders, retail supplier/inventory data, and fuel-consumption records. "
    "The three source domains remain separate and are not joined."
)

st.markdown("---")
st.subheader("Three problem statements")
problem_cols = st.columns(3)
with problem_cols[0]:
    st.markdown(
        "#### Regional order risk\n"
        "- Where are delivery problems concentrated?\n"
        "- Which regions, states, and cities show repeated delay exposure?\n"
        "- Which individual orders require review?"
    )
with problem_cols[1]:
    st.markdown(
        "#### Supplier & inventory\n"
        "- Which suppliers and warehouses show performance or stock pressure?\n"
        "- What do the source demand and stock records contain?"
    )
with problem_cols[2]:
    st.markdown(
        "#### Fuel consumption\n"
        "- What fuel was recorded per trip?\n"
        "- What vehicle, road, distance, and efficiency values exist in the fuel source?\n"
        "- Which route and travel-time fields are unavailable?"
    )

st.markdown("---")
st.subheader("Dataset information")
source_cols = st.columns(3)
with source_cols[0]:
    st.markdown("**Regional orders**\n- DataCoSupplyChainDataset.csv\n- Order region, state, city, ID\n- Delivery status, late flag, shipping days")
with source_cols[1]:
    st.markdown("**Supplier & inventory**\n- Retail_Supply_Chain_Analytics_Dataset.csv\n- Product, category, stock, supplier, warehouse\n- Demand and service measures when populated")
with source_cols[2]:
    st.markdown("**Fuel consumption**\n- supplyiq_final_dataset_filled_synthetic_fuel_weather.csv\n- Trip distance, efficiency, consumed fuel, price, cost\n- Source field marks records synthetic_demo_values")

st.subheader("Connected dataset status")
dataset_labels = {
    "supply_chain": "DataCoSupplyChainDataset.csv",
    "retail_analytics": "Retail_Supply_Chain_Analytics_Dataset.csv",
    "fuel_weather": "supplyiq_final_dataset_filled_synthetic_fuel_weather.csv",
}
for key, path in DATASETS.items():
    frame = load_dataset(key)
    with st.expander(dataset_labels.get(key, str(path))):
        if frame.empty:
            st.error("Missing or empty source CSV.")
        else:
            st.write(f"Records: {len(frame):,} · Columns: {len(frame.columns)}")
            st.write(", ".join(frame.columns))

st.markdown("---")
st.subheader("Current system status")
st.info("CURRENT PHASE: Frontend / Interactive Prototype")
st.markdown(
    "- Source CSV files remain read-only.\n"
    "- No model predictions, route alternatives, real-time feeds, or generated recommendations are connected.\n"
    "- The fuel source's added_data_status value is synthetic_demo_values; its data is not validated live operations."
)

st.markdown("---")
st.subheader("Planned backend integration")
st.warning("FUTURE PHASE: planned backend work begins after the frontend prototype is complete.")
st.markdown(
    "- data validation\n"
    "- data cleaning and imputation\n"
    "- EDA and business-context analysis\n"
    "- feature engineering\n"
    "- model development and evaluation\n"
    "- FastAPI / backend integration\n"
    "- advanced decision intelligence\n"
    "- RAG / AI assistant if implemented later"
)

st.caption("Features requiring missing data or backend services remain unavailable rather than using substitute values.")
