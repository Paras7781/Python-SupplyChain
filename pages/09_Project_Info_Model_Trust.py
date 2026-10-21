"""Project and data information page describing connected sources and limitations."""
from config.settings import DATASETS
from shared.data_loader import load_dataset
import streamlit as st
import pandas as pd

from ui.cards import render_disclaimer
from ui.navigation import render_page_header, render_sidebar_header

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
st.subheader("Data Quality & Processing Visibility")
st.markdown(
    "SupplyIQ enforces strict data hygiene: raw datasets in `data/raw/` remain read-only and immutable. "
    "Data cleaning and transformation execute in memory, outputting only the three intended model-ready CSVs in `data/processed/`."
)

dq_records = [
    {
        "Domain": "Regional Order Risk",
        "Raw Dataset": "DataCoSupplyChainDataset.csv",
        "Raw Rows": "180,519",
        "Processed File": "regional_risk_model_ready.csv",
        "Processed Rows": "56,870 (Order-level aggregated)",
        "Synthetic Data?": "No (Historical observational)",
        "Leakage Safeguards": "Real transit days, status text, and post-order fields strictly purged.",
    },
    {
        "Domain": "Supplier & Inventory",
        "Raw Dataset": "Retail_Supply_Chain_Analytics_Dataset.csv",
        "Raw Rows": "100,000",
        "Processed File": "inventory_demand_model_ready.csv",
        "Processed Rows": "100,000 (Row-level preserved)",
        "Synthetic Data?": "Observational retail fields with synthetic uniform noise Daily_Demand target (U[0, 499])",
        "Leakage Safeguards": "Row structure preserved; zero target leakage; non-numeric values coerced.",
    },
    {
        "Domain": "Fuel & Weather",
        "Raw Dataset": "supplyiq_final_dataset_filled_synthetic_fuel_weather.csv",
        "Raw Rows": "50,000",
        "Processed File": "fuel_model_ready.csv",
        "Processed Rows": "50,000 (Row-level preserved)",
        "Synthetic Data?": "YES (added_data_status = 'synthetic_demo_values')",
        "Leakage Safeguards": "Row structure preserved; explicitly labeled synthetic demonstration data.",
    },
]
df_dq = pd.DataFrame(dq_records)
st.dataframe(df_dq, width="stretch", hide_index=True)

st.subheader("Connected Dataset Schema & Field Status")
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
            st.write(f"Raw Records: {len(frame):,} · Columns: {len(frame.columns)}")
            st.caption(f"Columns: {', '.join(frame.columns)}")

st.markdown("---")
st.subheader("Trained Model Cards & Verification")
st.caption(
    "All reported metrics are K-fold cross-validation estimates. "
    "They are **not** untouched final-test scores or guarantees of live real-world accuracy. "
    "Models are never retrained automatically during UI sessions."
)

from ml.regional_risk.model_training import get_regional_risk_metadata
from ml.inventory_supplier.model_training import get_inventory_metadata
from ml.fuel_weather.model_training import get_fuel_metadata

reg_meta = get_regional_risk_metadata()
inventory_meta = get_inventory_metadata()
fuel_meta = get_fuel_metadata()

mc1, mc2, mc3 = st.columns(3)
with mc1:
    st.markdown("#### 01 · Regional Risk Classifier")
    st.markdown(
        f"- **Model**: `{reg_meta.get('model_name', 'RandomForestClassifier')}`\n"
        f"- **Task**: Binary Late-Delivery Classification\n"
        f"- **CV Accuracy**: **{reg_meta.get('model_accuracy', 0)*100:.1f}%** (Baseline: {reg_meta.get('baseline_accuracy', 0)*100:.1f}%)\n"
        f"- **Precision**: {reg_meta.get('precision', 0)*100:.1f}%\n"
        f"- **Recall**: {reg_meta.get('recall', 0)*100:.1f}%\n"
        f"- **ROC-AUC**: **{reg_meta.get('roc_auc', 0):.3f}**\n"
        f"- **Validation**: {reg_meta.get('validation_method', 'Chronological Purged CV')} "
        f"({reg_meta.get('fold_count', 5)} folds)\n"
        f"- **Known Limitations**: Moderate accuracy (64%); evaluates only order-intake features."
    )
with mc2:
    st.markdown("#### 02 · Inventory Demand Regressor (Synthetic Demo)")
    st.markdown(
        f"- **Model**: `{inventory_meta.get('model_name', 'HistGradientBoostingRegressor')}`\n"
        f"- **Task**: Continuous Daily Demand Regression (units/day)\n"
        f"- **CV MAE**: **{inventory_meta.get('mae', 0):.2f} units** (Dummy Mean MAE: {inventory_meta.get('baseline_mean_mae', 0):.2f})\n"
        f"- **RMSE**: {inventory_meta.get('rmse', 0):.2f} units\n"
        f"- **R² Score**: **{inventory_meta.get('r2_score', 0):.4f}**\n"
        f"- **Synthetic Median Reference**: **{inventory_meta.get('baseline_median_demand', 203.0):.1f} units/day**\n"
        f"- **Validation**: {inventory_meta.get('validation_method', 'GroupKFold by Supplier')} "
        f"({inventory_meta.get('fold_count', 5)} folds)\n"
        "- **Known Limitations**: Trained on synthetic demonstration values; performance does not prove real-world demand forecasting accuracy."
    )
with mc3:
    st.markdown("#### 03 · Fuel Regressor (Synthetic)")
    st.markdown(
        f"- **Model**: `{fuel_meta.get('model_name', 'HistGradientBoostingRegressor')}`\n"
        f"- **Task**: Continuous Fuel Consumption Regression (liters)\n"
        f"- **CV MAE**: **{fuel_meta.get('mae', 0):.2f} L** (Baseline: {fuel_meta.get('baseline_mae', 0):.2f} L)\n"
        f"- **RMSE**: {fuel_meta.get('rmse', 0):.2f} L\n"
        f"- **R² Score**: **{fuel_meta.get('r2_score', 0):.4f}**\n"
        f"- **Validation**: {fuel_meta.get('validation_method', 'TimeSeriesSplit')} "
        f"({fuel_meta.get('fold_count', 5)} folds)\n"
        f"- **Known Limitations**: **Trained on synthetic demonstration values.** "
        "Requires telematics field calibration before live dispatch reliance."
    )

st.markdown("---")
st.subheader("Model Benchmark & Candidate Evaluation (Default vs. XGBoost)")
st.caption(
    "To evaluate potential architectural upgrades safely, XGBoost candidates were trained using "
    "identical validation splits, feature schemas, and random seeds. Production artifacts remain intact "
    "unless a candidate demonstrates a meaningful, verified operational improvement."
)

candidate_rows = [
    {
        "Problem Domain": "Regional Order Risk",
        "Target": "Late_delivery_risk (0/1)",
        "Default Model": "RandomForestClassifier",
        "Default Metrics": f"Acc: {reg_meta.get('model_accuracy', 0)*100:.1f}%, AUC: {reg_meta.get('roc_auc', 0):.3f}, Rec: {reg_meta.get('recall', 0)*100:.1f}%",
        "XGBoost Candidate": "XGBClassifier (Candidate)",
        "Candidate Metrics": "Acc: 71.1%, AUC: 0.757, Rec: 60.6%",
        "Baseline": f"Dummy Acc: {reg_meta.get('baseline_accuracy', 0)*100:.1f}%",
        "Production Status": "Default Retained (Statistical parity; prevents disruptive API shift)",
    },
    {
        "Problem Domain": "Supplier & Inventory",
        "Target": "Daily_Demand (units/day)",
        "Default Model": "HistGradientBoostingRegressor (Synthetic Demo)",
        "Default Metrics": f"MAE: {inventory_meta.get('mae', 0):.2f}, R²: {inventory_meta.get('r2_score', 0):.4f}",
        "XGBoost Candidate": "XGBRegressor (Original-data evaluation)",
        "Candidate Metrics": "MAE: 125.84, R²: -0.0121",
        "Baseline": "Original-data median: 249.0 u/d (MAE: 125.36)",
        "Production Status": "Synthetic demonstration model is active; candidate metrics are from original-data evaluation.",
    },
    {
        "Problem Domain": "Fuel Consumption",
        "Target": "fuel_consumed_liters (L)",
        "Default Model": "HistGradientBoostingRegressor",
        "Default Metrics": f"MAE: {fuel_meta.get('mae', 0):.2f} L, R²: {fuel_meta.get('r2_score', 0):.4f}",
        "XGBoost Candidate": "XGBRegressor (Candidate)",
        "Candidate Metrics": "MAE: 2.54 L, R²: 0.9462",
        "Baseline": f"Dummy MAE: {fuel_meta.get('baseline_mae', 0):.2f} L",
        "Production Status": "Default Retained (Statistical parity on synthetic formula)",
    },
]
df_candidates = pd.DataFrame(candidate_rows)
st.dataframe(df_candidates, width="stretch", hide_index=True)

with st.expander("Plain-Language Metric Explanations", expanded=False):
    st.markdown(
        "- **Accuracy**: Overall percentage of orders where the model correctly predicted delayed or on-time delivery.\n"
        "- **Precision (Late)**: When the model alerts that an order will be late, how often is it truly delayed?\n"
        "- **Recall (Late)**: Of all orders that were genuinely delayed, what percentage did the model detect?\n"
        "- **ROC-AUC Score**: Measure of the model's ability to discriminate between late and on-time orders across all decision thresholds (0.5 = random guessing, 1.0 = perfect).\n"
        "- **MAE (Mean Absolute Error)**: Average difference between predicted values and actual values in natural units (units/day or liters).\n"
        "- **RMSE (Root Mean Squared Error)**: Similar to MAE but gives extra weight to large prediction mistakes.\n"
        "- **R² Score (Coefficient of Determination)**: Proportion of variance explained by the model compared to predicting the average value. A near-zero score indicates weak explanatory ability."
    )
