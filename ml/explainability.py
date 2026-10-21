"""
Explainability Module for SupplyIQ Machine Learning Models.
Provides individual and global SHAP (SHapley Additive exPlanations) for model transparency,
feature attribution, and plain-language driver breakdowns without changing model predictions.
"""
from __future__ import annotations

from functools import lru_cache
import logging
import numpy as np
import pandas as pd
import plotly.graph_objects as px_go
import sklearn

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    shap = None
    SHAP_AVAILABLE = False


# =====================================================================
# Feature Name Humanizer
# =====================================================================
def humanize_feature_name(raw_name: str) -> str:
    """Convert pipeline column names (e.g. 'num__Days for shipment (scheduled)')
    into clean, business-friendly terminology.
    """
    clean = raw_name
    if clean.startswith("num__"):
        clean = clean[5:]
    elif clean.startswith("cat__"):
        clean = clean[5:]

    name_mapping = {
        "Days for shipment (scheduled)": "Scheduled Window (Days)",
        "order_month": "Order Month",
        "order_dayofweek": "Order Day of Week",
        "order_hour": "Order Hour of Day",
        "is_weekend": "Weekend Order Flag",
        "Total_Units": "Total Order Units",
        "Total_Order_Value": "Total Order Value ($)",
        "Line_Items": "Line Items Count",
        "value_per_unit": "Unit Value Density ($/unit)",
        "trip_distance_km": "Trip Distance (km)",
        "load_weight_kg": "Cargo Load Weight (kg)",
        "temperature_c": "Ambient Temperature (°C)",
        "wind_speed_kmh": "Wind Speed (km/h)",
        "rainfall_mm": "Rainfall (mm)",
    }

    if clean in name_mapping:
        return name_mapping[clean]

    known_cat_prefixes = [
        ("Shipping Mode_", "Shipping Mode: "),
        ("Order Region_", "Destination Region: "),
        ("Market_", "Market Area: "),
        ("Customer Segment_", "Customer Segment: "),
        ("Product_Category_", "Product Category: "),
        ("Warehouse_Location_", "Warehouse: "),
        ("Transportation_Mode_", "Transport Mode: "),
        ("vehicle_type_", "Vehicle Type: "),
        ("fuel_type_", "Fuel Type: "),
        ("road_type_", "Road Type: "),
        ("weather_condition_", "Weather Condition: "),
    ]
    for prefix, display_prefix in known_cat_prefixes:
        if clean.startswith(prefix):
            val = clean[len(prefix):].replace("_", " ")
            return f"{display_prefix}{val}"

    # Handle general categorical prefixing (e.g. Category_Value)
    if "_" in clean:
        parts = clean.rsplit("_", 1)
        category = parts[0].replace("_", " ").title()
        val = parts[1].replace("_", " ")
        return f"{category}: {val}"

    return clean.replace("_", " ").title()


# =====================================================================
# Regional Risk SHAP Explanations
# =====================================================================
def explain_regional_risk_prediction(pipeline, input_dict: dict, top_n: int = 7) -> dict:
    """Generate an individual SHAP feature attribution explanation for a Regional Risk prediction.

    Parameters:
        pipeline: The fitted scikit-learn Pipeline (preprocessor + classifier).
        input_dict: Dictionary containing order-time attributes.
        top_n: Number of top driving factors to highlight.

    Returns:
        dict containing base_value, contributions list, plain-language text,
        and Plotly figure.
    """
    if not SHAP_AVAILABLE:
        return {
            "success": False,
            "error": "SHAP library is not installed.",
        }

    try:
        prep = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        feature_names = [humanize_feature_name(f) for f in prep.get_feature_names_out()]

        # Prepare single row DataFrame matching pipeline feature columns
        from ml.regional_risk.model_training import ALL_FEATURES
        row_df = pd.DataFrame([input_dict])[ALL_FEATURES]

        # Transform using the pipeline preprocessor
        raw_x = prep.transform(row_df)
        x_dense = raw_x.toarray() if hasattr(raw_x, "toarray") else np.asarray(raw_x, dtype=np.float64)

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(x_dense)

        # Handle binary classification shapes across RF and XGBoost
        # RandomForest outputs (n_samples, n_features, 2); class 1 is index 1.
        if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_for_late = shap_values[0, :, 1]
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_for_late = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            shap_for_late = shap_values[0]
        else:
            shap_for_late = np.asarray(shap_values).flatten()

        # Expected / base value
        expected_val = explainer.expected_value
        if isinstance(expected_val, (list, np.ndarray)) and len(expected_val) > 1:
            base_value = float(expected_val[1])
        else:
            base_value = float(expected_val)

        df_contribs = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_for_late,
            "abs_shap": np.abs(shap_for_late),
        }).sort_values("abs_shap", ascending=False)

        top_factors = df_contribs.head(top_n).copy()

        # Build human-readable descriptions
        narratives = []
        for _, r in top_factors.iterrows():
            pct = abs(r["shap_value"]) * 100
            if r["shap_value"] > 0:
                direction = f"increases late delivery risk by +{pct:.1f}%"
            else:
                direction = f"reduces late delivery risk by -{pct:.1f}%"
            narratives.append(f"**{r['feature']}**: {direction}")

        # Build Plotly horizontal bar chart
        plot_df = top_factors.sort_values("shap_value", ascending=True)
        colors = ["#22c55e" if val < 0 else "#ef4444" for val in plot_df["shap_value"]]

        fig = px_go.Figure(
            px_go.Bar(
                x=plot_df["shap_value"],
                y=plot_df["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.1%}" for v in plot_df["shap_value"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            title=f"Top {top_n} Factors Influencing Late Delivery Probability",
            xaxis_title="Contribution to Late Delivery Risk (SHAP Value)",
            yaxis_title=None,
            margin=dict(l=10, r=20, t=40, b=30),
            height=320,
            template="plotly_white",
        )
        fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#94a3b8")

        return {
            "success": True,
            "base_value": base_value,
            "top_factors": top_factors.to_dict(orient="records"),
            "narratives": narratives,
            "fig": fig,
            "disclaimer": (
                "Feature attribution reflects statistical associations extracted by the tree-based model "
                "from historical order records. It illustrates model sensitivity and does not constitute "
                "a definitive causal guarantee in live freight operations."
            ),
        }
    except Exception as exc:
        logger.warning("SHAP explanation failed: %s", exc)
        return {
            "success": False,
            "error": f"Could not generate SHAP explanation: {str(exc)}",
        }


def get_regional_risk_global_importance(pipeline, sample_df: pd.DataFrame, top_n: int = 10) -> dict:
    """Compute global mean |SHAP| feature importances across a background sample of orders."""
    if not SHAP_AVAILABLE:
        return {"success": False, "error": "SHAP library is not installed."}

    try:
        from ml.regional_risk.feature_engineering import create_regional_risk_features
        from ml.regional_risk.model_training import ALL_FEATURES

        feat_df = create_regional_risk_features(sample_df, drop_leakage=True)
        valid_cols = [c for c in ALL_FEATURES if c in feat_df.columns]
        X = feat_df[valid_cols].head(300)

        prep = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        feat_names = [humanize_feature_name(f) for f in prep.get_feature_names_out()]

        raw_x = prep.transform(X)
        x_dense = raw_x.toarray() if hasattr(raw_x, "toarray") else np.asarray(raw_x, dtype=np.float64)

        explainer = shap.TreeExplainer(clf)
        shap_vals = explainer.shap_values(x_dense)

        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            shap_for_late = shap_vals[:, :, 1]
        elif isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap_for_late = shap_vals[1]
        else:
            shap_for_late = shap_vals

        mean_abs = np.mean(np.abs(shap_for_late), axis=0)

        df_importance = pd.DataFrame({
            "feature": feat_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).head(top_n)

        # Plotly horizontal bar chart for global importance
        plot_df = df_importance.sort_values("mean_abs_shap", ascending=True)
        fig = px_go.Figure(
            px_go.Bar(
                x=plot_df["mean_abs_shap"],
                y=plot_df["feature"],
                orientation="h",
                marker_color="#3b82f6",
                text=[f"{v:.3f}" for v in plot_df["mean_abs_shap"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            title=f"Global Feature Importance Across {len(X)} Historical Orders",
            xaxis_title="Mean |SHAP Value| (Average Impact on Prediction Magnitude)",
            yaxis_title=None,
            margin=dict(l=10, r=20, t=40, b=30),
            height=340,
            template="plotly_white",
        )

        return {
            "success": True,
            "df_importance": df_importance,
            "fig": fig,
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}


# =====================================================================
# Fuel Consumption SHAP Explanations
# =====================================================================
def explain_fuel_prediction(pipeline, input_dict: dict, top_n: int = 6) -> dict:
    """Generate an individual SHAP feature attribution explanation for trip fuel consumption."""
    if not SHAP_AVAILABLE:
        return {"success": False, "error": "SHAP library is not installed."}

    try:
        from ml.fuel_weather.model_training import ALL_FEATURES
        prep = pipeline.named_steps["preprocessor"]
        reg = pipeline.named_steps["regressor"]
        feature_names = [humanize_feature_name(f) for f in prep.get_feature_names_out()]

        row_df = pd.DataFrame([input_dict])[ALL_FEATURES]
        raw_x = prep.transform(row_df)
        x_dense = raw_x.toarray() if hasattr(raw_x, "toarray") else np.asarray(raw_x, dtype=np.float64)

        explainer = shap.TreeExplainer(reg)
        shap_values = explainer.shap_values(x_dense)

        shap_vals = shap_values[0] if shap_values.ndim == 2 else shap_values
        base_value = float(explainer.expected_value) if np.isscalar(explainer.expected_value) else float(explainer.expected_value[0])

        df_contribs = pd.DataFrame({
            "feature": feature_names,
            "shap_value": shap_vals,
            "abs_shap": np.abs(shap_vals),
        }).sort_values("abs_shap", ascending=False)

        top_factors = df_contribs.head(top_n).copy()

        narratives = []
        for _, r in top_factors.iterrows():
            direction = "increases fuel consumption by" if r["shap_value"] > 0 else "reduces fuel consumption by"
            narratives.append(f"**{r['feature']}**: {direction} {abs(r['shap_value']):.2f} L")

        plot_df = top_factors.sort_values("shap_value", ascending=True)
        colors = ["#22c55e" if val < 0 else "#f97316" for val in plot_df["shap_value"]]

        fig = px_go.Figure(
            px_go.Bar(
                x=plot_df["shap_value"],
                y=plot_df["feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.2f} L" for v in plot_df["shap_value"]],
                textposition="auto",
            )
        )
        fig.update_layout(
            title=f"Top {top_n} Factors Influencing Predicted Fuel Consumption",
            xaxis_title="Contribution to Fuel Consumed in Litres (SHAP Value)",
            yaxis_title=None,
            margin=dict(l=10, r=20, t=40, b=30),
            height=300,
            template="plotly_white",
        )
        fig.add_vline(x=0, line_width=1.5, line_dash="dash", line_color="#94a3b8")

        return {
            "success": True,
            "base_value": base_value,
            "top_factors": top_factors.to_dict(orient="records"),
            "narratives": narratives,
            "fig": fig,
            "disclaimer": (
                "SYNTHETIC DATA NOTICE: Fuel feature attributions explain model behavior trained on "
                "synthetic demonstration records (Distance, Vehicle, Load, and Weather). "
                "Must be validated against real vehicle CAN-bus telemetry before operational dispatch reliance."
            ),
        }
    except Exception as exc:
        return {"success": False, "error": f"Fuel SHAP explanation failed: {str(exc)}"}
