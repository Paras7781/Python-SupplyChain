"""Reusable Plotly components styled for the SupplyIQ dark operations console."""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ui.cards import render_empty_state

COLOR_PALETTE = ["#22D3EE", "#34D399", "#FBBF24", "#FB7185", "#60A5FA", "#A78BFA"]
SEMANTIC_COLORS = {
    "Low": "#34D399",
    "Healthy": "#34D399",
    "Healthy Stock": "#34D399",
    "On-Time Delivery": "#34D399",
    "Advance Shipping": "#22D3EE",
    "Moderate": "#FBBF24",
    "Medium": "#FBBF24",
    "Low Stock Alert": "#FBBF24",
    "Delayed": "#FB7185",
    "Late Delivery": "#FB7185",
    "High": "#FB7185",
    "Critical": "#FB7185",
    "Out of Stock": "#FB7185",
}


def _color_map(values):
    """Use semantic colors for known operational states and a restrained palette otherwise."""
    unique_values = list(dict.fromkeys(values))
    # Preserve input order so fallback colors remain stable across chart redraws.
    return {
        value: SEMANTIC_COLORS.get(value, COLOR_PALETTE[index % len(COLOR_PALETTE)])
        for index, value in enumerate(unique_values)
    }


def apply_dark_layout(fig, title: str = ""):
    """Apply readable typography, quiet grid lines, and transparent card backgrounds."""
    fig.update_layout(
        title=dict(
            text=title,
            x=0,
            xanchor="left",
            font=dict(size=15, color="#E8F0F8", family="Inter, sans-serif"),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#B5C4D5", family="Inter, sans-serif", size=12),
        margin=dict(l=20, r=18, t=52 if title else 25, b=24),
        height=330,
        hoverlabel=dict(
            bgcolor="#152238",
            bordercolor="#3A536C",
            font=dict(color="#F1F5F9", size=12),
        ),
        legend=dict(
            bgcolor="rgba(17, 28, 46, .65)",
            bordercolor="rgba(38, 54, 75, .75)",
            borderwidth=1,
            font=dict(color="#C6D2E0", size=11),
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        xaxis=dict(
            gridcolor="rgba(92, 116, 143, .18)",
            zerolinecolor="rgba(92, 116, 143, .3)",
            linecolor="rgba(92, 116, 143, .3)",
            tickfont=dict(color="#AFC0D2", size=11),
            automargin=True,
        ),
        yaxis=dict(
            gridcolor="rgba(92, 116, 143, .18)",
            zerolinecolor="rgba(92, 116, 143, .3)",
            linecolor="rgba(92, 116, 143, .3)",
            tickfont=dict(color="#AFC0D2", size=11),
            automargin=True,
        ),
    )
    return fig


def render_donut_chart(df: pd.DataFrame, names_col: str, values_col: str, title: str):
    """Render a compact status distribution with business-state colors."""
    if df.empty or names_col not in df.columns or values_col not in df.columns:
        render_empty_state(f"No distribution data available for '{title}'.")
        return
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        hole=0.62,
        color=names_col,
        color_discrete_map=_color_map(df[names_col].astype(str)),
    )
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        textfont=dict(color="#F1F5F9", size=12),
        hovertemplate="%{label}<br>Count: %{value:,}<br>Share: %{percent}<extra></extra>",
        sort=False,
        marker=dict(line=dict(color="#111C2E", width=3)),
    )
    fig = apply_dark_layout(fig, title)
    fig.update_layout(
        legend=dict(orientation="h", y=-0.12, x=0, xanchor="left"),
        margin=dict(l=10, r=10, t=52, b=48),
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_bar_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    color_col: str = None,
    orientation: str = "v",
):
    """Render a vertical or horizontal categorical comparison."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        render_empty_state(f"No bar chart data available for '{title}'.")
        return
    color_mapping = (
        _color_map(df[color_col].astype(str))
        if color_col and color_col in df.columns
        else None
    )
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        orientation=orientation,
        color_discrete_map=color_mapping,
        color_discrete_sequence=COLOR_PALETTE,
        custom_data=[color_col] if color_col and color_col in df.columns else None,
    )
    numeric_column = x_col if orientation == "h" else y_col
    category_column = y_col if orientation == "h" else x_col
    # Plotly swaps the visual x/y roles for horizontal bars, so the tooltip follows that swap.
    fig.update_traces(
        marker_line_width=0,
        hovertemplate=f"{category_column}: %{{y}}<br>{numeric_column}: %{{x:,.1f}}<extra></extra>"
        if orientation == "h"
        else f"{category_column}: %{{x}}<br>{numeric_column}: %{{y:,.1f}}<extra></extra>",
    )
    fig = apply_dark_layout(fig, title)
    if orientation == "h":
        fig.update_layout(yaxis=dict(**fig.layout.yaxis.to_plotly_json(), autorange="reversed"))
    if color_col is None:
        fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_grouped_bar_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str):
    """Render a grouped comparison for metrics sharing a meaningful scale."""
    if df.empty or x_col not in df.columns:
        render_empty_state(f"No grouped comparison data available for '{title}'.")
        return
    fig = go.Figure()
    for idx, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(
                go.Bar(
                    x=df[x_col],
                    y=df[col],
                    name=col,
                    marker_color=COLOR_PALETTE[idx % len(COLOR_PALETTE)],
                    hovertemplate=f"{x_col}: %{{x}}<br>{col}: %{{y:,.1f}}<extra></extra>",
                )
            )
    fig.update_layout(barmode="group", bargap=0.28, legend=dict(orientation="h", y=1.12, x=0))
    fig = apply_dark_layout(fig, title)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_line_chart(df: pd.DataFrame, x_col: str, y_cols: list, title: str):
    """Render an interactive multi-series trend chart."""
    if df.empty or x_col not in df.columns:
        render_empty_state(f"No line chart data available for '{title}'.")
        return
    fig = go.Figure()
    for idx, col in enumerate(y_cols):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df[x_col],
                    y=df[col],
                    mode="lines",
                    name=col.replace("_", " "),
                    line=dict(width=2.5, color=COLOR_PALETTE[idx % len(COLOR_PALETTE)]),
                    hovertemplate=f"{x_col}: %{{x}}<br>{col.replace('_', ' ')}: %{{y:,.2f}}<extra></extra>",
                )
            )
    fig = apply_dark_layout(fig, title)
    if len(fig.data) > 1:
        fig.update_layout(legend=dict(orientation="h", y=1.12, x=0))
    else:
        fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def render_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str = None,
    size_col: str = None,
    title: str = "",
):
    """Render a supplier/route comparison scatter plot with focused tooltips."""
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        render_empty_state(f"No scatter plot data available for '{title}'.")
        return
    color_mapping = (
        _color_map(df[color_col].astype(str))
        if color_col and color_col in df.columns
        else None
    )
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=color_col if color_col and color_col in df.columns else None,
        size=size_col if size_col and size_col in df.columns else None,
        color_discrete_map=color_mapping,
        color_discrete_sequence=COLOR_PALETTE,
        hover_data=[column for column in [color_col, size_col] if column and column in df.columns],
    )
    fig.update_traces(
        marker=dict(opacity=0.82, line=dict(color="#0B1220", width=1)),
        hovertemplate="%{fullData.name}<br>"
        + f"{x_col}: %{{x:,.1f}}<br>{y_col}: %{{y:,.2f}}<extra></extra>",
    )
    fig = apply_dark_layout(fig, title)
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
