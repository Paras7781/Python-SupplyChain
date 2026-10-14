"""Reusable metric, alert, disclaimer, and empty-state components."""
from html import escape

import streamlit as st


def render_metric_card(
    title: str,
    value: str,
    delta: str = None,
    help_text: str = None,
    icon: str = None,
):
    """Render a compact branded KPI card without implying unverified comparisons."""
    icon_markup = (
        f'<span style="align-items:center;background:rgba(34,211,238,.1);border:1px solid rgba(34,211,238,.18);'
        f'border-radius:7px;color:#22D3EE;display:inline-flex;font-size:12px;height:22px;'
        f'justify-content:center;margin-right:7px;width:22px;">{escape(icon)}</span>'
        if icon
        else ""
    )
    delta_markup = (
        f'<div class="siq-metric-delta">{escape(str(delta))}</div>'
        if delta
        else '<div class="siq-metric-delta">&nbsp;</div>'
    )
    st.markdown(
        f"""
        <div class="siq-metric siq-metric-accent" title="{escape(help_text or '')}">
          <div class="siq-metric-label" style="align-items:center;display:flex;">
            {icon_markup}{escape(str(title))}
          </div>
          <div class="siq-metric-value">{escape(str(value))}</div>
          {delta_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_banner(title: str, text: str, alert_type: str = "warning"):
    """Render an alert using Streamlit's accessible native severity treatment."""
    message = f"**{title}**: {text}"
    if alert_type == "error":
        st.error(message)
    elif alert_type == "warning":
        st.warning(message)
    elif alert_type == "success":
        st.success(message)
    else:
        st.info(message)


def render_disclaimer(message: str = None):
    """Render a prominent, accessible notice about data provenance and limits."""
    msg = message or (
        "Metrics are calculated from the connected source records. Missing data is not replaced with invented values."
    )
    st.markdown(
        f"""
        <div style="background:rgba(251,191,36,.07);border:1px solid rgba(251,191,36,.22);
                    border-left:3px solid #FBBF24;border-radius:10px;padding:10px 14px;margin:0 0 18px;">
          <div style="color:#FBBF24;font-size:11px;font-weight:750;letter-spacing:.09em;">
            SOURCE DATA NOTICE
          </div>
          <div style="color:#C6D2E0;font-size:12px;line-height:1.5;margin-top:3px;">
            {escape(msg)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(message: str = "No data available matching the selected filter criteria."):
    """Render a clear empty state with a concise next step."""
    st.markdown(
        f"""
        <div style="background:#111C2E;border:1px dashed #34465E;border-radius:12px;
                    padding:24px 18px;text-align:center;margin:12px 0;">
          <div style="color:#D7E2EF;font-size:14px;font-weight:600;">{escape(message)}</div>
          <div style="color:#9AAFC4;font-size:12px;margin-top:6px;">
            Adjust the filters above to explore other records.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

