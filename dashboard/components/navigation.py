"""Shared SupplyIQ theme, brand, and page-header components."""
from html import escape

import streamlit as st


def apply_custom_theme():
    """Apply SupplyIQ's shared dark operations-console theme."""
    theme_css = """
        <style>
            :root {
                color-scheme: dark;
                --siq-canvas: #0B1220;
                --siq-surface: #111C2E;
                --siq-surface-raised: #152238;
                --siq-border: #26364B;
                --siq-accent: #22D3EE;
                --siq-positive: #34D399;
                --siq-warning: #FBBF24;
                --siq-critical: #FB7185;
                --siq-text: #F1F5F9;
                --siq-muted: #9AAFC4;
            }

            html, body, [class*="css"] {
                font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                    BlinkMacSystemFont, "Segoe UI", sans-serif;
            }
            .stApp {
                background: var(--siq-canvas);
                color: var(--siq-text);
            }
            [data-testid="stAppViewContainer"] {
                background:
                    radial-gradient(ellipse at 78% 0%, rgba(34, 211, 238, .055), transparent 34%),
                    var(--siq-canvas);
            }
            .main .block-container {
                width: 100%;
                max-width: 1480px;
                padding: 1.35rem clamp(1rem, 3vw, 2.75rem) 3rem;
            }
            [data-testid="stMainBlockContainer"] {
                width: 100%;
                max-width: 1480px;
                padding: 1.35rem clamp(.75rem, 2vw, 1.5rem) 3rem !important;
            }

            section[data-testid="stSidebar"] {
                background: #0D1727;
                border-right: 1px solid var(--siq-border);
                max-width: 17rem;
                min-width: 17rem;
                width: 17rem;
            }
            section[data-testid="stSidebar"] > div {
                background: #0D1727;
            }
            section[data-testid="stSidebar"] .block-container {
                padding: 1.35rem 1rem 1.5rem;
            }
            [data-testid="stSidebarNav"] {
                padding-top: .65rem;
            }
            [data-testid="stSidebarNav"] {
                display: none !important;
            }
            [data-testid="stSidebarNav"] a {
                border-radius: 10px;
                color: #A9B9CC;
                margin: 3px 0;
                transition: background .16s ease, color .16s ease;
            }
            [data-testid="stSidebarNav"] a:hover {
                background: #152238;
                color: var(--siq-text);
            }
            [data-testid="stSidebarNav"] a[aria-current="page"] {
                background: rgba(34, 211, 238, .12);
                box-shadow: inset 3px 0 0 var(--siq-accent);
                color: #E8FCFF;
                font-weight: 650;
            }
            a[data-testid="stPageLink-NavLink"] {
                border-radius: 9px;
                color: #A9B9CC;
                margin: 2px 0;
                transition: background .16s ease, color .16s ease;
            }
            a[data-testid="stPageLink-NavLink"]:hover {
                background: #152238;
                color: var(--siq-text);
            }
            a[data-testid="stPageLink-NavLink"][aria-current="page"] {
                background: rgba(34, 211, 238, .12);
                box-shadow: inset 3px 0 0 var(--siq-accent);
                color: #E8FCFF;
                font-weight: 650;
            }
            a[data-testid="stPageLink-NavLink"][aria-disabled="true"] {
                background: rgba(34, 211, 238, .12);
                box-shadow: inset 3px 0 0 var(--siq-accent);
                color: #E8FCFF;
                font-weight: 650;
                opacity: 1;
            }
            .siq-brand {
                display: flex;
                align-items: center;
                gap: 11px;
                margin: 0 0 5px;
            }
            .siq-brand-mark {
                display: grid;
                width: 38px;
                height: 38px;
                place-items: center;
                border: 1px solid rgba(34, 211, 238, .32);
                border-radius: 12px;
                background: linear-gradient(145deg, rgba(34, 211, 238, .18), rgba(17, 28, 46, .9));
                color: var(--siq-accent);
            }
            .siq-brand-name {
                color: var(--siq-text);
                font-size: 1.1rem;
                font-weight: 750;
                letter-spacing: -.035em;
                line-height: 1.1;
            }
            .siq-brand-subtitle {
                color: var(--siq-muted);
                font-size: .72rem;
                margin-top: 4px;
            }
            .siq-source-status {
                align-items: center;
                background: rgba(251, 191, 36, .07);
                border: 1px solid rgba(251, 191, 36, .18);
                border-radius: 9px;
                color: #D8C99B;
                display: flex;
                font-size: .72rem;
                gap: 8px;
                margin: 1.15rem 0 .8rem;
                padding: 8px 10px;
            }
            .siq-source-dot {
                background: var(--siq-warning);
                border-radius: 50%;
                height: 7px;
                width: 7px;
            }

            .siq-page-header {
                border-bottom: 1px solid rgba(38, 54, 75, .78);
                margin: 0 0 1.15rem;
                padding: 0 0 1rem;
            }
            .siq-eyebrow {
                color: var(--siq-accent);
                font-size: .69rem;
                font-weight: 750;
                letter-spacing: .11em;
                margin-bottom: .34rem;
                text-transform: uppercase;
            }
            .siq-page-title {
                color: var(--siq-text);
                font-size: clamp(1.55rem, 2.6vw, 2rem);
                font-weight: 720;
                letter-spacing: -.045em;
                line-height: 1.15;
                margin: 0;
            }
            h1.siq-page-title {
                color: var(--siq-text) !important;
                font-size: clamp(1.55rem, 2vw, 1.9rem) !important;
                font-weight: 720 !important;
                line-height: 1.15 !important;
            }
            .stMarkdown h1.siq-page-title {
                font-size: clamp(1.55rem, 2vw, 1.9rem) !important;
            }
            .siq-page-description {
                color: var(--siq-muted);
                font-size: .91rem;
                line-height: 1.5;
                margin: .45rem 0 0;
                max-width: 850px;
            }

            h2, h3, h4, h5 {
                color: var(--siq-text) !important;
                letter-spacing: -.025em;
            }
            h2 { font-size: 1.25rem !important; margin-top: 1.1rem !important; }
            h3 { font-size: 1.08rem !important; }
            p, li, label, [data-testid="stCaptionContainer"] {
                color: #CBD7E5;
            }
            [data-testid="stCaptionContainer"] { color: var(--siq-muted); }
            a { color: var(--siq-accent); }
            hr { border-color: var(--siq-border); margin: 1.35rem 0; }

            .siq-metric {
                background: linear-gradient(145deg, #142238 0%, #111C2E 78%);
                border: 1px solid var(--siq-border);
                border-radius: 14px;
                box-shadow: 0 8px 24px rgba(0, 0, 0, .14);
                height: 100%;
                min-height: 116px;
                padding: 16px 17px 14px;
                position: relative;
                transition: border-color .16s ease, transform .16s ease;
            }
            .siq-metric:hover {
                border-color: rgba(34, 211, 238, .48);
                transform: translateY(-2px);
            }
            .siq-metric-label {
                color: #A7B8CC;
                font-size: .77rem;
                font-weight: 600;
                letter-spacing: .015em;
            }
            .siq-metric-value {
                color: var(--siq-text);
                font-size: clamp(1.35rem, 2.1vw, 1.85rem);
                font-weight: 740;
                letter-spacing: -.045em;
                line-height: 1.15;
                margin-top: 10px;
                overflow-wrap: anywhere;
            }
            .siq-metric-delta {
                color: var(--siq-muted);
                font-size: .72rem;
                margin-top: 8px;
            }
            .siq-metric-accent {
                border-left: 3px solid var(--siq-accent);
            }

            .stButton > button, .stDownloadButton > button,
            [data-testid="stFormSubmitButton"] > button {
                background: #14263A;
                border: 1px solid #35516C;
                border-radius: 9px;
                color: #E8F8FC;
                font-weight: 650;
                min-height: 2.45rem;
                transition: background .16s ease, border-color .16s ease, transform .16s ease;
            }
            .stButton > button:hover, .stDownloadButton > button:hover,
            [data-testid="stFormSubmitButton"] > button:hover {
                background: #19364A;
                border-color: var(--siq-accent);
                color: #FFFFFF;
                transform: translateY(-1px);
            }
            button[kind="primary"], [data-testid="stFormSubmitButton"] button {
                background: #0E7490;
                border-color: #22AFC4;
                color: #F0FDFF;
            }
            button[kind="primary"]:hover, [data-testid="stFormSubmitButton"] button:hover {
                background: #0E899F;
            }

            [data-testid="stTextInput"] input,
            [data-testid="stNumberInput"] input,
            [data-testid="stSelectbox"] [data-baseweb="select"],
            [data-testid="stMultiSelect"] [data-baseweb="select"] {
                background-color: #111C2E;
            }
            [data-testid="stTextInput"] input,
            [data-testid="stNumberInput"] input {
                border-color: #34465E;
                border-radius: 8px;
                color: var(--siq-text);
            }
            [data-testid="stTextInput"] input:focus,
            [data-testid="stNumberInput"] input:focus {
                border-color: var(--siq-accent);
                box-shadow: 0 0 0 1px var(--siq-accent);
            }
            [data-testid="stSlider"] [role="slider"] { border-color: var(--siq-accent); }
            [data-testid="stForm"] {
                background: rgba(17, 28, 46, .72);
                border: 1px solid var(--siq-border);
                border-radius: 14px;
                padding: 1.15rem;
            }

            [data-testid="stDataFrame"], [data-testid="stTable"] {
                border: 1px solid var(--siq-border);
                border-radius: 11px;
                overflow: hidden;
            }
            [data-testid="stAlert"] {
                background: #111C2E;
                border-radius: 11px;
            }
            [data-testid="stChatMessage"] {
                background: #111C2E;
                border: 1px solid var(--siq-border);
                border-radius: 14px;
                padding: .8rem 1rem;
            }
            [data-testid="stChatInput"] textarea {
                background: #111C2E;
                border-color: #34465E;
                border-radius: 12px;
            }

            @media (max-width: 760px) {
                .main .block-container { padding: 1rem .85rem 2rem; }
                .siq-metric { min-height: 103px; padding: 13px; }
                [data-testid="stForm"] { padding: .8rem; }
            }
            @media (max-width: 960px) {
                [data-testid="stHorizontalBlock"] {
                    flex-wrap: wrap !important;
                    gap: 12px !important;
                }
                [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                    flex: 1 1 100% !important;
                    min-width: 100% !important;
                    width: 100% !important;
                }
            }
        </style>
        """
    st.markdown(theme_css, unsafe_allow_html=True)


def render_sidebar_header(active_page: str = "workspace"):
    """Render SupplyIQ branding, custom navigation, and source-data status."""
    apply_custom_theme()
    st.sidebar.markdown(
        """
        <div class="siq-brand">
          <div class="siq-brand-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" width="23" height="23" fill="none">
              <path d="M4 8.5 12 4l8 4.5v7L12 20l-8-4.5v-7Z" stroke="currentColor" stroke-width="1.6"/>
              <path d="m4.5 8.7 7.5 4.2 7.5-4.2M12 13v6.5M8 6.2l8 4.5" stroke="currentColor" stroke-width="1.5"/>
              <circle cx="12" cy="11" r="1.15" fill="currentColor"/>
            </svg>
          </div>
          <div>
            <div class="siq-brand-name">SupplyIQ</div>
            <div class="siq-brand-subtitle">Supply Chain Intelligence</div>
          </div>
        </div>
        <div class="siq-source-status"><span class="siq-source-dot"></span>Source data · original CSV files</div>
        """,
        unsafe_allow_html=True,
    )
    active_titles = {
        "workspace": "SupplyIQ workspace",
        "executive": "Executive overview",
        "regional": "Regional order explorer",
        "inventory": "Supplier & inventory intelligence",
        "fuel": "Fuel consumption records",
        "prediction": "Risk assessment",
        "scenario": "What-if simulator",
        "reports": "Reports & exports",
        "assistant": "AI assistant",
        "information": "Project information",
    }
    current_title = active_titles.get(active_page, "Workspace")
    st.sidebar.markdown(
        f"""
        <div style="background:rgba(34,211,238,.08);border:1px solid rgba(34,211,238,.2);
                    border-left:3px solid #22D3EE;border-radius:9px;margin:12px 0 10px;padding:8px 10px;">
          <div style="color:#75DDEB;font-size:9px;font-weight:750;letter-spacing:.1em;">CURRENT VIEW</div>
          <div style="color:#EAFBFE;font-size:12px;font-weight:650;margin-top:3px;">{escape(current_title)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if hasattr(st, "page_link"):
        st.sidebar.markdown(
            '<div style="color:#71859C;font-size:10px;font-weight:750;'
            'letter-spacing:.1em;margin:14px 0 6px;text-transform:uppercase;">Overview</div>',
            unsafe_allow_html=True,
        )
        _render_page_link("app.py", "Workspace", ":material/home:")
        _render_page_link(
            "pages/01_Executive_Overview.py",
            "Executive overview",
            ":material/dashboard:",
        )
        st.sidebar.markdown(
            '<div style="color:#71859C;font-size:10px;font-weight:750;'
            'letter-spacing:.1em;margin:16px 0 6px;text-transform:uppercase;">Business modules</div>',
            unsafe_allow_html=True,
        )
        _render_page_link(
            "pages/02_Regional_Risk.py",
            "Regional order explorer",
            ":material/public:",
        )
        _render_page_link(
            "pages/03_Inventory_Supplier.py",
            "Supplier & inventory",
            ":material/inventory_2:",
        )
        _render_page_link(
            "pages/04_Fuel_Weather_Analytics.py",
            "Fuel consumption",
            ":material/local_gas_station:",
        )
        st.sidebar.markdown(
            '<div style="color:#71859C;font-size:10px;font-weight:750;'
            'letter-spacing:.1em;margin:16px 0 6px;text-transform:uppercase;">Planned capabilities</div>',
            unsafe_allow_html=True,
        )
        _render_page_link(
            "pages/05_Risk_Prediction.py",
            "Risk assessment",
            ":material/rule:",
        )
        _render_page_link(
            "pages/06_What_If_Simulator.py",
            "What-if simulator",
            ":material/tune:",
        )
        _render_page_link(
            "pages/07_Reports_Exports.py",
            "Reports & exports",
            ":material/description:",
        )
        _render_page_link(
            "pages/08_AI_Assistant.py",
            "AI assistant",
            ":material/chat:",
        )
        _render_page_link(
            "pages/09_Project_Info_Model_Trust.py",
            "Project information",
            ":material/info:",
        )


def _render_page_link(page: str, label: str, icon: str):
    """Place one native page link inside the sidebar navigation rail."""
    with st.sidebar:
        st.page_link(page, label=label, icon=icon)


def render_page_header(section: str, title: str, description: str):
    """Render a compact, consistent title block for each application page."""
    st.markdown(
        f"""
        <header class="siq-page-header">
          <div class="siq-eyebrow">{escape(section)}</div>
          <h1 class="siq-page-title" style="font-size:clamp(1.55rem,2vw,1.9rem)!important;
              font-weight:720!important;line-height:1.15!important;">{escape(title)}</h1>
          <p class="siq-page-description">{escape(description)}</p>
        </header>
        """,
        unsafe_allow_html=True,
    )
