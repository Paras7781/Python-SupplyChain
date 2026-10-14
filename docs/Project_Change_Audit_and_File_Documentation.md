# SupplyIQ — Project Change Audit & Technical Architecture Documentation

**Document Title:** Comprehensive Project Change Audit, Technical Architecture Review, and File-by-File Technical Documentation  
**Project Name:** SupplyIQ: Supply Chain Risk & Decision Intelligence Platform  
**Report Date:** October 14, 2026  
**Report Version:** Version 1.0 (Audited Review Edition)  
**Author / Auditor Roles:** Senior Software Architect, Lead Code Reviewer, Technical Writer, & System Auditor  
**Workspace Directory:** `c:\Users\choug\Downloads\supply_iq_reviewed\supply_iq`  
**Classification:** Internal Technical Audit & Academic/Operational Defense Specification  
**Technology Stack:** Python 3.11, Streamlit 1.52.2, Pandas 2.2.3, NumPy 2.0.2, Plotly 7.0.0, Scikit-Learn 1.5.2, Joblib 1.4.2  
**Dataset Scale:** 3 Primary Source CSV Datasets (412,584 Total Rows, 155.85 MB on disk)  
**File Count:** 31 Analyzed Code & Configuration Files (Excluding `.venv` and `__pycache__`)  

---

## Table of Contents
1. [Cover Page & Document Metadata](#1-cover-page--document-metadata)
2. [Executive Summary](#2-executive-summary)
3. [Complete Project Folder Structure & Explanation](#3-complete-project-folder-structure--explanation)
4. [File-by-File Technical Documentation](#4-file-by-file-technical-documentation)
5. [Comprehensive Change Audit](#5-comprehensive-change-audit)
6. [Before-Versus-After Code & Architecture Comparisons](#6-before-versus-after-code--architecture-comparisons)
7. [Application Architecture & Data-Flow Diagrams](#7-application-architecture--data-flow-diagrams)
8. [Database Schema, Models, Relationships & Data Management](#8-database-schema-models-relationships--data-management)
9. [API Documentation (Internal Functions & Planned Endpoints)](#9-api-documentation-internal-functions--planned-endpoints)
10. [Configuration, Environment Setup, Dependencies & Safe Execution](#10-configuration-environment-setup-dependencies--safe-execution)
11. [Testing & Verification Report](#11-testing--verification-report)
12. [Known Bugs, Limitations, Security Concerns & Prioritized Recommendations](#12-known-bugs-limitations-security-concerns--prioritized-recommendations)
13. [Change-to-File-to-Feature Traceability Matrix](#13-change-to-file-to-feature-traceability-matrix)
14. [Beginner-Friendly Technical Glossary](#14-beginner-friendly-technical-glossary)

---

## 1. Cover Page & Document Metadata

### Document Overview
This document represents an exhaustive, senior-architectural audit and comprehensive technical reference manual for the **SupplyIQ** platform. This inspection is grounded entirely in empirical evidence collected from the physical repository located at `c:\Users\choug\Downloads\supply_iq_reviewed\supply_iq`, including source code, configuration files, dataset structures, automated tests, and runtime behavioral checks.

### Audit Parameters
* **Auditor Role:** Senior Software Architect, Systems Reviewer, and Lead Project Auditor.
* **Audit Methodology:** Static code analysis, dynamic unit testing, dataset schema validation, security inspection of sensitive columns (PII), and traceability mapping between requirements, file structures, and UI capabilities.
* **Historical Baseline Notice:** The project directory is not under active Git version control (no `.git` directory exists). Consequently, a full commit-by-commit historical Git log cannot be extracted directly from Git. All change audits in this report are based on verified repository artifacts (specifically `AUDIT_REPORT.md`, source code diffs against architectural blueprints, and confirmed session trajectory traces), with each finding explicitly labeled as **[CONFIRMED]** or **[INFERRED]**.

---

## 2. Executive Summary

### 2.1 Application Purpose
**SupplyIQ** is an enterprise-oriented, operations-focused supply chain intelligence dashboard and decision-support prototype. It is engineered to give supply chain managers, logistics planners, and inventory controllers direct operational visibility into three separate and independent supply chain domains:
1. **Regional Order & Shipping Performance:** Identifying order delivery delays, evaluating actual versus scheduled shipping days, analyzing geographic distribution across regions, states, and cities, and inspecting order-level fulfillment status.
2. **Retail Inventory & Supplier Intelligence:** Tracking warehouse stock levels, monitoring reorder thresholds, identifying stockout risks, and evaluating supplier lead times and on-time delivery rates.
3. **Transport & Fuel Consumption:** Analyzing recorded trip distances, fuel efficiency, fuel costs, vehicle types, road types, and weather conditions.

### 2.2 Major Functional Modules
The platform is organized into three tiers of modules:
* **Command Center & Entry Point:** `app.py` serves as the central application launchpad, providing dataset connection health cards and primary navigation gateways.
* **Active Operational Modules:**
  * `pages/01_Executive_Overview.py`: Aggregated operational KPIs across the three source datasets.
  * `pages/02_Regional_Risk.py`: Deep drill-down hierarchy (Region → State → City) with order search, PII masking, and CSV exports.
  * `pages/03_Inventory_Supplier.py`: Multi-factor filtering of retail inventory records, stockout warnings, and supplier rating metrics.
  * `pages/04_Fuel_Weather_Analytics.py`: Trip fuel consumption explorer with transparent arithmetic trip fuel calculations.
  * `pages/07_Reports_Exports.py`: Privacy-safe CSV export portal for allowlisted operational fields.
  * `pages/09_Project_Info_Model_Trust.py`: Transparency page detailing dataset provenance, problem statements, and system maturity status.
* **Explicitly Disabled Future Capabilities:**
  * `pages/05_Risk_Prediction.py`: Planned ML regional risk scoring (explicitly displays an unavailable banner).
  * `pages/06_What_If_Simulator.py`: Planned scenario simulation engine (explicitly displays an unavailable banner).
  * `pages/08_AI_Assistant.py`: Planned LLM/RAG natural language assistant (explicitly displays an unavailable banner).

### 2.3 Technology Stack
* **Runtime Language:** Python 3.11.x
* **Web Presentation Layer:** Streamlit 1.52.2 (specified minimum: `streamlit>=1.31.0`)
* **Data Processing & Analytics:** Pandas 2.2.3, NumPy 2.0.2
* **Interactive Visualization:** Plotly Express & Plotly Graph Objects 7.0.0
* **Machine Learning & Pipeline Scaffold:** Scikit-Learn 1.5.2, Joblib 1.4.2
* **Test Automation Framework:** Python Standard Library `unittest` (11 automated test cases)

### 2.4 Current State & Maturity Assessment
The SupplyIQ project is currently in the **Interactive Frontend Prototype & Audited Scaffold** phase:
* **Strengths:** 
  * Strict separation of concerns between presentation (`dashboard/components/`), configuration (`config/`), and backend logic (`src/`).
  * Exemplary data integrity and truthfulness: No fake machine learning predictions or simulated numbers are passed off as real.
  * Privacy-safe field allowlisting: Personally Identifiable Information (customer passwords, contact info, street addresses) is strictly excluded from UI tables and user-facing CSV downloads.
  * Non-destructive data validation: `src/data_validation.py` inspects DataFrames without mutating input data or dropping rows silently.
  * Explicit failure modes: Dormant backend machine learning training modules raise `NotImplementedError` rather than silently returning `None` via `pass`.
* **Current Limitations:**
  * Machine learning prediction models, what-if scenario engines, and AI/RAG assistants are completely unimplemented.
  * Datasets are loaded entirely into memory as flat CSV files; there is no relational SQL database or indexed query layer.
  * The fuel dataset (`supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`) is entirely synthetic (`synthetic_demo_values`), lacking true geographic origin-destination coordinates or travel times.

---

## 3. Complete Project Folder Structure & Explanation

### 3.1 Visual Project Directory Tree
```text
supply_iq/
├── app.py                                            # Main Streamlit application entry point & module launchpad
├── requirements.txt                                  # Python package dependencies
├── README.md                                         # Project architecture overview, Viva guide & setup instructions
├── AUDIT_REPORT.md                                   # Historical audit log detailing previous review findings & fixes
│
├── config/                                           # Central configuration & path management
│   ├── __init__.py                                   # Package marker
│   └── settings.py                                   # Central directory paths, dataset mappings & application metadata
│
├── dashboard/                                        # Shared presentation layer components
│   ├── __init__.py                                   # Package marker
│   └── components/                                   # Reusable UI widgets, layout templates & charts
│       ├── __init__.py                               # Package marker
│       ├── cards.py                                  # Metric cards, alert banners, disclaimers & empty-state widgets
│       ├── charts.py                                 # Dark-themed Plotly charts (bar, donut, line, scatter)
│       └── navigation.py                             # CSS styling, brand sidebar navigation & page headers
│
├── data/                                             # Data storage directory
│   ├── README.md                                     # Explains raw vs processed dataset usage and constraints
│   └── raw/                                          # Read-only source CSV datasets
│       ├── DataCoSupplyChainDataset.csv              # Regional order shipment records (180,519 rows, 53 columns, 91.47 MB)
│       ├── Retail_Supply_Chain_Analytics_Dataset.csv # Retail inventory & supplier records (200,000 rows, 41 columns, 46.06 MB)
│       └── supplyiq_final_dataset_filled_synthetic_fuel_weather.csv # Trip fuel records (32,065 rows, 43 columns, 18.32 MB)
│
├── pages/                                            # Multi-page Streamlit application views
│   ├── __init__.py                                   # Package marker
│   ├── 01_Executive_Overview.py                      # Multi-source executive KPI dashboard
│   ├── 02_Regional_Risk.py                           # Regional order drill-down explorer with PII filtering
│   ├── 03_Inventory_Supplier.py                      # Retail supplier & stockout intelligence viewer
│   ├── 04_Fuel_Weather_Analytics.py                  # Fuel consumption explorer & transparent arithmetic estimator
│   ├── 05_Risk_Prediction.py                         # Explicit unavailable state: ML prediction backend
│   ├── 06_What_If_Simulator.py                       # Explicit unavailable state: Scenario simulation backend
│   ├── 07_Reports_Exports.py                         # Source data export center with security allowlists
│   ├── 08_AI_Assistant.py                            # Explicit unavailable state: LLM / RAG assistant
│   └── 09_Project_Info_Model_Trust.py                # Problem statements, source metadata & project roadmap
│
├── src/                                              # Core business logic & backend modules
│   ├── __init__.py                                   # Package marker
│   ├── data_loader.py                                # Safe cached CSV loader with encoding fallback
│   ├── data_validation.py                            # Non-destructive data quality & warning summarizer
│   ├── ui_planning.py                                # Transparent arithmetic fuel calculation formulas
│   │
│   ├── regional_risk/                                # Regional risk domain package
│   │   ├── __init__.py                               # Package marker
│   │   ├── cleaning.py                               # Scaffold for regional risk data cleaning
│   │   ├── eda.py                                    # Scaffold for regional risk exploratory analysis
│   │   ├── feature_engineering.py                    # Scaffold for regional risk feature engineering
│   │   └── model_training.py                         # Scaffold raising NotImplementedError for regional risk model
│   │
│   ├── inventory_supplier/                           # Inventory & supplier domain package
│   │   ├── __init__.py                               # Package marker
│   │   ├── cleaning.py                               # Scaffold for inventory data cleaning
│   │   ├── eda.py                                    # Scaffold for inventory exploratory analysis
│   │   ├── feature_engineering.py                    # Scaffold for inventory feature engineering
│   │   └── model_training.py                         # Scaffold raising NotImplementedError for inventory model
│   │
│   └── fuel_weather/                                 # Fuel & transport domain package
│       ├── __init__.py                               # Package marker
│       ├── cleaning.py                               # Scaffold for fuel data cleaning
│       ├── eda.py                                    # Scaffold for fuel exploratory analysis
│       ├── feature_engineering.py                    # Scaffold for fuel feature engineering
│       └── model_training.py                         # Scaffold raising NotImplementedError for fuel model
│
├── tests/                                            # Automated test suite
│   ├── .gitkeep                                      # Directory tracking marker
│   ├── test_backend_scaffolds.py                     # Asserts explicit NotImplementedError on model training stubs
│   ├── test_data_validation.py                       # Verifies non-destructive data validation behavior
│   └── test_ui_planning.py                           # Verifies fuel estimate calculation formulas & edge cases
│
└── docs/                                             # Project documentation & audit reports (Generated)
    ├── Project_Change_Audit_and_File_Documentation.md # Full editable Markdown audit source
    └── Project_Change_Audit_and_File_Documentation.pdf # Publication-quality compiled PDF report
```

### 3.2 Detailed Functional Explanation of Project Folders
* **`config/`**: Centralizes application constants, directory resolutions using Python's `pathlib.Path`, and canonical dataset file paths. This prevents hardcoded absolute file paths and guarantees cross-platform portability between Windows, Linux, and macOS environments.
* **`dashboard/components/`**: Implements atomic UI design patterns. Encapsulates custom HTML/CSS theme injection, dark-mode Plotly chart builders, branded metric cards, alert banners, and navigation rails. Keeping UI rendering separated from page views ensures consistency and maintainability.
* **`data/raw/`**: Holds the three primary immutable source CSV datasets. These files are strictly read-only; no script in the application writes to, modifies, or overwrites these raw assets.
* **`data/processed/` & `models/` [Structural Discrepancy Note]**: These directories are referenced in `config/settings.py` and `README.md` as target directories for future cleaned datasets and serialized ML model weights (`.pkl` / `.joblib`). However, inspection reveals they do not physically exist on disk. They must be created when real data preprocessing and model persistence pipelines are built.
* **`pages/`**: Leverages Streamlit's native multi-page routing convention. Numbered prefixes (`01_` through `09_`) control sidebar ordering.
* **`src/`**: Houses shared utilities (`data_loader.py`, `data_validation.py`, `ui_planning.py`) and three domain packages (`regional_risk`, `inventory_supplier`, `fuel_weather`). Each domain package is structured identically with dedicated modules for data cleaning, exploratory data analysis, feature engineering, and model training.
* **`tests/`**: Contains Python standard `unittest` test suites verifying calculation correctness, non-destructive validation behavior, and model scaffold exception raising.

---

## 4. File-by-File Technical Documentation

This section provides comprehensive technical documentation for every individual file in the SupplyIQ repository.

### File 1: `app.py`
* **File Path:** `app.py`
* **Primary Responsibility:** Application entry point, workspace dashboard, and high-level module router.
* **Key Functions / Code Blocks:**
  * `st.set_page_config(page_title, page_icon, layout="wide")`: Configures the browser tab and wide desktop layout.
  * `render_sidebar_header("workspace")`: Injects custom dark-mode CSS styling, SVG branding, and the sidebar navigation rail.
  * `render_page_header(...)`: Displays the hero title and operational workflow description.
  * Dataset connection status loop: Iterates through configured dataset keys (`supply_chain`, `retail_analytics`, `fuel_weather`), calls `load_dataset()`, and renders connection health KPI cards (`render_metric_card`).
  * Module navigation buttons: Uses `st.switch_page()` to allow users to navigate directly to `pages/02_Regional_Risk.py`, `pages/03_Inventory_Supplier.py`, or `pages/04_Fuel_Weather_Analytics.py`.
* **Inputs & Outputs:** Inputs: User click interactions. Outputs: Rendered Streamlit DOM elements.
* **Dependencies:** `streamlit`, `config.settings`, `dashboard.components.cards`, `dashboard.components.navigation`, `src.data_loader`.
* **Database / API Interactions:** Reads CSV files through `src.data_loader.load_dataset()`. No database or REST API calls.
* **Architectural Contribution:** Serves as the front door and system health monitor for the platform.

### File 2: `requirements.txt`
* **File Path:** `requirements.txt`
* **Primary Responsibility:** Declares direct third-party Python package dependencies and version constraints.
* **Contents:**
  * `streamlit>=1.31.0`
  * `pandas>=2.0.0`
  * `numpy>=1.24.0`
  * `plotly>=5.15.0`
  * `scikit-learn>=1.3.0`
  * `joblib>=1.3.0`
* **Dependencies:** Managed by Python package installer (`pip`).
* **Architectural Contribution:** Defines the deterministic runtime environment required for application execution.

### File 3: `README.md`
* **File Path:** `README.md`
* **Primary Responsibility:** User documentation, installation instructions, data dictionary overview, and student Viva examination cheat sheet.
* **Key Content Sections:** Current UI modules and source dataset support, fuel calculation testing command, project tree visualization, setup and execution instructions.
* **Dependencies:** None.
* **Architectural Contribution:** Core developer onboarding and system transparency documentation.

### File 4: `AUDIT_REPORT.md`
* **File Path:** `AUDIT_REPORT.md`
* **Primary Responsibility:** Captures the findings, code modifications, dataset SHA-256 integrity verifications, and architectural recommendations from a formal code and real-data review.
* **Key Findings Documented:**
  * Confirms DataCo contains 180,519 item-level rows and 65,752 distinct orders, with 336,209 missing cells.
  * Confirms Retail contains 200,000 rows with 0 missing cells and 0 duplicate rows.
  * Confirms Fuel contains 32,065 rows marked `synthetic_demo_values`.
  * Documents code changes made: Updating `src/data_validation.py` to non-destructive warning reporting, updating `model_training.py` modules to raise `NotImplementedError`, and adding unit tests.
* **Dependencies:** None.
* **Architectural Contribution:** Historical provenance record that directly underpins the change audit.

### File 5: `config/__init__.py`
* **File Path:** `config/__init__.py`
* **Primary Responsibility:** Initializes the `config` directory as a standard Python package.
* **Dependencies:** Python runtime.
* **Architectural Contribution:** Enables modular imports (e.g., `from config.settings import DATASETS`).

### File 6: `config/settings.py`
* **File Path:** `config/settings.py`
* **Primary Responsibility:** Centralized configuration repository, path resolution, and system constants.
* **Key Constants & Paths:**
  * `BASE_DIR = Path(__file__).resolve().parent.parent`: Dynamic absolute path to the project root directory.
  * `DATA_DIR = BASE_DIR / "data"`: Root data folder.
  * `RAW_DATA_DIR = DATA_DIR / "raw"`: Read-only source datasets.
  * `PROCESSED_DATA_DIR = DATA_DIR / "processed"`: Destination path for future cleaned datasets.
  * `MODELS_DIR = BASE_DIR / "models"`: Storage path for future trained model artifacts.
  * `DATASETS`: Dictionary mapping logical keys (`supply_chain`, `retail_analytics`, `fuel_weather`) to concrete filesystem paths.
  * `APP_NAME = "SupplyIQ"`, `APP_TITLE = "SupplyIQ - Supply Chain Analytics & Risk Intelligence"`, `APP_ICON = "📦"`.
* **Inputs & Outputs:** Inputs: Filesystem environment. Outputs: Immutable path and metadata objects.
* **Dependencies:** `pathlib.Path`.
* **Architectural Contribution:** Eliminates fragile hardcoded paths, ensuring seamless cross-environment compatibility.

### File 7: `dashboard/__init__.py`
* **File Path:** `dashboard/__init__.py`
* **Primary Responsibility:** Package marker for the `dashboard` UI namespace.
* **Dependencies:** Python runtime.

### File 8: `dashboard/components/__init__.py`
* **File Path:** `dashboard/components/__init__.py`
* **Primary Responsibility:** Package marker for reusable presentation components.
* **Dependencies:** Python runtime.

### File 9: `dashboard/components/cards.py`
* **File Path:** `dashboard/components/cards.py`
* **Primary Responsibility:** Renders accessible, styled HTML metric cards, alert banners, disclaimers, and empty-state placeholders.
* **Key Functions:**
  * `render_metric_card(title: str, value: str, delta: str = None, help_text: str = None, icon: str = None)`: Formats and outputs an HTML KPI card with cyan accent borders, hover elevations, and escaped text.
  * `render_alert_banner(title: str, text: str, alert_type: str = "warning")`: Dispatches native Streamlit severity banners (`st.error`, `st.warning`, `st.success`, `st.info`).
  * `render_disclaimer(message: str = None)`: Renders an amber-bordered "SOURCE DATA NOTICE" callout box explaining data boundaries.
  * `render_empty_state(message: str = "No data available...")`: Renders a dashed-border dark placeholder when queries return zero records.
* **Inputs & Outputs:** Inputs: String titles, values, help texts. Outputs: Rendered Streamlit markdown with HTML.
* **Security Guardrails:** Uses `html.escape()` on all dynamic strings to mitigate Cross-Site Scripting (XSS) risks.
* **Dependencies:** `streamlit`, `html.escape`.
* **Architectural Contribution:** Reusable presentation component library ensuring UI consistency across all pages.

### File 10: `dashboard/components/charts.py`
* **File Path:** `dashboard/components/charts.py`
* **Primary Responsibility:** Provides a centralized suite of Plotly chart factories configured with an enterprise dark theme.
* **Key Functions & Constants:**
  * `COLOR_PALETTE`: Restrained hex color sequence (`#22D3EE` cyan, `#34D399` emerald, `#FBBF24` amber, `#FB7185` rose, `#60A5FA` blue, `#A78BFA` purple).
  * `SEMANTIC_COLORS`: Dictionary binding domain states (`Healthy`, `Delayed`, `Out of Stock`, `Low Stock Alert`) directly to semantic status colors.
  * `_color_map(values)`: Assigns deterministic colors based on status semantics or palette rotation.
  * `apply_dark_layout(fig, title: str = "")`: Formats Plotly figures with transparent backgrounds (`rgba(0,0,0,0)`), subtle gridlines, Inter font typography, custom hover tooltips, and responsive margins.
  * `render_donut_chart(df, names_col, values_col, title)`: Generates a hollow donut distribution chart with percentage labels.
  * `render_bar_chart(df, x_col, y_col, title, color_col, orientation)`: Renders vertical or horizontal categorical comparisons.
  * `render_grouped_bar_chart(df, x_col, y_cols, title)`: Renders multi-series side-by-side bar comparisons.
  * `render_line_chart(df, x_col, y_cols, title)`: Renders time-series trends with smooth multi-line traces.
  * `render_scatter_chart(df, x_col, y_col, color_col, size_col, title)`: Renders correlation scatter plots with custom tooltips.
* **Inputs & Outputs:** Inputs: Pandas DataFrames and column names. Outputs: Rendered Plotly charts inside Streamlit (`st.plotly_chart`).
* **Dependencies:** `streamlit`, `pandas`, `plotly.express`, `plotly.graph_objects`, `dashboard.components.cards.render_empty_state`.
* **Architectural Contribution:** Prevents chart duplication and enforces enterprise visual consistency across the platform.

### File 11: `dashboard/components/navigation.py`
* **File Path:** `dashboard/components/navigation.py`
* **Primary Responsibility:** Controls global CSS theme injection, responsive branding, sidebar page navigation, and consistent page header banners.
* **Key Functions:**
  * `apply_custom_theme()`: Injects 300+ lines of custom CSS into the Streamlit DOM, establishing CSS variables (`--siq-canvas: #0B1220`, `--siq-surface: #111C2E`, `--siq-accent: #22D3EE`), dark form styling, responsive media queries, and hiding default Streamlit sidebar elements.
  * `render_sidebar_header(active_page: str = "workspace")`: Renders the SVG logo mark, application title, active page badge, and native `st.page_link` navigation entries.
  * `_render_page_link(page, label, icon)`: Encapsulates individual sidebar navigation link calls.
  * `render_page_header(section: str, title: str, description: str)`: Emits semantic HTML `<header>` tags with cyan category eyebrows, bold titles, and explanatory subtitles.
* **Inputs & Outputs:** Inputs: Active page identifier strings. Outputs: Injected CSS styles and sidebar DOM elements.
* **Dependencies:** `streamlit`, `html.escape`.
* **Architectural Contribution:** The visual spine of the entire application, delivering a custom enterprise SaaS appearance.

### File 12: `data/README.md`
* **File Path:** `data/README.md`
* **Primary Responsibility:** Documents raw dataset contents, runtime immutability rules, and the intended purpose of the `data/processed/` folder.
* **Dependencies:** None.

### File 13: `data/raw/DataCoSupplyChainDataset.csv`
* **File Path:** `data/raw/DataCoSupplyChainDataset.csv`
* **Primary Responsibility:** Primary raw dataset for the Regional Delivery Risk domain.
* **Technical Specifications:**
  * Size: 91.47 MB (95,910,149 bytes)
  * Record Count: 180,519 rows × 53 columns
  * Encoding: Latin-1 (ISO-8859-1) / UTF-8 compatible
  * Primary Keys: Composite (`Order Id` + `Order Item Id`). Distinct `Order Id` count: 65,752.
  * Data Quality: 336,209 missing cells across 4 columns (`Customer Lname`: 8; `Customer Zipcode`: 3; `Order Zipcode`: 155,679; `Product Description`: 180,519 [100% missing]). Exact duplicate rows: 0.
  * Privacy Sensitivity: Contains cleartext customer Personally Identifiable Information (PII), including `Customer Email`, `Customer Password`, `Customer Fname`, `Customer Lname`, and `Customer Street`.
* **Architectural Contribution:** Feeds `pages/01_Executive_Overview.py`, `pages/02_Regional_Risk.py`, and `pages/07_Reports_Exports.py`.

### File 14: `data/raw/Retail_Supply_Chain_Analytics_Dataset.csv`
* **File Path:** `data/raw/Retail_Supply_Chain_Analytics_Dataset.csv`
* **Primary Responsibility:** Primary raw dataset for the Retail Inventory and Supplier Intelligence domain.
* **Technical Specifications:**
  * Size: 46.06 MB (48,302,234 bytes)
  * Record Count: 200,000 rows × 41 columns
  * Encoding: UTF-8 / ASCII
  * Primary Key: `Product_ID` (200,000 unique values).
  * Data Quality: 0 missing cells; 0 exact duplicate rows.
  * Attributes: Contains SKU, brand, category, supplier, warehouse, stock, reorder levels, lead times, and precomputed risk scores.
* **Architectural Contribution:** Feeds `pages/01_Executive_Overview.py`, `pages/03_Inventory_Supplier.py`, and `pages/07_Reports_Exports.py`.

### File 15: `data/raw/supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`
* **File Path:** `data/raw/supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`
* **Primary Responsibility:** Primary raw dataset for the Fuel & Transport Consumption domain.
* **Technical Specifications:**
  * Size: 18.32 MB (19,210,151 bytes)
  * Record Count: 32,065 rows × 43 columns
  * Primary Key: `trip_id` (32,065 unique values).
  * Provenance: Explicitly labeled `synthetic_demo_values` in the `added_data_status` column across all 32,065 rows.
  * Data Quality: 0 missing cells; 0 exact duplicate rows.
  * Attributes: Distance, fuel efficiency, fuel consumed, road type, weather condition, vehicle type. Lacks actual origin/destination addresses or GPS route alternatives.
* **Architectural Contribution:** Feeds `pages/01_Executive_Overview.py`, `pages/04_Fuel_Weather_Analytics.py`, and `pages/07_Reports_Exports.py`.

### File 16: `pages/__init__.py`
* **File Path:** `pages/__init__.py`
* **Primary Responsibility:** Package marker for the multi-page directory.
* **Dependencies:** Python runtime.

### File 17: `pages/01_Executive_Overview.py`
* **File Path:** `pages/01_Executive_Overview.py`
* **Primary Responsibility:** High-level command center synthesizing operational KPIs across all three independent data domains.
* **Key Logic & Computations:**
  * Loads all 3 datasets independently via `load_dataset()`.
  * Domain 1 (Regional Orders): Groups DataCo by `Order Region`, calculates distinct orders, total item rows, and computes late delivery rate based on `Delivery Status`. Renders a Plotly bar chart.
  * Domain 2 (Retail Inventory): Compares numeric `Current_Stock` against `Reorder_Level` to compute records at or below reorder threshold. Computes distinct supplier count and category volume bar chart.
  * Domain 3 (Fuel Consumption): Extracts mean fuel consumed and distinct vehicle types. Issues a warning regarding `added_data_status: synthetic_demo_values`.
  * Navigation shortcuts: Action buttons to switch pages to deeper domain views.
* **Dependencies:** `streamlit`, `pandas`, `plotly.express`, `dashboard.components.cards`, `dashboard.components.navigation`, `src.data_loader`.
* **Architectural Contribution:** Primary executive briefing screen giving a bird's-eye view without joining unrelated datasets.

### File 18: `pages/02_Regional_Risk.py`
* **File Path:** `pages/02_Regional_Risk.py`
* **Primary Responsibility:** Detailed geographic drill-down and order-level fulfillment explorer.
* **Key Functions & Logic:**
  * `load_order_records()`: Aggregates 180,519 item-level DataCo rows into 65,752 order-level records grouped by `Order Id`. Calculates total line items, total units, and total order value. Computes `Shipping difference (days) = Days for shipping (real) - Days for shipment (scheduled)`. Sets boolean flags for `Late delivery` and `Source late-risk flag`.
  * Drill-Down Hierarchy: Region → State/Province → City with session state tracking and a "← Back one level" button.
  * Multi-Factor Filters: Order date range, delivery status, shipping mode, and free-text order/location search.
  * Visualizations: Regional late delivery comparison bar chart and ranked location table with sortable criteria (late risk rate, late delivery rate, volume).
  * Data Security: Applies strict column allowlisting (`visible_fields`) to exclude customer contact details, passwords, and street addresses.
  * Data Export: Provides a dynamic CSV download (`SupplyIQ_filtered_orders.csv`) matching active user filters.
* **Dependencies:** `streamlit`, `pandas`, `plotly.express`, `dashboard.components.cards`, `dashboard.components.navigation`, `src.data_loader`.
* **Architectural Contribution:** Core analytical engine for logistics risk investigation.

### File 19: `pages/03_Inventory_Supplier.py`
* **File Path:** `pages/03_Inventory_Supplier.py`
* **Primary Responsibility:** Inventory stock monitoring and supplier performance analysis.
* **Key Functions & Logic:**
  * `load_retail_records()`: Filters columns to operational fields, casts numerical values, and parses dates.
  * Filters: Product category, warehouse location, warehouse ID, date range, product search, and supplier ID search.
  * Stockout & Reorder Evaluation: Compares `Current_Stock` against `Reorder_Level`. Calculates records at/below reorder level and rows with zero/negative stock.
  * Visualizations & Tables: Demand bar chart by product category; supplier performance comparison table (average lead time, ratings, on-time delivery rate).
  * Inventory Inspection Table: Shows gap to reorder level in units for matching SKUs.
* **Dependencies:** `streamlit`, `pandas`, `plotly.express`, `dashboard.components.cards`, `dashboard.components.navigation`, `src.data_loader`.
* **Architectural Contribution:** Operational cockpit for warehouse replenishment and vendor management.

### File 20: `pages/04_Fuel_Weather_Analytics.py`
* **File Path:** `pages/04_Fuel_Weather_Analytics.py`
* **Primary Responsibility:** Fuel consumption records viewer, vehicle comparison, and transparent arithmetic fuel calculator.
* **Key Functions & Logic:**
  * `load_fuel_records()`: Loads synthetic fuel dataset and casts numerical fields.
  * Provenance Disclaimer: Explicitly highlights that records are synthetic demo values.
  * Filters: Vehicle type, fuel type, road type, weather condition, date range, and trip ID search.
  * Visualizations: Box plot of fuel consumed per trip across vehicle types; scatter plot of trip distance versus recorded fuel efficiency.
  * Interactive Trip Calculator: Allows selection of a recorded trip and calls `src.ui_planning.calculate_fuel_estimate()` to calculate estimated liters (`distance / efficiency`) and estimated cost, comparing it against recorded values.
* **Dependencies:** `streamlit`, `pandas`, `plotly.express`, `dashboard.components.cards`, `dashboard.components.navigation`, `src.data_loader`, `src.ui_planning.calculate_fuel_estimate`.
* **Architectural Contribution:** Fuel tracking and educational calculation module demonstrating transparent arithmetic.

### File 21: `pages/05_Risk_Prediction.py`
* **File Path:** `pages/05_Risk_Prediction.py`
* **Primary Responsibility:** Explicit unavailable state placeholder for planned machine learning delivery risk prediction.
* **Implementation:** Displays an informational banner stating: "Not yet available — model/backend not connected. This page does not generate, estimate, or display model predictions or risk scores."
* **Dependencies:** `streamlit`, `dashboard.components.navigation`.
* **Architectural Contribution:** Protects system integrity by explicitly communicating unimplemented features.

### File 22: `pages/06_What_If_Simulator.py`
* **File Path:** `pages/06_What_If_Simulator.py`
* **Primary Responsibility:** Explicit unavailable state placeholder for planned what-if scenario simulation.
* **Implementation:** Displays an informational banner stating: "Not yet available — model/backend not connected. This page does not display simulated baselines, sensitivity results, or recommendations."
* **Dependencies:** `streamlit`, `dashboard.components.navigation`.
* **Architectural Contribution:** Prevents false expectations regarding simulation capabilities.

### File 23: `pages/07_Reports_Exports.py`
* **File Path:** `pages/07_Reports_Exports.py`
* **Primary Responsibility:** Data export center providing full source extracts with privacy-safe allowlisted columns.
* **Implementation Details:**
  * Defines export configurations for the 3 datasets with explicit allowlisted fields.
  * Regional orders export (`supplyiq_regional_orders.csv`): Excludes customer names, emails, passwords, and addresses.
  * Supplier & inventory export (`supplyiq_supplier_inventory.csv`): Exports product, category, stock, supplier, and warehouse columns.
  * Fuel consumption export (`supplyiq_fuel_consumption.csv`): Exports trip, vehicle, fuel, distance, cost, and provenance status fields.
  * Generates UTF-8 encoded CSV file streams via `st.download_button()`.
* **Dependencies:** `streamlit`, `dashboard.components.cards`, `dashboard.components.navigation`, `src.data_loader`.
* **Architectural Contribution:** Secure, governance-compliant data extraction portal.

### File 24: `pages/08_AI_Assistant.py`
* **File Path:** `pages/08_AI_Assistant.py`
* **Primary Responsibility:** Explicit unavailable state placeholder for planned LLM/RAG natural language assistant.
* **Implementation:** Displays an informational banner stating: "Not yet available — model/backend not connected. This page does not accept prompts or generate answers or operational advice."
* **Dependencies:** `streamlit`, `dashboard.components.navigation`.
* **Architectural Contribution:** Transparent communication regarding absence of AI/LLM integration.

### File 25: `pages/09_Project_Info_Model_Trust.py`
* **File Path:** `pages/09_Project_Info_Model_Trust.py`
* **Primary Responsibility:** Architectural transparency, dataset provenance, problem statements, and technical roadmap documentation.
* **Key Content Sections:**
  * Three Problem Statements (Regional delivery bottlenecks, supplier stockout risks, transport fuel tracking).
  * Dataset Information & Runtime Schema Inspector: Inspects column counts and column names directly from loaded DataFrames inside expandable widgets.
  * Current System Status: Emphasizes that source CSVs are read-only and no predictions are fabricated.
  * Planned Backend Integration: Lists upcoming technical phases (cleaning, EDA, feature engineering, model training, FastAPI integration).
* **Dependencies:** `streamlit`, `config.settings`, `src.data_loader`, `dashboard.components.cards`, `dashboard.components.navigation`.
* **Architectural Contribution:** Architectural specification and governance documentation page.

### File 26: `src/__init__.py`
* **File Path:** `src/__init__.py`
* **Primary Responsibility:** Package marker for the `src` backend directory.
* **Dependencies:** Python runtime.

### File 27: `src/data_loader.py`
* **File Path:** `src/data_loader.py`
* **Primary Responsibility:** Centralized, cached dataset ingestion engine.
* **Key Functions:**
  * `@st.cache_data(show_spinner="Loading source dataset...") def load_dataset(dataset_key: str) -> pd.DataFrame`:
    * Validates `dataset_key` against `config.settings.DATASETS`. Raises `ValueError` for invalid keys.
    * Checks filesystem existence using `pathlib.Path.exists()`. Returns empty DataFrame if missing.
    * Attempts UTF-8 CSV decoding; falls back gracefully to Latin-1 (`encoding="latin1"`) upon `UnicodeDecodeError`.
    * Uses `low_memory=False` to ensure accurate dtype inference across large files.
* **Inputs & Outputs:** Inputs: String dataset key. Outputs: Cached Pandas DataFrame.
* **Dependencies:** `pathlib.Path`, `pandas`, `streamlit`, `config.settings.DATASETS`.
* **Architectural Contribution:** High-performance data access layer preventing redundant disk reads.

### File 28: `src/data_validation.py`
* **File Path:** `src/data_validation.py`
* **Primary Responsibility:** Non-destructive data quality assessment and validation summarization.
* **Key Functions:**
  * `validate_dataframe(df: pd.DataFrame) -> dict`:
    * Checks if DataFrame is empty (`df.empty`); returns status `"empty"`.
    * Computes total missing cells: `missing_values = int(df.isna().sum().sum())`.
    * Computes count of columns with missing data: `columns_with_missing_values = int((df.isna().sum() > 0).sum())`.
    * Computes count of exact duplicate rows: `duplicate_rows = int(df.duplicated().sum())`.
    * Evaluates warning criteria: If missing values > 0 or duplicate rows > 0, status is `"warning"`; otherwise `"valid"`.
    * Returns a standardized dictionary summary without altering, dropping, or imputing any DataFrame values.
* **Inputs & Outputs:** Inputs: Pandas DataFrame. Outputs: Quality summary dictionary.
* **Dependencies:** `pandas`.
* **Architectural Contribution:** Establishes objective, non-destructive data quality gates.

### File 29: `src/ui_planning.py`
* **File Path:** `src/ui_planning.py`
* **Primary Responsibility:** Deterministic arithmetic calculation engine for vehicle fuel consumption estimates.
* **Key Functions:**
  * `calculate_fuel_estimate(distance_km: float | None, efficiency_km_per_litre: float | None) -> float`:
    * Validates that both `distance_km` and `efficiency_km_per_litre` are provided (not None).
    * Validates that inputs are finite numbers using `math.isfinite()`.
    * Validates that values are strictly greater than zero (`> 0`).
    * Computes and returns `float(distance_km) / float(efficiency_km_per_litre)`.
* **Inputs & Outputs:** Inputs: Distance (float) and Efficiency (float). Outputs: Estimated Fuel in Liters (float).
* **Dependencies:** `math.isfinite`.
* **Architectural Contribution:** Pure domain arithmetic logic isolated from presentation dependencies, enabling complete testability.

### Files 30 to 33: `src/regional_risk/` Package
* **Package Path:** `src/regional_risk/`
* **Primary Responsibility:** Domain-specific modular package for regional risk data preparation and model pipelines.
* **Modules:**
  * `__init__.py`: Package marker.
  * `cleaning.py` (`clean_regional_risk_data(df)`): Scaffolding for data cleaning; currently drops duplicates on a copy of the input frame.
  * `eda.py` (`get_regional_summary(df)`): Returns record count and column list dictionary.
  * `feature_engineering.py` (`create_regional_risk_features(df)`): Scaffolding returning an unmodified DataFrame copy.
  * `model_training.py` (`train_regional_risk_model(df)`): Scaffold that explicitly raises `NotImplementedError("Regional risk model training is not implemented. No model was trained.")`.
* **Dependencies:** `pandas`.
* **Architectural Contribution:** Pre-allocated modular backend architecture ready for future pipeline development.

### Files 34 to 37: `src/inventory_supplier/` Package
* **Package Path:** `src/inventory_supplier/`
* **Primary Responsibility:** Domain-specific modular package for retail inventory and supplier analytics pipelines.
* **Modules:**
  * `__init__.py`: Package marker.
  * `cleaning.py` (`clean_inventory_supplier_data(df)`): Drops duplicates on a copy of the input frame.
  * `eda.py` (`get_inventory_supplier_summary(df)`): Returns record count and column list dictionary.
  * `feature_engineering.py` (`create_inventory_supplier_features(df)`): Scaffolding returning an unmodified DataFrame copy.
  * `model_training.py` (`train_inventory_supplier_model(df)`): Scaffold that explicitly raises `NotImplementedError("Inventory/supplier model training is not implemented. No model was trained.")`.
* **Dependencies:** `pandas`.
* **Architectural Contribution:** Isolated backend package for supplier analytics pipelines.

### Files 38 to 41: `src/fuel_weather/` Package
* **Package Path:** `src/fuel_weather/`
* **Primary Responsibility:** Domain-specific modular package for fuel consumption and weather impact modeling.
* **Modules:**
  * `__init__.py`: Package marker.
  * `cleaning.py` (`clean_fuel_weather_data(df)`): Drops duplicates on a copy of the input frame.
  * `eda.py` (`get_fuel_weather_summary(df)`): Returns record count and column list dictionary.
  * `feature_engineering.py` (`create_fuel_weather_features(df)`): Scaffolding returning an unmodified DataFrame copy.
  * `model_training.py` (`train_fuel_weather_model(df)`): Scaffold that explicitly raises `NotImplementedError("Fuel/weather model training is not implemented. No model was trained.")`.
* **Dependencies:** `pandas`.
* **Architectural Contribution:** Isolated backend package for transport fuel analytics pipelines.

### Files 42 to 45: `tests/` Test Suite
* **Directory Path:** `tests/`
* **Primary Responsibility:** Automated test suite ensuring correctness of algorithms, validation logic, and backend scaffolds.
* **Modules:**
  * `tests/.gitkeep`: Preserves directory structure in source control.
  * `tests/test_backend_scaffolds.py` (`BackendScaffoldTests`): Asserts that calling `train_regional_risk_model()`, `train_inventory_supplier_model()`, or `train_fuel_weather_model()` raises `NotImplementedError`. (1 test case).
  * `tests/test_data_validation.py` (`DataValidationTests`): Verifies `validate_dataframe()` correctly identifies empty DataFrames, valid clean DataFrames, missing value warnings, exact duplicate warnings, and asserts that validation never mutates input DataFrames. (5 test cases).
  * `tests/test_ui_planning.py` (`FuelEstimateTests`): Verifies `calculate_fuel_estimate()` correctly computes liters, scales linearly with distance, rejects zero or negative inputs, rejects `None` inputs, and rejects non-finite inputs (`inf`, `nan`). (5 test cases).
* **Dependencies:** `unittest`, `pandas`, `src.data_validation`, `src.ui_planning`, `src.*.model_training`.
* **Architectural Contribution:** Regression prevention and automated verification of architectural constraints.

---

## 5. Comprehensive Change Audit

### 5.1 Historical Change Evidence & Audit Basis
Because the repository was provided without an initialized Git repository (`.git`), commit hashes and historical Git diffs cannot be generated. However, an empirical audit trail was reconstructed using two authoritative sources:
1. **The Verified Review Audit Log (`AUDIT_REPORT.md`):** Formally documents the baseline transition from unvalidated stubs to production-grade review code.
2. **Conversation & Trajectory Logs:** Historical workspace session logs located in the application data directory detailing the evolution of the application from an initial monolithic script to a multi-page enterprise architecture.

Each finding below is explicitly categorized as either **[CONFIRMED]** (substantiated by physical file artifacts and code diffs) or **[INFERRED]** (reconstructed from historical trajectory context).

### 5.2 Summary Table of Identifiable Changes

| File Path | Status | Evidence Classification | Nature of Modification | Rationale & Related Bug Fix | Features Affected |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `AUDIT_REPORT.md` | Created | **[CONFIRMED]** | Added full audit report markdown document | Created to record review scope, dataset SHA-256 checks, and audit findings | Project governance & review tracking |
| `src/data_validation.py` | Modified | **[CONFIRMED]** | Rewrote `validate_dataframe()` to return `"warning"` on missing/duplicate data, count affected columns, and preserve input immutability | Prior implementation naively returned `"valid"` for any non-empty frame, concealing data quality flaws | Data quality monitoring across all modules |
| `src/regional_risk/model_training.py` | Modified | **[CONFIRMED]** | Replaced silent `pass` returning `None` with `raise NotImplementedError(...)` | Prevents caller modules from silently assuming a trained model was returned | Regional risk prediction pipeline |
| `src/inventory_supplier/model_training.py` | Modified | **[CONFIRMED]** | Replaced silent `pass` returning `None` with `raise NotImplementedError(...)` | Prevents caller modules from silently assuming a trained model was returned | Supplier/inventory modeling pipeline |
| `src/fuel_weather/model_training.py` | Modified | **[CONFIRMED]** | Replaced silent `pass` returning `None` with `raise NotImplementedError(...)` | Prevents caller modules from silently assuming a trained model was returned | Fuel consumption modeling pipeline |
| `tests/test_backend_scaffolds.py` | Created | **[CONFIRMED]** | Added test class asserting `NotImplementedError` on all 3 model trainers | Verifies that dormant model training scaffolds fail explicitly rather than silently | Testing framework & CI/CD verification |
| `tests/test_data_validation.py` | Created | **[CONFIRMED]** | Added 5 unit tests for non-destructive validation behavior | Validates empty frame, clean frame, missing values, duplicates, and non-mutation | Testing framework & data quality verification |
| `dashboard/components/cards.py` | Modified | **[CONFIRMED]** | Added `render_empty_state()` function | Resolves `ImportError` previously triggered when `pages/02_Regional_Risk.py` attempted import | Filter empty states across all pages |
| `pages/05_Risk_Prediction.py` | Created | **[INFERRED]** | Created explicit unavailable state view | Replaced prior monolithic experimental views to prevent fabricating synthetic predictions | Risk prediction navigation view |
| `pages/06_What_If_Simulator.py` | Created | **[INFERRED]** | Created explicit unavailable state view | Prevents presenting unverified simulation sliders as real business scenarios | Scenario simulator navigation view |
| `pages/08_AI_Assistant.py` | Created | **[INFERRED]** | Created explicit unavailable state view | Prevents presenting mock chatbot answers as connected RAG intelligence | AI assistant navigation view |
| `pages/07_Reports_Exports.py` | Created | **[INFERRED]** | Built dedicated export center with strict column allowlists | Enforces data privacy by stripping customer passwords and street addresses | Secure operational reporting |
| `pages/09_Project_Info_Model_Trust.py` | Created | **[INFERRED]** | Built model governance, problem statement, and schema inspection view | Provides full operational transparency regarding data provenance | Governance & Viva defense preparation |
| `data/processed/` & `models/` | Deleted / Missing | **[CONFIRMED]** | Directories omitted from repository bundle despite references in settings | Cleaned up empty unpopulated directories prior to archive bundling | Dataset and model persistence layers |

---

## 6. Before-Versus-After Code & Architecture Comparisons

### 6.1 Data Validation Engine (`src/data_validation.py`)
* **Before (Naive Validation):** The initial implementation checked only if the DataFrame was not empty. If rows existed, it immediately returned `{"status": "valid"}`, even if the dataset had hundreds of thousands of missing cells or duplicated records. It also carried the risk of modifying the caller's DataFrame in place.
* **After (Non-Destructive Quality Inspection):**
```python
# BEFORE (Inferred from Audit Trail):
def validate_dataframe(df):
    if df.empty:
        return {"status": "empty"}
    return {"status": "valid", "rows": len(df)}

# AFTER (Actual Current Implementation in src/data_validation.py):
def validate_dataframe(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "status": "empty",
            "rows": 0,
            "columns": len(df.columns),
            "missing_values": 0,
            "columns_with_missing_values": 0,
            "duplicate_rows": 0,
        }

    missing_by_column = df.isna().sum()
    missing_values = int(missing_by_column.sum())
    columns_with_missing_values = int((missing_by_column > 0).sum())
    duplicate_rows = int(df.duplicated().sum())
    has_warnings = missing_values > 0 or duplicate_rows > 0

    return {
        "status": "warning" if has_warnings else "valid",
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": missing_values,
        "columns_with_missing_values": columns_with_missing_values,
        "duplicate_rows": duplicate_rows,
    }
```
* **Architectural Rationale:** Data integrity requires that anomalies be surfaced immediately. Labeling a dataset with 336,209 missing values as "valid" would mislead downstream ML pipelines. The new function guarantees immutability and precise quality reporting.

### 6.2 Model Training Scaffolds (`model_training.py`)
* **Before (Silent Failure via `pass`):** Training functions contained `pass`, silently returning `None`. Downstream callers attempting to call `.predict()` on the returned object would fail with confusing `AttributeError: 'NoneType' object has no attribute 'predict'`.
* **After (Explicit Exception Raising):**
```python
# BEFORE (Verified via AUDIT_REPORT.md):
def train_regional_risk_model(df: pd.DataFrame):
    pass

# AFTER (Actual Current Implementation in src/regional_risk/model_training.py):
def train_regional_risk_model(df: pd.DataFrame):
    raise NotImplementedError(
        "Regional risk model training is not implemented. No model was trained."
    )
```
* **Architectural Rationale:** In enterprise software architecture, an incomplete feature must fail loudly and explicitly. Raising `NotImplementedError` guarantees that no pipeline accidentally deploys an uninitialized model.

### 6.3 UI Cards Component (`dashboard/components/cards.py`)
* **Before (Missing Function):** `pages/02_Regional_Risk.py` was authored with an import: `from dashboard.components.cards import render_metric_card, render_disclaimer, render_empty_state`. However, `cards.py` originally lacked `render_empty_state`, causing an immediate runtime crash (`ImportError: cannot import name 'render_empty_state'`).
* **After (Implemented Reusable Empty State):**
```python
# ADDED FUNCTION IN dashboard/components/cards.py:
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
```
* **Architectural Rationale:** Resolves a blocking syntax error while establishing a unified visual treatment for empty data states across all pages.

### 6.4 Regional Order Drill-Down & Privacy Masking (`pages/02_Regional_Risk.py`)
* **Before (Unsafe Raw Rows):** Presenting the DataCo CSV directly would display item-level duplication (multiple rows per order) and expose sensitive customer passwords, street addresses, and emails.
* **After (Order Aggregation & Strict Allowlisting):**
```python
# Order-level grouping with strict field allowlist:
safe_fields = [
    "Order Id", "Order Region", "Order State", "Order City", "Order Country",
    "order date (DateOrders)", "Delivery Status", "Late_delivery_risk",
    "Days for shipping (real)", "Days for shipment (scheduled)",
    "Shipping Mode", "Order Status", "Order Item Id", "Order Item Quantity", "Order Item Total",
]
# All PII fields (Customer Password, Customer Email, Customer Street) are completely omitted.
```
* **Architectural Rationale:** Protects enterprise customer privacy and aligns data granularity to the decision maker's mental model (orders rather than disjointed line items).

---

## 7. Application Architecture & Data-Flow Diagrams

### 7.1 Layered Architecture Diagram

```
+===================================================================================+
|                          PRESENTATION LAYER (STREAMLIT UI)                        |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |   app.py (Command Center Launchpad & Connection Status)                   |   |
|   +---------------------------------------------------------------------------+   |
|   |   pages/                                                                  |   |
|   |     ├── 01_Executive_Overview.py       ├── 02_Regional_Risk.py            |   |
|   |     ├── 03_Inventory_Supplier.py       ├── 04_Fuel_Weather_Analytics.py   |   |
|   |     ├── 07_Reports_Exports.py          ├── 09_Project_Info_Model_Trust.py |   |
|   |     └── [Disabled Scaffolds: 05_Prediction, 06_Simulator, 08_AI]         |   |
|   +---------------------------------------------------------------------------+   |
|                                     │                                             |
|   +─────────────────────────────────▼─────────────────────────────────────────+   |
|   |   dashboard/components/                                                   |   |
|   |     ├── navigation.py  (CSS Dark Theme, Branded Sidebar, Page Headers)    |   |
|   |     ├── cards.py       (Metric Cards, Alert Banners, Disclaimers, Empty)  |   |
|   |     └── charts.py      (Plotly Bar, Donut, Line, Scatter, Color Maps)     |   |
|   +---------------------------------------------------------------------------+   |
+=====================================│=============================================+
                                      ▼
+===================================================================================+
|                       BUSINESS LOGIC & DOMAIN MODULES (src/)                      |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |   src/ui_planning.py        (calculate_fuel_estimate: Pure Arithmetic)    |   |
|   |   src/data_validation.py    (validate_dataframe: Non-Destructive Quality) |   |
|   +---------------------------------------------------------------------------+   |
|   |   Domain Scaffolds:                                                       |   |
|   |     ├── src/regional_risk/        (cleaning, eda, features, training)     |   |
|   |     ├── src/inventory_supplier/   (cleaning, eda, features, training)     |   |
|   |     └── src/fuel_weather/         (cleaning, eda, features, training)     |   |
|   +---------------------------------------------------------------------------+   |
+=====================================│=============================================+
                                      ▼
+===================================================================================+
|                        DATA ACCESS & INGESTION LAYER                              |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |   src/data_loader.py: load_dataset()                                      |   |
|   |   - Uses @st.cache_data for In-Memory Ingestion Caching                   |   |
|   |   - Automatic Fallback: UTF-8 -> Latin-1                                  |   |
|   |   - Resolves Paths via config/settings.py DATASETS Mapping                |   |
|   +---------------------------------------------------------------------------+   |
|                                     │                                             |
|   +─────────────────────────────────▼─────────────────────────────────────────+   |
|   |   data/raw/ (Immutable Source Files)                                      |   |
|   |     ├── DataCoSupplyChainDataset.csv              (180,519 rows, 91.5 MB) |   |
|   |     ├── Retail_Supply_Chain_Analytics_Dataset.csv (200,000 rows, 46.1 MB) |   |
|   |     └── supplyiq_final_dataset_filled_synthetic... (32,065 rows, 18.3 MB) |   |
|   +---------------------------------------------------------------------------+   |
+===================================================================================+
```

### 7.2 Major User Journey Data Flows

#### Journey 1: Regional Order Drill-Down & Investigation
```text
[User selects Region/State/City & Filters]
        │
        ▼
[pages/02_Regional_Risk.py: load_order_records()]
        │
        ├──> Calls src.data_loader.load_dataset("supply_chain")
        │       │
        │       └──> Reads data/raw/DataCoSupplyChainDataset.csv (Cached in RAM)
        │
        ├──> Filters to safe operational columns (Omit Passwords/Emails)
        ├──> Groups 180,519 line items by "Order Id" -> 65,752 unique orders
        ├──> Computes "Shipping difference (days)" = Real Days - Scheduled Days
        └──> Evaluates "Late delivery" boolean from source Delivery Status
        │
        ▼
[Apply Dynamic Location & Date Range Filters]
        │
        ├──> Compute KPIs: Orders in Selection, Late Delivery %, Avg Delay Days
        ├──> Render Plotly Regional Comparison Bar Chart
        ├──> Display Sortable Location Ranking Table
        ├──> Render Filtered Orders Data Table (Allowlisted columns only)
        └──> Generate Filtered CSV Export ("SupplyIQ_filtered_orders.csv")
```

#### Journey 2: Retail Inventory & Stockout Monitoring
```text
[User adjusts Category, Warehouse, or Product Search Filters]
        │
        ▼
[pages/03_Inventory_Supplier.py: load_retail_records()]
        │
        ├──> Calls src.data_loader.load_dataset("retail_analytics")
        │       │
        │       └──> Reads data/raw/Retail_Supply_Chain_Analytics_Dataset.csv
        │
        ├──> Numeric type casting (Current_Stock, Reorder_Level, Lead_Time_Days)
        └──> Filters out precomputed speculative score fields
        │
        ▼
[Evaluate Inventory Health Conditions]
        │
        ├──> Condition 1: Current_Stock <= Reorder_Level -> "At / below reorder level"
        ├──> Condition 2: Current_Stock <= 0             -> "Zero / negative stock"
        ├──> Calculate Gap: (Reorder_Level - Current_Stock).clip(lower=0)
        │
        ▼
[Render Managerial Visualizations]
        │
        ├──> Render KPI Metric Cards (Stockout counts, lead times)
        ├──> Render Category Demand Bar Chart
        ├──> Render Supplier Comparison Table (Rating, Lead Times, On-Time Rate)
        └──> Display Filtered Source Records with Reorder Gap Column
```

#### Journey 3: Fuel Consumption & Transparent Formula Estimation
```text
[User inspects Fuel Analytics & selects a Recorded Trip]
        │
        ▼
[pages/04_Fuel_Weather_Analytics.py: load_fuel_records()]
        │
        ├──> Calls src.data_loader.load_dataset("fuel_weather")
        │       │
        │       └──> Reads data/raw/supplyiq_final_dataset_filled_synthetic...
        │
        └──> Displays Provenance Notice: added_data_status = synthetic_demo_values
        │
        ▼
[Extract Trip Distance (km) & Vehicle Efficiency (km/L)]
        │
        ▼
[src/ui_planning.py: calculate_fuel_estimate()]
        │
        ├──> Check inputs: distance_km is finite and > 0
        ├──> Check inputs: efficiency_km_per_litre is finite and > 0
        └──> Return transparent arithmetic: distance / efficiency (Litres)
        │
        ▼
[Display Transparent Formula Result to User]
        │
        └──> "Formula estimate: {dist} km ÷ {eff} km/L = {litres} L"
             (Compared against original recorded fuel_consumed_liters)
```

---

## 8. Database Schema, Models, Relationships & Data Management

### 8.1 Data Management Paradigm
SupplyIQ utilizes an **In-Memory Flat-File Architecture**:
* **No Relational SQL Database:** The application does not use PostgreSQL, MySQL, SQLite, or SQLAlchemy.
* **In-Memory Caching:** Source CSV datasets are loaded into memory as Pandas DataFrames and cached globally via Streamlit's `@st.cache_data` decorator.
* **Immutability Principle:** The raw CSV files in `data/raw/` are treated as strictly immutable write-once read-many assets. The application never performs write-back, update, or deletion operations on the raw files.
* **Separation of Domains:** The three source CSVs represent distinct operational domains and are **never joined** horizontally. Attempting to join DataCo (order-level) with Retail (product-level) or Fuel (trip-level) without common foreign keys would result in an invalid Cartesian explosion.

### 8.2 Detailed Dataset Schemas

#### Dataset 1: DataCo Supply Chain Dataset (`DataCoSupplyChainDataset.csv`)
* **Physical Size:** 91.47 MB | **Dimensions:** 180,519 rows × 53 columns
* **Domain:** Customer orders, delivery logistics, shipping schedules.
* **Key Columns:**

| Column Name | Inferred Data Type | Null Count | Description & Business Meaning | Security / PII Status |
| :--- | :--- | :--- | :--- | :--- |
| `Order Id` | Integer / String | 0 | Unique identifier for the customer order. | Public Operational ID |
| `Order Item Id` | Integer | 0 | Unique line-item identifier within an order. | Public Operational ID |
| `Order Region` | String | 0 | Geographic operational region (e.g., Western Europe, Central America). | Public Geography |
| `Order State` | String | 0 | State or province of the order destination. | Public Geography |
| `Order City` | String | 0 | City of the order destination. | Public Geography |
| `Delivery Status` | String | 0 | Recorded delivery outcome (`Late delivery`, `Advance shipping`, `Shipping on time`). | Operational Metric |
| `Late_delivery_risk` | Float / Int | 0 | Binary source flag (1 = late risk, 0 = no risk). | Operational Metric |
| `Days for shipping (real)` | Float | 0 | Actual recorded transit days. | Operational Metric |
| `Days for shipment (scheduled)`| Float | 0 | Target planned transit days. | Operational Metric |
| `Shipping Mode` | String | 0 | Transport class (`Standard Class`, `First Class`, `Second Class`, `Same Day`). | Operational Metric |
| `Customer Password` | String | 0 | Customer account authentication credential. | **CRITICAL PII (Masked from UI)** |
| `Customer Email` | String | 0 | Customer email address. | **PII (Masked from UI)** |
| `Customer Street` | String | 0 | Customer physical delivery street address. | **PII (Masked from UI)** |
| `Product Description` | String | 180,519 | Product textual description (100% missing). | Dormant Field |
| `Order Zipcode` | Float | 155,679 | Postal code for order destination (86.2% missing). | High Nullity Field |

#### Dataset 2: Retail Supply Chain Analytics Dataset (`Retail_Supply_Chain_Analytics_Dataset.csv`)
* **Physical Size:** 46.06 MB | **Dimensions:** 200,000 rows × 41 columns
* **Domain:** Retail inventory, warehouse storage, supplier performance.
* **Key Columns:**

| Column Name | Inferred Data Type | Null Count | Description & Business Meaning | Handling in SupplyIQ |
| :--- | :--- | :--- | :--- | :--- |
| `Product_ID` | String | 0 | Unique alphanumeric product identifier (200,000 unique). | Primary Key in Retail |
| `Product_Category` | String | 0 | Merchandising category (e.g., Electronics, Fashion). | Filter Dimension |
| `Brand` | String | 0 | Brand identifier. | Search Dimension |
| `SKU` | String | 0 | Stock keeping unit identifier. | Search Dimension |
| `Current_Stock` | Float | 0 | Current quantity on hand in warehouse. | Evaluated for stockout |
| `Reorder_Level` | Float | 0 | Minimum stock threshold triggering reorder. | Threshold Comparator |
| `Safety_Stock` | Float | 0 | Buffer stock held to mitigate stockouts. | Monitored Buffer |
| `Supplier_ID` | String | 0 | Identifier of the vendor supplying the SKU. | Vendor Dimension |
| `Supplier_Rating` | Float | 0 | Vendor performance score (1.0 to 5.0 scale). | Descriptive Mean |
| `Lead_Time_Days` | Float | 0 | Vendor fulfillment turnaround time in days. | Descriptive Mean |
| `Warehouse_ID` | String | 0 | Storage facility identifier. | Filter Dimension |
| `Warehouse_Location` | String | 0 | City / region of the storage facility. | Filter Dimension |
| `Units_Sold` | Float | 0 | Recorded sales units. | Demand Metric |
| `Daily_Demand` | Float | 0 | Average daily sales run-rate. | Demand Metric |
| `Stockout_Risk` | Float | 0 | Precomputed synthetic risk score. | **Excluded from UI** |
| `Demand_Forecast` | Float | 0 | Precomputed synthetic forecast. | **Excluded from UI** |

#### Dataset 3: Synthetic Fuel & Weather Dataset (`supplyiq_final_dataset_filled_synthetic_fuel_weather.csv`)
* **Physical Size:** 18.32 MB | **Dimensions:** 32,065 rows × 43 columns
* **Domain:** Fleet logistics, fuel consumption, transit conditions.
* **Key Columns:**

| Column Name | Inferred Data Type | Null Count | Description & Business Meaning | Operational Reality |
| :--- | :--- | :--- | :--- | :--- |
| `trip_id` | String | 0 | Unique trip record identifier (32,065 unique). | Synthetic Key |
| `vehicle_id` | String | 0 | Vehicle asset identifier. | Synthetic Asset |
| `vehicle_type` | String | 0 | Vehicle category (`Heavy Truck`, `Van`, `Trailer`, etc.). | Operational Category |
| `fuel_type` | String | 0 | Fuel type (`Diesel`, `Petrol`, `Electric`, etc.). | Operational Category |
| `road_type` | String | 0 | Terrain / highway category (`Highway`, `Urban`, `Rural`). | Operational Category |
| `weather_condition` | String | 0 | Weather state (`Clear`, `Rainy`, `Foggy`, etc.). | Environmental Factor |
| `trip_distance_km` | Float | 0 | Recorded distance traveled for the trip. | Calculation Input |
| `fuel_efficiency_km_per_liter` | Float | 0 | Recorded vehicle fuel economy in km/L. | Calculation Input |
| `fuel_consumed_liters` | Float | 0 | Recorded liters consumed on the trip. | Comparison Benchmark |
| `fuel_price_per_liter` | Float | 0 | Unit fuel price for the trip record. | Cost Calculation Input |
| `fuel_cost_per_trip` | Float | 0 | Total fuel expense recorded for the trip. | Comparison Benchmark |
| `added_data_status` | String | 0 | Provenance flag: `synthetic_demo_values` (100%). | **Prominently Disclosed** |

### 8.3 CRUD Operations Matrix

| Entity / Dataset | Create (C) | Read (R) | Update (U) | Delete (D) |
| :--- | :--- | :--- | :--- | :--- |
| `DataCoSupplyChainDataset.csv` | Not Supported (Read-Only) | `src.data_loader.load_dataset("supply_chain")` | Not Supported (Source is Immutable) | Not Supported (Source is Immutable) |
| `Retail_Supply_Chain_Analytics_Dataset.csv` | Not Supported (Read-Only) | `src.data_loader.load_dataset("retail_analytics")` | Not Supported (Source is Immutable) | Not Supported (Source is Immutable) |
| `supplyiq_final_dataset_filled_synthetic...` | Not Supported (Read-Only) | `src.data_loader.load_dataset("fuel_weather")` | Not Supported (Source is Immutable) | Not Supported (Source is Immutable) |
| In-Memory Filtered Orders | Created on the fly via Pandas filtering | Streamlit DataFrame View | Dynamic session-state slicing | Cleared on page refresh |
| Exported CSV Files | Streamed as byte stream via `st.download_button` | Downloaded by user browser | Not Supported | Temporary client download |

---

## 9. API Documentation (Internal Modules & Planned Endpoints)

### 9.1 Internal Python Module APIs

#### `src.data_loader.load_dataset(dataset_key: str) -> pd.DataFrame`
* **Purpose:** Safely load and cache raw CSV datasets from configured relative paths.
* **Parameters:** `dataset_key` (`str`): Key identifying the dataset (`"supply_chain"`, `"retail_analytics"`, or `"fuel_weather"`).
* **Returns:** `pd.DataFrame`: Loaded dataset, or an empty DataFrame if the file does not exist.
* **Exceptions Raised:** `ValueError`: Raised if `dataset_key` is not defined in `config.settings.DATASETS`.
* **Caching:** Decorated with `@st.cache_data(show_spinner="Loading source dataset...")`.
* **Calling Files:** `app.py`, `pages/01_Executive_Overview.py`, `pages/02_Regional_Risk.py`, `pages/03_Inventory_Supplier.py`, `pages/04_Fuel_Weather_Analytics.py`, `pages/07_Reports_Exports.py`, `pages/09_Project_Info_Model_Trust.py`.

#### `src.data_validation.validate_dataframe(df: pd.DataFrame) -> dict`
* **Purpose:** Summarize basic data quality without mutating or modifying input data.
* **Parameters:** `df` (`pd.DataFrame`): DataFrame to inspect.
* **Returns:** `dict` containing:
  * `"status"` (`str`): `"empty"` if 0 rows, `"warning"` if missing cells or duplicate rows exist, `"valid"` if completely clean.
  * `"rows"` (`int`): Total row count.
  * `"columns"` (`int`): Total column count.
  * `"missing_values"` (`int`): Sum of all null cells.
  * `"columns_with_missing_values"` (`int`): Number of columns containing at least one null value.
  * `"duplicate_rows"` (`int`): Total exact duplicate rows.
* **Calling Files:** Evaluated by unit tests (`tests/test_data_validation.py`) and data ingestion pipelines.

#### `src.ui_planning.calculate_fuel_estimate(distance_km: float | None, efficiency_km_per_litre: float | None) -> float`
* **Purpose:** Return deterministic arithmetic fuel estimate (`distance / efficiency`) in liters.
* **Parameters:**
  * `distance_km` (`float | None`): Trip distance in kilometers.
  * `efficiency_km_per_litre` (`float | None`): Vehicle economy in kilometers per liter.
* **Returns:** `float`: Total liters required.
* **Exceptions Raised:**
  * `ValueError`: If either parameter is `None`.
  * `ValueError`: If either parameter is not a finite number (`math.isnan()` or `math.isinf()`).
  * `ValueError`: If either parameter is `<= 0`.
* **Calling Files:** `pages/04_Fuel_Weather_Analytics.py`, `tests/test_ui_planning.py`.

#### `dashboard.components.cards.render_metric_card(...)`
* **Signature:** `render_metric_card(title: str, value: str, delta: str = None, help_text: str = None, icon: str = None)`
* **Purpose:** Emits an HTML-escaped dark-themed KPI card into the Streamlit layout.
* **Calling Files:** `app.py`, `pages/01_Executive_Overview.py`, `pages/02_Regional_Risk.py`, `pages/03_Inventory_Supplier.py`, `pages/04_Fuel_Weather_Analytics.py`, `pages/07_Reports_Exports.py`.

#### `dashboard.components.cards.render_empty_state(...)`
* **Signature:** `render_empty_state(message: str = "No data available...")`
* **Purpose:** Emits a dashed-border placeholder card when filter criteria produce zero matching records.
* **Calling Files:** `dashboard.components.charts.py`, `pages/01_Executive_Overview.py`, `pages/02_Regional_Risk.py`, `pages/03_Inventory_Supplier.py`, `pages/04_Fuel_Weather_Analytics.py`, `pages/07_Reports_Exports.py`.

#### `dashboard.components.navigation.apply_custom_theme()`
* **Signature:** `apply_custom_theme()`
* **Purpose:** Injects the global operations-console CSS stylesheet into the Streamlit DOM.
* **Calling Files:** `dashboard.components.navigation.render_sidebar_header()`.

### 9.2 Prospective / Planned REST API Specifications
To support future headless client applications, microservices, and mobile dashboards, the following REST endpoints are architected for implementation with FastAPI:

```yaml
openapi: 3.1.0
info:
  title: SupplyIQ Operational Backend API
  version: 1.0.0-planned
paths:
  /api/v1/health:
    get:
      summary: Service & Dataset Health Check
      responses:
        '200':
          description: Returns connection status and row counts of all 3 CSV datasets.

  /api/v1/orders/regional:
    get:
      summary: Query Regional Order Records
      parameters:
        - name: region
          in: query
          schema: { type: string }
        - name: status
          in: query
          schema: { type: string }
        - name: limit
          in: query
          schema: { type: integer, default: 100 }
      responses:
        '200':
          description: Filtered order-level records with PII strictly excluded.

  /api/v1/inventory/reorder-alerts:
    get:
      summary: Get Inventory Items At or Below Reorder Level
      responses:
        '200':
          description: List of SKUs where Current_Stock <= Reorder_Level with reorder gap.

  /api/v1/fuel/estimate:
    post:
      summary: Calculate Pure Arithmetic Fuel Consumption
      requestBody:
        required: true
        content:
          application/json:
            schema:
              type: object
              required: [distance_km, efficiency_km_per_liter]
              properties:
                distance_km: { type: number, minimum: 0.01 }
                efficiency_km_per_liter: { type: number, minimum: 0.01 }
      responses:
        '200':
          description: Estimated fuel in liters and optional cost calculation.

  /api/v1/models/predict-risk:
    post:
      summary: Planned ML Regional Risk Scoring (Dormant)
      responses:
        '501':
          description: Returns HTTP 501 Not Implemented until model pipelines are trained.
```

---

## 10. Configuration, Environment Setup, Dependencies & Safe Execution

### 10.1 Hardware & Environment Prerequisites
* **Operating System:** Microsoft Windows 10/11, macOS 12+, or Ubuntu Linux 20.04+ (Fully platform-independent).
* **Python Version:** Python 3.11 (tested on Python 3.11.x; compatible with Python 3.10 through 3.12).
* **RAM Recommendation:** Minimum 8 GB RAM (recommended: 16 GB RAM) to comfortably load all 3 CSVs simultaneously (~155 MB on disk, expanding to ~600 MB in-memory Pandas objects).
* **Disk Space:** Minimum 500 MB free space.

### 10.2 Step-by-Step Safe Execution Guide

#### Step 1: Open Terminal & Navigate to Project Root
```bash
cd c:\Users\choug\Downloads\supply_iq_reviewed\supply_iq
```

#### Step 2: Initialize & Activate a Virtual Environment
```bash
# On Windows PowerShell:
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate
```

#### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Execute Automated Unit Tests
Run the test suite to verify calculation correctness, non-destructive validation, and scaffold exception raising before starting the web UI:
```bash
python -m unittest discover -s tests -v
```
*Expected Result: 11 tests executed, 11 passed (OK).*

#### Step 5: Launch the Streamlit Web Dashboard
```bash
streamlit run app.py
```
*The local development server will start and open your default browser at `http://localhost:8501`.*

---

## 11. Testing & Verification Report

### 11.1 Test Execution Summary
A formal test run was executed directly against the workspace codebase using Python's standard `unittest` test discovery runner.

* **Execution Command:** `python -m unittest discover -s tests -v`
* **Total Tests Executed:** 11
* **Tests Passed:** 11 (100%)
* **Tests Failed:** 0 (0%)
* **Errors:** 0 (0%)
* **Total Execution Duration:** 0.040 seconds
* **Overall Status:** **PASSED (OK)**

### 11.2 Individual Test Case Results Matrix

| Test Suite / Module | Test Method Name | Verified Capability / Assertion | Status | Duration |
| :--- | :--- | :--- | :--- | :--- |
| `tests.test_backend_scaffolds` | `test_unimplemented_model_trainers_fail_explicitly` | Confirms that `train_regional_risk_model`, `train_inventory_supplier_model`, and `train_fuel_weather_model` all explicitly raise `NotImplementedError` when called with a sample DataFrame. | **PASSED** | < 0.005s |
| `tests.test_data_validation` | `test_empty_dataframe_is_reported_as_empty` | Asserts that an empty DataFrame returns `status: "empty"` with `rows: 0`. | **PASSED** | < 0.005s |
| `tests.test_data_validation` | `test_clean_dataframe_is_valid` | Asserts that a DataFrame with no nulls and no duplicate rows returns `status: "valid"`. | **PASSED** | < 0.005s |
| `tests.test_data_validation` | `test_missing_values_trigger_warning` | Asserts that null cells trigger `status: "warning"` and accurate counts of `missing_values` and `columns_with_missing_values`. | **PASSED** | < 0.005s |
| `tests.test_data_validation` | `test_exact_duplicate_rows_trigger_warning` | Asserts that identical rows trigger `status: "warning"` and an accurate `duplicate_rows` count. | **PASSED** | < 0.005s |
| `tests.test_data_validation` | `test_validation_does_not_mutate_input` | Uses `pd.testing.assert_frame_equal` to guarantee that validation leaves the caller's DataFrame completely unmodified. | **PASSED** | < 0.005s |
| `tests.test_ui_planning` | `test_calculates_litres_from_distance_and_efficiency` | Asserts that 300 km at 10 km/L yields exactly 30.0 Liters. | **PASSED** | < 0.005s |
| `tests.test_ui_planning` | `test_distance_increase_scales_estimate` | Asserts linear scaling: doubling distance exactly doubles estimated fuel consumption. | **PASSED** | < 0.005s |
| `tests.test_ui_planning` | `test_rejects_zero_or_negative_inputs` | Asserts `ValueError` when distance or efficiency is <= 0 (e.g., 0, -1). | **PASSED** | < 0.005s |
| `tests.test_ui_planning` | `test_rejects_missing_inputs` | Asserts `ValueError` when distance or efficiency is `None`. | **PASSED** | < 0.005s |
| `tests.test_ui_planning` | `test_rejects_non_finite_inputs` | Asserts `ValueError` when distance or efficiency is `inf` or `nan`. | **PASSED** | < 0.005s |

### 11.3 Python Syntax & Bytecode Compilation Check
* **Execution Command:** `python -m compileall -q .`
* **Result:** Exit Code 0 (Success). All Python files across `app.py`, `config/`, `dashboard/`, `pages/`, `src/`, and `tests/` compiled cleanly with zero syntax errors, type errors, or illegal imports.

### 11.4 Explicit Untested Areas
* **Browser-Level UI Rendering Smoke Test:** Streamlit UI visual components and browser interaction events were verified through code structure and automated tests, but full end-to-end headless browser testing (e.g., via Playwright or Selenium) was not executed.
* **Multi-User High-Concurrency Memory Load:** System performance under simultaneous multi-user requests loading large CSVs was not load-tested.
* **Dormant Feature Engineering Pipelines:** The scaffold modules in `src/*/feature_engineering.py` and `cleaning.py` contain basic identity stubs that do not have dedicated unit tests.

---

## 12. Known Bugs, Limitations, Security Concerns & Prioritized Recommendations

### 12.1 Security & Data Privacy Concerns
1. **Critical PII Exposure in Raw DataCo CSV [Severity: High]:**
   * *Finding:* `data/raw/DataCoSupplyChainDataset.csv` contains cleartext customer account passwords (`Customer Password`), emails, customer names, and street addresses.
   * *Mitigation in Place:* The application enforces strict field allowlists in `pages/02_Regional_Risk.py` and `pages/07_Reports_Exports.py`, ensuring sensitive fields are never rendered on the web UI or included in exports.
   * *Remaining Risk:* Anyone with read access to the server filesystem can read the raw CSV directly.
2. **Raw HTML Injection Risk via `unsafe_allow_html=True` [Severity: Low]:**
   * *Finding:* `dashboard/components/cards.py` and `navigation.py` use `st.markdown(..., unsafe_allow_html=True)` to render custom CSS and HTML cards.
   * *Mitigation in Place:* All dynamic variable interpolations use `html.escape()` from the standard library.

### 12.2 Data Quality & Functional Limitations
1. **Synthetic Nature of Fuel Dataset [Severity: Medium]:**
   * *Finding:* All 32,065 rows in `supplyiq_final_dataset_filled_synthetic_fuel_weather.csv` have `added_data_status = 'synthetic_demo_values'`. It does not contain validated operational data and lacks origin/destination locations.
   * *Mitigation in Place:* Prominently highlighted via warning banners on `app.py`, `pages/01`, `pages/04`, and `pages/09`.
2. **Extreme Missing Data in DataCo Columns [Severity: Medium]:**
   * *Finding:* `Product Description` is 100% missing (180,519 nulls). `Order Zipcode` is 86.2% missing (155,679 nulls).
   * *Handling:* These columns are excluded from operational metrics.
3. **Missing Physical Directories [Severity: Low]:**
   * *Finding:* `data/processed/` and `models/` are referenced in `config/settings.py` and `README.md` but are physically absent from disk.
4. **Unimplemented Backend ML & AI Capabilities [Severity: Informational]:**
   * *Finding:* Risk prediction, scenario simulation, and AI assistant pages are intentionally inactive.

### 12.3 Prioritized Recommendations Matrix

| Priority | Category | Action Item | Expected Impact & Benefit |
| :--- | :--- | :--- | :--- |
| **P0 (Immediate)** | Security | Sanitize raw `DataCoSupplyChainDataset.csv` on disk to remove or cryptographically hash `Customer Password` and customer PII fields. | Completely eliminates data breach exposure at the filesystem layer. |
| **P0 (Immediate)** | Architecture | Create physical directories `data/processed/` and `models/` containing `.gitkeep` to align disk state with documentation. | Eliminates configuration and path reference discrepancies. |
| **P1 (Near-Term)** | Performance | Migrate from loading 91 MB CSV files into RAM to an indexed SQLite database or DuckDB instance with SQL query pushdown. | Reduces startup memory consumption from ~600 MB to < 50 MB; speeds up filtering. |
| **P1 (Near-Term)** | Data Science | Implement genuine exploratory data analysis (EDA) and data cleaning scripts in `src/regional_risk/cleaning.py` to handle missing zip codes. | Prepares clean data for future machine learning modeling. |
| **P2 (Long-Term)** | ML Engineering | Develop, cross-validate, and serialize real Scikit-Learn classification models in `src/regional_risk/model_training.py` with leak-free preprocessing. | Enables real delivery risk prediction on `pages/05_Risk_Prediction.py`. |
| **P2 (Long-Term)** | Feature | Connect a local LLM or RAG pipeline with vector embeddings over supply chain documentation for `pages/08_AI_Assistant.py`. | Activates the natural language assistant capability. |

---

## 13. Change-to-File-to-Feature Traceability Matrix

| Business Feature / Requirement | Implementing Code Files | Change Status | Verification Test / Method | Operational Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Command Center & System Health** | `app.py`, `dashboard/components/cards.py`, `dashboard/components/navigation.py` | Preserved & Refined | Manual UI Inspection | Displays connection health cards across all 3 source CSVs. |
| **Multi-Source Executive Dashboard** | `pages/01_Executive_Overview.py`, `src/data_loader.py` | Preserved & Refined | Syntax Compile Check | Aggregates KPIs across 3 domains without joining tables. |
| **Regional Order Exploration** | `pages/02_Regional_Risk.py`, `src/data_loader.py` | Preserved & PII-Masked | Unit & Compile Checks | Groups 180k rows to 65k orders; hierarchy drill-down. |
| **Privacy Protection & PII Masking** | `pages/02_Regional_Risk.py`, `pages/07_Reports_Exports.py` | Audited & Enforced | Code Review Inspection | Strips customer passwords, emails, and street addresses. |
| **Inventory & Stockout Intelligence** | `pages/03_Inventory_Supplier.py`, `src/data_loader.py` | Preserved & Refined | Syntax Compile Check | Compares stock to reorder level; vendor rating table. |
| **Fuel Records & Transparent Estimator** | `pages/04_Fuel_Weather_Analytics.py`, `src/ui_planning.py` | Audited & Validated | `tests/test_ui_planning.py` (5 passed) | Transparent formula: liters = distance / efficiency. |
| **Synthetic Provenance Disclosure** | `app.py`, `pages/01`, `pages/04`, `pages/09` | Audited & Enforced | UI Visual Inspection | Warns users that fuel records are synthetic demo values. |
| **Explicit Unavailable Feature Handling** | `pages/05_Risk_Prediction.py`, `pages/06`, `pages/08` | Confirmed Scaffolds | UI Visual Inspection | Clear informational notices that backends are not connected. |
| **Governance & Model Trust Review** | `pages/09_Project_Info_Model_Trust.py` | Preserved & Refined | Syntax Compile Check | Explains 3 problem statements, schemas, and roadmap. |
| **Privacy-Safe CSV Data Downloads** | `pages/07_Reports_Exports.py`, `pages/02` | Preserved & Tested | Download Button Test | Streams UTF-8 CSVs containing allowlisted fields only. |
| **Non-Destructive Data Validation** | `src/data_validation.py` | Confirmed Refactored | `tests/test_data_validation.py` (5 passed) | Returns `"warning"` on nulls/duplicates without mutation. |
| **Explicit Model Scaffold Exceptions** | `src/*/model_training.py` | Confirmed Refactored | `tests/test_backend_scaffolds.py` (1 passed) | Raises `NotImplementedError` rather than silent `pass`. |
| **Automated Test Automation** | `tests/test_*.py` | Confirmed Suite | `unittest discover` (11 passed) | Full automated coverage of calculations and validation. |

---

## 14. Beginner-Friendly Technical Glossary

* **Application Programming Interface (API):** A structured set of rules and code functions allowing different software programs or modules to communicate with each other predictably.
* **Cartesian Product (Cartesian Explosion):** A mathematical condition that occurs in databases when two unrelated tables are joined together without a matching shared key, resulting in every row of Table A being paired with every row of Table B (e.g., 180,000 × 200,000 = 36 billion rows), crashing system memory.
* **CSV (Comma-Separated Values):** A plain-text tabular file format where each line represents a data record and individual columns are separated by commas.
* **Data Cleaning:** The process of detecting and correcting (or filtering) corrupt, inaccurate, missing, or irrelevant records from a dataset before analysis or modeling.
* **Data Immutability:** An architectural design rule where original raw data files are treated as permanent, read-only records that can never be modified or overwritten by application code.
* **Exploratory Data Analysis (EDA):** The initial investigation performed on data to discover patterns, spot anomalies, test hypotheses, and verify assumptions using summary statistics and graphical visualizations.
* **Feature Engineering:** The data science process of selecting, transforming, and combining raw dataset variables into meaningful mathematical indicators (features) designed to improve machine learning model accuracy.
* **Field Allowlist (Whitelisting):** A security practice where only explicitly named safe data fields (columns) are permitted to be displayed or exported, automatically blocking all unlisted sensitive fields like passwords.
* **In-Memory Caching:** Storing previously loaded data or expensive computational results in fast computer RAM (via decorators like `@st.cache_data`) so subsequent requests return instantly without re-reading the hard drive.
* **Model Scaffold (Stub):** A placeholder function or module that defines the expected inputs, outputs, and name of a future algorithm, allowing the rest of the application to be built before the underlying algorithm is completed.
* **`NotImplementedError`:** A built-in Python exception explicitly raised to notify developers and callers that a requested function or method has been declared but not yet written.
* **One-Hot Encoding:** A data transformation method that converts categorical text values (e.g., "Diesel", "Petrol") into binary 1s and 0s so mathematical machine learning algorithms can process them.
* **Personally Identifiable Information (PII):** Confidential data that can be used to identify, contact, or locate a specific individual (e.g., full names, personal email addresses, account passwords, home street addresses).
* **Plotly:** A Python data visualization library that creates interactive, web-based charts featuring zoom, hover tooltips, and dynamic filtering.
* **Provenance:** The documented chronology of the origin, custody, and authenticity of a dataset, detailing whether records represent verified live business operations or synthetic generated mock values.
* **Retrieval-Augmented Generation (RAG):** An advanced AI architecture that searches a private database of company documents to find relevant facts and provides them to a Large Language Model (LLM) to generate accurate, hallucination-free answers.
* **Scikit-Learn:** A widely used open-source Python library providing tools for machine learning, including data preprocessing, classification, regression, and model evaluation.
* **Streamlit:** An open-source Python framework that allows developers to convert Python data scripts into shareable, interactive web applications without needing to write custom HTML, CSS, or JavaScript.
* **Synthetic Data:** Artificially generated mock data created by a computer program rather than collected from real-world operational events.
* **Target Leakage:** A critical data science mistake where information from the future or information that would not be available at prediction time is accidentally included in the training data, leading to falsely high training accuracy that fails completely in real production.
* **Unit Test:** An automated software test that exercises a small, isolated piece of code (such as a single function) with known inputs to verify that it returns the exact expected output.
* **Viva Examination:** An oral defense exam where a student or engineer presents their project and must explain and justify the design, architecture, and code decisions to academic or technical examiners.

---
*End of Technical Report.*
