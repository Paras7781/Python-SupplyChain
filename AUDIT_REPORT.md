# SupplyIQ project review and real-data audit

## Scope

Reviewed the Streamlit entry point, all nine pages, shared dashboard components, dataset configuration/loader, validation helpers, UI planning utilities, dormant EDA/cleaning/feature/model-training modules, README, and tests. The three original CSVs were read directly from the uploaded archive and copied byte-for-byte into this review copy.

## Findings

- The application is a Streamlit frontend with three source-backed workflows: regional order records, retail inventory/supplier records, and fuel-consumption records. The prediction, scenario, and AI/RAG pages are explicit unavailable states, not implementations.
- The fuel CSV marks all 32,065 records as `synthetic_demo_values`. These are not validated operational fuel observations. Its schema has no real route alternatives, origins/destinations, or travel-time field.
- DataCo has 180,519 item-level rows and 65,752 distinct order IDs. It has missing values (336,209 missing cells), including some customer ZIP/product description/customer-name fields. The regional UI groups item rows by Order Id and uses a privacy-safe field allowlist; export columns also omit customer contact details, passwords, and street addresses.
- The retail CSV has 200,000 rows and no missing cells or exact duplicate rows. Some SKU, supplier, and warehouse identifiers repeat as expected for repeated source records; Product_ID is unique in this file.
- The fuel CSV has 32,065 rows, no missing cells or exact duplicate rows, unique trip IDs, and all rows marked synthetic. The audited numeric fields had no parse failures, infinities, or negative values. Zero values exist in some retail quantities/rates and should be interpreted in context rather than silently changed.
- The three CSVs remain separate; the app does not join unrelated datasets.
- The repository includes dormant model-training modules that previously returned `None` silently via `pass`, and a validator that labelled any nonempty dataframe `valid` even when it contained missing values or duplicate rows.
- The uploaded archive bundles a local `.venv` and generated Python cache files. Those are environment-specific and should not be distributed with source code.

## Changes made in this review copy

- `src/data_validation.py`: validation now reports `warning` when missing values or exact duplicate rows exist, distinguishes the number of columns with missing values, and does not mutate the input frame.
- `src/regional_risk/model_training.py`, `src/inventory_supplier/model_training.py`, `src/fuel_weather/model_training.py`: unimplemented training functions now raise clear `NotImplementedError` messages rather than silently returning `None`.
- Added tests for validation behavior and explicit model-training scaffold behavior.
- Added this audit report.
- Original raw CSV bytes were preserved exactly. No data cleaning or replacement was applied to the raw files.

## Verification

- Python compile check: passed for `app.py`, `config`, `dashboard`, `pages`, `src`, and `tests`.
- Unit tests: 11 passed (5 fuel estimate tests, 5 data validation tests, 1 explicit backend-scaffold test).
- CSV integrity: SHA-256 verified against the archive entries for all three raw datasets; all matched byte-for-byte.
- Streamlit page smoke test: not rerun in this review container because Streamlit is not installed in the container's active Python environment. The project requirements include Streamlit; install `requirements.txt` and run the app in its intended environment for a full runtime smoke test.

## Recommended next steps

1. Keep raw CSVs immutable and perform cleaning into a separate processed-data directory.
2. Before modeling, establish a data dictionary and validate business meaning, not just numeric ranges. Treat the fuel dataset as synthetic unless its provenance is independently established.
3. Audit target leakage before using fields such as source risk labels, forecast fields, optimization scores, or other precomputed scores as model inputs.
4. Keep PII out of UI tables and exports; restrict access to raw DataCo files because they contain customer-related fields, including a password column.
5. Implement real model pipelines only after EDA, split strategy, baseline metrics, and leakage checks are documented.
