"""
Data Loading Module for SupplyIQ.
Loads datasets safely using configured relative paths, with encoding fallbacks,
contract validation, file-mtime-aware caching, and privacy protection.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

from config.settings import DATASETS

# Dataset column contracts: required columns per business use-case
DATASET_CONTRACTS = {
    "supply_chain": [
        "Order Id",
        "Order Region",
        "Order State",
        "Order City",
        "order date (DateOrders)",
        "Delivery Status",
        "Late_delivery_risk",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Shipping Mode",
    ],
    "retail_analytics": [
        "Product_ID",
        "Product_Category",
        "Brand",
        "SKU",
        "Current_Stock",
        "Reorder_Level",
        "Safety_Stock",
        "Supplier_ID",
        "Supplier_Rating",
        "Warehouse_ID",
        "Warehouse_Location",
        "Daily_Demand",
        "Units_Sold",
    ],
    "fuel_weather": [
        "trip_id",
        "vehicle_id",
        "vehicle_type",
        "fuel_type",
        "road_type",
        "weather_condition",
        "trip_distance_km",
        "fuel_efficiency_km_per_liter",
        "fuel_consumed_liters",
        "fuel_price_per_liter",
        "fuel_cost_per_trip",
    ],
}

# Sensitive customer fields that must never be exposed in UI, logs, or exports
SENSITIVE_CUSTOMER_FIELDS = [
    "Customer Email",
    "Customer Password",
    "Customer Fname",
    "Customer Lname",
    "Customer Street",
    "Customer Zipcode",
]

# Internal cache keyed by dataset, file signature, and PII handling mode.
_MEMORY_CACHE: dict[
    tuple[str, str, bool],
    tuple[tuple[int, int], pd.DataFrame],
] = {}


def sanitize_privacy_fields(df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
    """Return a copy of the dataframe with sensitive PII removed.

    Does not mutate the input dataframe.
    """
    if df.empty or not drop:
        return df.copy()

    sanitized = df.copy()
    columns_to_remove = [col for col in SENSITIVE_CUSTOMER_FIELDS if col in sanitized.columns]
    if columns_to_remove:
        sanitized = sanitized.drop(columns=columns_to_remove)
    return sanitized


def check_dataset_contract(df: pd.DataFrame, dataset_key: str) -> dict:
    """Verify that the loaded DataFrame satisfies its contract.

    Returns a dictionary with status, missing_required_columns, and extra details.
    """
    if dataset_key not in DATASET_CONTRACTS:
        return {
            "valid": False,
            "status": "unknown_dataset",
            "missing_required_columns": [],
        }

    if df.empty:
        return {
            "valid": False,
            "status": "empty",
            "missing_required_columns": DATASET_CONTRACTS[dataset_key],
        }

    required = DATASET_CONTRACTS.get(dataset_key, [])
    missing = [col for col in required if col not in df.columns]
    return {
        "valid": len(missing) == 0,
        "status": "valid" if len(missing) == 0 else "contract_violation",
        "missing_required_columns": missing,
        "total_columns": len(df.columns),
        "total_rows": len(df),
    }


def read_raw_csv_file(file_path: Path | str) -> pd.DataFrame:
    """Read a CSV file with automatic encoding fallback (UTF-8 then Latin-1).

    Raises FileNotFoundError if the file does not exist.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Source file not found at: {path}")

    try:
        return pd.read_csv(path, encoding="utf-8", low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin1", low_memory=False)


def load_raw_dataset(dataset_key: str, sanitize_pii: bool = False) -> pd.DataFrame:
    """Load a raw dataset by key, with in-memory caching keyed on file mtime.

    Parameters:
        dataset_key: 'supply_chain', 'retail_analytics', or 'fuel_weather'
        sanitize_pii: if True, drops known sensitive customer columns
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key: '{dataset_key}'. Available: {list(DATASETS.keys())}")

    file_path = DATASETS[dataset_key]
    if not Path(file_path).exists():
        return pd.DataFrame()

    path = Path(file_path).resolve()
    stat = path.stat()
    signature = (stat.st_mtime_ns, stat.st_size)
    cache_key = (dataset_key, str(path), sanitize_pii)

    # Reuse a cached frame only while the source path, size, and modified time match.
    if cache_key in _MEMORY_CACHE:
        cached_signature, cached_df = _MEMORY_CACHE[cache_key]
        if cached_signature == signature:
            return cached_df.copy()

    df = read_raw_csv_file(path)
    if sanitize_pii:
        df = sanitize_privacy_fields(df)

    _MEMORY_CACHE[cache_key] = (signature, df.copy())
    return df.copy()


@st.cache_data(show_spinner="Loading source dataset...")
def _load_dataset_cached(
    dataset_key: str,
    file_path: str,
    modified_ns: int,
    file_size: int,
) -> pd.DataFrame:
    """Read a dataset with cache keys tied to the current source-file version."""
    # Streamlit uses these otherwise-unused arguments to distinguish file versions.
    del dataset_key, modified_ns, file_size
    return read_raw_csv_file(file_path)


def load_dataset(dataset_key: str) -> pd.DataFrame:
    """Load a raw dataset by key for Streamlit apps.

    Preserves the existing DataFrame return contract and invalidates cached
    results when the configured source file changes.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key: '{dataset_key}'. Available: {list(DATASETS.keys())}")

    file_path = Path(DATASETS[dataset_key])
    if not file_path.exists():
        return pd.DataFrame()

    path = file_path.resolve()
    stat = path.stat()
    # Pass file metadata as cache keys so edits to a CSV trigger a fresh read.
    return _load_dataset_cached(
        dataset_key,
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
    )


def clear_dataset_caches() -> None:
    """Clear process-local and Streamlit dataset caches."""
    _MEMORY_CACHE.clear()
    _load_dataset_cached.clear()
