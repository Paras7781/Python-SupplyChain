"""Lightweight, non-destructive validation summaries for source DataFrames."""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.data_loader import DATASET_CONTRACTS, SENSITIVE_CUSTOMER_FIELDS


DATASET_NUMERIC_COLUMNS = {
    "supply_chain": [
        "Order Id",
        "Late_delivery_risk",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
    ],
    "retail_analytics": [
        "Current_Stock",
        "Reorder_Level",
        "Safety_Stock",
        "Supplier_Rating",
        "Daily_Demand",
        "Units_Sold",
    ],
    "fuel_weather": [
        "trip_distance_km",
        "fuel_efficiency_km_per_liter",
        "fuel_consumed_liters",
        "fuel_price_per_liter",
        "fuel_cost_per_trip",
    ],
}

DATASET_DATETIME_COLUMNS = {
    "supply_chain": ["order date (DateOrders)"],
    "retail_analytics": ["Date"],
    "fuel_weather": ["timestamp"],
}


def _validate_expected_types(df: pd.DataFrame, dataset_key: str | None) -> dict[str, int]:
    """Count non-null values that cannot be parsed to configured column types."""
    if dataset_key is None:
        return {}

    violations: dict[str, int] = {}
    for column in DATASET_NUMERIC_COLUMNS.get(dataset_key, []):
        if column in df.columns:
            # Count only non-empty values that failed conversion; genuine nulls are separate.
            parsed = pd.to_numeric(df[column], errors="coerce")
            invalid = int((df[column].notna() & parsed.isna()).sum())
            if invalid:
                violations[column] = invalid

    for column in DATASET_DATETIME_COLUMNS.get(dataset_key, []):
        if column in df.columns:
            # Apply the same distinction to dates without changing the source frame.
            parsed = pd.to_datetime(df[column], errors="coerce")
            invalid = int((df[column].notna() & parsed.isna()).sum())
            if invalid:
                violations[column] = invalid

    return violations


def validate_dataframe(df: pd.DataFrame, dataset_key: str | None = None) -> dict:
    """Summarize data quality without modifying or silently cleaning data.

    Status is ``empty`` for a zero-row frame, ``warning`` when missing values or
    exact duplicate rows are present, and ``valid`` only when neither is found.
    """
    contract_known = dataset_key is None or dataset_key in DATASET_CONTRACTS
    missing_contract = (
        [column for column in DATASET_CONTRACTS[dataset_key] if column not in df.columns]
        if dataset_key in DATASET_CONTRACTS
        else []
    )
    contract_valid = contract_known and not missing_contract
    type_violations = _validate_expected_types(df, dataset_key)

    if df.empty:
        return {
            "status": "empty",
            "rows": 0,
            "columns": len(df.columns),
            "missing_values": 0,
            "columns_with_missing_values": 0,
            "duplicate_rows": 0,
            "sensitive_columns_detected": [],
            "type_violations": {},
            "contract_valid": contract_valid,
            "missing_required_columns": missing_contract,
        }

    missing_by_column = df.isna().sum()
    missing_values = int(missing_by_column.sum())
    columns_with_missing = {col: int(cnt) for col, cnt in missing_by_column.items() if cnt > 0}
    columns_with_missing_values = int(len(columns_with_missing))
    duplicate_rows = int(df.duplicated().sum())
    has_warnings = missing_values > 0 or duplicate_rows > 0 or bool(type_violations)

    # A broken required-column contract takes precedence over ordinary quality warnings.
    sensitive_found = [col for col in SENSITIVE_CUSTOMER_FIELDS if col in df.columns]
    if not contract_valid:
        status = "contract_violation"
    else:
        status = "warning" if has_warnings else "valid"

    return {
        "status": status,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": missing_values,
        "columns_with_missing_values": columns_with_missing_values,
        "missing_breakdown": columns_with_missing,
        "duplicate_rows": duplicate_rows,
        "sensitive_columns_detected": sensitive_found,
        "contract_valid": contract_valid,
        "missing_required_columns": missing_contract,
        "type_violations": type_violations,
    }


def validate_numeric_ranges(
    df: pd.DataFrame,
    columns_to_check: list[str],
    allow_zero: bool = True,
    allow_negative: bool = False,
) -> dict[str, dict]:
    """Validate numeric ranges for specified columns without altering data.

    Checks for infinities, NaNs, negatives, and zero values.
    """
    results: dict[str, dict] = {}
    for col in columns_to_check:
        if col not in df.columns:
            results[col] = {"present": False, "valid": False, "reason": "Column missing"}
            continue

        series = pd.to_numeric(df[col], errors="coerce")
        nans = int(series.isna().sum())
        infs = int(np.isinf(series).sum())
        negatives = int((series < 0).sum())
        zeros = int((series == 0).sum())

        is_valid = True
        reasons = []
        if infs > 0:
            is_valid = False
            reasons.append(f"{infs} infinite values")
        if not allow_negative and negatives > 0:
            is_valid = False
            reasons.append(f"{negatives} negative values")
        if not allow_zero and zeros > 0:
            is_valid = False
            reasons.append(f"{zeros} zero values")

        results[col] = {
            "present": True,
            "valid": is_valid,
            "total_count": len(series),
            "nan_count": nans,
            "inf_count": infs,
            "negative_count": negatives,
            "zero_count": zeros,
            "min": float(series.min()) if not series.empty and pd.notna(series.min()) else None,
            "max": float(series.max()) if not series.empty and pd.notna(series.max()) else None,
            "reasons": reasons,
        }
    return results


def get_dataset_provenance(dataset_key: str) -> dict:
    """Return provenance facts and data nature for the specified dataset."""
    provenance_map = {
        "supply_chain": {
            "dataset_key": "supply_chain",
            "source_file": "DataCoSupplyChainDataset.csv",
            "provenance_type": "observed_historical",
            "description": "Historical DataCo supply chain transactions with order and shipment details.",
            "synthetic": False,
            "granularity": "item_level",
            "disclaimer": "Contains customer identifiers and delivery status labels.",
        },
        "retail_analytics": {
            "dataset_key": "retail_analytics",
            "source_file": "Retail_Supply_Chain_Analytics_Dataset.csv",
            "provenance_type": "observed_retail",
            "description": "Retail supply chain transactions across warehouses, suppliers, and SKUs.",
            "synthetic": False,
            "granularity": "sku_transaction_level",
            "disclaimer": "Contains pre-existing benchmark scores and stock/demand observations.",
        },
        "fuel_weather": {
            "dataset_key": "fuel_weather",
            "source_file": "supplyiq_final_dataset_filled_synthetic_fuel_weather.csv",
            "provenance_type": "synthetic_demonstration",
            "description": "Trip distance, efficiency, weather, and fuel consumption records.",
            "synthetic": True,
            "granularity": "trip_level",
            "disclaimer": "Marked synthetic_demo_values in source. Not validated live operations.",
        },
    }
    return provenance_map.get(
        dataset_key,
        {
            "dataset_key": dataset_key,
            "provenance_type": "unknown",
            "synthetic": False,
            "disclaimer": "Provenance unverified.",
        },
    )
