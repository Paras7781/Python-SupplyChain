"""Lightweight, non-destructive validation summaries for source DataFrames."""
from __future__ import annotations

import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> dict:
    """Summarize basic data quality without modifying or silently cleaning data.

    Status is ``empty`` for a zero-row frame, ``warning`` when missing values or
    exact duplicate rows are present, and ``valid`` only when neither is found.
    """
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
