"""
Inventory & Supplier - Data Cleaning Module.
Cleans inventory records, vendor lead times, and stockout history datasets.
"""
import pandas as pd


def clean_inventory_supplier_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean inventory and supplier analytics dataset.
    """
    if df.empty:
        return df
    cleaned_df = df.copy()
    cleaned_df.drop_duplicates(inplace=True)
    return cleaned_df
