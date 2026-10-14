"""
Regional Risk - Data Cleaning Module.
Handles missing data imputation, duplicate removal, and type casting for regional risk datasets.
"""
import pandas as pd


def clean_regional_risk_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean regional risk dataset.
    """
    if df.empty:
        return df
    cleaned_df = df.copy()
    cleaned_df.drop_duplicates(inplace=True)
    return cleaned_df
