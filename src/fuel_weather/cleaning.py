"""
Fuel & Weather - Data Cleaning Module.
Cleans fuel price logs and weather impact observations datasets.
"""
import pandas as pd


def clean_fuel_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean fuel and weather dataset.
    """
    if df.empty:
        return df
    cleaned_df = df.copy()
    cleaned_df.drop_duplicates(inplace=True)
    return cleaned_df
