"""
Data Loading Module for SupplyIQ.
Loads datasets safely using configured relative paths.
"""
from pathlib import Path
import pandas as pd
import streamlit as st
from config.settings import DATASETS


@st.cache_data(show_spinner="Loading source dataset...")
def load_dataset(dataset_key: str) -> pd.DataFrame:
    """
    Load a raw dataset by key.
    
    Parameters:
        dataset_key (str): Key of the dataset ('supply_chain', 'retail_analytics', 'fuel_weather')
        
    Returns:
        pd.DataFrame: Loaded dataset dataframe or empty dataframe if file missing.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key: '{dataset_key}'. Available: {list(DATASETS.keys())}")
    
    file_path = DATASETS[dataset_key]
    if not Path(file_path).exists():
        print(f"Warning: File not found at {file_path}")
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(file_path, encoding="utf-8", low_memory=False)
        return df
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding="latin1", low_memory=False)
        return df
