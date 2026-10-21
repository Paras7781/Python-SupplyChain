"""
SupplyIQ - Synthetic Daily Demand Generator.

Generates realistic, learnable synthetic Daily_Demand values based on prediction-time
operational features (Product_Category, Warehouse_Location, Seasonal_Demand_Index)
with reproducible Gaussian noise.

Key Principles:
1. Canonical active dataset: Operates strictly on the single authoritative file:
   data/raw/Retail_Supply_Chain_Analytics_Dataset.csv.
2. Backup preservation: Strictly verifies that data/backup/Retail_Supply_Chain_Analytics_Dataset_original.csv
   exists and matches its verified SHA-256 hash. Never modifies or overwrites the backup.
3. No duplicate datasets: Does NOT write to data/synthetic/ or create competing CSVs.
4. Non-destructive by default: Running without '--apply' performs a safe dry-run validation check.
   The active CSV is updated ONLY when '--apply' is explicitly passed.
5. Uses prediction-time available features only:
   - Does NOT use Units_Sold, Monthly_Demand, Demand_Forecast, or original Daily_Demand.
   - Does NOT use arbitrary IDs (Product_ID, row indices).
6. Uses a fixed random seed (seed=42) and documented noise (std=30.0) for full reproducibility.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CANONICAL_ACTIVE_CSV = PROJECT_ROOT / "data" / "raw" / "Retail_Supply_Chain_Analytics_Dataset.csv"
ORIGINAL_BACKUP_CSV = PROJECT_ROOT / "data" / "backup" / "Retail_Supply_Chain_Analytics_Dataset_original.csv"

# Known SHA256 of original unmodified backup
EXPECTED_ORIGINAL_BACKUP_SHA256 = "d885d2506fd48aa08b38012d8fc707e86eaaff985d410fa9eb077d40b272ef81"

# Documented formula parameters
CATEGORY_BASE_DEMAND = {
    "Food": 350.0,         # Fast-moving consumer goods: high turnover
    "Health": 275.0,       # Essential pharmaceuticals & personal care: steady high demand
    "Fashion": 220.0,      # Apparel & footwear: moderate turnover
    "Home": 160.0,         # Home goods & furnishings: lower volume
    "Electronics": 110.0,  # Consumer electronics: high unit price, lower daily volume
}

WAREHOUSE_FACTORS = {
    "North": 1.10,  # Major regional distribution hub (+10%)
    "East": 1.05,   # Secondary high-density hub (+5%)
    "South": 0.95,  # Regional satellite warehouse (-5%)
    "West": 0.90,   # Remote hub (-10%)
}

NOISE_STD_DEV = 30.0
RANDOM_SEED = 42


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def generate_synthetic_demand(df: pd.DataFrame, seed: int = RANDOM_SEED) -> np.ndarray:
    """
    Generate synthetic daily demand using documented formula:
      Daily_Demand = max(0, round(category_base * warehouse_factor * seasonal_factor + noise))

    Parameters:
        df: DataFrame containing Product_Category, Warehouse_Location, Seasonal_Demand_Index.
        seed: Random seed for reproducible Gaussian noise.

    Returns:
        1D numpy array of non-negative integers.
    """
    # 1. Base demand by product category
    base = df["Product_Category"].map(CATEGORY_BASE_DEMAND).to_numpy(dtype=float)
    if np.isnan(base).any():
        raise ValueError("Unknown Product_Category encountered in dataset.")

    # 2. Regional warehouse location multiplier
    wh_factor = df["Warehouse_Location"].map(WAREHOUSE_FACTORS).to_numpy(dtype=float)
    if np.isnan(wh_factor).any():
        raise ValueError("Unknown Warehouse_Location encountered in dataset.")

    # 3. Seasonal demand intensity factor:
    # Seasonal_Demand_Index ranges from 0 to 100 with mean ~50.
    # Index 0   -> factor 0.50 (low season)
    # Index 50  -> factor 1.00 (average season)
    # Index 100 -> factor 1.50 (peak season)
    seasonal_idx = df["Seasonal_Demand_Index"].to_numpy(dtype=float)
    seasonal_factor = 0.50 + (seasonal_idx / 100.0)

    # 4. Deterministic signal
    deterministic_demand = base * wh_factor * seasonal_factor

    # 5. Reproducible Gaussian noise
    rng = np.random.default_rng(seed=seed)
    noise = rng.normal(loc=0.0, scale=NOISE_STD_DEV, size=len(df))

    # 6. Combined demand rounded to non-negative integers
    synthetic_demand = np.clip(np.round(deterministic_demand + noise), 0, None).astype(int)
    return synthetic_demand


def main():
    parser = argparse.ArgumentParser(
        description="SupplyIQ - Synthetic Daily Demand Generator for Canonical Active Dataset."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Explicitly apply regenerated Daily_Demand to the active dataset (default is dry-run validation).",
    )
    args = parser.parse_args()

    print("=" * 80)
    print(" SupplyIQ - Synthetic Daily Demand Generator")
    print("=" * 80)

    # Safety Check 1: Verified original backup must exist and remain untampered
    if not ORIGINAL_BACKUP_CSV.exists():
        raise FileNotFoundError(
            f"SAFETY CRITICAL ERROR: Original backup not found at {ORIGINAL_BACKUP_CSV}. "
            "Refusing to proceed without a verified recovery backup."
        )

    backup_sha = compute_sha256(ORIGINAL_BACKUP_CSV)
    print(f"Verified Original Backup: {ORIGINAL_BACKUP_CSV}")
    print(f"  Backup SHA-256:         {backup_sha}")
    if backup_sha != EXPECTED_ORIGINAL_BACKUP_SHA256:
        raise ValueError(
            f"SAFETY CRITICAL ERROR: Backup SHA-256 ({backup_sha}) differs from expected "
            f"({EXPECTED_ORIGINAL_BACKUP_SHA256}). Original backup may be corrupted. Aborting!"
        )
    print("  -> PASSED: Original backup integrity verified. Backup will not be modified.\n")

    # Safety Check 2: Canonical active CSV must exist
    if not CANONICAL_ACTIVE_CSV.exists():
        raise FileNotFoundError(f"Active dataset not found at {CANONICAL_ACTIVE_CSV}")

    active_sha_pre = compute_sha256(CANONICAL_ACTIVE_CSV)
    print(f"Canonical Active Dataset: {CANONICAL_ACTIVE_CSV}")
    print(f"  Current Active SHA-256: {active_sha_pre}")

    active_df = pd.read_csv(CANONICAL_ACTIVE_CSV)
    print(f"  Shape: {active_df.shape[0]:,} rows, {active_df.shape[1]} columns")

    # Generate demand using the documented formula
    print("\nEvaluating synthetic demand formula:")
    print("  Formula: Daily_Demand = max(0, round(Category_Base * Warehouse_Factor * Seasonal_Factor + Noise))")
    print(f"  Random seed: {RANDOM_SEED}, Noise std dev: {NOISE_STD_DEV}")
    print(f"  Category bases: {CATEGORY_BASE_DEMAND}")
    print(f"  Warehouse factors: {WAREHOUSE_FACTORS}")
    print("  Seasonal factor: 0.50 + (Seasonal_Demand_Index / 100.0)")

    generated_demand = generate_synthetic_demand(active_df, seed=RANDOM_SEED)

    # Validate generated values
    current_demand = active_df["Daily_Demand"].to_numpy()
    matches_current = np.array_equal(generated_demand, current_demand)
    print(f"\nFormula Check:")
    print(f"  Generated values match current active Daily_Demand: {matches_current}")
    print(f"  Min: {generated_demand.min()}, Max: {generated_demand.max()}, Mean: {generated_demand.mean():.2f}, Std: {generated_demand.std():.2f}")

    if args.apply:
        print("\n[--apply specified] Writing synthetic demand to active CSV...")
        updated_df = active_df.copy()
        updated_df["Daily_Demand"] = generated_demand

        # Confirm 40 non-target columns did not change
        mismatched_cols = [c for c in active_df.columns if c != "Daily_Demand" and not active_df[c].equals(updated_df[c])]
        if mismatched_cols:
            raise RuntimeError(f"Safety check failed: unexpected non-target columns changed: {mismatched_cols}")

        updated_df.to_csv(CANONICAL_ACTIVE_CSV, index=False)
        post_sha = compute_sha256(CANONICAL_ACTIVE_CSV)
        print(f"  -> Successfully updated {CANONICAL_ACTIVE_CSV}")
        print(f"  -> Post-update SHA-256: {post_sha}")
    else:
        print("\n[Dry-Run Mode] No changes made to disk. To apply, run with '--apply'.")

    # Re-verify backup was NEVER touched
    post_backup_sha = compute_sha256(ORIGINAL_BACKUP_CSV)
    assert post_backup_sha == EXPECTED_ORIGINAL_BACKUP_SHA256, "CRITICAL: Backup file was modified!"
    print("Verified: Original backup remains 100% untouched and intact.")
    print("=" * 80)


if __name__ == "__main__":
    main()
