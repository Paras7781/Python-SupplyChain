"""
Candidate Model Training and Benchmarking Module for SupplyIQ.
Trains, evaluates, and serializes XGBoost candidate models alongside existing models
without overwriting production artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from sklearn.model_selection import TimeSeriesSplit, GroupKFold

from config.settings import MODELS_DIR
from ml.regional_risk.feature_engineering import create_regional_risk_features
from ml.regional_risk.model_training import (
    CATEGORICAL_FEATURES as RR_CAT,
    NUMERIC_FEATURES as RR_NUM,
    ALL_FEATURES as RR_ALL,
    TARGET_COLUMN as RR_TARGET,
)
from ml.inventory_supplier.model_training import (
    CATEGORICAL_FEATURES as INV_CAT,
    NUMERIC_FEATURES as INV_NUM,
    ALL_FEATURES as INV_ALL,
    TARGET_COLUMN as INV_TARGET,
)
from ml.fuel_weather.model_training import (
    CATEGORICAL_FEATURES as FUEL_CAT,
    NUMERIC_FEATURES as FUEL_NUM,
    ALL_FEATURES as FUEL_ALL,
    TARGET_COLUMN as FUEL_TARGET,
)

CANDIDATES_DIR = MODELS_DIR / "candidates"


def ensure_candidates_dir() -> Path:
    CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
    return CANDIDATES_DIR


# =====================================================================
# 1. Regional Risk XGBoost Candidate
# =====================================================================
def build_regional_risk_xgboost_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                RR_NUM,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                RR_CAT,
            ),
        ]
    )

    clf = xgb.XGBClassifier(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])


def train_regional_risk_xgboost(df: pd.DataFrame, sample_size: int = 30000) -> dict:
    """Train and evaluate an XGBClassifier candidate on the Regional Risk dataset."""
    ensure_candidates_dir()
    order_df = df.copy()
    if sample_size and len(order_df) > sample_size:
        order_df = order_df.sample(n=sample_size, random_state=42)

    feat_df = create_regional_risk_features(order_df, drop_leakage=True)
    valid_mask = feat_df[RR_TARGET].notna() & feat_df[RR_TARGET].isin([0, 1])
    feat_df = feat_df[valid_mask].copy()

    X = feat_df[RR_ALL].reset_index(drop=True)
    y = feat_df[RR_TARGET].astype(int).reset_index(drop=True)

    date_col = next((c for c in ("Order_Date", "order date (DateOrders)") if c in feat_df.columns), None)
    if date_col:
        dates = pd.to_datetime(feat_df[date_col], errors="coerce")
        order = np.argsort(dates.to_numpy(), kind="stable")
        X = X.iloc[order].reset_index(drop=True)
        y = y.iloc[order].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for tr_idx, te_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        fold_model = build_regional_risk_xgboost_pipeline()
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)
        probas = fold_model.predict_proba(X_te)[:, 1]

        accs.append(accuracy_score(y_te, preds))
        precs.append(precision_score(y_te, preds, zero_division=0))
        recs.append(recall_score(y_te, preds, zero_division=0))
        f1s.append(f1_score(y_te, preds, zero_division=0))
        aucs.append(roc_auc_score(y_te, probas))

    # Fit final model on all data
    final_pipeline = build_regional_risk_xgboost_pipeline()
    final_pipeline.fit(X, y)

    artifact_path = CANDIDATES_DIR / "regional_risk_model_xgboost.joblib"
    metadata_path = CANDIDATES_DIR / "regional_risk_xgboost_metadata.json"

    joblib.dump(final_pipeline, artifact_path)

    metadata = {
        "model_name": "XGBClassifier (Candidate)",
        "task": "binary_classification",
        "target": RR_TARGET,
        "validation_method": "TimeSeriesSplit",
        "fold_count": 5,
        "sample_size": len(X),
        "xgboost_version": xgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "model_accuracy": float(np.mean(accs)),
        "model_accuracy_std": float(np.std(accs)),
        "precision": float(np.mean(precs)),
        "precision_std": float(np.std(precs)),
        "recall": float(np.mean(recs)),
        "recall_std": float(np.std(recs)),
        "f1_score": float(np.mean(f1s)),
        "f1_score_std": float(np.std(f1s)),
        "roc_auc": float(np.mean(aucs)),
        "roc_auc_std": float(np.std(aucs)),
        "is_candidate": True,
        "artifact_path": str(artifact_path.name),
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# =====================================================================
# 2. Inventory Demand XGBoost Candidate
# =====================================================================
def build_inventory_xgboost_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                INV_NUM,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                INV_CAT,
            ),
        ]
    )

    reg = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg),
    ])


def train_inventory_demand_xgboost(df: pd.DataFrame, sample_size: int = 30000) -> dict:
    """Train and evaluate an XGBRegressor candidate on the Inventory Demand dataset."""
    ensure_candidates_dir()
    train_df = df.copy()
    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    X = train_df[INV_ALL].reset_index(drop=True)
    y = train_df[INV_TARGET].reset_index(drop=True)
    groups = train_df["Supplier_ID"].reset_index(drop=True) if "Supplier_ID" in train_df.columns else None

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(X, y, groups))

    maes, rmses, r2s = [], [], []
    for tr_idx, te_idx in splits:
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        fold_model = build_inventory_xgboost_pipeline()
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)

        maes.append(mean_absolute_error(y_te, preds))
        rmses.append(root_mean_squared_error(y_te, preds))
        r2s.append(r2_score(y_te, preds))

    final_pipeline = build_inventory_xgboost_pipeline()
    final_pipeline.fit(X, y)
    final_pipeline.baseline_median_demand_ = float(y.median())

    artifact_path = CANDIDATES_DIR / "inventory_demand_model_xgboost.joblib"
    metadata_path = CANDIDATES_DIR / "inventory_demand_xgboost_metadata.json"

    joblib.dump(final_pipeline, artifact_path)

    metadata = {
        "model_name": "XGBRegressor (Candidate)",
        "task": "regression",
        "target": INV_TARGET,
        "validation_method": "GroupKFold by Supplier",
        "fold_count": 5,
        "sample_size": len(X),
        "xgboost_version": xgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "baseline_median_demand": float(y.median()),
        "mae": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_score": float(np.mean(r2s)),
        "r2_score_std": float(np.std(r2s)),
        "is_candidate": True,
        "artifact_path": str(artifact_path.name),
        "notes": "XGBoost does not outperform the training median baseline because Daily_Demand is independent discrete uniform random noise.",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# =====================================================================
# 3. Fuel Consumption XGBoost Candidate
# =====================================================================
def build_fuel_xgboost_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                FUEL_NUM,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]),
                FUEL_CAT,
            ),
        ]
    )

    reg = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", reg),
    ])


def train_fuel_xgboost(df: pd.DataFrame, sample_size: int = 30000) -> dict:
    """Train and evaluate an XGBRegressor candidate on the Fuel dataset."""
    ensure_candidates_dir()
    train_df = df.copy()
    if sample_size and len(train_df) > sample_size:
        train_df = train_df.sample(n=sample_size, random_state=42)

    valid_fuel = train_df[FUEL_TARGET] > 0
    train_df = train_df[valid_fuel].copy()

    if "timestamp" in train_df.columns:
        train_df["_dt"] = pd.to_datetime(train_df["timestamp"], errors="coerce")
        train_df = train_df.sort_values("_dt").reset_index(drop=True)

    X = train_df[FUEL_ALL].reset_index(drop=True)
    y = train_df[FUEL_TARGET].reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=5)
    maes, rmses, r2s = [], [], []

    for tr_idx, te_idx in tscv.split(X):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_te, y_te = X.iloc[te_idx], y.iloc[te_idx]
        fold_model = build_fuel_xgboost_pipeline()
        fold_model.fit(X_tr, y_tr)
        preds = fold_model.predict(X_te)

        maes.append(mean_absolute_error(y_te, preds))
        rmses.append(root_mean_squared_error(y_te, preds))
        r2s.append(r2_score(y_te, preds))

    final_pipeline = build_fuel_xgboost_pipeline()
    final_pipeline.fit(X, y)

    artifact_path = CANDIDATES_DIR / "fuel_model_xgboost.joblib"
    metadata_path = CANDIDATES_DIR / "fuel_xgboost_metadata.json"

    joblib.dump(final_pipeline, artifact_path)

    metadata = {
        "model_name": "XGBRegressor (Candidate)",
        "task": "regression",
        "target": FUEL_TARGET,
        "validation_method": "TimeSeriesSplit",
        "fold_count": 5,
        "sample_size": len(X),
        "xgboost_version": xgb.__version__,
        "scikit_learn_version": sklearn.__version__,
        "trained_timestamp": datetime.now(timezone.utc).isoformat(),
        "mae": float(np.mean(maes)),
        "mae_std": float(np.std(maes)),
        "rmse": float(np.mean(rmses)),
        "rmse_std": float(np.std(rmses)),
        "r2_score": float(np.mean(r2s)),
        "r2_score_std": float(np.std(r2s)),
        "is_candidate": True,
        "is_synthetic": True,
        "artifact_path": str(artifact_path.name),
        "notes": "Trained on synthetic demonstration records (added_data_status = 'synthetic_demo_values').",
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata
