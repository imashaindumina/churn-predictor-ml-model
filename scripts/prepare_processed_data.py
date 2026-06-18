import os
import sys
import pandas as pd

# Make src module importable safely across different systems
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.load_data import load_data
from src.features.build_features import build_features

# Corrected paths matching your local repository ecosystem
RAW = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = "data/processed/telco_churn_processed.csv"

print("=== STARTING OFFLINE DATA PREPARATION PIPELINE ===")

# 1. Ingest Data safely using your custom load module
if not os.path.exists(RAW):
    raise FileNotFoundError(f"Raw source file not found at: {RAW}")

print(f"📥 Ingesting raw baseline dataset from: {RAW}")
df = load_data(RAW)

# 2. Process Features using your optimized 10-feature production engine
# This will handle types, mappings, and preserve the "Churn" column seamlessly!
print("🔧 Applying VIF-optimized feature mutations and target alignments...")
df_processed = build_features(df, target_col="Churn")

# 3. Defensive Sanity Checking to enforce data quality constraints
assert "Churn" in df_processed.columns, "Target validation error: 'Churn' column was dropped!"
assert df_processed["Churn"].isna().sum() == 0, "Data Integrity Violation: Churn contains NaNs!"
assert set(df_processed["Churn"].unique()) <= {0, 1}, "Domain Constraint Failure: Churn contains invalid states!"

# 4. Serialize and save the finalized historical records
os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)

print(f"✅ [SUCCESS] Clean processed dataset saved to: {OUT}")
print(f"📊 Matrix Structural Footprint Shape: {df_processed.shape}")