#!/usr/bin/env python3
"""
PRODUCTION RE-TRAINING ORCHESTRATOR - 10-Feature High-Recall Execution Pipeline
=============================================================================
"""

import os
import sys
import time
import argparse
import pandas as pd
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# === Fix import path for local modules ===
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core components alignment
from src.data.load_data import load_data  
from src.features.build_features import build_features  
from src.utils.validate_data import validate_telco_data  

def main(args):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    
    # CRITICAL WINDOWS FIX: Force MLflow to accept filesystem tracking
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    
    mlruns_path = args.mlflow_uri or os.path.join(project_root, "mlruns")
    
    if not mlruns_path.startswith("file://") and not mlruns_path.startswith("http"):
        from pathlib import Path
        mlruns_path = Path(mlruns_path).as_uri()
        
    mlflow.set_tracking_uri(mlruns_path)
    mlflow.set_experiment(args.experiment)

    print("\n=== STARTING AUTOMATED RETRAINING PIPELINE ORCHESTRATOR ===")

    with mlflow.start_run():
        # Log strategic infrastructure parameters
        mlflow.log_param("model_engine", "xgboost")
        mlflow.log_param("classification_threshold", args.threshold)
        mlflow.log_param("test_split_ratio", args.test_size)

        # STAGE 1: Ingest Data
        print(f"📥 Ingesting baseline dataset from: {args.input}")
        df = load_data(args.input)
        print(f"✅ Data loaded safely: Row-count: {df.shape[0]} | Columns: {df.shape[1]}")

        # STAGE 2: Great Expectations Validation Payload Gate
        print("🔍 Invoking runtime data validation engine...")
        is_valid, failed_checks = validate_telco_data(df)
        mlflow.log_metric("data_quality_pass_gate", int(is_valid))

        if not is_valid:
            import json
            mlflow.log_text(json.dumps(failed_checks, indent=2), artifact_file="failed_expectations.json")
            raise ValueError(f"❌ Structural Data Validation Gate Failed! Violated Criteria: {failed_checks}")
        print("✅ Data validation gate passed cleanly.")

        # STAGE 3: Core Feature Engineering Pipeline
        print("🛠️ Applying feature transformations and schema alignment...")
        target = args.target
        df_ml_ready = build_features(df, target_col=target)
        print(f"✅ Feature alignment complete. Operational Column Footprint: {df_ml_ready.shape[1]}")

        # STAGE 4: Stratified Dataset Split Execution
        print("📊 Formatting train/test splits...")
        X = df_ml_ready.drop(columns=[target])
        y = df_ml_ready[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )
        print(f"✅ Splits configured -> Train size: {X_train.shape[0]} samples | Test size: {X_test.shape[0]} samples")

        # STAGE 5: Class Imbalance Ratio Tuning
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📈 Imbalance weight mapped: {scale_pos_weight:.2f}")

        # STAGE 6: Model Ingestion with Tuned Optuna Trial 25 Hyperparameters
        print("🤖 Training champion XGBoost estimator...")
        champion_params = {
            "n_estimators": 522,
            "learning_rate": 0.1626,
            "max_depth": 5,
            "subsample": 0.8997,
            "colsample_bytree": 0.5038,
            "min_child_weight": 8,
            "gamma": 3.8200,
            "reg_alpha": 2.3861,
            "reg_lambda": 4.4400,
            "random_state": 42,
            "n_jobs": -1,
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss"
        }
        
        mlflow.log_params(champion_params)
        model = XGBClassifier(**champion_params)

        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        mlflow.log_metric("training_duration_seconds", train_time)
        print(f"✅ Model convergence achieved in {train_time:.2f}s")

        # STAGE 7: Performance Evaluation & Inference Threshold Optimization
        print("📊 Running model inference performance validation...")
        t1 = time.time()
        proba = model.predict_proba(X_test)[:, 1]
        
        y_pred = (proba >= args.threshold).astype(int)
        inference_time = time.time() - t1
        mlflow.log_metric("inference_duration_seconds", inference_time)

        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, proba)

        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        # STAGE 8: Log Dataset and Feature Metadata
        feature_cols = list(X.columns)
        mlflow.log_text("\n".join(feature_cols), artifact_file="feature_columns.txt")
        
        train_ds = mlflow.data.from_pandas(df_ml_ready, source="training_pipeline_source")
        mlflow.log_input(train_ds, context="training")

        mlflow.xgboost.log_model(model, artifact_path="model")

        print("\n🔥 [SUCCESS] End-to-End Orchestrated Pipeline Run Completed!")
        print(f"🎯 Execution Score Summary -> Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | AUC: {roc_auc:.3f}")
        print(classification_report(y_test, y_pred, digits=3))

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run Production Retraining Pipeline Suite")
    p.add_argument("--input", type=str, default="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    p.add_argument("--target", type=str, default="Churn")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--experiment", type=str, default="Telco Churn - Production Pipeline")
    p.add_argument("--mlflow_uri", type=str, default=None)

    args = p.parse_args()
    main(args)