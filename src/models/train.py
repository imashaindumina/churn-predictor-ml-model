"""
TRAINING PIPELINE - Automated Retraining & MLflow Tracking Ecosystem
"""

import os
import mlflow
import pandas as pd
import mlflow.lightgbm
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.data.load_data import load_data
from src.features.build_features import build_features

def train_model(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Trains the optimal 10-feature LightGBM model and logs metadata via MLflow.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 1. Stratified Train-Test Split matching Notebook 02 footprint
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # 2. Injecting the optimal LightGBM tuned hyperparameters from our 50-trial study
    champion_params = {
        "n_estimators": 624,
        "learning_rate": 0.0421,
        "max_depth": 4,
        "num_leaves": 31,
        "subsample": 0.7853,
        "colsample_bytree": 0.6412,
        "min_child_samples": 22,
        "reg_alpha": 1.4231,
        "reg_lambda": 2.8541,
        "random_state": 42,
        "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
        "verbose": -1
    }

    print("🚀 Initializing MLflow Active Experiment Tracking for Tuned LightGBM...")
    mlflow.set_experiment("Telco Churn - Production Pipeline")

    with mlflow.start_run():
        model = lgb.LGBMClassifier(**champion_params)
        model.fit(X_train, y_train)
        
        # 3. Predict Probabilities using verified production decision threshold (0.45)
        proba = model.predict_proba(X_test)[:, 1]
        THRESHOLD = 0.45
        y_pred = (proba >= THRESHOLD).astype(int)

        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        # 4. Log configuration parameters and execution metrics to MLflow
        mlflow.log_params(champion_params)
        mlflow.log_param("tuned_classification_threshold", THRESHOLD)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1_score", f1)

        # 5. Serialize and log the active model artifact securely
        mlflow.lightgbm.log_model(model, "model")

        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print("\n🔥 [SUCCESS] Model Automated Retraining Pipeline Completed!")
        print(f"Metrics -> Accuracy: {acc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")


if __name__ == "__main__":
    print("\n=== STARTING AUTOMATED RETRAINING PIPELINE ===")

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    try:
        print(f"📥 Loading raw dataset from: {data_path}")
        raw_df = load_data(data_path)

        print("🛠️ Processing features through the 10-feature matrix...")
        processed_df = build_features(raw_df)

        train_model(processed_df, target_col="Churn")

    except Exception as pipeline_error:
        print(f"❌ [PIPELINE CRASH] Operational failure in training orchestration: {str(pipeline_error)}")