import os
import mlflow
import pandas as pd
import mlflow.xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from src.data.load_data import load_data
from src.features.build_features import build_features

def train_model(df: pd.DataFrame, target_col: str = "Churn"):
    """
    Trains your VIF-optimized 70.6% Accuracy / 91.7% Recall XGBoost Champion model 
    and systematically logs all metadata, parameters, and code components via MLflow.
    """
    # Isolate independent features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Strategic Stratified Train-Test Split matching your Notebook 02 state
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Calculate scale_pos_weight dynamically to handle the 74/26 class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # Injecting the EXACT tuned hyperparameters found during your Optuna Trial 25 execution
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

    print("🚀 Initializing MLflow Active Experiment Tracking for Tuned XGBoost...")
    mlflow.set_experiment("Telco Churn - Production Pipeline")

    with mlflow.start_run():
        # Initialize and Train the dynamic XGBoost Champion Estimator
        model = XGBClassifier(**champion_params)
        model.fit(X_train, y_train)
        
        # Predict Probabilities to execute your business threshold tuning (THRESHOLD = 0.3)
        proba = model.predict_proba(X_test)[:, 1]
        THRESHOLD = 0.3
        y_pred = (proba >= THRESHOLD).astype(int)

        # Calculate exact classification performance metrics matching your benchmarks
        acc = accuracy_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred, pos_label=1)
        prec = precision_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        # Log your explicit tuned configuration hyperparameters to MLflow UI dashboard
        mlflow.log_params(champion_params)
        mlflow.log_param("tuned_classification_threshold", THRESHOLD)

        # Log system model evaluation performance tracking scores
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1_score", f1)

        # Serialize and log the active trained pipeline binary artifact directly into MLflow registry
        mlflow.xgboost.log_model(model, "model")

        # Log dataset so it shows in MLflow UI cleanly
        train_ds = mlflow.data.from_pandas(df, source="training_data")
        mlflow.log_input(train_ds, context="training")

        print("\n🔥 [SUCCESS] Model Automated Retraining Pipeline Completed!")
        print(f"Metrics -> Accuracy: {acc:.4f} | Recall (Catch Rate): {rec:.4f} | F1: {f1:.4f}")


# This main block must be completely outside the function, aligned to the far left!
if __name__ == "__main__":
    print("\n=== STARTING AUTOMATED RETRAINING PIPELINE ===")

    # 1. Resolve Path to the RAW Dataset inside your project
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, "data", "raw", "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    try:
        # 2. Ingest Data using your custom load module
        print(f"📥 Loading raw dataset from: {data_path}")
        raw_df = load_data(data_path)

        # 3. Process Features using your optimized 10-feature production engine
        processed_df = build_features(raw_df)

        # 4. Fire Retraining and log everything to MLflow UI cleanly
        train_model(processed_df, target_col="Churn")

    except Exception as pipeline_error:
        print(f"❌ [PIPELINE CRASH] Operational failure in training orchestration: {str(pipeline_error)}")