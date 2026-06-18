import os
import pandas as pd
import mlflow
from src.features.build_features import build_features

# === MODEL LOADING CONFIGURATION ===
MODEL_DIR = "/app/model"
model = None

try:
    # Try to load the trained model in MLflow pyfunc format
    model = mlflow.pyfunc.load_model(MODEL_DIR)
    print(f"✅ Production Model loaded successfully from {MODEL_DIR}")
except Exception as e:
    print(f"ℹ️ Container path not found ({MODEL_DIR}). Searching local mlruns fallback...")
    try:
        import glob
        local_model_paths = glob.glob("./mlruns/*/*/artifacts/model")
        if local_model_paths:
            latest_model = max(local_model_paths, key=os.path.getmtime)
            model = mlflow.pyfunc.load_model(latest_model)
            print(f"✅ Fallback Success: Loaded latest MLflow model from {latest_model}")
        else:
            raise Exception("No registered MLflow models found in local directory.")
    except Exception as fallback_error:
        print(f"❌ Dynamic model loading failed: {fallback_error}")

def predict(input_dict: dict) -> str:
    """
    Main optimized prediction function for customer churn inference.
    Processes the minimal required input payload using our consistent 10-feature structure
    and applies the tuned 0.3 probability threshold for high-recall safety.
    """
    if model is None:
        return "Error: Churn Prediction Engine is offline (Model not loaded)."

    try:
        # Step 1: Convert raw input dict to single-row Pandas DataFrame
        df_raw = pd.DataFrame([input_dict])
        
        # Step 2: Formulate the exact 10 numerical training tracking features safely
        df_ml_ready = build_features(df_raw, target_col="Churn")
        
        # Step 3: Extract prediction probabilities using MLflow pipeline wrapper
        # Note: pyfunc model returns a numpy array or dataframe of predictions/probabilities
        # XGBoost pyfunc predict_proba generally surfaces via continuous scoring or probabilities matrix
        try:
            # For standard logged booster models, predict handles raw score or class probabilities
            # To be strictly safe with our tuned threshold, we fetch the positive class probability
            if hasattr(model.metadata, "flavor_backend") and model.metadata.flavor_backend == "xgboost":
                # Raw underlying model extraction if standard pyfunc methods abstract class probs
                raw_booster = model._model_impl.xgboost_model
                proba = raw_booster.predict_proba(df_ml_ready)[0][1]
            else:
                # Standard fallback or direct continuous output inference
                preds_raw = model.predict(df_ml_ready)
                if hasattr(preds_raw, "iloc"):
                    proba = float(preds_raw.iloc[0])
                elif isinstance(preds_raw, list) or hasattr(preds_raw, "tolist"):
                    proba = float(preds_raw[0])
                else:
                    proba = float(preds_raw)
        except Exception:
            # Safe ultimate backup prediction format mapping
            preds_raw = model.predict(df_ml_ready)
            proba = float(preds_raw[0]) if hasattr(preds_raw, "__len__") else float(preds_raw)

        # Step 4: Apply our high-recall operational boundary (THRESHOLD = 0.3)
        THRESHOLD = 0.3
        
        # Step 5: Convert to Business-Friendly Output String matching Riyad's exact specs
        if proba >= THRESHOLD:
            return "Likely to churn"      # High risk - needs immediate retention action
        else:
            return "Not likely to churn"  # Low risk - maintain normal operation

    except Exception as inference_error:
        return f"Prediction Failure: {str(inference_error)}"