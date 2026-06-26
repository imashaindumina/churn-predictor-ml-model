import os
import pandas as pd
import mlflow
from google import genai
from src.features.build_features import build_features

# Load Model
model = None
LOCAL_ARTIFACT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "artifacts", "model"))
if os.path.exists(LOCAL_ARTIFACT_DIR):
    try:
        model = mlflow.pyfunc.load_model(LOCAL_ARTIFACT_DIR)
        print(f"✅ Success: Loaded production model from local artifacts.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

def predict_churn(input_dict: dict) -> float:
    """Returns the raw churn probability score."""
    if model is None:
        return 0.0
    try:
        df_raw = pd.DataFrame([input_dict])
        df_ml_ready = build_features(df_raw, target_col="Churn")
        
        if hasattr(model.metadata, "flavor_backend") and model.metadata.flavor_backend == "xgboost":
            raw_booster = model._model_impl.xgboost_model
            proba = raw_booster.predict_proba(df_ml_ready)[0][1]
        else:
            preds_raw = model.predict(df_ml_ready)
            proba = float(preds_raw.iloc[0]) if hasattr(preds_raw, "iloc") else float(preds_raw[0])
        return float(proba)
    except Exception:
        return 0.0

def generate_ai_email(input_dict: dict) -> str:
    """Triggers Gemini Pro to write a highly personalized retention email."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ Setup Error: GEMINI_API_KEY missing in environment variables."
    
    try:
        # Initialize the official new Google GenAI Client
        client = genai.Client(api_key=api_key)
        
        # Build contextual prompt from customer features
        prompt = f"""
        You are an expert customer retention agent for a premium telecom company.
        A high-value customer has been flagged by our Predictive AI as 'Likely to Churn' (risk of leaving the company).
        
        Customer Profile Data:
        - Gender: {input_dict.get('gender')}
        - Senior Citizen: {'Yes' if input_dict.get('SeniorCitizen') == 1 else 'No'}
        - Contract Type: {input_dict.get('Contract')}
        - Internet Service: {input_dict.get('InternetService')}
        - Monthly Charges: ${input_dict.get('MonthlyCharges')}
        - Total Tenure: {input_dict.get('tenure')} months
        
        Task:
        Write a highly persuasive, professional, and personalized retention email to this customer. 
        Offer a strategic incentive (e.g., a special discount or contract upgrade) based on their specific profile to convince them to stay. Keep the tone warm, welcoming, and premium. Do not use generic placeholders.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ Agentic AI Error: Failed to generate email. Details: {str(e)}"