"""
INFERENCE LAYER - Production Model Serving & Agentic AI Ingestion
"""

import os
import pickle
import pandas as pd
from google import genai
from src.features.build_features import build_features

# 1. Load the optimal production LightGBM model directly from the models cache
model = None
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../..", "models", "customer_churn_lgb_model.pkl"))

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)
        print(f"✅ Success: Loaded optimized LightGBM model from local models cache.")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")

def predict_churn(input_dict: dict) -> float:
    """
    Returns the raw churn probability score using the tuned LightGBM architecture.
    """
    if model is None:
        return 0.0
    try:
        df_raw = pd.DataFrame([input_dict])
        df_ml_ready = build_features(df_raw, target_col="Churn")
        
        # Execute prediction mapping directly against the binary estimator probabilities array
        proba = model.predict_proba(df_ml_ready)[0][1]
        return float(proba)
    except Exception:
        return 0.0

def generate_ai_email(input_dict: dict) -> str:
    """
    Triggers Gemini Pro to write a highly personalized retention email using the 10 core features.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "❌ Setup Error: GEMINI_API_KEY missing in environment variables."
    
    try:
        client = genai.Client(api_key=api_key)
        
        # 3. Formulate the prompt using exclusively the available 10 operational parameters
        prompt = f"""
        You are an expert customer retention agent for a premium telecom company.
        A customer has been flagged by our Predictive AI as 'Likely to Churn' due to high-risk usage parameters.
        
        Customer Profile Data (Top 10 Metrics):
        - Monthly Charges: ${input_dict.get('MonthlyCharges')}
        - Account Tenure: {input_dict.get('tenure')} months
        - Payment Setup: {input_dict.get('PaymentMethod')}
        - Contract Structure: {input_dict.get('Contract')}
        - Security Subscription: {'Active' if input_dict.get('OnlineSecurity') == 'Yes' else 'Inactive/None'}
        - Multiple Phone Lines: {'Active' if input_dict.get('MultipleLines') == 'Yes' else 'Inactive/None'}
        - Paperless Billing: {'Enabled' if input_dict.get('PaperlessBilling') == 'Yes' else 'Disabled'}
        - Dedicated Technical Support: {'Active' if input_dict.get('TechSupport') == 'Yes' else 'Inactive/None'}
        - Online Cloud Backup: {'Active' if input_dict.get('OnlineBackup') == 'Yes' else 'Inactive/None'}
        - Senior Citizen Demographic Status: {'Yes' if input_dict.get('SeniorCitizen') == 1 else 'No'}
        
        Task:
        Write a highly persuasive, professional, and personalized retention email to this customer. 
        Offer a strategic incentive (e.g., a custom loyalty credit, premium tech support extension, or billing upgrade plan) calculated to align with their profile constraints. Keep the tone warm, welcoming, and premium. Do not use generic placeholders.
        """
        
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ Agentic AI Error: Failed to generate email. Details: {str(e)}"