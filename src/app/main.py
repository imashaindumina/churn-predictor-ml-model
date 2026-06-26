"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================
Features: VIF-Optimized 10-Feature Inference Matrix & Live Gemini Pro Agentic Email Generation.
"""

import os
import sys
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel, Field
import gradio as gr
from dotenv import load_dotenv

# Ensure environment variables are loaded (.env)
load_dotenv()

# Ensure local source path utilities are correctly discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
# predict_churn සහ generate_ai_email අලුත් ලෙයර් එකෙන් ලෝඩ් කරගන්නවා
from src.serving.inference import predict_churn, generate_ai_email  

# Initialize FastAPI application instance
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Operational Production API exposing our High-Recall Tuned XGBoost Champion Model with Live Gemini Pro Agentic Emails.",
    version="2.1.0"
)

# === HEALTH CHECK ENDPOINT ===
@app.get("/")
def root():
    return {
        "status": "Active & Healthy",
        "model_version": "2.1.0-Gemini-Agentic-Enabled",
        "documentation_endpoint": "/docs",
        "web_ui_endpoint": "/ui"
    }

# === REQUEST DATA SCHEMA (18 BASELINE INPUTS FOR COMPATIBILITY) ===
class CustomerData(BaseModel):
    gender: str = Field(..., examples=["Female"])
    Partner: str = Field(..., examples=["No"])
    Dependents: str = Field(..., examples=["No"])
    SeniorCitizen: int = Field(..., examples=[0])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    tenure: int = Field(..., examples=[1])
    MonthlyCharges: float = Field(..., examples=[85.0])
    TotalCharges: float = Field(..., examples=[85.0])

# === MAIN PREDICTION API ENDPOINT (FASTAPI COMPATIBILITY) ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main API ingestion endpoint for backend requests.
    """
    try:
        payload_dict = data.model_dump()
        # Raw Churn probability එක ගන්නවා
        proba = predict_churn(payload_dict)
        THRESHOLD = 0.3
        result_string = "Likely to churn" if proba >= THRESHOLD else "Not likely to churn"
        
        return {
            "prediction": result_string,
            "churn_probability": round(proba, 4),
            "agentic_action_available": True if result_string == "Likely to churn" else False
        }
    except Exception as e:
        return {"error": f"Internal Inference Framework Core Exception: {str(e)}"}


# ===================================================================== #


# === GRADIO INTERFACE LOGIC BRIDGES ===

def process_ui_submission(*args):
    """
    Handles the initial ML Churn evaluation.
    If high risk, pops open the Generative AI control options.
    """
    input_keys = [
        "gender", "Partner", "Dependents", "SeniorCitizen", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges"
    ]
    data_payload = dict(zip(input_keys, args))
    
    # 1. Run ML Model Prediction
    proba = predict_churn(data_payload)
    THRESHOLD = 0.3
    
    if proba >= THRESHOLD:
        status_md = f"## ⚠️ Operational Status: **Likely to Churn** (Risk Score: {proba:.2f})"
        # Churn වෙනවා නම්: බටන් එකයි, ඊමේල් ටෙක්ස්ට් ඒරියා එකයි පෙන්වනවා (visible=True)
        return status_md, gr.update(visible=True), gr.update(visible=True, value="")
    else:
        status_md = f"## ✅ Operational Status: **Not likely to Churn** (Risk Score: {proba:.2f})"
        # Churn වෙන්නේ නැත්නම්: බටන් සහ ටෙක්ස්ට් ඒරියා හංගනවා (visible=False)
        return status_md, gr.update(visible=False), gr.update(visible=False, value="")

def generate_agentic_email_ui(*args):
    """
    Triggered when the user clicks the AI generation button.
    Calls Gemini Pro Live via the inference layer.
    """
    input_keys = [
        "gender", "Partner", "Dependents", "SeniorCitizen", "PhoneService", "MultipleLines",
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
        "PaperlessBilling", "PaymentMethod", "tenure", "MonthlyCharges", "TotalCharges"
    ]
    data_payload = dict(zip(input_keys, args))
    # Gemini එකෙන් ලයිව්ම ලියන ඊමේල් එක රිටර්න් කරනවා
    return generate_ai_email(data_payload)


# === NEXT-LEVEL GRADIO UI ARCHITECTURE (BLOCKS) ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔮 Live Telco Agentic Churn Predictor Engine")
    gr.Markdown("### ML-Powered Churn Analysis Platform coupled with Live Google Gemini Pro Generation.")
    
    with gr.Row():
        # Input Columns (වම් පැත්ත)
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Enter Customer Metrics")
            gender = gr.Dropdown(["Male", "Female"], label="Gender", value="Male")
            Partner = gr.Dropdown(["Yes", "No"], label="Partner", value="No")
            Dependents = gr.Dropdown(["Yes", "No"], label="Dependents", value="No")
            SeniorCitizen = gr.Dropdown([0, 1], label="Senior Citizen Status (1=Yes, 0=No)", value=0)
            PhoneService = gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes")
            MultipleLines = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No")
            InternetService = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic")
            OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
            OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No")
            DeviceProtection = gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No")
            TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
            StreamingTV = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes")
            StreamingMovies = gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes")
            Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month")
            PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
            PaymentMethod = gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check")
            tenure = gr.Number(label="Tenure (Total Active Months)", value=1, minimum=0, maximum=120)
            MonthlyCharges = gr.Number(label="Monthly Statement Unit Charges ($)", value=85.0, minimum=0, maximum=200)
            TotalCharges = gr.Number(label="Total Aggregated Ledger Charges ($)", value=85.0, minimum=0, maximum=20000)
            
            submit_btn = gr.Button("Evaluate Customer Churn Risk", variant="primary")
            
        # Output Columns (දකුණු පැත්ත)
        with gr.Column(scale=1):
            gr.Markdown("### 🛡️ AI Control & Strategy Center")
            
            # 1. Prediction Result Box
            output_status = gr.Markdown("## Status: Awaiting Input Evaluation...")
            
            # 2. Agentic Trigger Button (Hidden by default, shows only if likely to churn)
            email_btn = gr.Button("🤖 Generate AI Personalized Retention Email via Gemini Pro", variant="secondary", visible=False)
            
            # 3. Live AI Generated Text Area (Hidden by default)
            output_email = gr.Textbox(label="Live Gemini Pro Generated Retention Draft", lines=18, interactive=False, visible=False)

    # Inputs Matrix Wrapper for functions mapping
    ui_inputs = [
        gender, Partner, Dependents, SeniorCitizen, PhoneService, MultipleLines,
        InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies, Contract,
        PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
    ]
    
    # Wire Actions
    submit_btn.click(fn=process_ui_submission, inputs=ui_inputs, outputs=[output_status, email_btn, output_email])
    email_btn.click(fn=generate_agentic_email_ui, inputs=ui_inputs, outputs=output_email)

# Mount Gradio into FastAPI Core Router
app = gr.mount_gradio_app(app, demo, path="/ui")