"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================
Integrated with VIF-Optimized 10-Feature Pipeline & High-Recall operational constraints.
"""

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel, Field
import gradio as gr

# Ensure local source path utilities are correctly discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.serving.inference import predict  # Safe decoupled ML inference layer

# Initialize FastAPI application instance
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Operational Production API exposing our High-Recall Tuned XGBoost Champion Model.",
    version="2.0.0"
)

# === HEALTH CHECK ENDPOINT ===
@app.get("/")
def root():
    """
    Health check endpoint for container infrastructure monitoring and load balancer validation flags.
    """
    return {
        "status": "Active & Healthy",
        "model_version": "2.0.0-VIF-Optimized",
        "documentation_endpoint": "/docs",
        "web_ui_endpoint": "/ui"
    }

# === REQUEST DATA SCHEMA (18 BASELINE INPUTS FOR COMPATIBILITY) ===
class CustomerData(BaseModel):
    # Demographics
    gender: str = Field(..., examples=["Female"])
    Partner: str = Field(..., examples=["No"])
    Dependents: str = Field(..., examples=["No"])
    SeniorCitizen: int = Field(..., examples=[0])
    
    # Phone services
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    
    # Internet services  
    InternetService: str = Field(..., examples=["Fiber optic"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["No"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["Yes"])
    StreamingMovies: str = Field(..., examples=["Yes"])
    
    # Account information
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    
    # Numeric features
    tenure: int = Field(..., examples=[1])
    MonthlyCharges: float = Field(..., examples=[85.0])
    TotalCharges: float = Field(..., examples=[85.0])

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    """
    Main ingestion endpoint. Processes full operational request payload, passes it 
    safely down to the decoupled inference pipeline, and enforces the 0.3 threshold boundary.
    """
    try:
        # Pydantic v2 safe syntax transformation dict extraction
        payload_dict = data.model_dump()
        
        # Dispatch transaction directly to inference mapping loop
        result_string = predict(payload_dict)
        
        return {"prediction": result_string}
    except Exception as e:
        return {"error": f"Internal Inference Framework Core Exception: {str(e)}"}


# ===================================================================== #


# === GRADIO WEB INTERFACE BRIDGE ===
def gradio_interface(
    gender, Partner, Dependents, SeniorCitizen, PhoneService, MultipleLines,
    InternetService, OnlineSecurity, OnlineBackup, DeviceProtection,
    TechSupport, StreamingTV, StreamingMovies, Contract,
    PaperlessBilling, PaymentMethod, tenure, MonthlyCharges, TotalCharges
):
    """
    Form transformation wrapper designed to marshal sequential inputs into strict dictionary schemas.
    """
    data_payload = {
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "SeniorCitizen": int(SeniorCitizen),
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "tenure": int(tenure),
        "MonthlyCharges": float(MonthlyCharges),
        "TotalCharges": float(TotalCharges),
    }
    
    # Fire calculation using standard decoupled prediction architecture
    result = predict(data_payload)
    return str(result)

# === GRADIO UI MANAGEMENT ENGINE ===
demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Dropdown(["Yes", "No"], label="Partner", value="No"),
        gr.Dropdown(["Yes", "No"], label="Dependents", value="No"),
        gr.Dropdown([0, 1], label="Senior Citizen Status (1=Yes, 0=No)", value=0),
        
        gr.Dropdown(["Yes", "No"], label="Phone Service", value="Yes"),
        gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Lines", value="No"),
        
        gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="Fiber optic"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Online Backup", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Device Protection", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming TV", value="Yes"),
        gr.Dropdown(["Yes", "No", "No internet service"], label="Streaming Movies", value="Yes"),
        
        gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"),
        gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes"),
        gr.Dropdown([
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ], label="Payment Method", value="Electronic check"),
        
        gr.Number(label="Tenure (Total Active Months)", value=1, minimum=0, maximum=120),
        gr.Number(label="Monthly Statement Unit Charges ($)", value=85.0, minimum=0, maximum=200),
        gr.Number(label="Total Aggregated Ledger Charges ($)", value=85.0, minimum=0, maximum=20000),
    ],
    outputs=gr.Textbox(label="Operational Churn Decision Output", lines=2),
    title="🔮 Live Telco Customer Churn Predictor Engine",
    description="""
    ### ML-Powered Enterprise Customer Churn Analysis Platform
    Fill in the active field data records below to fetch real-time algorithmic insights regarding account retention status.
    This module features a VIF-Optimized high-recall inference matrix designed to intercept attrition threats before they impact revenue.
    """,
    examples=[
        ["Female", "No", "No", 0, "Yes", "No", "Fiber optic", "No", "No", "No", "No", "Yes", "Yes", "Month-to-month", "Yes", "Electronic check", 1, 85.0, 85.0],
        ["Male", "Yes", "Yes", 0, "Yes", "Yes", "DSL", "Yes", "Yes", "Yes", "Yes", "No", "No", "Two year", "No", "Credit card (automatic)", 60, 45.0, 2700.0]
    ],
    theme=gr.themes.Soft()
)

# === MOUNT GRADIO APP GATEWAY ===
app = gr.mount_gradio_app(app, demo, path="/ui")