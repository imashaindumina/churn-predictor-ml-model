"""
FASTAPI + GRADIO SERVING APPLICATION - Production-Ready ML Model Serving
========================================================================
Features: Fixed High-Contrast Status Boxes & Premium Midnight Corporate Theme.
"""

import os
import sys
from fastapi import FastAPI
from pydantic import BaseModel, Field
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.serving.inference import predict_churn, generate_ai_email  

app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="Operational Production API exposing our High-Recall Tuned LightGBM Champion Model with Live Gemini Pro Agentic Emails.",
    version="2.1.0"
)

OPTIMAL_THRESHOLD = 0.45

# === HEALTH CHECK ENDPOINT ===
@app.get("/")
def root():
    return {
        "status": "Active & Healthy",
        "model_version": "2.1.0-Gemini-Agentic-Enabled",
        "documentation_endpoint": "/docs",
        "web_ui_endpoint": "/ui"
    }

# === REQUEST DATA SCHEMA ===
class CustomerData(BaseModel):
    MonthlyCharges: float = Field(..., examples=[85.0])
    tenure: int = Field(..., examples=[12])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    OnlineSecurity: str = Field(..., examples=["No"])
    MultipleLines: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    TechSupport: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["No"])
    SeniorCitizen: int = Field(..., examples=[0])

# === MAIN PREDICTION API ENDPOINT ===
@app.post("/predict")
def get_prediction(data: CustomerData):
    try:
        payload_dict = data.model_dump()
        proba = predict_churn(payload_dict)
        result_string = "Likely to churn" if proba >= OPTIMAL_THRESHOLD else "Not likely to churn"
        
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
    input_keys = [
        "MonthlyCharges", "tenure", "PaymentMethod", "OnlineSecurity", 
        "MultipleLines", "Contract", "PaperlessBilling", "TechSupport", 
        "OnlineBackup", "SeniorCitizen"
    ]
    data_payload = dict(zip(input_keys, args))
    proba = predict_churn(data_payload)
    
    if proba >= OPTIMAL_THRESHOLD:
        # FIXED: Removed dynamic markdown <b> tags and hardcoded explicit dark high-contrast style sheets override
        status_md = f"""
        <div style='background-color: #fce4e4 !important; border-left: 6px solid #d32f2f !important; padding: 18px !important; border-radius: 8px !important;'>
            <h2 style='color: #c62828 !important; margin: 0 !important; font-size: 22px !important; font-weight: bold !important;'>⚠️ Operational Status: CHURN RISK DETECTED</h2>
            <p style='color: #b71c1c !important; margin: 8px 0 0 0 !important; font-size: 16px !important; font-weight: 600 !important;'>Prediction Matrix: Customer is Likely to Churn</p>
            <p style='color: #c62828 !important; margin: 4px 0 0 0 !important; font-size: 14px !important;'>Risk Profile Score: <span style='color: #b71c1c !important; font-weight: bold !important; font-size: 16px !important;'>{proba:.4f}</span> (Action Required via Gemini Pro Layer)</p>
        </div>
        """
        return status_md, gr.update(visible=True), gr.update(visible=True, value="")
    else:
        status_md = f"""
        <div style='background-color: #e8f5e9 !important; border-left: 6px solid #2e7d32 !important; padding: 18px !important; border-radius: 8px !important;'>
            <h2 style='color: #1b5e20 !important; margin: 0 !important; font-size: 22px !important; font-weight: bold !important;'>✅ Operational Status: ACCOUNT STABLE</h2>
            <p style='color: #1b5e20 !important; margin: 8px 0 0 0 !important; font-size: 16px !important; font-weight: 600 !important;'>Prediction Matrix: Customer is Not Likely to Churn</p>
            <p style='color: #2e7d32 !important; margin: 4px 0 0 0 !important; font-size: 14px !important;'>Risk Profile Score: <span style='color: #1b5e20 !important; font-weight: bold !important; font-size: 16px !important;'>{proba:.4f}</span> (Maintains Baseline Threshold)</p>
        </div>
        """
        return status_md, gr.update(visible=False), gr.update(visible=False, value="")

def generate_agentic_email_ui(*args):
    input_keys = [
        "MonthlyCharges", "tenure", "PaymentMethod", "OnlineSecurity", 
        "MultipleLines", "Contract", "PaperlessBilling", "TechSupport", 
        "OnlineBackup", "SeniorCitizen"
    ]
    data_payload = dict(zip(input_keys, args))
    return generate_ai_email(data_payload)


# === LUXURY HIGH-CONTRAST SYSTEM CSS CUSTOM THEME ===
custom_css = """
#predict-btn { background: linear-gradient(135deg, #e65100 0%, #ff6d00 100%) !important; border: none !important; color: white !important; font-weight: bold !important; font-size: 16px !important; text-transform: uppercase; padding: 12px !important; }
#predict-btn:hover { background: linear-gradient(135deg, #ff6d00 0%, #ff9100 100%) !important; box-shadow: 0 4px 15px rgba(255,109,0,0.4) !important; }
#email-btn { background: linear-gradient(135deg, #0091ea 0%, #00b0ff 100%) !important; border: none !important; color: white !important; font-weight: bold !important; text-transform: uppercase; }
#email-btn:hover { box-shadow: 0 4px 15px rgba(0,145,234,0.4) !important; }

/* FIXED: Changed from total black to a beautiful dark modern midnight slate */
.gradio-container { background-color: #111827 !important; font-family: 'Segoe UI', system-ui, sans-serif !important; }
.gradio-container-5-0 { background-color: #111827 !important; }
"""

# === STREAMLINED GRADIO UI ARCHITECTURE (BLOCKS) ===
with gr.Blocks(theme=gr.themes.Soft(primary_hue="orange", secondary_hue="slate"), css=custom_css) as demo:
    
    gr.Markdown(
        """
        # 🔮 Live Telco Agentic Churn Predictor Engine
        ### Enterprise-Grade Customer Risk Mitigation Platform powered by LightGBM & Google Gemini Pro.
        ---
        """
    )
    
    with gr.Row():
        # Input Columns (Left Column)
        with gr.Column(scale=11):
            gr.Markdown("### 📋 Customer Profile Attributes")
            
            with gr.Group():
                gr.Markdown("**💳 Financial & Subscription Metrics**")
                with gr.Row():
                    MonthlyCharges = gr.Number(label="Monthly Unit Charges ($)", value=85.0, minimum=0, maximum=200)
                    tenure = gr.Number(label="Tenure (Active Months)", value=12, minimum=0, maximum=120)
                
                with gr.Row():
                    Contract = gr.Dropdown(["Month-to-month", "One year", "Two year"], label="Contract Type", value="Month-to-month")
                    PaymentMethod = gr.Dropdown(["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], label="Payment Method", value="Electronic check")
                    PaperlessBilling = gr.Dropdown(["Yes", "No"], label="Paperless Billing", value="Yes")
            
            with gr.Group():
                gr.Markdown("**🛡️ Service & Account Configurations**")
                with gr.Row():
                    OnlineSecurity = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Security", value="No")
                    OnlineBackup = gr.Dropdown(["Yes", "No", "No internet service"], label="Online Cloud Backup", value="No")
                    TechSupport = gr.Dropdown(["Yes", "No", "No internet service"], label="Tech Support", value="No")
                with gr.Row():
                    MultipleLines = gr.Dropdown(["Yes", "No", "No phone service"], label="Multiple Phone Lines", value="No")
                    SeniorCitizen = gr.Dropdown([0, 1], label="Senior Citizen Status (1=Yes, 0=No)", value=0)
            
            submit_btn = gr.Button("Evaluate Customer Churn Risk", variant="primary", elem_id="predict-btn")
            
        # Output Columns (Right Column)
        with gr.Column(scale=9):
            gr.Markdown("### ⚡ AI Control & Strategy Hub")
            
            with gr.Group():
                # Base Placeholder Status
                output_status = gr.HTML("<div style='background-color: #1f2937; border-left: 6px solid #4b5563; padding: 16px; border-radius: 8px;'><h2 style='color: #9ca3af; margin: 0; font-size: 20px;'>Status: Awaiting Input Evaluation...</h2></div>")
            
            # Agentic Trigger Button (Hidden by default)
            email_btn = gr.Button("🤖 Generate AI Personalized Retention Email via Gemini Pro", variant="secondary", elem_id="email-btn", visible=False)
            
            # Live AI Generated Text Area (Hidden by default)
            output_email = gr.Textbox(label="Live Gemini Pro Generated Retention Draft", lines=17, interactive=False, visible=False)

    # Resolved components mapping vector
    resolved_inputs = [
        MonthlyCharges, tenure, PaymentMethod, OnlineSecurity, 
        MultipleLines, Contract, PaperlessBilling, TechSupport, 
        OnlineBackup, SeniorCitizen
    ]
    
    # Wire Interactive Actions
    submit_btn.click(fn=process_ui_submission, inputs=resolved_inputs, outputs=[output_status, email_btn, output_email])
    email_btn.click(fn=generate_agentic_email_ui, inputs=resolved_inputs, outputs=output_email)

# Mount Gradio into FastAPI Core Router
app = gr.mount_gradio_app(app, demo, path="/ui")