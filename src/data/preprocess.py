"""
PREPROCESSING LAYER - Streamlined Production Inference Pipeline
"""

import pandas as pd
import numpy as np

def preprocess_data(raw_payload: dict) -> np.ndarray:
    """
    Transforms a single customer dictionary payload into a structured 
    Top 10 array compatible with the tuned deployment model.
    """
    
    # 1. Parse baseline numerical metrics safely
    monthly_charges_val = float(raw_payload.get('MonthlyCharges', 0.0))
    tenure_val = int(raw_payload.get('tenure', 0))
    senior_citizen_val = int(raw_payload.get('SeniorCitizen', 0))
    
    # 2. Map categorical string values directly to dynamic continuous model states
    payment_method_val = 1 if raw_payload.get('PaymentMethod') == 'Electronic check' else 0
    online_security_val = 1 if raw_payload.get('OnlineSecurity') == 'Yes' else 0
    multiple_lines_val = 1 if raw_payload.get('MultipleLines') == 'Yes' else 0
    contract_one_year_val = 1 if raw_payload.get('Contract') == 'One year' else 0
    paperless_billing_val = 1 if raw_payload.get('PaperlessBilling') == 'Yes' else 0
    tech_support_val = 1 if raw_payload.get('TechSupport') == 'Yes' else 0
    online_backup_val = 1 if raw_payload.get('OnlineBackup') == 'Yes' else 0

    # 3. Formulate the exact 10-feature matrix mirroring model training order
    processed_vector = [
        monthly_charges_val,
        tenure_val,
        payment_method_val,
        online_security_val,
        multiple_lines_val,
        contract_one_year_val,
        paperless_billing_val,
        tech_support_val,
        online_backup_val,
        senior_citizen_val
    ]
    
    return np.array(processed_vector).reshape(1, -1)