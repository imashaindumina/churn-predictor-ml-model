import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the exact structural cleaning and feature purging pipeline 
    from Notebook 01 to stabilize the dataset for modeling.
    """
    # 1. Drop customerID due to zero predictive power
    df = df.drop(columns=['customerID'], errors='ignore')

    # 2. Handle 11 hidden blanks in TotalCharges and drop those rows
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df = df.dropna(subset=['TotalCharges'])

    # 3. Explicit Binary Mapping (Yes/No & Male/Female -> 1/0)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    binary_mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map(lambda x: binary_mapping.get(x, x)).astype(int)

    # 4. Multi class One-Hot Encoding 
    multi_cat_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
        'Contract', 'PaymentMethod'
    ]
    # Filter only existing categorical columns to avoid errors
    available_cats = [col for col in multi_cat_cols if col in df.columns]
    if available_cats:
        df = pd.get_dummies(df, columns=available_cats, drop_first=True, dtype=int)

    # 5. Global Feature Purging based on your VIF Audit (Crucial Step!)
    # Collapse 'No internet service' columns first to match your exact structural state
    no_internet_cols = [col for col in df.columns if 'No internet service' in col]
    if no_internet_cols:
        df['No_internet_service'] = df[no_internet_cols].any(axis=1).astype(int)
        df = df.drop(columns=no_internet_cols)

    # Drop MultipleLines_No phone service if present
    if 'MultipleLines_No phone service' in df.columns:
        df = df.drop(columns=['MultipleLines_No phone service'])

    # Now purge high VIF columns to globally stabilize as per section 7.2
    vif_purge_cols = ['MonthlyCharges', 'No_internet_service']
    df = df.drop(columns=vif_purge_cols, errors='ignore')

    # Reset index to guarantee clean continuous row tracking
    df = df.reset_index(drop=True)

    return df