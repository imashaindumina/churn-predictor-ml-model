import pandas as pd

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Production-ready feature pipeline. Takes the minimal user input payload,
    applies identical transformations to training, and outputs the 10 core features
    PLUS the target column if it exists in the raw data.
    """
    df = df.copy()
    
    # 1. Handle SeniorCitizen (Ensure it is standard int)
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)
        
    # 2. Manual Binary Encoding for PaperlessBilling (Yes: 1, No: 0)
    if 'PaperlessBilling' in df.columns:
        df['PaperlessBilling'] = df['PaperlessBilling'].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    # 3. Apply explicit manual One-Hot Encoding values to match training columns perfectly.
    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)
    df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['InternetService_No'] = (df['InternetService'] == 'No').astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['OnlineSecurity_Yes'] = (df['OnlineSecurity'] == 'Yes').astype(int)

    # 4. Final strict feature alignment table
    FINAL_10_FEATURES = [
        'tenure', 'TotalCharges', 'SeniorCitizen',
        'Contract_One year', 'Contract_Two year',
        'InternetService_Fiber optic', 'InternetService_No',
        'PaymentMethod_Electronic check', 'OnlineSecurity_Yes',
        'PaperlessBilling'
    ]
    
    # Fix TotalCharges dtype to float safely just before serving
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)

    # CRITICAL FIX: If the target column (Churn) is in the dataframe (like during training),
    # we must keep it so train_model can split it correctly!
    output_cols = FINAL_10_FEATURES.copy()
    if target_col in df.columns:
        # Convert text Churn ("Yes"/"No") to binary (1/0) if it is still string format
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].map({"No": 0, "Yes": 1}).fillna(0).astype(int)
        output_cols.append(target_col)

    # Filter and re-index safely
    df_final = df.reindex(columns=output_cols, fill_value=0)
    
    return df_final