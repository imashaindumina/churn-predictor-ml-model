"""
FEATURE ENGINEERING LAYER - Operational Column Alignment Matrix
"""

import pandas as pd

def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Slices and maps the pandas dataframe layout to match the optimal 10-feature structure.
    """
    df = df.copy()
    
    # 1. Map raw metrics directly to structured categorical states
    if 'SeniorCitizen' in df.columns:
        df['SeniorCitizen'] = df['SeniorCitizen'].fillna(0).astype(int)
        
    if 'PaperlessBilling' in df.columns:
        df['PaperlessBilling'] = df['PaperlessBilling'].map({"No": 0, "Yes": 1}).fillna(0).astype(int)

    # 2. Build one-hot structural flags explicitly
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['OnlineSecurity_Yes'] = (df['OnlineSecurity'] == 'Yes').astype(int)
    df['MultipleLines_Yes'] = (df['MultipleLines'] == 'Yes').astype(int)
    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['TechSupport_Yes'] = (df['TechSupport'] == 'Yes').astype(int)
    df['OnlineBackup_Yes'] = (df['OnlineBackup'] == 'Yes').astype(int)

    # 3. Establish strict definitive feature importance column sequence
    FINAL_10_FEATURES = [
        'MonthlyCharges', 'tenure', 'PaymentMethod_Electronic check', 
        'OnlineSecurity_Yes', 'MultipleLines_Yes', 'Contract_One year', 
        'PaperlessBilling', 'TechSupport_Yes', 'OnlineBackup_Yes', 'SeniorCitizen'
    ]
    
    output_cols = FINAL_10_FEATURES.copy()
    
    # 4. Inject target column safely if pipeline tracks training data
    if target_col in df.columns:
        if df[target_col].dtype == 'object':
            df[target_col] = df[target_col].map({"No": 0, "Yes": 1}).fillna(0).astype(int)
        output_cols.append(target_col)

    return df.reindex(columns=output_cols, fill_value=0)