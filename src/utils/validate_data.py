"""
VALIDATION LAYER - Lightweight Operational Data Quality Engine
"""

from typing import Tuple, List
import pandas as pd

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validates the structured 10-feature streaming parameters before feature processing.
    """
    print("🔍 Starting native production data validation pipeline...")
    failed_checks = []
    
    # 1. Column Existence and Null Constraints (Updated to match exact Top 10 Core list)
    REQUIRED_COLUMNS = [
        'MonthlyCharges', 'tenure', 'PaymentMethod', 'OnlineSecurity', 
        'MultipleLines', 'Contract', 'PaperlessBilling', 'TechSupport', 
        'OnlineBackup', 'SeniorCitizen'
    ]
    
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            failed_checks.append(f"MissingColumn -> '{col}' does not exist.")
        elif df[col].isnull().any():
            failed_checks.append(f"NullValuesFound -> '{col}' contains unmapped NaN fields.")
            
    if failed_checks:
        return False, failed_checks

    # 2. Categorical Business Logic Domain Rules Set Validation
    valid_contracts = {"Month-to-month", "One year", "Two year"}
    invalid_contracts = df[~df["Contract"].isin(valid_contracts)]["Contract"].unique()
    if len(invalid_contracts) > 0:
        failed_checks.append(f"InvalidSetConstraint -> 'Contract' contains invalid categories: {invalid_contracts}")
        
    valid_billing = {"Yes", "No"}
    invalid_billing = df[~df["PaperlessBilling"].isin(valid_billing)]["PaperlessBilling"].unique()
    if len(invalid_billing) > 0:
        failed_checks.append(f"InvalidSetConstraint -> 'PaperlessBilling' contains invalid categories: {invalid_billing}")

    # 3. Numerical Boundary Verification Range Checks
    # Tenure range boundary enforcement [0 to 120 months]
    invalid_tenure = df[(df["tenure"] < 0) | (df["tenure"] > 120)]
    if not invalid_tenure.empty:
        failed_checks.append(f"OutOfRange -> 'tenure' values outside bounds [0, 120]. Count: {len(invalid_tenure)}")
        
    # MonthlyCharges lower boundary check [Should not be negative]
    invalid_charges = df[df["MonthlyCharges"] < 0]
    if not invalid_charges.empty:
        failed_checks.append(f"OutOfRange -> 'MonthlyCharges' contains negative violations. Count: {len(invalid_charges)}")

    # 4. Final Verification Evaluation Status Reporting
    if not failed_checks:
        print(f"✅ Input Payload validation PASSED: All operational checks successful.")
        return True, []
    else:
        print(f"❌ Input Payload validation FAILED: {len(failed_checks)} critical assertions failed.")
        return False, failed_checks