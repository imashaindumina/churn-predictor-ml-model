from typing import Tuple, List
import pandas as pd

def validate_telco_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Production-grade lightweight data quality validation engine built natively.
    Validates streaming features and column criteria before feature engineering.
    """
    print("🔍 Starting native production data validation pipeline...")
    failed_checks = []
    
    # 1. Column Existence and Null Checks
    REQUIRED_COLUMNS = [
        "tenure", "TotalCharges", "SeniorCitizen", "Contract",
        "InternetService", "PaymentMethod", "OnlineSecurity", "PaperlessBilling"
    ]
    
    print("   📋 Validating required columns and null constraints...")
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            failed_checks.append(f"MissingColumn -> '{col}' does not exist.")
        elif df[col].isnull().any():
            failed_checks.append(f"NullValuesFound -> '{col}' contains unmapped NaN fields.")
            
    # If columns are missing, return early to prevent index errors
    if failed_checks:
        return False, failed_checks

    # 2. Business Logic Set Constraints
    print("   💼 Validating operational domain categorical rules...")
    
    valid_contracts = {"Month-to-month", "One year", "Two year"}
    invalid_contracts = df[~df["Contract"].isin(valid_contracts)]["Contract"].unique()
    if len(invalid_contracts) > 0:
        failed_checks.append(f"InvalidSetConstraint -> 'Contract' contains invalid categories: {invalid_contracts}")
        
    valid_internet = {"DSL", "Fiber optic", "No"}
    invalid_internet = df[~df["InternetService"].isin(valid_internet)]["InternetService"].unique()
    if len(invalid_internet) > 0:
        failed_checks.append(f"InvalidSetConstraint -> 'InternetService' contains invalid categories: {invalid_internet}")

    valid_billing = {"Yes", "No"}
    invalid_billing = df[~df["PaperlessBilling"].isin(valid_billing)]["PaperlessBilling"].unique()
    if len(invalid_billing) > 0:
        failed_checks.append(f"InvalidSetConstraint -> 'PaperlessBilling' contains invalid categories: {invalid_billing}")

    # 3. Numerical Scale Range Boundaries
    print("   📊 Verifying numerical bounds and scale parameters...")
    
    # Tenure check (0 to 120 months max telecom range)
    invalid_tenure = df[(df["tenure"] < 0) | (df["tenure"] > 120)]
    if not invalid_tenure.empty:
        failed_checks.append(f"OutOfRange -> 'tenure' contains values outside bounds [0, 120]. Count: {len(invalid_tenure)}")
        
    # TotalCharges safe numeric parsing and lower boundary check
    total_charges_numeric = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    invalid_charges = df[total_charges_numeric < 0]
    if not invalid_charges.empty:
        failed_checks.append(f"OutOfRange -> 'TotalCharges' contains negative boundary violations. Count: {len(invalid_charges)}")

    # Final Evaluation Summary Output
    if not failed_checks:
        print(f"✅ Input Payload validation PASSED: All operational checks successful.")
        return True, []
    else:
        print(f"❌ Input Payload validation FAILED: {len(failed_checks)} critical assertions failed.")
        print(f"   Violated Criteria details: {failed_checks}")
        return False, failed_checks