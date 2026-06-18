import pandas as pd
import os

def load_data(file_path: str) -> pd.DataFrame:
    
    #Loads raw Telco CSV data into a pandas DataFrame safely.

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Specified raw dataset path not found: {file_path}")
    
    return pd.read_csv(file_path)