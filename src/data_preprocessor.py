import re

import pandas as pd

def check_dateFormat(df):
    print("Checking date format:")
    print("-" * 30)
    print(f"Sample dates: {df['TransactionMonth'].iloc[0]}")
    print(f"Data type of 'date' column: {df['TransactionMonth'].dtype}")
    print(f"  Target format: YYYY-MM-DD (string or date object)")


def missing_values(df):
    before = len(df)
    # Drop rows missing values
    df = df.dropna()

    removed = before - len(df)
    print(f"Removed {removed} rows with missing data")
    print(f"Remaining: {len(df)}")  
    return df

def normalize_date(df, column):
    df[column] = pd.to_datetime(df[column]).dt.normalize()

    print("Dates normalized")
    print(f"dtype: {df[column].dtype}")
    print(f"\nDate range: {df[column].min()} to {df[column].max()}")

    return df


import os

def save_cleaned_data(cleaned_data, data_name):
    # 1. Correct the target path to point from the project root directory
    output_path = f'data/processed/cleaned_{data_name}.csv'
    
    # 2. Ensure the directory structure is automatically created 
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 3. Save the dataframe (Using index=False keeps your column count clean)
    cleaned_data.to_csv(output_path, index=False)

    print(f"Shape of cleaned {data_name}: {cleaned_data.shape}")
    print(f"First 5 rows of cleaned {data_name}:")
    print(cleaned_data.head())

    return cleaned_data

# Ensure this block exists at the very bottom of src/data_preprocessor.py
if __name__ == "__main__":
    print("🚀 Preprocessor script started...")
    
    # 1. Load the raw data from the root folder
    df = pd.read_csv('MachineLearningRating_v3.txt', sep='|', parse_dates=['TransactionMonth'], na_values=[' ', 'Not specified'])
    
    # 2. Add your data cleaning steps here (the logic from the middle of your notebook)
    # Example:
    # df_clean = normalize_date(df, 'TransactionMonth')
    # df_clean = df_clean.dropna(subset=['some_important_column'])
    
    # Make sure whatever your final dataframe variable is called, it matches below:
    df_clean = df  # <-- Replace 'df' with your actual final cleaning steps / variable
    
    print("Processing complete. Attempting to save...")
    
    # 3. Save the data safely
    save_cleaned_data(df_clean, 'insurance_data')
    
    print("🏁 Preprocessor script finished successfully!")
