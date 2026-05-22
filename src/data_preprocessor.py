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


def save_cleaned_data(cleaned_data, data_name):
    #save cleaned data to csv file
    cleaned_data.to_csv(f'../data/processed/cleaned_{data_name}.csv', index=True)

    #print the shape of the cleaned data
    print(f"Shape of cleaned {data_name}: {cleaned_data.shape}")

    #print the first 5 rows of the cleaned data
    print(f"First 5 rows of cleaned {data_name}:")
    print(cleaned_data.head())

    return cleaned_data

