import pandas as pd
import numpy as np

raw_path = "data/raw/metro_passenger_feedback_dataset.csv"

# Load dataset
df = pd.read_csv(raw_path)

print("--- 1. NUMBER OF ROWS AND COLUMNS ---")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}\n")

print("--- 2. COLUMN NAMES AND DATA TYPES ---")
print(df.dtypes)
print("\nFirst 5 rows:")
print(df.head())
print()

print("--- 3. MISSING VALUES IN EACH COLUMN ---")
print(df.isnull().sum())
print()

print("--- 4. DUPLICATE ROWS ---")
print(f"Exact duplicate rows: {df.duplicated().sum()}")
print()

print("--- 5. DUPLICATE FEEDBACK_ID VALUES ---")
if 'feedback_id' in df.columns:
    print(f"Duplicate feedback_id count: {df['feedback_id'].duplicated().sum()}")
    print("Sample duplicate IDs:")
    print(df[df['feedback_id'].duplicated(keep=False)].sort_values('feedback_id'))
print()

print("--- 6. UNIQUE VALUES FOR CATEGORICAL COLUMNS ---")
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
for col in categorical_cols:
    print(f"Unique values in '{col}' ({df[col].nunique()} unique):")
    print(df[col].value_counts(dropna=False))
    print("-" * 40)
print()

print("--- 7. INVALID OR UNEXPECTED RATING VALUES ---")
if 'rating' in df.columns:
    print("Unique rating values:")
    print(df['rating'].value_counts(dropna=False))
print()

print("--- 8. INVALID SENTIMENT VALUES ---")
if 'sentiment' in df.columns:
    print("Unique sentiment values:")
    print(df['sentiment'].value_counts(dropna=False))
print()

print("--- 9. INVALID DATES ---")
date_cols = [c for c in df.columns if 'date' in c.lower()]
for c in date_cols:
    print(f"Checking date column '{c}':")
    invalid_dates = pd.to_datetime(df[c], errors='coerce').isnull() & df[c].notnull()
    print(f"Invalid date format entries count: {invalid_dates.sum()}")
    if invalid_dates.sum() > 0:
        print("Sample invalid dates:")
        print(df[invalid_dates][c])
print()
