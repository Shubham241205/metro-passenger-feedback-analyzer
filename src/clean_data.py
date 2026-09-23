"""
Metro Passenger Feedback Sentiment & Topic Analyzer
Stage 2: Data Cleaning Module

This script loads the raw metro passenger feedback dataset, performs data validation
and cleaning operations according to specified project guidelines, and exports the 
cleaned dataset to `data/processed/metro_feedback_cleaned.csv`.

Author: Mini Project Team
"""

import os
import pandas as pd

def clean_dataset(raw_csv_path: str, output_csv_path: str):
    """
    Reads the raw feedback CSV, applies cleaning transformations, 
    and saves the cleaned CSV to the processed data directory.
    """
    print("=" * 60)
    print("      METRO PASSENGER FEEDBACK - DATA CLEANING PIPELINE")
    print("=" * 60)

    # 1. Load Raw Dataset
    if not os.path.exists(raw_csv_path):
        raise FileNotFoundError(f"Raw dataset file not found at: {raw_csv_path}")

    print(f"\n[1/6] Loading raw dataset from: {raw_csv_path}")
    df_raw = pd.read_csv(raw_csv_path)
    initial_record_count = len(df_raw)
    print(f"      Initial record count: {initial_record_count} rows, {df_raw.shape[1]} columns")

    df = df_raw.copy()

    # Track cleaning operations
    operations_log = []

    # 2. Remove Exact Duplicate Rows
    duplicates_before = df.duplicated().sum()
    if duplicates_before > 0:
        df = df.drop_duplicates()
        operations_log.append(f"Removed {duplicates_before} exact duplicate row(s).")
    else:
        operations_log.append("Checked for exact duplicate rows: 0 duplicates found.")

    # 3. Remove Duplicate feedback_id Records (Keeping First Valid Record)
    if 'feedback_id' in df.columns:
        duplicate_ids_before = df['feedback_id'].duplicated().sum()
        if duplicate_ids_before > 0:
            df = df.drop_duplicates(subset=['feedback_id'], keep='first')
            operations_log.append(f"Removed {duplicate_ids_before} duplicate feedback_id record(s), keeping first occurrence.")
        else:
            operations_log.append("Checked for duplicate feedback_id values: 0 duplicates found.")

    # 4. Trim Leading & Trailing Whitespace in Text and Categorical Fields
    string_columns = df.select_dtypes(include=['object']).columns
    trimmed_cols_count = 0
    for col in string_columns:
        # Strip leading and trailing whitespaces
        df[col] = df[col].astype(str).str.strip()
        trimmed_cols_count += 1
    operations_log.append(f"Stripped leading and trailing whitespaces across {trimmed_cols_count} text/categorical columns.")

    # 5. Handle Missing Values
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    if total_nulls > 0:
        # Drop rows with missing crucial fields like feedback_text or rating
        df = df.dropna(subset=['feedback_text', 'rating', 'feedback_id'])
        operations_log.append(f"Handled missing values: Removed rows with critical missing fields ({total_nulls} total missing cells).")
    else:
        operations_log.append("Checked missing values: 0 missing values present in dataset.")

    # 6. Validate and Format Rating (Numeric and between 1 and 5)
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
        invalid_ratings = df[(df['rating'] < 1) | (df['rating'] > 5) | (df['rating'].isnull())]
        if len(invalid_ratings) > 0:
            df = df[(df['rating'] >= 1) & (df['rating'] <= 5)]
            operations_log.append(f"Removed {len(invalid_ratings)} row(s) with invalid rating values.")
        else:
            operations_log.append("Validated ratings: All ratings are numeric integers in the valid range [1 to 5].")
        df['rating'] = df['rating'].astype(int)

    # 7. Validate and Standardize feedback_date
    if 'feedback_date' in df.columns:
        parsed_dates = pd.to_datetime(df['feedback_date'], errors='coerce')
        invalid_dates = parsed_dates.isnull()
        if invalid_dates.sum() > 0:
            df = df[~invalid_dates]
            operations_log.append(f"Removed {invalid_dates.sum()} row(s) with invalid date values.")
        else:
            operations_log.append("Validated feedback dates: All dates are valid ISO YYYY-MM-DD dates.")
        df['feedback_date'] = parsed_dates.dt.strftime('%Y-%m-%d')

    # Calculate final stats
    final_record_count = len(df)
    records_removed = initial_record_count - final_record_count

    # 8. Export Cleaned Dataset
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f"\n[2/6] Successfully saved cleaned dataset to: {output_csv_path}")

    # 9. Print Data Cleaning Summary Report
    print("\n" + "=" * 60)
    print("                 DATA CLEANING SUMMARY REPORT")
    print("=" * 60)
    print(f" - Records before cleaning : {initial_record_count}")
    print(f" - Records removed         : {records_removed}")
    print(f" - Records remaining       : {final_record_count}")
    print("\nCleaning Operations Performed:")
    for idx, op in enumerate(operations_log, 1):
        print(f"  {idx}. {op}")

    print("\nFinal Column List ({0} columns):".format(len(df.columns)))
    for col in df.columns:
        print(f"  - {col} ({df[col].dtype})")
    print("=" * 60)

    return df, initial_record_count, records_removed, final_record_count, operations_log

if __name__ == "__main__":
    raw_file = "data/raw/metro_passenger_feedback_dataset.csv"
    processed_file = "data/processed/metro_feedback_cleaned.csv"
    clean_dataset(raw_file, processed_file)
