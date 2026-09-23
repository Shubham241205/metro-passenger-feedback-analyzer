"""
Metro Passenger Feedback Sentiment & Topic Analyzer
Stage 4: NLP Preprocessing Module

This script performs Natural Language Processing (NLP) preprocessing on the cleaned metro feedback dataset.
It takes `data/processed/metro_feedback_cleaned.csv` as input, applies NLP text cleaning steps,
and saves the resulting dataset to `data/processed/metro_feedback_nlp.csv`.

Author: Mini Project Team
"""

import os
import re
import string
import pandas as pd
import nltk

# NLTK imports
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def ensure_nltk_resources():
    """
    Checks for required NLTK data resources and downloads them cleanly if not found.
    This prevents missing resource errors during tokenization, stop word removal, and lemmatization.
    """
    required_resources = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    print("[1/4] Checking and downloading required NLTK resources...")
    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
            print(f"      - NLTK resource '{resource_name}' is already available.")
        except LookupError:
            print(f"      - Downloading missing NLTK resource: '{resource_name}'...")
            nltk.download(resource_name, quiet=True)
    print("      All NLTK resources ready.\n")

def preprocess_text(text: str, stop_words: set, lemmatizer: WordNetLemmatizer) -> str:
    """
    Applies NLP preprocessing to a single feedback text string in exact sequence:
    1. Convert text to lowercase
    2. Remove URLs
    3. Remove punctuation
    4. Remove unnecessary special characters
    5. Normalize extra whitespace
    6. Tokenize the text
    7. Remove English stop words
    8. Lemmatize the remaining words
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Step 1: Convert text to lowercase
    text_lower = text.lower()

    # Step 2: Remove URLs (http, https, www)
    text_no_url = re.sub(r'https?://\S+|www\.\S+', '', text_lower)

    # Step 3: Remove punctuation
    text_no_punct = text_no_url.translate(str.maketrans('', '', string.punctuation))

    # Step 4: Remove unnecessary special characters (keep alphanumeric and spaces)
    text_clean_chars = re.sub(r'[^a-z0-9\s]', '', text_no_punct)

    # Step 5: Normalize extra whitespace (replace multiple spaces with single space)
    text_normalized = re.sub(r'\s+', ' ', text_clean_chars).strip()

    # Step 6: Tokenize the text into individual words
    tokens = word_tokenize(text_normalized)

    # Step 7: Remove English stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Step 8: Lemmatize the remaining words (convert words to base dictionary form)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    # Join clean tokens back into a single processed text string
    return " ".join(lemmatized_tokens)

def run_nlp_preprocessing(input_file: str, output_file: str):
    """
    Main function to load cleaned feedback dataset, create `processed_text` column,
    verify dataset integrity, and save to `metro_feedback_nlp.csv`.
    """
    print("=" * 70)
    print("        METRO PASSENGER FEEDBACK - STAGE 4: NLP PREPROCESSING")
    print("=" * 70)

    # Ensure all NLTK dependencies are available
    ensure_nltk_resources()

    # Initialize stop words and lemmatizer
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # 1. Load cleaned dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    print(f"[2/4] Loading cleaned dataset from: {input_file}")
    df = pd.read_csv(input_file)
    initial_rows = len(df)
    initial_cols = len(df.columns)
    print(f"      Loaded {initial_rows} records with {initial_cols} columns.\n")

    # 2. Apply NLP preprocessing to feedback_text
    print("[3/4] Running NLP text preprocessing pipeline on 'feedback_text'...")
    df['processed_text'] = df['feedback_text'].apply(
        lambda text: preprocess_text(text, stop_words, lemmatizer)
    )

    records_processed = len(df)
    empty_processed_count = (df['processed_text'].str.strip() == "").sum()

    # 3. Display verification metrics before saving
    print("\n" + "=" * 70)
    print("               PREPROCESSING VERIFICATION METRICS")
    print("=" * 70)
    print(f" - Total records in dataset     : {initial_rows}")
    print(f" - Total records processed        : {records_processed}")
    print(f" - Empty processed_text values   : {empty_processed_count}")
    print(f" - Original feedback_text preserved: {'Yes' if 'feedback_text' in df.columns else 'No'}")
    print(f" - New processed_text created     : {'Yes' if 'processed_text' in df.columns else 'No'}")
    print(f" - Total columns in final dataset : {len(df.columns)} (Original: {initial_cols})")
    print("=" * 70)

    # Display 5 sample comparisons
    print("\n--- 5 SAMPLE RECORDS (ORIGINAL vs PROCESSED TEXT) ---")
    samples = df[['feedback_id', 'feedback_text', 'processed_text']].head(5)
    for idx, row in samples.iterrows():
        print(f"\n[ID: {row['feedback_id']}]")
        print(f"  Original  : {row['feedback_text']}")
        print(f"  Processed : {row['processed_text']}")
    print("-" * 70)

    # 4. Save processed dataset to target CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n[4/4] Processed dataset saved successfully to:\n      {output_file}")
    print("=" * 70)

    # Safety Assertions / Verification
    assert len(df) == 2000, f"Error: Row count changed! Expected 2000, got {len(df)}"
    assert 'feedback_text' in df.columns, "Error: Original feedback_text column was removed!"
    assert 'processed_text' in df.columns, "Error: processed_text column was not created!"
    assert len(df.columns) == initial_cols + 1, f"Error: Expected {initial_cols + 1} columns, got {len(df.columns)}"

    return df

if __name__ == "__main__":
    cleaned_data_path = "data/processed/metro_feedback_cleaned.csv"
    nlp_output_path = "data/processed/metro_feedback_nlp.csv"
    run_nlp_preprocessing(cleaned_data_path, nlp_output_path)
