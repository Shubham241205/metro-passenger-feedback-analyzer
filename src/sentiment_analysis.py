"""
Metro Passenger Feedback Sentiment & Topic Analyzer
Stage 5: Sentiment Analysis Module

This module loads the NLP preprocessed dataset (`data/processed/metro_feedback_nlp.csv`),
converts `processed_text` into numerical features using TF-IDF Vectorization,
trains a Logistic Regression classifier to predict sentiment (Positive, Neutral, Negative),
evaluates model performance, plots the confusion matrix, and saves trained model artifacts.

Author: Mini Project Team
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Scikit-Learn imports
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

def run_sentiment_analysis(input_file: str,
                           report_file: str,
                           plot_file: str,
                           model_dir: str):
    """
    Main function to execute sentiment classification pipeline.
    """
    print("=" * 70)
    print("        METRO PASSENGER FEEDBACK - STAGE 5: SENTIMENT ANALYSIS")
    print("=" * 70)

    # 1. Load Preprocessed Dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    print(f"\n[1/7] Loading preprocessed dataset from: {input_file}")
    df = pd.read_csv(input_file)
    print(f"      Total records loaded: {len(df)}")

    # 2. Target Variable Validation
    print("\n[2/7] Validating Target Variable ('sentiment')...")
    missing_sentiments = df['sentiment'].isnull().sum()
    unique_sentiments = df['sentiment'].unique()
    class_counts = df['sentiment'].value_counts()

    print(f"      - Missing sentiment values: {missing_sentiments}")
    print(f"      - Unique sentiment classes: {list(unique_sentiments)}")
    print("      - Class distribution:")
    for cls, count in class_counts.items():
        print(f"        * {cls}: {count} records ({count / len(df) * 100:.1f}%)")

    # Ensure feature column has no NaNs
    df['processed_text'] = df['processed_text'].fillna('')

    # Define Features (X) and Target (y)
    # IMPORTANT: Strictly use processed_text as input feature to avoid data leakage
    X = df['processed_text']
    y = df['sentiment']

    # 3. Train-Test Split (80% Train, 20% Test, Stratified)
    print("\n[3/7] Splitting dataset into training (80%) and testing (20%) sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
    print(f"      - Training records : {len(X_train)}")
    print(f"      - Testing records  : {len(X_test)}")

    # 4. TF-IDF Feature Extraction
    print("\n[4/7] Converting processed text into TF-IDF numerical features...")
    vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 2),
        min_df=2
    )
    
    # Fit vectorizer on training data and transform both train and test
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    num_features = X_train_tfidf.shape[1]
    print(f"      - TF-IDF Vocabulary / Feature Count: {num_features}")

    # 5. Model Training (Logistic Regression)
    print("\n[5/7] Training Logistic Regression Classifier...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'
    )
    model.fit(X_train_tfidf, y_train)
    print("      Model training complete.")

    # 6. Evaluation on Test Set
    print("\n[6/7] Predicting & Evaluating Model on Test Set...")
    y_pred = model.predict(X_test_tfidf)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    clf_rep = classification_report(y_test, y_pred, digits=4)
    labels = ['Negative', 'Neutral', 'Positive']
    cm = confusion_matrix(y_test, y_pred, labels=labels)

    print("\n" + "=" * 70)
    print("                  MODEL EVALUATION METRICS")
    print("=" * 70)
    print(f" Accuracy  : {acc:.4f} ({acc * 100:.2f}%)")
    print(f" Precision : {prec:.4f}")
    print(f" Recall    : {rec:.4f}")
    print(f" F1-Score  : {f1:.4f}")
    print("\nClassification Report:\n" + clf_rep)
    print("=" * 70)

    # Save Evaluation Report to file
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("METRO PASSENGER FEEDBACK - SENTIMENT ANALYSIS EVALUATION REPORT\n")
        f.write("=" * 65 + "\n")
        f.write(f"Training Records : {len(X_train)}\n")
        f.write(f"Testing Records  : {len(X_test)}\n")
        f.write(f"TF-IDF Features  : {num_features}\n")
        f.write("-" * 65 + "\n")
        f.write(f"Accuracy  : {acc:.4f}\n")
        f.write(f"Precision : {prec:.4f}\n")
        f.write(f"Recall    : {rec:.4f}\n")
        f.write(f"F1-Score  : {f1:.4f}\n")
        f.write("-" * 65 + "\n")
        f.write("Classification Report:\n")
        f.write(clf_rep + "\n")
        f.write("-" * 65 + "\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(cm) + "\n")

    print(f"      Saved evaluation report to: {report_file}")

    # Plot and Save Confusion Matrix
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title("Sentiment Classification Confusion Matrix")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"      Saved confusion matrix figure to: {plot_file}")

    # Save Trained Model Artifacts
    os.makedirs(model_dir, exist_ok=True)
    vectorizer_path = os.path.join(model_dir, "sentiment_tfidf_vectorizer.pkl")
    model_path = os.path.join(model_dir, "sentiment_logistic_regression.pkl")

    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(model, model_path)
    print(f"      Saved TF-IDF Vectorizer to: {vectorizer_path}")
    print(f"      Saved Logistic Regression Model to: {model_path}")

    # 7. Sample Test Predictions (10 Examples)
    print("\n[7/7] Displaying 10 Sample Predictions from Test Set:")
    print("=" * 70)
    
    test_indices = X_test.index[:10]
    sample_df = df.loc[test_indices].copy()
    sample_df['predicted_sentiment'] = y_pred[:10]

    for idx, (_, row) in enumerate(sample_df.iterrows(), 1):
        print(f"\nExample {idx} [ID: {row['feedback_id']}]:")
        print(f"  Feedback Text     : {row['feedback_text']}")
        print(f"  Actual Sentiment  : {row['sentiment']}")
        print(f"  Predicted Sentiment: {row['predicted_sentiment']}")
    print("-" * 70)

    return {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": num_features,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "report_path": report_file,
        "plot_path": plot_file,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path
    }

if __name__ == "__main__":
    nlp_csv = "data/processed/metro_feedback_nlp.csv"
    report_out = "results/reports/sentiment_evaluation.txt"
    plot_out = "results/figures/sentiment_confusion_matrix.png"
    models_out = "models/trained_models"

    run_sentiment_analysis(nlp_csv, report_out, plot_out, models_out)
