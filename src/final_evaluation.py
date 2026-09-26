"""
Metro Passenger Feedback Sentiment & Topic Analyzer
Stage 7: Final Model Evaluation Module

This script compiles the final evaluation report combining both Sentiment Analysis (Stage 5)
and Topic Analysis (Stage 6) results without retraining any models or modifying datasets.

Author: Mini Project Team
"""

import os
import pandas as pd
import numpy as np

def generate_final_evaluation_report(topics_csv_path: str, report_output_path: str):
    """
    Reads the processed dataset and writes a comprehensive final evaluation report.
    """
    print("=" * 75)
    print("      METRO PASSENGER FEEDBACK - STAGE 7: FINAL MODEL EVALUATION")
    print("=" * 75)

    if not os.path.exists(topics_csv_path):
        raise FileNotFoundError(f"Input file not found: {topics_csv_path}")

    df = pd.read_csv(topics_csv_path)
    total_records = len(df)

    # 1. Topic Distribution Stats
    topic_counts = df['discovered_topic'].value_counts()
    topic_percents = df['discovered_topic'].value_counts(normalize=True) * 100

    # 2. Cross Tabulations
    ct_category = pd.crosstab(df['discovered_topic'], df['issue_category'])
    ct_subtopic = pd.crosstab(df['discovered_topic'], df['subtopic'])

    # 3. 20 Representative Examples
    sample_20 = df[['feedback_id', 'feedback_text', 'sentiment', 'discovered_topic', 'issue_category', 'subtopic']].head(20)

    # 4. Generate Text Report Content
    os.makedirs(os.path.dirname(report_output_path), exist_ok=True)
    with open(report_output_path, 'w', encoding='utf-8') as f:
        f.write("=================================================================================\n")
        f.write("      METRO PASSENGER FEEDBACK SENTIMENT & TOPIC ANALYZER\n")
        f.write("                  FINAL MODEL EVALUATION REPORT\n")
        f.write("=================================================================================\n\n")

        # SECTION 1: DATASET & PREPROCESSING SUMMARY
        f.write("1. DATASET & PREPROCESSING INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Passenger Feedback Records : {total_records}\n")
        f.write(f"Total Columns                    : {len(df.columns)}\n")
        f.write("NLP Preprocessing Steps Applied  : Lowercasing, URL removal, Punctuation removal,\n")
        f.write("                                   Special character cleaning, Whitespace normalization,\n")
        f.write("                                   Tokenization, English Stop Words removal,\n")
        f.write("                                   WordNet Lemmatization, Template noise filtering.\n\n")

        # SECTION 2: SENTIMENT ANALYSIS EVALUATION
        f.write("2. SENTIMENT MODEL EVALUATION (STAGE 5)\n")
        f.write("-" * 80 + "\n")
        f.write("Algorithm              : TF-IDF Vectorizer + Logistic Regression\n")
        f.write("Train/Test Split       : 80% Training (1,600 records), 20% Testing (400 records)\n")
        f.write("Split Strategy         : Stratified Split (random_state=42)\n")
        f.write("Feature Count (TF-IDF) : 424 features\n\n")
        
        f.write("Overall Sentiment Metrics (Test Set):\n")
        f.write("  - Accuracy  : 0.8700 (87.00%)\n")
        f.write("  - Precision : 0.8573 (Weighted)\n")
        f.write("  - Recall    : 0.8700 (Weighted)\n")
        f.write("  - F1-Score  : 0.8496 (Weighted)\n\n")

        f.write("Detailed Classification Report:\n")
        f.write("              precision    recall  f1-score   support\n")
        f.write("    Negative     0.8247    0.9938    0.9014       161\n")
        f.write("     Neutral     0.6786    0.3065    0.4222        62\n")
        f.write("    Positive     0.9494    0.9548    0.9521       177\n\n")
        f.write("    accuracy                         0.8700       400\n")
        f.write("   macro avg     0.8176    0.7517    0.7586       400\n")
        f.write("weighted avg     0.8573    0.8700    0.8496       400\n\n")

        # SECTION 3: CONFUSION MATRIX INTERPRETATION
        f.write("3. CONFUSION MATRIX INTERPRETATION & LIMITATIONS\n")
        f.write("-" * 80 + "\n")
        f.write("Confusion Matrix Grid:\n")
        f.write("  Actual \\ Predicted    Negative    Neutral    Positive    Total Actual\n")
        f.write("    Actual Negative       160          1          0            161\n")
        f.write("    Actual Neutral         34         19          9             62\n")
        f.write("    Actual Positive         0          8        169            177\n\n")

        f.write("Key Observations & Class-Wise Performance:\n")
        f.write("  1. High Polarity Accuracy: The model shows near-perfect separation between Positive (F1: 0.952) \n")
        f.write("     and Negative (F1: 0.901) classes with zero cross-misclassifications between them.\n")
        f.write("  2. Neutral Class Limitation: Neutral sentiment achieved a lower recall of 30.65% (19/62 correct).\n")
        f.write("     Over 54.8% (34/62) of Neutral feedback records were misclassified as Negative.\n")
        f.write("     Reason: Neutral feedback often contains polite suggestions (e.g. 'Please review fare') \n")
        f.write("     which share vocabulary with negative complaints ('review', 'fare', 'cleanliness').\n\n")

        # SECTION 4: TOPIC MODEL EVALUATION
        f.write("4. TOPIC MODEL EVALUATION (STAGE 6 - NMF)\n")
        f.write("-" * 80 + "\n")
        f.write("Algorithm           : Non-negative Matrix Factorization (NMF)\n")
        f.write("Number of Topics    : 8 Discovered Topics\n")
        f.write("Evaluation Method   : Unsupervised semantic topic coherence & reference label cross-tabulation.\n")
        f.write("                      (Note: Classification accuracy is not computed for unsupervised NMF).\n\n")

        f.write("Discovered Topics Distribution:\n")
        for topic_name, count in topic_counts.items():
            pct = topic_percents[topic_name]
            f.write(f"  - {topic_name:<45} : {count:3d} records ({pct:5.2f}%)\n")
        f.write("\n")

        # SECTION 5: TOPIC vs REFERENCE LABELS COMPARISON
        f.write("5. DISCOVERED TOPICS vs REFERENCE LABELS COMPARISON\n")
        f.write("-" * 80 + "\n")
        f.write("Cross-Tabulation (Discovered Topic vs Original Issue Category):\n\n")
        f.write(ct_category.to_string() + "\n\n")

        f.write("Cross-Tabulation (Discovered Topic vs Original Subtopic):\n\n")
        f.write(ct_subtopic.to_string() + "\n\n")

        # SECTION 6: 20 REPRESENTATIVE EXAMPLES
        f.write("6. 20 REPRESENTATIVE FEEDBACK EXAMPLES\n")
        f.write("-" * 80 + "\n")
        for idx, row in sample_20.iterrows():
            f.write(f"[{row['feedback_id']}] Sentiment: {row['sentiment']:<8} | Discovered Topic: {row['discovered_topic']}\n")
            f.write(f"  Ground-Truth Category: {row['issue_category']} | Subtopic: {row['subtopic']}\n")
            f.write(f"  Feedback Text: \"{row['feedback_text']}\"\n\n")

        # SECTION 7: STRENGTHS, LIMITATIONS & OVERALL FINDINGS
        f.write("7. PROJECT STRENGTHS, LIMITATIONS & OVERALL FINDINGS\n")
        f.write("-" * 80 + "\n")
        f.write("Strengths:\n")
        f.write("  - Robust end-to-end pipeline from data cleaning to NLP preprocessing, sentiment prediction, and topic discovery.\n")
        f.write("  - High overall sentiment classification accuracy (87.00%) with 0% data leakage.\n")
        f.write("  - Clean NMF topic discovery eliminating synthetic survey template noise via custom stopword filtering.\n")
        f.write("  - Clear semantic alignment between discovered topics and real operational metro service categories.\n\n")
        f.write("Limitations:\n")
        f.write("  - Neutral sentiment detection is challenging due to lexical overlap with mild negative complaints.\n")
        f.write("  - Unsupervised NMF topic modeling requires human interpretation for topic label assignment.\n\n")
        f.write("Overall Findings:\n")
        f.write("  The NLP pipeline effectively categorizes passenger sentiment and uncovers primary operational issues\n")
        f.write("  (e.g., overcrowding, staff behavior, pricing, equipment maintenance) to inform metro transit improvements.\n")
        f.write("=================================================================================\n")

    print(f"\n[1/1] Final evaluation report successfully generated at:\n      {report_output_path}")

if __name__ == "__main__":
    topics_csv = "data/processed/metro_feedback_topics.csv"
    final_report = "results/reports/final_model_evaluation.txt"
    generate_final_evaluation_report(topics_csv, final_report)
