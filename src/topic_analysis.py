"""
Metro Passenger Feedback Sentiment & Topic Analyzer
Stage 6: Enhanced Topic Analysis Module

This script implements an enhanced NMF (Non-negative Matrix Factorization) topic modeling pipeline.
It extracts TF-IDF Unigram & Bigram features from `processed_text`, strips generic survey template words 
(e.g., 'overall', 'today', 'experience', 'trip', 'please', 'improve'), discovers 8 distinct operational topics, 
assigns dominant topics to feedback records, and exports all reports and visualizations.

Workflow:
  Feedback Text -> Better NLP representation -> TF-IDF (Unigrams + Bigrams) 
  -> Remove generic/template words -> NMF Topic Modeling -> 8 Meaningful Topics -> Topic Assignment

Author: Mini Project Team
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

# Scikit-Learn imports for NLP & Topic Modeling
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

def run_enhanced_topic_analysis(input_file: str,
                                output_csv: str,
                                report_file: str,
                                plot_file: str,
                                model_dir: str,
                                n_topics: int = 8):
    """
    Main function to execute refined NMF topic modeling.
    """
    print("=" * 75)
    print("       METRO PASSENGER FEEDBACK - ENHANCED TOPIC ANALYSIS (NMF)")
    print("=" * 75)

    # 1. Load Preprocessed Dataset
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input dataset not found at: {input_file}")

    print(f"\n[1/6] Loading preprocessed dataset from: {input_file}")
    df = pd.read_csv(input_file)
    total_records = len(df)
    print(f"      Total records loaded: {total_records}")

    # 2. Validate processed_text
    print("\n[2/6] Validating 'processed_text' column...")
    missing_text_count = df['processed_text'].isnull().sum()
    df['processed_text'] = df['processed_text'].fillna('')
    empty_text_count = (df['processed_text'].str.strip() == '').sum()

    print(f"      - Missing text values : {missing_text_count}")
    print(f"      - Empty text values   : {empty_text_count}")
    print(f"      - Total valid records : {total_records - empty_text_count}")

    # 3. Build Custom Stopwords List (Standard English + Generic Survey Template Noise)
    print("\n[3/6] Defining custom stop words to filter out survey template noise...")
    base_stopwords = list(stopwords.words('english'))
    generic_template_words = [
        'overall', 'today', 'experience', 'trip', 'please', 'improve', 
        'feel', 'needed', 'appeared', 'during', 'my', 'would', 'make',
        'could', 'also', 'really', 'much', 'like', 'get', 'see'
    ]
    custom_stopwords = list(set(base_stopwords + generic_template_words))
    print(f"      - Total stop words (English + Template filter): {len(custom_stopwords)}")

    # 4. Extract TF-IDF Features (Unigrams + Bigrams)
    print("\n[4/6] Extracting TF-IDF Unigrams + Bigrams from 'processed_text'...")
    vectorizer = TfidfVectorizer(
        max_features=2000,
        ngram_range=(1, 2),
        min_df=2,
        stop_words=custom_stopwords
    )
    X_tfidf = vectorizer.fit_transform(df['processed_text'])
    feature_names = vectorizer.get_feature_names_out()
    print(f"      - TF-IDF Feature Matrix Shape: {X_tfidf.shape}")

    # 5. Fit NMF Model for Topic Discovery
    print(f"\n[5/6] Fitting NMF Topic Model for {n_topics} Topics...")
    nmf_model = NMF(
        n_components=n_topics,
        random_state=42,
        init='nndsvda',
        max_iter=500
    )
    
    W = nmf_model.fit_transform(X_tfidf)
    H = nmf_model.components_

    # Extract Top Words & Map Human-Readable Topic Names
    topic_keywords = {}
    for topic_idx, topic_weights in enumerate(H):
        top_indices = topic_weights.argsort()[:-11:-1]
        topic_keywords[topic_idx] = [feature_names[i] for i in top_indices]

    topic_name_map = {
        0: "Coach & Infrastructure Maintenance",
        1: "Escalator & Lift Operational Maintenance",
        2: "Train Service Delays & Inconvenience",
        3: "Daily Commuter Fare Structure & Review",
        4: "Ticket Pricing & Short Distance Fares",
        5: "Station Staff Behavior & Security",
        6: "Station Seating & Signage Amenities",
        7: "Train & Platform Overcrowding"
    }

    # Assign dominant topic ID and readable topic name
    df['dominant_topic_id'] = W.argmax(axis=1)
    df['discovered_topic'] = df['dominant_topic_id'].map(topic_name_map)

    topic_counts = df['discovered_topic'].value_counts()
    topic_percents = df['discovered_topic'].value_counts(normalize=True) * 100

    print("\n" + "=" * 75)
    print("             REFINED DISCOVERED TOPICS & TOP KEYWORDS")
    print("=" * 75)
    for t_id, name in topic_name_map.items():
        words_str = ", ".join(topic_keywords[t_id])
        cnt = topic_counts.get(name, 0)
        pct = topic_percents.get(name, 0.0)
        print(f"Topic {t_id + 1}: {name}")
        print(f"  Count: {cnt} records ({pct:.1f}%)")
        print(f"  Top Words: {words_str}\n")
    print("=" * 75)

    # 6. Export Dataset, Models, Figures & Reports
    print("\n[6/6] Exporting Updated Dataset, Plots, and Evaluation Report...")
    
    # 6a. Export CSV
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"      - Saved updated dataset with 'discovered_topic' to:\n        {output_csv}")

    # 6b. Export NMF Model & Vectorizer
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(model_dir, "topic_tfidf_vectorizer.pkl"))
    joblib.dump(nmf_model, os.path.join(model_dir, "topic_nmf_model.pkl"))
    print(f"      - Saved Topic Vectorizer & NMF Model to: {model_dir}")

    # 6c. Export Visualization
    os.makedirs(os.path.dirname(plot_file), exist_ok=True)
    plt.figure(figsize=(10, 6))
    bars = plt.barh(topic_counts.index, topic_counts.values, color='#3498db', edgecolor='black')
    plt.xlabel("Number of Feedback Records")
    plt.ylabel("Discovered Topic")
    plt.title("Distribution of Refined Discovered Topics in Metro Passenger Feedback (NMF)")
    plt.gca().invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 3, bar.get_y() + bar.get_height()/2, f'{int(width)} ({width/total_records*100:.1f}%)', 
                 va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"      - Saved topic distribution plot to: {plot_file}")

    # 6d. Cross-tabulation Comparison with issue_category
    cross_tab = pd.crosstab(df['discovered_topic'], df['issue_category'])

    # 6e. Save Text Report
    os.makedirs(os.path.dirname(report_file), exist_ok=True)
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("METRO PASSENGER FEEDBACK - STAGE 6: REFINED TOPIC ANALYSIS REPORT\n")
        f.write("=" * 75 + "\n")
        f.write(f"Number of Records Analyzed : {total_records}\n")
        f.write(f"Number of Topics Discovered: {n_topics}\n")
        f.write("Workflow: Feedback Text -> Better NLP -> TF-IDF (Unigrams+Bigrams) -> Template Stopword Filter -> NMF -> Topic Assignment\n")
        f.write("=" * 75 + "\n\n")

        f.write("DISCOVERED TOPICS & TOP KEYWORDS:\n")
        f.write("-" * 75 + "\n")
        for t_id, name in topic_name_map.items():
            words_str = ", ".join(topic_keywords[t_id])
            cnt = topic_counts.get(name, 0)
            pct = topic_percents.get(name, 0.0)
            f.write(f"Topic {t_id + 1}: {name}\n")
            f.write(f"  Count: {cnt} records ({pct:.2f}%)\n")
            f.write(f"  Top 10 Words: {words_str}\n\n")

        f.write("CROSS-TABULATION COMPARISON (DISCOVERED TOPIC vs ORIGINAL ISSUE CATEGORY):\n")
        f.write("-" * 75 + "\n")
        f.write(cross_tab.to_string() + "\n\n")

    print(f"      - Saved topic analysis report to: {report_file}")

    # 7. Sample Assigned Records
    print("\n--- 10 SAMPLE ASSIGNED RECORDS (ORIGINAL TEXT vs DISCOVERED TOPIC) ---")
    sample_rows = df[['feedback_id', 'feedback_text', 'discovered_topic', 'issue_category', 'subtopic']].head(10)
    for idx, row in sample_rows.iterrows():
        print(f"\n[ID: {row['feedback_id']}]")
        print(f"  Feedback Text    : {row['feedback_text']}")
        print(f"  Discovered Topic : {row['discovered_topic']}")
        print(f"  Issue Category   : {row['issue_category']}")
        print(f"  Subtopic         : {row['subtopic']}")
    print("-" * 75)

    return df

if __name__ == "__main__":
    nlp_csv = "data/processed/metro_feedback_nlp.csv"
    topic_csv = "data/processed/metro_feedback_topics.csv"
    report_out = "results/reports/topic_analysis.txt"
    plot_out = "results/figures/topic_distribution.png"
    models_out = "models/trained_models"

    run_enhanced_topic_analysis(nlp_csv, topic_csv, report_out, plot_out, models_out, n_topics=8)
