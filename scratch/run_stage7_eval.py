import pandas as pd
import numpy as np

# Load the topic-annotated dataset
df = pd.read_csv("data/processed/metro_feedback_topics.csv")

print("Total records:", len(df))
print("Columns:", df.columns.tolist())

print("\n--- DISCOVERED TOPICS DISTRIBUTION ---")
print(df['discovered_topic'].value_counts())
print(df['discovered_topic'].value_counts(normalize=True) * 100)

print("\n--- TOPIC VS ISSUE_CATEGORY CROSS TAB ---")
ct_cat = pd.crosstab(df['discovered_topic'], df['issue_category'])
print(ct_cat)

print("\n--- TOPIC VS SUBTOPIC CROSS TAB ---")
ct_sub = pd.crosstab(df['discovered_topic'], df['subtopic'])
print(ct_sub)

print("\n--- 20 REPRESENTATIVE EXAMPLES ---")
sample20 = df[['feedback_id', 'feedback_text', 'sentiment', 'discovered_topic', 'issue_category', 'subtopic']].head(20)
for idx, row in sample20.iterrows():
    print(f"[{row['feedback_id']}] Sentiment: {row['sentiment']} | Topic: {row['discovered_topic']} | Category: {row['issue_category']} | Subtopic: {row['subtopic']}")
    print(f"  Text: {row['feedback_text']}\n")
