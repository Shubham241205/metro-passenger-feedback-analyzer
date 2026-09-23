import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF

df = pd.read_csv("data/processed/metro_feedback_nlp.csv")
df['processed_text'] = df['processed_text'].fillna('')

tfidf = TfidfVectorizer(max_features=1500, min_df=2, ngram_range=(1, 2))
X_tfidf = tfidf.fit_transform(df['processed_text'])
feature_names = tfidf.get_feature_names_out()

nmf = NMF(n_components=8, random_state=42, init='nndsvda', max_iter=500)
W = nmf.fit_transform(X_tfidf)
df['topic_id'] = W.argmax(axis=1)

topic_names = {
    0: "Punctuality & Service Delays",
    1: "Maintenance & Station Cleanliness",
    2: "Ticket Pricing & Short Journey Cost",
    3: "Daily Commuter Fare Structure",
    4: "Staff Politeness & Security Assistance",
    5: "Escalator & Lift Facilities Maintenance",
    6: "Passenger Comfort & Water Facilities",
    7: "Train Overcrowding & Capacity"
}

df['discovered_topic'] = df['topic_id'].map(topic_names)

print("--- CROSS TABULATION: DISCOVERED TOPIC vs ISSUE CATEGORY ---")
ct = pd.crosstab(df['discovered_topic'], df['issue_category'])
print(ct)
