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
H = nmf.components_

print("--- TOP 10 WORDS PER TOPIC ---")
for topic_idx, topic in enumerate(H):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

df['dominant_topic_id'] = W.argmax(axis=1)
print("\n--- TOPIC COUNTS ---")
print(df['dominant_topic_id'].value_counts().sort_index())
