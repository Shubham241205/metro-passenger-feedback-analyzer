import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("data/processed/metro_feedback_nlp.csv")
df['processed_text'] = df['processed_text'].fillna('')

# 1. Define custom generic/template stop words to filter out survey/boilerplate noise
base_stopwords = list(stopwords.words('english'))
generic_template_words = [
    'overall', 'today', 'experience', 'trip', 'please', 'improve', 
    'feel', 'needed', 'appeared', 'during', 'my', 'would', 'make',
    'could', 'also', 'really', 'much', 'like', 'get', 'see'
]
custom_stopwords = list(set(base_stopwords + generic_template_words))

# 2. TF-IDF Vectorizer with Unigrams + Bigrams & Custom Stopwords
tfidf = TfidfVectorizer(
    max_features=2000,
    min_df=2,
    ngram_range=(1, 2),
    stop_words=custom_stopwords
)
X_tfidf = tfidf.fit_transform(df['processed_text'])
feature_names = tfidf.get_feature_names_out()

# 3. NMF Topic Modeling (using 8 topics)
nmf = NMF(n_components=8, random_state=42, init='nndsvda', max_iter=500)
W = nmf.fit_transform(X_tfidf)
H = nmf.components_

print("--- REFINED NMF TOPICS (8 TOPICS) ---")
for topic_idx, topic in enumerate(H):
    top_words_idx = topic.argsort()[:-11:-1]
    top_words = [feature_names[i] for i in top_words_idx]
    print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")

df['dominant_topic_id'] = W.argmax(axis=1)
print("\n--- TOPIC DISTRIBUTION ---")
print(df['dominant_topic_id'].value_counts().sort_index())
