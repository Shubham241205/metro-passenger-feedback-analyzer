import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from nltk.corpus import stopwords

df = pd.read_csv("data/processed/metro_feedback_nlp.csv")
df['processed_text'] = df['processed_text'].fillna('')

base_stopwords = list(stopwords.words('english'))
generic_template_words = [
    'overall', 'today', 'experience', 'trip', 'please', 'improve', 
    'feel', 'needed', 'appeared', 'during', 'my', 'would', 'make',
    'could', 'also', 'really', 'much', 'like', 'get', 'see'
]
custom_stopwords = list(set(base_stopwords + generic_template_words))

tfidf = TfidfVectorizer(
    max_features=2000,
    min_df=2,
    ngram_range=(1, 2),
    stop_words=custom_stopwords
)
X_tfidf = tfidf.fit_transform(df['processed_text'])
feature_names = tfidf.get_feature_names_out()

for n in [9, 10]:
    nmf = NMF(n_components=n, random_state=42, init='nndsvda', max_iter=500)
    W = nmf.fit_transform(X_tfidf)
    H = nmf.components_
    print(f"\n=================== {n} TOPICS ===================")
    for topic_idx, topic in enumerate(H):
        top_words_idx = topic.argsort()[:-11:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        print(f"Topic {topic_idx + 1}: {', '.join(top_words)}")
