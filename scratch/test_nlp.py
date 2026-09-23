import nltk
import re
import string
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # 1. Convert text to lowercase
    text = str(text).lower()
    # 2. Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # 3. Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 4. Remove unnecessary special characters
    text = re.sub(r'[^a-z0-9\s]', '', text)
    # 5. Normalize extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # 6. Tokenize text
    tokens = word_tokenize(text)
    # 7. Remove English stop words
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # 8. Lemmatize remaining words
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return " ".join(lemmatized_tokens)

sample_text = "Overall, The ticket machine was not working and caused a queue at https://example.com!"
print("Original:", sample_text)
print("Processed:", preprocess_text(sample_text))
