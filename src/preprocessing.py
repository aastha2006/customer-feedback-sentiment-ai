
import re
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from .config import VECTORIZER_PATH, MAX_FEATURES

class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, max_features=MAX_FEATURES):
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, 
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )

    def clean_text(self, text):
        """
        Cleans text by removing special characters, lowercasing, etc.
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def fit(self, X, y=None):
        """
        Fits the vectorizer on the cleaned text.
        """
        # Ensure input is a list or series of strings
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
        
        cleaned_X = [self.clean_text(text) for text in X]
        self.vectorizer.fit(cleaned_X)
        return self

    def transform(self, X):
        """
        Transforms the text using the fitted vectorizer.
        """
        if isinstance(X, pd.DataFrame):
            X = X.iloc[:, 0].tolist()
            
        cleaned_X = [self.clean_text(text) for text in X]
        return self.vectorizer.transform(cleaned_X)

    def save(self, filepath=VECTORIZER_PATH):
        """Saves the preprocessor to disk."""
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath=VECTORIZER_PATH):
        """Loads the preprocessor from disk."""
        return joblib.load(filepath)
