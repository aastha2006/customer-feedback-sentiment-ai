
import pandas as pd
import numpy as np
from .config import DATASET_PATH
from .utils import logger

def load_data(filepath=DATASET_PATH):
    """
    Loads dataset from CSV if it exists, otherwise generates dummy data.
    """
    if filepath.exists():
        logger.info(f"Loading data from {filepath}")
        try:
            df = pd.read_csv(filepath)
            
            # Normalize column names
            df.columns = [c.lower().strip() for c in df.columns]
            
            # Column mapping
            text_cols = ['review', 'content', 'body', 'reviews', 'comment']
            label_cols = ['label', 'target', 'class', 'rating', 'score']
            
            # Rename text column
            if 'text' not in df.columns:
                for col in text_cols:
                    if col in df.columns:
                        df = df.rename(columns={col: 'text'})
                        break
            
            # Rename sentiment column
            if 'sentiment' not in df.columns:
                for col in label_cols:
                    if col in df.columns:
                        df = df.rename(columns={col: 'sentiment'})
                        break

            # Ensure required columns exist
            if 'text' not in df.columns or 'sentiment' not in df.columns:
                raise ValueError(f"Dataset must contain 'text' and 'sentiment' columns. Found: {df.columns.tolist()}")
            # Drop duplicates and NaNs
            df = df.dropna(subset=['text', 'sentiment'])
            df = df.drop_duplicates()
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    else:
        logger.warning(f"Dataset not found at {filepath}. Generating dummy data.")
        return generate_dummy_data()

def generate_dummy_data(n_samples=100):
    """
    Generates synthetic data for testing.
    """
    data = {
        'text': [
            "I love this product, it works great!", 
            "Terrible experience, would not recommend.", 
            "It's okay, nothing special.",
            "Fast shipping and good quality.",
            "Broken upon arrival, very disappointed.",
            "Average product for the price.",
            "Excellent customer service!",
            "Waste of money.",
            "Does the job.",
            "Best purchase ever!"
        ] * (n_samples // 10),
        'sentiment': [
            "Positive", "Negative", "Neutral",
            "Positive", "Negative", "Neutral",
            "Positive", "Negative", "Neutral",
            "Positive"
        ] * (n_samples // 10)
    }
    
    df = pd.DataFrame(data)
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df
