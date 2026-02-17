
import requests
import pandas as pd
import io
import os
from .config import DATASET_PATH
from .utils import logger

# URL for a subset or full IMDb dataset (CSV)
# Using a reliable GitHub raw link for IMDb dataset (50k reviews)
DATASET_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"

def download_data():
    """
    Downloads the IMDb dataset and saves it to the raw data directory.
    """
    logger.info(f"Downloading dataset from {DATASET_URL}...")
    try:
        response = requests.get(DATASET_URL)
        response.raise_for_status()
        
        # Read content
        content = response.content.decode('utf-8')
        df = pd.read_csv(io.StringIO(content))
        
        logger.info(f"Dataset downloaded. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Renaissance columns to match our schema if needed
        # Expected: 'text', 'sentiment'
        # IMDb dataset usually has 'review' and 'sentiment'
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
        
        # Normalize sentiment column
        # Ensure values are 'positive'/'negative' or 'Positive'/'Negative'
        # Our model expects: 'Positive', 'Negative', 'Neutral'
        # IMDb is usually 'positive', 'negative'
        df['sentiment'] = df['sentiment'].str.capitalize()
        
        # Save
        DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATASET_PATH, index=False)
        logger.info(f"Dataset saved to {DATASET_PATH}")
        
    except Exception as e:
        logger.error(f"Failed to download data: {e}")
        raise

if __name__ == "__main__":
    download_data()
