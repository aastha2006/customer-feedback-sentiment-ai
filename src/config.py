
import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_PATH = RAW_DATA_DIR / "dataset.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "sentiment_model.joblib"
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

# Logs
LOGS_DIR = PROJECT_ROOT / "logs"
LOG_FILE = LOGS_DIR / "app.log"

# Model parameters
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256  # DistilBERT max length
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
RANDOM_STATE = 42
TEST_SIZE = 0.2

import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths for HF model
BERT_MODEL_DIR = MODELS_DIR / "distilbert_sentiment"

# API
API_TITLE = "Customer Feedback Sentiment Analysis API"
API_VERSION = "1.0.0"
API_HOST = "0.0.0.0"
API_PORT = 8000
