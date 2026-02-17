
import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from .config import BERT_MODEL_DIR, MAX_LEN, DEVICE
from .utils import logger

class ModelPredictor:
    _instance = None
    _model = None
    _tokenizer = None
    _device = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelPredictor, cls).__new__(cls)
            # Lazy initialization: Model is loaded on first prediction
            # cls._instance._initialize() 
        return cls._instance

    def _initialize(self):
        """
        Loads the model and tokenizer from disk.
        """
        logger.info("Loading BERT model artifacts...")
        try:
            self._device = torch.device(DEVICE)
            self._tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_DIR)
            self._model = DistilBertForSequenceClassification.from_pretrained(BERT_MODEL_DIR)
            self._model.to(self._device)
            self._model.eval()
            
            # Label mapping (must match training)
            self.id2label = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
            
            logger.info("BERT Model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise RuntimeError("Failed to load model artifacts.")

    def predict(self, text):
        """
        Predicts sentiment for a single text.
        Returns: (sentiment, confidence)
        """
        if not text:
            return None, 0.0

        if self._model is None:
             self._initialize()

        try:
            inputs = self._tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Get max probability and label index
            confidence, label_idx = torch.max(probs, dim=1)
            
            sentiment = self.id2label[label_idx.item()]
            return sentiment, confidence.item()
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Error", 0.0

    def predict_batch(self, texts):
        """
        Predicts sentiment for a list of texts.
        Returns: List of dictionaries [{'sentiment': 'Positive', 'confidence': 0.95}, ...]
        """
        results = []
        # Process in batches if necessary, but for simplicity here we loop or use small batches
        # For true batch processing with BERT, we should tokenize properly as a batch
        
        if self._model is None:
            self._initialize()
        
        try:
            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=MAX_LEN
            ).to(self._device)
            
            with torch.no_grad():
                outputs = self._model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            confidences, label_idxs = torch.max(probs, dim=1)
            
            for i in range(len(texts)):
                sentiment = self.id2label[label_idxs[i].item()]
                confidence = confidences[i].item()
                results.append({'sentiment': sentiment, 'confidence': confidence})
                
            return results

        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            # Fallback
            return [{'sentiment': 'Error', 'confidence': 0.0}] * len(texts)

# Singleton instance
predictor = ModelPredictor()
