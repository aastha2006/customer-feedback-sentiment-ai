
import os
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from .data_ingestion import load_data
from .dataset import SentimentDataset
from .config import MODEL_NAME, MAX_LEN, BATCH_SIZE, EPOCHS, LEARNING_RATE, RANDOM_STATE, TEST_SIZE, BERT_MODEL_DIR
from .utils import logger

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model():
    logger.info("Starting BERT training pipeline...")
    
    # Check GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    df = load_data()
    X = df['text'].tolist()
    y = df['sentiment'].tolist()
    logger.info(f"Loaded {len(df)} samples.")

    # 2. Split Data
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 3. Tokenizer
    logger.info(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)
    
    # 4. Datasets
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer, MAX_LEN)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    
    # 5. Model
    logger.info(f"Loading model: {MODEL_NAME}")
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)
    model.to(device)
    
    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=str(BERT_MODEL_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=LEARNING_RATE,
        fp16=True if device == "cuda" else False, # Enable fp16 if on GPU
    )
    
    # 7. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 8. Train
    logger.info("Starting training...")
    trainer.train()
    
    # 9. Evaluate
    logger.info("Evaluating...")
    eval_result = trainer.evaluate()
    logger.info(f"Evaluation results: {eval_result}")
    
    # 10. Save
    logger.info(f"Saving model to {BERT_MODEL_DIR}")
    model.save_pretrained(BERT_MODEL_DIR)
    tokenizer.save_pretrained(BERT_MODEL_DIR)
    logger.info("Training complete.")

if __name__ == "__main__":
    train_model()
