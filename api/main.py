
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import time

from src.predict import predictor
from src.config import API_TITLE, API_VERSION
from src.utils import logger

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    text: str = Field(..., description="The customer feedback text to analyze", min_length=1)

class PredictionResponse(BaseModel):
    sentiment: str
    confidence: float

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., description="List of feedback texts", min_length=1)

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


# --- App Instance ---
app = FastAPI(title=API_TITLE, version=API_VERSION)
start_time = time.time()

# --- Endpoints ---

@app.get("/", response_model=HealthResponse)
def health_check():
    """Returns system health status."""
    return {
        "status": "healthy",
        "version": API_VERSION,
        "uptime_seconds": time.time() - start_time
    }

@app.post("/predict", response_model=PredictionResponse)
def predict_sentiment(request: PredictionRequest):
    """
    Predicts sentiment for a single text.
    """
    logger.info(f"Received prediction request for text: {request.text[:50]}...")
    try:
        sentiment, confidence = predictor.predict(request.text)
        return {"sentiment": sentiment, "confidence": confidence}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_predict", response_model=BatchPredictionResponse)
def batch_predict_sentiment(request: BatchPredictionRequest):
    """
    Predicts sentiment for a batch of texts.
    """
    logger.info(f"Received batch prediction request for {len(request.texts)} items.")
    try:
        results = predictor.predict_batch(request.texts)
        predictions = [
            {"sentiment": r["sentiment"], "confidence": r["confidence"]} for r in results
        ]
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Error during batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
