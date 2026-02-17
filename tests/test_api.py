
import pytest
import pathlib
from fastapi.testclient import TestClient
from api.main import app
from src.predict import predictor

client = TestClient(app)

# --- Mocking for Prediction Logic (Optional but good practice) ---
# For end-to-end testing, we'll assume the model is trained or uses dummy data logic.
# However, to be robust, we should check if model exists or mock it.
# Since we might run 'make test' before 'make train', let's handle that.

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_prediction_endpoint_validation():
    # Empty text
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # Validation error

    # Missing field
    response = client.post("/predict", json={})
    assert response.status_code == 422

# NOTE: The following tests depend on the model being trained.
# In a real CI/CD, we would train a dummy model or mock the predictor.
@pytest.mark.skipif(not pathlib.Path("models/sentiment_model.joblib").exists(), 
                    reason="Model not trained yet")
def test_prediction_flow():
    response = client.post("/predict", json={"text": "This is great!"})
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

@pytest.mark.skipif(not pathlib.Path("models/sentiment_model.joblib").exists(), 
                    reason="Model not trained yet")
def test_batch_prediction_flow():
    payload = {"texts": ["Good product", "Bad product"]}
    response = client.post("/batch_predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["sentiment"] is not None
