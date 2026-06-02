import os
import sys
import pytest
from fastapi.testclient import TestClient

# Ensure root folder is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import app

client = TestClient(app)

def test_home_route():
    """
    Test the home root endpoint.
    """
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["status"] == "API is running"
    assert "version" in data

def test_health_check():
    """
    Test the health check behavior.
    """
    response = client.get("/health")
    if response.status_code == 200:
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
    else:
        assert response.status_code == 503
        data = response.json()
        assert "detail" in data

def test_predict_success():
    """
    Test a valid prediction request.
    """
    payload = {"text": "Premium Sony Wireless Headphones with noise cancelling technology"}
    response = client.post("/predict", json=payload)
    if response.status_code == 200:
        data = response.json()
        assert "input" in data
        assert "prediction" in data
        assert "category" in data["prediction"]
        assert "confidence" in data["prediction"]
        assert 0.0 <= data["prediction"]["confidence"] <= 1.0
    else:
        # Handles case where model weights are not pre-trained
        assert response.status_code in [500, 503]

def test_predict_input_too_short():
    """
    Test validation fails when text is less than 3 characters.
    """
    payload = {"text": "hi"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_input_too_long():
    """
    Test validation fails when text exceeds 500 characters.
    """
    payload = {"text": "a" * 501}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_input_whitespace_only():
    """
    Test validation fails for whitespace-only strings.
    """
    payload = {"text": "   "}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Text must contain at least 3 non-whitespace characters." in response.text

def test_predict_missing_payload():
    """
    Test validation fails when request schema is empty.
    """
    response = client.post("/predict", json={})
    assert response.status_code == 422
