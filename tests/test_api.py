import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# mock GCS model loading before app imports
with patch('google.cloud.storage.Client') as mock_storage:
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob

    # load real model from local file for testing
    def mock_download(filename):
        import shutil
        # use github-actions-key location as reference — model is local
        local_model = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'best_model.pkl')
        if os.path.exists(local_model):
            shutil.copy(local_model, filename)
        else:
            # create a minimal mock pipeline if no local model
            from sklearn.pipeline import Pipeline
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            import numpy as np

            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=1000)),
                ('classifier', LogisticRegression())
            ])
            texts = [
                "stock market business economy",
                "football basketball sports game",
                "president government world politics",
                "technology science computer nasa"
            ]
            labels = ["Business", "Sports", "World", "Sci_Tech"]
            pipeline.fit(texts, labels)
            with open(filename, 'wb') as f:
                pickle.dump(pipeline, f)

    mock_blob.download_to_filename.side_effect = mock_download

    from app import app

client = TestClient(app)

# ── SDE layer tests ───────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_returns_ok_status():
    response = client.get("/health")
    assert response.json()["status"] == "ok"

def test_model_info_returns_200():
    response = client.get("/model/info")
    assert response.status_code == 200

def test_model_info_has_accuracy():
    response = client.get("/model/info")
    assert "accuracy" in response.json()

def test_model_info_has_categories():
    response = client.get("/model/info")
    data = response.json()
    assert "categories" in data
    assert len(data["categories"]) == 4

def test_predict_returns_200():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    assert response.status_code == 200

def test_predict_returns_category():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    assert "category" in response.json()

def test_predict_returns_confidence():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    assert "confidence" in response.json()

def test_predict_returns_all_scores():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    assert "all_scores" in response.json()

def test_short_text_returns_422():
    response = client.post("/predict", json={"text": "hi"})
    assert response.status_code == 422

def test_empty_text_returns_422():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422

# ── MLE layer tests ───────────────────────────────────────────────────────────

def test_confidence_between_0_and_1():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    confidence = response.json()["confidence"]
    assert 0.0 <= confidence <= 1.0

def test_all_scores_sum_to_1():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    scores = response.json()["all_scores"]
    total = sum(scores.values())
    assert abs(total - 1.0) < 0.01

def test_category_is_valid():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    category = response.json()["category"]
    assert category in ["World", "Sports", "Business", "Sci_Tech"]

def test_confidence_not_too_low():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    confidence = response.json()["confidence"]
    assert confidence >= 0.20
