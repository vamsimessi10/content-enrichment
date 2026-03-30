import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import pickle
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# mock everything before app imports
with patch('google.cloud.storage.Client') as mock_storage, \
     patch('sentence_transformers.SentenceTransformer') as mock_embedder_class, \
     patch('zipfile.ZipFile') as mock_zip:

    # mock GCS
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_storage.return_value.bucket.return_value = mock_bucket
    mock_bucket.blob.return_value = mock_blob
    mock_blob.download_to_filename.return_value = None

    # mock embedder
    mock_embedder = MagicMock()
    mock_embedder.encode.return_value = np.random.rand(1, 384)
    mock_embedder_class.return_value = mock_embedder

    # mock zip
    mock_zip.return_value.__enter__ = MagicMock(return_value=MagicMock())
    mock_zip.return_value.__exit__ = MagicMock(return_value=False)

    # create minimal mock classifier
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    import pickle

    classifier = LinearSVC()
    texts = [
        "stock market business economy finance",
        "football basketball sports game championship",
        "president government world politics international",
        "technology science computer nasa research"
    ]
    labels = ["Business", "Sports", "World", "Sci_Tech"]

    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(texts)
    classifier.fit(X.toarray(), labels)

    with open("/tmp/v2_classifier.pkl", "wb") as f:
        pickle.dump(classifier, f)

    def mock_download(filename):
        if "classifier" in filename:
            import shutil
            shutil.copy("/tmp/v2_classifier.pkl", filename)

    mock_blob.download_to_filename.side_effect = mock_download

    # mock predict to return valid results
    mock_embedder.encode.return_value = np.random.rand(1, 384)

    from app import app

# patch the module level embedder and classifier
import app as app_module
mock_clf = MagicMock()
mock_clf.predict.return_value = ["Business"]
mock_clf.decision_function.return_value = np.array([[0.1, 0.2, 0.8, 0.3]])
mock_clf.classes_ = ["Business", "Sci_Tech", "Sports", "World"]
app_module.classifier = mock_clf
app_module.embedder = mock_embedder

client = TestClient(app)

# ── SDE layer tests ───────────────────────────────────────────────────────────

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200

def test_health_returns_ok_status():
    response = client.get("/health")
    assert response.json()["status"] == "ok"

def test_health_returns_v2():
    response = client.get("/health")
    assert response.json()["version"] == "2.0.0"

def test_model_info_returns_200():
    response = client.get("/model/info")
    assert response.status_code == 200

def test_model_info_has_accuracy():
    response = client.get("/model/info")
    assert "accuracy" in response.json()

def test_model_info_has_embedding_model():
    response = client.get("/model/info")
    assert "embedding_model" in response.json()

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

def test_predict_returns_model_version():
    response = client.post("/predict", json={"text": "Federal Reserve raises interest rates amid inflation concerns"})
    assert "model_version" in response.json()
    assert response.json()["model_version"] == "2.0.0"

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

def test_model_info_has_embedding_dim():
    response = client.get("/model/info")
    assert "embedding_dim" in response.json()
    assert response.json()["embedding_dim"] == 384
