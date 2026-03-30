
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import pickle
import zipfile
import os
import logging
import time
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "content-enrichment-488220"
BUCKET = "content-enrichment-488220"

app = FastAPI(
    title="Content Enrichment API",
    description="Classifies text content into categories using HuggingFace embeddings",
    version="2.0.0"
)

def load_models_from_gcs():
    logger.info("Loading v2 models from GCS...")
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET)

    # load classifier
    bucket.blob("models/v2_classifier.pkl").download_to_filename("/tmp/v2_classifier.pkl")
    with open("/tmp/v2_classifier.pkl", "rb") as f:
        classifier = pickle.load(f)
    logger.info("Classifier loaded ✅")

    # load embedder
    bucket.blob("models/v2_embedder.zip").download_to_filename("/tmp/v2_embedder.zip")
    with zipfile.ZipFile("/tmp/v2_embedder.zip", "r") as z:
        z.extractall("/tmp/v2_embedder")
    embedder = SentenceTransformer("/tmp/v2_embedder")
    logger.info("Embedder loaded ✅")

    return embedder, classifier

embedder, classifier = load_models_from_gcs()

MODEL_METADATA = {
    "model_name": "linear_svc_on_minilm_embeddings",
    "version": "2.0.0",
    "embedding_model": "all-MiniLM-L6-v2",
    "embedding_dim": 384,
    "accuracy": 0.8880,
    "f1_weighted": 0.8878,
    "categories": ["World", "Sports", "Business", "Sci_Tech"],
    "trained_on": "AG News Dataset (10k sample)",
    "improvement_over_v1": "semantic embeddings replace TF-IDF"
}

class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Federal Reserve raises interest rates"
            }
        }

class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict
    input_text: str
    model_version: str

@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}

@app.get("/model/info")
def model_info():
    return MODEL_METADATA

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()

    if not request.text or len(request.text.strip()) < 10:
        raise HTTPException(
            status_code=422,
            detail="Text too short. Please provide at least 10 characters."
        )

    try:
        # embed text
        embedding = embedder.encode([request.text])

        # predict category
        prediction = classifier.predict(embedding)[0]

        # get confidence scores
        # LinearSVC does not have predict_proba
        # use decision function instead
        decision = classifier.decision_function(embedding)[0]
        categories = classifier.classes_

        # convert decision scores to probabilities via softmax
        import numpy as np
        exp_scores = np.exp(decision - np.max(decision))
        probabilities = exp_scores / exp_scores.sum()

        all_scores = {
            cat: round(float(prob), 4)
            for cat, prob in zip(categories, probabilities)
        }
        confidence = round(float(max(probabilities)), 4)

        latency_ms = round((time.time() - start_time) * 1000, 2)
        logger.info({
            "event": "prediction",
            "input_length": len(request.text),
            "predicted_category": prediction,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "model_version": "2.0.0"
        })

        return PredictResponse(
            category=prediction,
            confidence=confidence,
            all_scores=all_scores,
            input_text=request.text,
            model_version="2.0.0"
        )

    except Exception as e:
        logger.error({
            "event": "prediction_error",
            "error": str(e),
            "input": request.text
        })
        raise HTTPException(status_code=500, detail=str(e))
