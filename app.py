
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import pickle
import os
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ID = "content-enrichment-488220"
BUCKET = "content-enrichment-488220"
MODEL_PATH = "models/best_model.pkl"

app = FastAPI(
    title="Content Enrichment API",
    description="Classifies text content into categories using NLP",
    version="1.0.0"
)

def load_model_from_gcs():
    logger.info(f"Loading model from gs://{BUCKET}/{MODEL_PATH}")
    gcs_client = storage.Client(project=PROJECT_ID)
    bucket = gcs_client.bucket(BUCKET)
    blob = bucket.blob(MODEL_PATH)
    blob.download_to_filename("model.pkl")
    with open("model.pkl", "rb") as f:
        pipeline = pickle.load(f)
    logger.info("Model loaded successfully ✅")
    return pipeline

pipeline = load_model_from_gcs()

MODEL_METADATA = {
    "model_name": "logistic_regression_c1",
    "version": "1.0.0",
    "accuracy": 0.9221,
    "f1_weighted": 0.9219,
    "categories": ["World", "Sports", "Business", "Sci_Tech"],
    "trained_on": "AG News Dataset (120k articles)"
}

class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "Apple reports record quarterly earnings"
            }
        }

class PredictResponse(BaseModel):
    category: str
    confidence: float
    all_scores: dict
    input_text: str

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}

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
        prediction = pipeline.predict([request.text])[0]
        probabilities = pipeline.predict_proba([request.text])[0]
        categories = pipeline.classes_
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
            "latency_ms": latency_ms
        })

        return PredictResponse(
            category=prediction,
            confidence=confidence,
            all_scores=all_scores,
            input_text=request.text
        )

    except Exception as e:
        logger.error({
            "event": "prediction_error",
            "error": str(e),
            "input": request.text
        })
        raise HTTPException(status_code=500, detail=str(e))
