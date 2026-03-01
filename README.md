# Content Enrichment

News article classifier built as an end-to-end ML pipeline. Takes any text input and returns a category (World, Sports, Business, Sci_Tech) with a confidence score. Trained on 120,000 AG News articles, deployed as a live REST API on GCP Cloud Run.

Live API: https://content-enrichment-177416861217.us-central1.run.app/docs

---

## What it does

Send a POST request with any news text and get back a category prediction with confidence scores for all four classes.

Example request:

```json
POST /predict
{
  "text": "Federal Reserve raises interest rates amid inflation concerns"
}
```

Example response:

```json
{
  "category": "Business",
  "confidence": 0.8915,
  "all_scores": {
    "Business": 0.8915,
    "World": 0.0918,
    "Sci_Tech": 0.0119,
    "Sports": 0.0049
  },
  "input_text": "Federal Reserve raises interest rates amid inflation concerns"
}
```

---

## Architecture

```
AG News (HuggingFace)
        |
        v
  Data Cleaning
        |
        v
  BigQuery (permanent storage)
  GCS (backup)
        |
        v
  Model Training (5 models, MLflow tracking on Dagshub)
        |
        v
  Best model saved to GCS
        |
        v
  FastAPI app (app.py)
        |
        v
  Docker container
        |
        v
  Artifact Registry (GCP)
        |
        v
  Cloud Run (live public API)
        |
        v
  GitHub Actions (CI/CD on every push to main)
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Returns service status |
| GET | /model/info | Returns model metadata and accuracy |
| POST | /predict | Classifies text into one of four categories |

---

## Model

Five models were trained and compared using MLflow experiment tracking:

| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| Logistic Regression C=1.0 | 92.21% | 92.19% |
| Logistic Regression C=10.0 | 91.64% | 91.63% |
| Naive Bayes | 90.78% | 90.74% |
| Logistic Regression C=0.1 | 90.50% | 90.46% |
| Random Forest 100 | 89.99% | 89.94% |

Logistic Regression C=1.0 was selected. Per-category F1 scores: Sports 97%, World 93%, Business 89%, Sci_Tech 90%.

Pipeline: TF-IDF vectorizer (50,000 features, bigrams, English stop words removed) followed by Logistic Regression.

Experiment runs tracked at: https://dagshub.com/vamsimessi10/content-enrichment.mlflow

---

## Stack

- Python 3.11
- FastAPI + uvicorn
- scikit-learn 1.6.1
- Google Cloud Platform: BigQuery, GCS, Cloud Run, Artifact Registry, Cloud Build
- Docker
- GitHub Actions (CI/CD)
- MLflow + Dagshub (experiment tracking)

---

## Data

AG News dataset from HuggingFace. 120,000 training articles and 7,600 test articles across four categories. Stored in BigQuery at:

```
content-enrichment-488220.content_enrichment.ag_news_raw
```

CSV backup at:

```
gs://content-enrichment-488220/data/ag_news_clean.csv
```

---

## Project Structure

```
content-enrichment/
    app.py                    FastAPI application, loads model from GCS on startup
    Dockerfile                Container definition, Python 3.11 slim base
    requirements.txt          Pinned dependencies
    .dockerignore             Excludes credentials, model files, cache
    .gitignore                Excludes credentials, model files, logs
    .github/
        workflows/
            deploy.yml        CI/CD pipeline definition
```

---

## CI/CD

Every push to main triggers the GitHub Actions pipeline:

1. Authenticate to GCP using a service account key stored as a GitHub Secret
2. Build Docker image on the GitHub Actions runner
3. Push image to Artifact Registry (us-central1)
4. Deploy to Cloud Run

The full pipeline takes around 3-4 minutes from push to live deployment.

---

## Local Development

Clone the repo:

```bash
git clone https://github.com/vamsimessi10/content-enrichment.git
cd content-enrichment
```

The app loads the model from GCS on startup so you need GCP credentials available locally:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
```

Build and run with Docker:

```bash
docker build -t content-enrichment .
docker run -p 8000:8000 -e GOOGLE_CLOUD_PROJECT=content-enrichment-488220 content-enrichment
```

Test the API:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "SpaceX launches Starship on fourth test flight"}'
```

---

## GCP Resources

| Resource | Location |
|----------|----------|
| GCP Project | content-enrichment-488220 |
| BigQuery dataset | content_enrichment |
| GCS bucket | gs://content-enrichment-488220 |
| Model file | gs://content-enrichment-488220/models/best_model.pkl |
| Artifact Registry | us-central1-docker.pkg.dev/content-enrichment-488220/content-enrichment |
| Cloud Run service | us-central1 |

---

## What is next

- pytest test suite for all endpoints
- Cloud Logging dashboard and alerting
- v2: replace TF-IDF with HuggingFace transformer embeddings, expected accuracy improvement to 94-96%
- v3: LangChain agent layer using the classifier as a tool
