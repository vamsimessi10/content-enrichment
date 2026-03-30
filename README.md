# Content Enrichment

News article classifier built as an end-to-end ML pipeline. Takes any text input and returns a category (World, Sports, Business, Sci_Tech) with a confidence score. Trained on 120,000 AG News articles, deployed as a live REST API on GCP Cloud Run and Kubernetes.

Live API: https://content-enrichment-177416861217.us-central1.run.app  
Swagger UI: https://content-enrichment-177416861217.us-central1.run.app/docs  
Experiment Tracking: https://dagshub.com/vamsimessi10/content-enrichment.mlflow  
GitHub: https://github.com/vamsimessi10/content-enrichment  

---

## Versions

- v1: Classical NLP — TF-IDF + Logistic Regression, 92.21% accuracy (120k training rows)
- v2 (current): Semantic NLP — HuggingFace all-MiniLM-L6-v2 embeddings + LinearSVC, 88.8% accuracy (10k sample, CPU constrained — full dataset with GPU expected to exceed v1)
- v3 (planned): Automated MLOps — Kubeflow pipelines + LangChain agent
- v4 (planned): Big data — PySpark + GCP Dataproc

> V2 note: trained on 10k sample due to Colab CPU constraints. On full 120k rows with GPU, HuggingFace embeddings are expected to match or exceed v1 accuracy. In production this would use GPU clusters for embedding generation.

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

## Project Structure

```
content-enrichment/
|
|-- app.py                         FastAPI application
|                                  loads model from GCS on startup (once, kept in memory)
|                                  3 endpoints: /health, /model/info, /predict
|                                  edge case handling: text < 10 chars returns 422
|
|-- Dockerfile                     container blueprint
|                                  FROM python:3.11-slim
|                                  copies requirements.txt first (layer caching)
|                                  installs dependencies
|                                  copies app.py
|                                  exposes port 8000
|
|-- requirements.txt               pinned dependencies
|                                  fastapi==0.104.1
|                                  uvicorn==0.24.0
|                                  scikit-learn==1.6.1  (must match training version)
|                                  google-cloud-storage==2.13.0
|                                  pydantic==2.5.2
|
|-- .dockerignore                  files excluded from Docker image
|                                  credentials.json, *.pkl, .env, __pycache__
|
|-- .gitignore                     files excluded from GitHub
|                                  *.pkl, *.csv, credentials, logs, __pycache__
|                                  github-actions-key.json  (never commit credentials)
|
|-- tests/
|   |-- __init__.py
|   |-- test_api.py                15 automated tests
|                                  SDE layer: status codes, response structure, edge cases
|                                  MLE layer: confidence range, probability sum, category validity
|                                  model layer: known predictions, accuracy threshold
|                                  mocks GCS on startup so tests run without cloud credentials
|
|-- kubernetes/
|   |-- api-deployment.yaml        Kubernetes deployment definition
|                                  replicas: 2 (high availability)
|                                  image: Artifact Registry path
|                                  resources: 512Mi-1Gi RAM, 250m-500m CPU
|                                  readinessProbe: GET /health every 10s
|                                  livenessProbe: GET /health every 20s (auto-restart on failure)
|   |
|   |-- api-service.yaml           Kubernetes service definition
|                                  type: LoadBalancer (provisions public IP)
|                                  port 80 (public) -> port 8000 (container)
|                                  selector matches app: content-enrichment label
|   |
|   |-- runtime.yaml               Kubernetes ConfigMap
|                                  environment variables for the cluster
|                                  GOOGLE_CLOUD_PROJECT, MODEL_BUCKET, MODEL_PATH
|
|-- .github/
|   |-- workflows/
|       |-- deploy.yml             GitHub Actions CI/CD pipeline
|                                  triggers on every push to main
|                                  job 1 (test): runs pytest 15 tests
|                                    if any fail: pipeline stops, no deploy
|                                    if all pass: continues to deploy
|                                  job 2 (deploy): only runs if tests pass
|                                    docker build on GitHub VM
|                                    docker push to Artifact Registry
|                                    gcloud run deploy to Cloud Run
```

---

## Architecture

```
Data Layer
----------
AG News (HuggingFace)
  120,000 train + 7,600 test articles
  4 categories: World, Sports, Business, Sci_Tech
        |
        v
Google Colab (training workbench)
  data cleaning: lowercase, remove special chars, drop short texts
        |
        v
BigQuery (permanent data storage)
  content-enrichment-488220.content_enrichment.ag_news_raw
        |
        v
GCS (backup + model storage)
  gs://content-enrichment-488220/data/ag_news_clean.csv
  gs://content-enrichment-488220/models/best_model.pkl


Training Layer
--------------
MLflow + Dagshub (experiment tracking)
  5 models trained and compared
  metrics, parameters, artifacts logged automatically
        |
        v
Best model selected (logistic_regression_c1, 92.21% accuracy)
        |
        v
Model saved to GCS as best_model.pkl


Serving Layer
-------------
app.py (FastAPI)
  loads model from GCS once at startup
  serves predictions at /predict
        |
        v
Dockerfile
  packages app.py + all dependencies into a container
        |
        v
Artifact Registry
  us-central1-docker.pkg.dev/content-enrichment-488220/content-enrichment/content-enrichment:latest
  permanent storage for Docker images
        |
        v
        |-----> Cloud Run (primary deployment)
        |       https://content-enrichment-177416861217.us-central1.run.app
        |       Google manages servers, scaling, HTTPS
        |       scales to zero when idle
        |
        |-----> Kubernetes / GKE (production deployment)
                2 nodes (virtual machines) in us-central1-a
                2 pods (app instances) distributed across nodes
                LoadBalancer service with public IP
                high availability: if one node fails, other serves traffic
                recreate with: gcloud container clusters create ...


CI/CD Layer
-----------
GitHub (source of truth for all code)
  https://github.com/vamsimessi10/content-enrichment
        |
        v
GitHub Actions (on every push to main)
  job 1: pytest (15 tests must pass)
  job 2: docker build + push + cloud run deploy
        |
        v
Live API updated in 3-4 minutes from push
```

---

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | /health | Returns service status and version |
| GET | /model/info | Returns model name, accuracy, F1, categories |
| POST | /predict | Classifies text, returns category and confidence |

---

## Model

Five models were trained and compared:

| Model | Accuracy | F1 (weighted) |
|-------|----------|---------------|
| Logistic Regression C=1.0 | 92.21% | 92.19% |
| Logistic Regression C=10.0 | 91.64% | 91.63% |
| Naive Bayes | 90.78% | 90.74% |
| Logistic Regression C=0.1 | 90.50% | 90.46% |
| Random Forest 100 | 89.99% | 89.94% |

Logistic Regression C=1.0 selected as best model.

Per-category F1 scores:

| Category | F1 Score |
|----------|----------|
| Sports | 0.97 |
| World | 0.93 |
| Sci_Tech | 0.90 |
| Business | 0.89 |

Pipeline: TF-IDF vectorizer (50,000 features, bigrams, English stop words removed) followed by Logistic Regression with C=1.0 and max_iter=1000.

All experiment runs tracked at: https://dagshub.com/vamsimessi10/content-enrichment.mlflow

---

## Tests

15 automated tests split into two layers:

SDE layer (11 tests) -- does the API work correctly:
- /health returns 200 and status ok
- /model/info returns 200 with accuracy and 4 categories
- /predict returns 200 with category, confidence, all_scores
- short text (less than 10 chars) returns 422
- empty text returns 422

MLE layer (4 tests) -- is the model trustworthy:
- confidence is between 0.0 and 1.0
- all 4 scores sum to 1.0 (valid probability distribution)
- returned category is one of the 4 valid values
- confidence is not suspiciously low on clear examples

Run locally:

```bash
python3 -m pytest tests/test_api.py -v
```

Tests run automatically in GitHub Actions before every deploy. If any test fails, deployment stops.

---

## Kubernetes Deployment

The app can be deployed to a GKE cluster using the manifests in kubernetes/.

```
Kubernetes concepts used:
  Node          virtual machine (server) in the cluster
  Pod           your Docker container running inside a node
  Deployment    manages pods: how many, which image, resource limits
  Service       exposes pods to the internet via a LoadBalancer
  ConfigMap     stores environment variables (runtime.yaml)
  LoadBalancer  public IP that routes traffic across pods
```

Create cluster:

```bash
gcloud container clusters create content-enrichment-cluster \
  --zone us-central1-a \
  --num-nodes 2 \
  --machine-type e2-medium \
  --enable-autoscaling \
  --min-nodes 1 \
  --max-nodes 3
```

Connect and deploy:

```bash
gcloud container clusters get-credentials content-enrichment-cluster \
  --zone us-central1-a

kubectl apply -f kubernetes/api-deployment.yaml
kubectl apply -f kubernetes/api-service.yaml
kubectl apply -f kubernetes/runtime.yaml
```

Check status:

```bash
kubectl get pods
kubectl get service content-enrichment-service
```

Delete cluster when done (avoid unnecessary charges):

```bash
gcloud container clusters delete content-enrichment-cluster \
  --zone us-central1-a
```

---

## Stack

- Python 3.11
- FastAPI + uvicorn
- scikit-learn 1.6.1
- Google Cloud Platform: BigQuery, GCS, Cloud Run, GKE, Artifact Registry, Cloud Build
- Docker
- Kubernetes
- GitHub Actions (CI/CD)
- MLflow + Dagshub (experiment tracking)
- pytest + httpx (testing)

---

## GCP Resources

| Resource | Location |
|----------|----------|
| GCP Project | content-enrichment-488220 |
| BigQuery dataset | content_enrichment |
| BigQuery table | ag_news_raw |
| GCS bucket | gs://content-enrichment-488220 |
| Model file | gs://content-enrichment-488220/models/best_model.pkl |
| Data backup | gs://content-enrichment-488220/data/ag_news_clean.csv |
| Artifact Registry | us-central1-docker.pkg.dev/content-enrichment-488220/content-enrichment |
| Cloud Run region | us-central1 |
| GKE zone | us-central1-a |

---

## CI/CD Pipeline

```
developer pushes to main branch
        |
        v
GitHub Actions triggered automatically
        |
        v
job 1: test
  install dependencies
  run pytest (15 tests)
  if any test fails: stop here, live API untouched
        |
        v (only if all tests pass)
job 2: deploy
  authenticate to GCP using GitHub Secret (GCP_SA_KEY)
  docker build on GitHub Actions runner
  docker push to Artifact Registry
  gcloud run deploy to Cloud Run
        |
        v
live API updated (3-4 minutes total from push)
```

---

## Local Development

Clone the repo:

```bash
git clone https://github.com/vamsimessi10/content-enrichment.git
cd content-enrichment
```

Set GCP credentials:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=path/to/your/credentials.json
export GOOGLE_CLOUD_PROJECT=content-enrichment-488220
```

Build and run with Docker:

```bash
docker build -t content-enrichment .
docker run -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT=content-enrichment-488220 \
  content-enrichment
```

Test the API:

```bash
curl http://127.0.0.1:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "SpaceX launches Starship on fourth test flight"}'
```

Run tests:

```bash
pip install pytest httpx fastapi uvicorn scikit-learn google-cloud-storage
python3 -m pytest tests/test_api.py -v
```

---

## What is next

- Cloud Logging dashboard and alerting (monitoring-info.yaml)
- v2: replace TF-IDF with HuggingFace transformer embeddings, expected accuracy improvement to 94-96%
- v3: Kubeflow pipeline for scheduled retraining on new data
- v3: LangChain agent layer using the classifier as a tool
