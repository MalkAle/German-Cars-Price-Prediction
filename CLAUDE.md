# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

**Locally (without Docker):**
```bash
streamlit run app/ger_cars_app.py
```

**With Docker (requires BuildKit secrets for Google API credentials):**
```bash
docker build \
  --secret id=api_key,env=API_KEY \
  --secret id=search_engine_id,env=SEARCH_ENGINE_ID \
  -t german_cars_app .
docker run -p 8501:8501 german_cars_app
```

The app runs on port 8501.

## Regenerating the ML Model

The trained model (`eda/complete_model.joblib`) is built by running:
```bash
cd eda
python ger_cars_model.py
```

This reads `eda/car_data_ML.csv`, trains a KNN regressor per car model, and saves `complete_model.joblib`.

## Architecture

The project has two distinct phases:

**Training phase (`eda/`):**
- `ger_cars_model.py` — reads `car_data_ML.csv`, fits a separate `KNeighborsRegressor` pipeline (MinMaxScaler + OneHotEncoder + KNN) for each car model, filters out models with R² < 0.4, and serializes everything into `complete_model.joblib` via joblib.
- `complete_model.joblib` — a dict with structure: `{'models': {model_name: {'ml_model': pipeline, 'r2': float, 'mape': float, 'model_data': DataFrame}}}`.

**App phase (`app/`):**
- `ger_cars_app.py` — single-file Streamlit app. Loads `complete_model.joblib` at startup (path resolved relative to `__file__` via `pathlib`), renders a sidebar for user input, runs prediction, fetches Google Images via Custom Search API, and plots a 3D scatter + histogram using Plotly.
- Google Image Search requires `API_KEY` and `SEARCH_ENGINE_ID` environment variables (loaded via `python-dotenv` from `.env`).

## Deployment

The app is deployed on AWS ECS Fargate (region `eu-central-1`) as a container image hosted in Amazon ECR (`german_cars_app`). The CI/CD workflow (in `.github/workflows/backup/`) triggers on push to `main`, authenticates via OIDC to assume an IAM role, builds the Docker image with API secrets injected via BuildKit, and pushes to ECR.

The ECS task definition is at `.github/workflows/task-definition.json` (1 vCPU, 3 GB RAM).

## Key Constraints

- Python version in container: 3.9. Pin dependency versions to match `requirements.txt` exactly.
- The `.env` file containing Google API credentials is written into the container at build time via Docker BuildKit secrets — it is not committed to the repo.
