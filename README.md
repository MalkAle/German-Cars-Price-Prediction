# German Cars Price Prediction App

A Streamlit web application that predicts used car prices in Germany using K-Nearest Neighbor regression, trained on a 2023 Kaggle dataset.

## Data Source

[Germany Used Cars Dataset 2023](https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023/) — available on Kaggle.

> **Note:** The deployed model includes Volkswagen models only to reduce deployment cost. The full dataset covers multiple German manufacturers.

## Project Structure

```
.
├── app/
│   └── ger_cars_app.py       # Streamlit application
├── eda/
│   ├── ger_cars_model.py     # Model training script
│   ├── car_data.csv          # Raw dataset (not committed — large file)
│   ├── car_data_ML.csv       # Preprocessed dataset (not committed — large file)
│   └── complete_model.joblib # Trained model artifact (not committed — large file)
├── Dockerfile
└── requirements.txt
```

## Regenerating the Model

The trained model file (`eda/complete_model.joblib`) is required to run the app but is not committed to the repository. To regenerate it:

```bash
cd eda
python ger_cars_model.py
```

This reads `eda/car_data_ML.csv`, fits a separate KNN regression pipeline (MinMaxScaler + OneHotEncoder + KNN) for each car model, and writes `complete_model.joblib`. Models with R² < 0.4 are filtered out.

## Running Locally

Make sure `eda/complete_model.joblib` exists before starting the app.

```bash
streamlit run app/ger_cars_app.py
```

The app runs on [http://localhost:8501](http://localhost:8501).

For Google Image Search to work, create a `.env` file in the project root:

```
API_KEY=your_google_api_key
SEARCH_ENGINE_ID=your_custom_search_engine_id
```

## Running with Docker

The Docker build uses BuildKit secrets to inject Google API credentials at build time without committing them to the image.

```bash
docker build \
  --secret id=api_key,env=API_KEY \
  --secret id=search_engine_id,env=SEARCH_ENGINE_ID \
  -t german_cars_app .

docker run -p 8501:8501 german_cars_app
```
