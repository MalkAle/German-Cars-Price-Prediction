# syntax=docker/dockerfile:1.4
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=api_key \ 
    --mount=type=secret,id=search_engine_id \ 
    echo "API_KEY=$(cat /run/secrets/api_key)" >> .env && \ 
    echo "SEARCH_ENGINE_ID=$(cat /run/secrets/search_engine_id)" >> .env

# Project root inside the container
WORKDIR /german_cars

# Install dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy app
COPY app ./app

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit from project root; app handles model path via Path(__file__)
CMD ["streamlit", "run", "app/ger_cars_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
