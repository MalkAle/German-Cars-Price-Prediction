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

# Copy app and Streamlit config
COPY app ./app
COPY .streamlit ./.streamlit

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Config is fully driven by .streamlit/config.toml; no CLI overrides that
# would shadow the TOML booleans (CLI flag "false" is a truthy string in Python)
CMD ["streamlit", "run", "app/ger_cars_app.py"]
