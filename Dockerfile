# syntax=docker/dockerfile:1.4
FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
  && rm -rf /var/lib/apt/lists/*

# Project root inside the container
WORKDIR /german_cars

# Install dependencies first
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy folders
COPY app ./app
COPY eda ./eda

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Run Streamlit app from the project root (no need for app/ CWD anymore)
CMD ["streamlit", "run", "app/ger_cars_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
