FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=secret,id=api_key \
    --mount=type=secret,id=search_engine_id \
    echo "API_KEY=$(cat /run/secrets/api_key)" >> .env && \
    echo "SEARCH_ENGINE_ID=$(cat /run/secrets/search_engine_id)" >> .env

#docker build --secret id=api_key,env=API_KEY --secret id=search_engine_id,env=SEARCH_ENGINE_ID --no-cache --progress=plain -t german-cars-app .

COPY Dockerfile requirements.txt /app /app/
COPY requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR /app

CMD ["streamlit", "run", "ger_cars_app.py", "--server.port=8501", "--server.address=0.0.0.0"]