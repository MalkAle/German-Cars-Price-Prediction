FROM python:3.9-slim

RUN --mount=type=secret,id=api_key \
    --mount=type=secret,id=search_engine_id \
    echo "api_key=$(cat /run/secrets/api_key)" >> .env \
    && echo "search_engine_id=$(cat /run/secrets/search_engine_id)" >> .env

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

#Remote repo
#Clone complete repo
#RUN git clone https://github.com/MalkAle/German-Cars-Price-Prediction.git . 
#Clone fist level files and the app folder only   
RUN git clone --no-checkout https://github.com/MalkAle/German-Cars-Price-Prediction.git \
    && cd German-Cars-Price-Prediction \
    && git sparse-checkout init --cone \
    && git sparse-checkout set app \
    && git checkout @ 

#Local machine
#COPY . .

WORKDIR /German-Cars-Price-Prediction

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

WORKDIR /German-Cars-Price-Prediction/app

CMD ["streamlit", "run", "ger_cars_app.py", "--server.port=8501", "--server.address=0.0.0.0"]