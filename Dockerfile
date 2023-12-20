FROM python:3.9-slim-buster

EXPOSE 8501

RUN apt-get update && apt-get install -y 

WORKDIR /app

COPY . /app

RUN pip3 install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "ger_cars_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
