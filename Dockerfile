FROM python:3.9-slim-buster

EXPOSE 8501

RUN apt-get update && apt-get install -y \
    pip3 install \
        numpy \
        pandas \
        scikit-learn \
        joblib \
        streamlit \
        plotly \
        sigfig

