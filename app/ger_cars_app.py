import io
import json
import os
from datetime import date, timedelta

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from joblib import load
from sigfig import round as rd_sigfig


S3_BUCKET = os.getenv('S3_BUCKET')


## Functions

@st.cache_resource
def _s3():
    return boto3.client('s3', region_name='eu-central-1')


@st.cache_resource
def _load_index():
    # Fetches index.json: {model_name: {key, r2, mape}} — tiny, loaded once at startup
    response = _s3().get_object(Bucket=S3_BUCKET, Key='index.json')
    return json.loads(response['Body'].read())


@st.cache_resource
def _load_model_artifacts(key_):
    # Fetches the sklearn pipeline and training data for one model, cached per key
    s3 = _s3()
    resp = s3.get_object(Bucket=S3_BUCKET, Key=f'models/{key_}.joblib')
    pipeline_ = load(io.BytesIO(resp['Body'].read()))
    resp = s3.get_object(Bucket=S3_BUCKET, Key=f'data/{key_}.parquet')
    model_data_ = pd.read_parquet(io.BytesIO(resp['Body'].read()))
    model_data_ = model_data_.astype({'price_in_euro': 'int', 'power_ps': 'int', 'mileage_in_km': 'int'})
    return pipeline_, model_data_


def _user_input_options(model_data_):
    # Creates select options for user input based on the chosen model's data
    user_input_options_ = {}
    user_input_options_['powers'] = np.sort(model_data_['power_ps'].unique())
    user_input_options_['fuel_types'] = np.sort(model_data_['fuel_type'].unique())
    user_input_options_['transmission_types'] = np.sort(model_data_['transmission_type'].unique())
    max_mileage_ = rd_sigfig(max(model_data_['mileage_in_km']), sigfigs=1)
    min_mileage_ = rd_sigfig(min(model_data_['mileage_in_km']), sigfigs=1) if min(model_data_['mileage_in_km']) > 1000 else 0
    mileage_intervals_1_ = np.arange(min_mileage_, 10**4, 10**3)
    mileage_intervals_2_ = np.arange(10**4, 10**5, 10**4)
    mileage_intervals_3_ = np.arange(10**5, max_mileage_ + 2*10**4, 2*10**4)
    user_input_options_['mileage_intervals'] = np.concatenate([mileage_intervals_1_,
                                                               mileage_intervals_2_,
                                                               mileage_intervals_3_])
    return user_input_options_


def _write_sidebar(index_, end_date_):
    # Renders the sidebar and returns user inputs + the S3 key for the selected model
    with st.sidebar:
        st.header('Car Data')
        st.markdown('Please provide input variables for the price prediction below')
        user_input_ = {}
        user_input_['model'] = st.selectbox('Model', list(index_.keys()))
        key_ = index_[user_input_['model']]['key']
        _, model_data_ = _load_model_artifacts(key_)
        user_input_options_ = _user_input_options(model_data_)
        user_input_['power'] = st.selectbox('Power HP', user_input_options_['powers'])
        user_input_['fuel_type'] = st.selectbox('Fuel Type', user_input_options_['fuel_types'])
        user_input_['transmission_type'] = st.selectbox('Transmission', user_input_options_['transmission_types'])
        car_age_max_ = max(model_data_['car_age'])
        timedelta_max_ = timedelta(days=car_age_max_ * 365)
        start_date_ = end_date_ - timedelta_max_
        user_input_['registration_date'] = st.date_input('Registration Date',
                                                          value=end_date_,
                                                          min_value=start_date_,
                                                          max_value=end_date_)
        user_input_['mileage'] = st.selectbox('Mileage in km', user_input_options_['mileage_intervals'])
        return user_input_, key_


def _car_age(end_date_, registration_date_):
    # Calculates car age in years from registration date
    delta_ = relativedelta(end_date_, registration_date_)
    car_age_ = (delta_.years * 12 + delta_.months) / 12
    return car_age_


def _datapoint(user_input_, end_date_):
    # Creates a single-row DataFrame from user input for prediction
    car_age_ = _car_age(end_date_, user_input_['registration_date'])
    X_ = pd.DataFrame([{
        'power_ps':          user_input_['power'],
        'transmission_type': user_input_['transmission_type'],
        'fuel_type':         user_input_['fuel_type'],
        'mileage_in_km':     user_input_['mileage'],
        'car_age':           car_age_,
    }])
    return X_


def _write_prediction(prediction_, r2_score_, model_data_, mape_):
    # Displays the prediction result with model quality indicator
    if r2_score_ < 0.65:
        st.error('Prediction calculated! Model quality is poor.', icon="🚩")
    elif r2_score_ >= 0.65 and r2_score_ < 0.85:
        st.warning('Prediction calculated! Model quality is ok.', icon="⚠️")
    else:
        st.success('Prediction calculated! Model quality is good.', icon="✅")
    with st.container(border=True):
        st.write('## Predicted price:', prediction_, '€')
        st.write('The prediction is calculated using K Nearest Neighbor Regression')
        st.write('R2 score for this prediction is', r2_score_, ', Mean Absolute Percentage Error for this prediction is', mape_, '%')
        st.write('ML model is based on', len(model_data_), 'samples.')


@st.cache_data
def _search_images(query_, num_images_):
    # Searches for model images via Google Custom Search API
    load_dotenv()
    api_key_ = os.getenv('API_KEY')
    search_engine_id_ = os.getenv('SEARCH_ENGINE_ID')
    search_engine_url_ = 'https://www.googleapis.com/customsearch/v1'
    params_ = {'q': query_,
               'key': api_key_,
               'cx': search_engine_id_,
               'searchType': 'image',
               'imgType': 'photo',
               'num': 10,
               }
    images = []
    domains = []
    search_response_ = requests.get(search_engine_url_, params=params_)
    if search_response_.status_code == 200:
        search_results_ = search_response_.json()['items']
        for item in search_results_:
            if item['link'].split('/')[2] not in domains:
                images.append(item['link'])
                domains.append(item['link'].split('/')[2])
                if len(images) == num_images_:
                    break
        return images
    else:
        st.write("Error", search_response_.status_code)


def _write_model_images(images_, model_):
    # Displays a row of model images
    st.write('## Google Images for', model_)
    try:
        col_names_images_ = [i for i in range(len(images_))]
        cols_images = st.columns(len(images_))
        for col in col_names_images_:
            cols_images[col].image(images_[col])
    except TypeError:
        None


def _write_3d_scatter(model_data_):
    # Plots a 3D scatter of all data for the selected model
    with st.container(border=True):
        st.write('## Visual Representation of Data for this Model')
        labels_ = {'mileage_in_km': 'Mileage in km',
                   'price_in_euro': 'Price in €',
                   'car_age': 'Car Age',
                   'power_ps': 'Power in hp',
                   'fuel_type': 'Fuel Type',
                   'transmission_type': 'Transmission Type',
                   }
        fig = px.scatter_3d(model_data_,
                            x='mileage_in_km',
                            z='price_in_euro',
                            y='car_age',
                            size='power_ps',
                            color='fuel_type',
                            symbol='transmission_type',
                            labels=labels_,
                            color_discrete_sequence=px.colors.qualitative.G10,
                            )
        fig.update_layout(height=800)
        st.plotly_chart(fig, use_container_width=True)


def _write_histogram(model_data_):
    # Plots a histogram of prices for the selected model
    with st.container(border=True):
        st.write('## Distribution of Prices for this Model')
        labels_ = {'price_in_euro': "Price in €", "count": "Counts"}
        fig_1 = px.histogram(model_data_,
                             x='price_in_euro',
                             labels=labels_)
        st.plotly_chart(fig_1, use_container_width=True)


if __name__ == "__main__":
    st.set_page_config(page_title='Car Prices Prediction App for Germany 2023',
                       layout='wide',
                       initial_sidebar_state="expanded")
    st.title('Car Prices Prediction App for Germany 2023')
    st.write('Based on the dataset from Kaggle.com (https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023/)')
    st.write('See the Github Repository: https://github.com/MalkAle/German-Cars-Price-Prediction')

    end_date = date(day=30, month=6, year=2023)
    num_images = 5

    # Fetch model index (tiny JSON, loaded once and shared across all sessions)
    index = _load_index()

    # Sidebar renders controls; returns user selections + S3 key for chosen model
    (user_input, key) = _write_sidebar(index, end_date)

    # Load pipeline + data for the selected model (S3 fetch only on first selection of each model)
    (ML_model, model_data) = _load_model_artifacts(key)

    # Metrics come from the index — no extra S3 fetch needed
    r2_score = np.round(index[user_input['model']]['r2'], decimals=2)
    mape = np.round(index[user_input['model']]['mape'] * 100, decimals=2)

    X = _datapoint(user_input, end_date)
    prediction = int(sum(ML_model.predict(X)))

    images = _search_images(user_input['model'], num_images)
    _write_model_images(images, user_input['model'])
    _write_prediction(prediction, r2_score, model_data, mape)
    _write_histogram(model_data)
    _write_3d_scatter(model_data)
