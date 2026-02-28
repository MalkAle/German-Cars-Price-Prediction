import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from pathlib import Path
import plotly.express as px
from sigfig import round as rd_sigfig
import requests
from datetime import timedelta, date
from dateutil.relativedelta import relativedelta


# Directory containing this file: /german_cars/app
CURRENT_DIR = Path(__file__).resolve().parent
# Project root: /german_cars
PROJECT_ROOT = CURRENT_DIR.parent
# Path to model file: /german_cars/eda/complete_model.joblib
MODEL_PATH = PROJECT_ROOT / "eda" / "complete_model.joblib"

## Functions
@st.cache_data
def _load_data():
   # This function loads the data for all car models
   try:
      loaded_data_ = load(MODEL_PATH)
      return loaded_data_
   except (ImportError, FileNotFoundError):
      st.error('Model load error', icon="🚨")
      
def _user_input_options(loaded_data_,model_):
   # This function creates select options for user input
   user_input_options_ = {}
   model_data_ = _model_data(model_,loaded_data_)
   user_input_options_['model'] = loaded_data_['models']
   user_input_options_['powers'] = np.sort(model_data_['power_ps'].unique())
   user_input_options_['fuel_types'] = np.sort(model_data_['fuel_type'].unique())
   user_input_options_['transmission_types'] = np.sort(model_data_['transmission_type'].unique())
   max_mileage_ = rd_sigfig(max(model_data_['mileage_in_km']),sigfigs=1) 
   min_mileage_ = rd_sigfig(min(model_data_['mileage_in_km']),sigfigs=1) if min(model_data_['mileage_in_km'])>1000 else 0 
   mileage_intervals_1_ = np.arange(min_mileage_,10**4,10**3)
   mileage_intervals_2_ = np.arange(10**4,10**5,10**4)
   mileage_intervals_3_ = np.arange(10**5,max_mileage_+2*10**4,2*10**4)
   user_input_options_['mileage_intervals'] = np.concatenate([mileage_intervals_1_,
                                       mileage_intervals_2_,
                                       mileage_intervals_3_])
   return user_input_options_

def _model_data(model_,loaded_data_):
   # This function fishes the model-specific data for from the complete loaded data and sets datatypes
   model_data_ = loaded_data_['models'][model_]['model_data']
   return model_data_.astype({'price_in_euro': 'int',
                              'power_ps':'int',
                              'mileage_in_km':'int'})

def _write_sidebar(loaded_data_,end_date_):
   # This function writes the sidebar and gets user inputs
   with st.sidebar:
      st.header('Car Data')
      st.markdown('Please provide input variables for the price prediction below')
      user_input_ = {}
      models_ = loaded_data_['models'] #creates list of all car models in the loaded data
      user_input_['model'] = st.selectbox('Model',models_) #saves car model chosen by user
      model_data_ = _model_data(user_input_['model'],loaded_data_) #picks the data for the car model choses by user
      user_input_options_ = _user_input_options(loaded_data_,user_input_['model']) #creates input options for user based in the chosen car model
      user_input_['power'] = st.selectbox('Power HP',user_input_options_['powers'])
      user_input_['fuel_type'] = st.selectbox('Fuel Type',user_input_options_['fuel_types'])
      user_input_['transmission_type'] = st.selectbox('Transmission',user_input_options_['transmission_types'])
      car_age_max_ = max(model_data_['car_age'])
      timedelta_max_ = timedelta(days=car_age_max_*365)
      start_date_ = end_date_ - timedelta_max_
      user_input_['registration_date'] = st.date_input('Registration Date',
                                          value=end_date_,
                                          min_value=start_date_,
                                          max_value=end_date_)
      user_input_['mileage'] = st.selectbox('Mileage in km',user_input_options_['mileage_intervals'])
      return (user_input_,model_data_)

def _ml_model(model_,loaded_data_):
   # This function retrieves the ml model and the model metrics from loaded data
   ML_model_ = loaded_data_['models'][model_]['ml_model']
   r2_score_ = np.round(loaded_data_['models'][model_]['r2'],decimals=2)
   mape_ = np.round(loaded_data_['models'][model_]['mape']*100,decimals=2)
   return (ML_model_,r2_score_,mape_)  

def _car_age(end_date_,registration_date_):
   # This function calculates the car age from registration date
   delta_ = relativedelta(end_date_, registration_date_)
   car_age_ = (delta_.years*12 + delta_.months)/12
   return car_age_

def _datapoint(user_input_,model_data_,end_date_):
   # This fuction creates a single datapoint from user input
   car_age_ = _car_age(end_date_,user_input_['registration_date'])
   X_ = pd.DataFrame([{
      'power_ps':          user_input_['power'],
      'transmission_type': user_input_['transmission_type'],
      'fuel_type':         user_input_['fuel_type'],
      'mileage_in_km':     user_input_['mileage'],
      'car_age':           car_age_,
   }])
   return X_

def _write_prediction(prediction_,r2_score_,model_data_,mape_):
   # This fuction writes the prediction results
   if r2_score_ < 0.65:
      st.error('Prediction calculated! Model quality is poor.', icon="🚩")
   elif r2_score_ >=0.65 and r2_score_ < 0.85:
      st.warning('Prediction calculated! Model quality is ok.',icon="⚠️")
   else:
      st.success('Prediction calculated! Model quality is good.',icon="✅")
   with st.container(border=True):
      st.write('## Predicted price:', prediction_, '€')
      st.write('The prediction is calculated using K Nearest Neighbor Regression')
      st.write('R2 score for this prediction is', r2_score_, ', Mean Absolute Percentage Error for this prediction is', mape_, '%')
      st.write('ML model is based on', len(model_data_) ,'samples.')

@st.cache_data
def _search_images(query_,num_images_):
   # This function searches for the model's image using google custom search engine
   #Loading secrets from environment
   load_dotenv()
   api_key_ = os.getenv('API_KEY')
   search_engine_id_ = os.getenv('SEARCH_ENGINE_ID')
   search_engine_url_ = 'https://www.googleapis.com/customsearch/v1'
   params_ = {'q': query_,
            'key': api_key_,
            'cx': search_engine_id_,
            'searchType': 'image',
            'imgType':'photo',
            'num': 10,
            } 
   images = []
   domains = []
   search_response_ = requests.get(search_engine_url_,params=params_)
   if search_response_.status_code == 200:
      search_results_ = search_response_.json()['items']
      # This section prevents duplicates by allowing to download only one image form every domain
      for item in search_results_:
         if item['link'].split('/')[2] not in domains:
            images.append(item['link'])
            domains.append(item['link'].split('/')[2])
            if len(images)==num_images_:
               break
      return images
   else:
      st.write("Error",search_response_.status_code)
      

def _write_model_images(images_,model_):
   # This function writes a row of model images in the page
   st.write('## Google Images for', model_)
   try:
      col_names_images_ = [i for i in range(len(images_))]
      cols_images = st.columns(len(images_))
      for col in col_names_images_:
         cols_images[col].image(images_[col])
   except TypeError:
      None

def _write_3d_scatter(model_data_):
   # This function plots a 3D scatterplot for all the data of the model chosen by user
   with st.container(border=True):
      st.write('## Visual Representation of Data for this Model')
      labels_ = {'mileage_in_km':'Mileage in km',
               'price_in_euro':'Price in €',
               'car_age':'Car Age',
               'power_ps':'Power in hp',
               'fuel_type':'Fuel Type',
               'transmission_type':'Transmission Type',
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
      st.plotly_chart(fig,use_container_width=True)

def _write_histogram(model_data_):
   # This function plots a histogram of prices for the car model chosen by user
   with st.container(border=True):
      st.write('## Distribution of Prices for this Model')
      labels_={'price_in_euro': "Price in €", "count": "Counts"}
      fig_1 = px.histogram(model_data_, 
                           x='price_in_euro',
                           #nbins=len(model_data) if len(model_data)<100 else 100,
                           labels=labels_)

      st.plotly_chart(fig_1, use_container_width=True)

if __name__ == "__main__":
   # Page intro
   st.set_page_config(page_title='Car Prices Prediction App for Germany 2023',
                     layout='wide',
                     initial_sidebar_state="expanded",)
   st.title('Car Prices Prediction App for Germany 2023')
   #st.write('Version 1.1.0')
   st.write('Based on the dataset from Kaggle.com (https://www.kaggle.com/datasets/wspirat/germany-used-cars-dataset-2023/)')
   st.write('See the Github Repository: https://github.com/MalkAle/German-Cars-Price-Prediction')

   # Some variable defintions 
   end_date = date(day=30,month=6,year=2023)
   num_images = 5

   ## Function calls
   # Load data
   loaded_data = _load_data()

   # Crate sidebar with user input
   (user_input,model_data) = _write_sidebar(loaded_data,end_date)
   # Get the ml model from the loaded data for the car model selected by user
   (ML_model,r2_score,mape) = _ml_model(user_input['model'],loaded_data)
   # Creates a single datapoint for prediction from user input
   X = _datapoint(user_input,model_data,end_date)      
   # Calculates price prediction from loaded ml model for the single datapoint
   prediction = int(sum(ML_model.predict(X)))
   # Gets urls of the photos from Google search for the car model selected by user
   images = _search_images(user_input['model'],num_images)
   # Displays images from image urls
   _write_model_images(images,user_input['model'])
   # Writes calculated predictions for single datapoint
   _write_prediction(prediction,r2_score,model_data,mape)
   # Plot a histogram with price distibution for the car model selected by user
   _write_histogram(model_data)
   # Plots a 3D scatteplot for all the data for the car model selected by user
   _write_3d_scatter(model_data)
