# %%
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import mean_absolute_percentage_error as mape_score
from sklearn.metrics import r2_score
 
from joblib import dump

# %%

if __name__ == "__main__":
    # Read Data 
    car_data_ML = pd.read_csv('car_data_ML.csv')

    # %%
    # For testing only
    #car_data_ML = car_data_ML[car_data_ML['brand']=='Volkswagen']
    #car_data_ML = car_data_ML[car_data_ML['model']=='Fiat Scudo']
    #car_data_ML = car_data_ML[(car_data_ML['model']=='Alfa Romeo Mito')]

    # %%
    models = car_data_ML['model'].unique()

    # %%
    complete_model = {'models':{},'good_models':{}} 
    r2 = []
    mape = []
    good_models = []
    bad_models = []
    # %%
    for model in models:
        # Creating the slice from model sub-dataset
        car_data_ML_slice = pd.DataFrame(car_data_ML[car_data_ML['model']==model])

        # Dropping "Unnamed" and "brand columns
        car_data_ML_slice.drop(columns=['Unnamed: 0','brand','model'],inplace=True)

        # Separating dataset into train and test data
        y = car_data_ML_slice['price_in_euro']
        X = car_data_ML_slice.drop(columns=['price_in_euro'])

        # Train-Test-Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

        # Creating Pre-Processor
        preprocessor = make_column_transformer(
            (MinMaxScaler(), make_column_selector(dtype_exclude=object)),
            (OneHotEncoder(), make_column_selector(dtype_include=object))
        )

        # Defining ML Algorithm
        reg_alg = KNeighborsRegressor()#algorithm='brute',metric='manhattan', weights='uniform')

        # Creating the model
        ml_model = make_pipeline(preprocessor, reg_alg)

        # Fitting Model
        ml_model_fit = ml_model.fit(X_train, y_train)

        #Creating predictions for X_test
        predictions_train = ml_model.predict(X_train)
        predictions_test = ml_model.predict(X_test)

        #"""
        # Printing Score (R2) 
        print(f'''{model}:
        Training set accuracy: {np.round(r2_score(y_train, predictions_train), 2)} based on {len(X_train)} samples\n\
        Test set accuracy: {np.round(r2_score(y_test, predictions_test), 2)} based on {len(X_test)} samples.
        -------------------------------------------------------------------------------------''')
        #"""

        #Mininmum R2 score to be used in app
        r2_min = 0.4
        
        #print(r2_score(y_test, predictions))
        if r2_score(y_test, predictions_test) >= r2_min:
            # Regression score output
            r2 = r2_score(y_test, predictions_test)
            mape = mape_score(y_test, predictions_test)
            good_models.append(model)
            complete_model['models'][model] = {'ml_model': ml_model_fit,'r2': r2, 'mape': mape, 'model_data': car_data_ML_slice}

            
        else:
            bad_models.append(model)
   
    print('\n\nCalculations finished\n')
    print(f'{len(good_models)} models made it into the finale ML model, {len(bad_models)} did not make it.\n')
     # Saving Model
    dump(complete_model, 'complete_model.joblib') 
    print('Model saved\n\n')


