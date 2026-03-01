import argparse
import io
import json
import re

import boto3
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.metrics import mean_absolute_percentage_error as mape_score
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def _safe_key(name):
    """Convert a car model name to a safe S3 key segment (e.g. 'BMW 3-Series' -> 'bmw_3_series')."""
    return re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_')


def _upload(s3, bucket, key, data_bytes):
    s3.put_object(Bucket=bucket, Key=key, Body=data_bytes)
    print(f'  Uploaded s3://{bucket}/{key}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train per-model KNN pipelines and upload artifacts to S3.')
    parser.add_argument('--bucket', required=True, help='S3 bucket name (from CloudFormation ModelArtifactsBucketName output)')
    args = parser.parse_args()

    s3 = boto3.client('s3', region_name='eu-central-1')

    # Read data
    car_data_ML = pd.read_csv('car_data_ML.csv')
    models = car_data_ML['model'].unique()

    index = {}
    good_models = []
    bad_models = []
    r2_min = 0.4

    for model in models:
        car_data_ML_slice = pd.DataFrame(car_data_ML[car_data_ML['model'] == model])
        car_data_ML_slice.drop(columns=['Unnamed: 0', 'brand', 'model'], inplace=True)

        y = car_data_ML_slice['price_in_euro']
        X = car_data_ML_slice.drop(columns=['price_in_euro'])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=333)

        preprocessor = make_column_transformer(
            (MinMaxScaler(), make_column_selector(dtype_exclude=object)),
            (OneHotEncoder(), make_column_selector(dtype_include=object))
        )
        ml_model = make_pipeline(preprocessor, KNeighborsRegressor())
        ml_model_fit = ml_model.fit(X_train, y_train)

        predictions_train = ml_model.predict(X_train)
        predictions_test = ml_model.predict(X_test)

        print(f'''{model}:
        Training set accuracy: {np.round(r2_score(y_train, predictions_train), 2)} based on {len(X_train)} samples
        Test set accuracy:     {np.round(r2_score(y_test, predictions_test), 2)} based on {len(X_test)} samples
        ---------------------------------------------------------------------------------''')

        if r2_score(y_test, predictions_test) >= r2_min:
            r2 = r2_score(y_test, predictions_test)
            mape = mape_score(y_test, predictions_test)
            key = _safe_key(model)
            good_models.append(model)

            # Upload sklearn pipeline as joblib
            buf = io.BytesIO()
            dump(ml_model_fit, buf)
            buf.seek(0)
            _upload(s3, args.bucket, f'models/{key}.joblib', buf.read())

            # Upload training data as parquet
            buf = io.BytesIO()
            car_data_ML_slice.to_parquet(buf, index=False)
            buf.seek(0)
            _upload(s3, args.bucket, f'data/{key}.parquet', buf.read())

            index[model] = {'key': key, 'r2': r2, 'mape': mape}
        else:
            bad_models.append(model)

    # Upload index (model list + scores, read at app startup)
    _upload(s3, args.bucket, 'index.json', json.dumps(index, indent=2).encode())

    print(f'\nCalculations finished')
    print(f'{len(good_models)} models uploaded, {len(bad_models)} did not make the cut.\n')
