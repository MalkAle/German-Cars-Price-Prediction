"""
Shared test configuration and fixtures.

Streamlit is mocked at the sys.modules level before the app module is imported
so that @st.cache_data becomes a no-op identity decorator and st.* calls in
the module body don't require a running Streamlit server.
"""
import sys
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

# --- Mock streamlit before any app import ---
_st_mock = MagicMock()
_st_mock.cache_data = lambda func: func   # identity decorator
sys.modules["streamlit"] = _st_mock


# --- Shared fixtures ---

@pytest.fixture
def sample_model_data():
    """Minimal DataFrame that mirrors the columns present in a real model slice."""
    # Column order must match the positional assignment in _datapoint:
    # [power_ps, transmission_type, fuel_type, mileage_in_km, car_age]
    return pd.DataFrame({
        "price_in_euro":       [10000, 15000, 20000, 8000, 25000],
        "power_ps":            [100,   150,   200,   90,   250],
        "transmission_type":   ["Manual", "Automatic", "Manual", "Automatic", "Manual"],
        "fuel_type":           ["Gasoline", "Diesel", "Gasoline", "Electric", "Diesel"],
        "mileage_in_km":       [50000, 80000, 120000, 30000, 200000],
        "car_age":             [3.5,   5.0,   7.0,   2.0,   10.0],
    })


@pytest.fixture
def loaded_data(sample_model_data):
    """Mock of the complete_model.joblib dict structure."""
    ml_model_mock = MagicMock()
    ml_model_mock.predict.return_value = [12000]

    return {
        "models": {
            "VW Golf": {
                "ml_model": ml_model_mock,
                "r2":        0.754321,
                "mape":      0.123456,
                "model_data": sample_model_data,
            },
            "BMW 3er": {
                "ml_model": MagicMock(),
                "r2":        0.902,
                "mape":      0.089,
                "model_data": sample_model_data.copy(),
            },
        }
    }


@pytest.fixture
def end_date():
    return date(2023, 6, 30)


@pytest.fixture
def user_input(end_date):
    return {
        "model":             "VW Golf",
        "power":             150,
        "fuel_type":         "Diesel",
        "transmission_type": "Automatic",
        "registration_date": date(2018, 6, 30),
        "mileage":           80000,
    }
