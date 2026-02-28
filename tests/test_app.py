"""
Unit tests for pure-logic functions in app/ger_cars_app.py.

Streamlit is mocked in conftest.py before this module is imported.
"""
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# conftest.py installs the streamlit mock before this import runs
from app.ger_cars_app import (
    _car_age,
    _datapoint,
    _ml_model,
    _model_data,
    _search_images,
    _user_input_options,
)


# ---------------------------------------------------------------------------
# _car_age
# ---------------------------------------------------------------------------

class TestCarAge:
    def test_exact_years(self):
        reg = date(2018, 6, 30)
        end = date(2023, 6, 30)
        assert _car_age(end, reg) == pytest.approx(5.0)

    def test_fractional_year(self):
        reg = date(2020, 1, 1)
        end = date(2023, 7, 1)   # 3 years 6 months = 3.5
        assert _car_age(end, reg) == pytest.approx(3.5)

    def test_zero_age(self):
        same = date(2023, 6, 30)
        assert _car_age(same, same) == pytest.approx(0.0)

    def test_one_month(self):
        reg = date(2023, 5, 30)
        end = date(2023, 6, 30)
        # 1 month → 1/12
        assert _car_age(end, reg) == pytest.approx(1 / 12)


# ---------------------------------------------------------------------------
# _model_data
# ---------------------------------------------------------------------------

class TestModelData:
    def test_returns_dataframe(self, loaded_data):
        result = _model_data("VW Golf", loaded_data)
        assert isinstance(result, pd.DataFrame)

    def test_correct_model_slice(self, loaded_data, sample_model_data):
        result = _model_data("VW Golf", loaded_data)
        assert len(result) == len(sample_model_data)

    def test_numeric_columns_cast_to_int(self, loaded_data):
        result = _model_data("VW Golf", loaded_data)
        assert result["price_in_euro"].dtype == int
        assert result["power_ps"].dtype == int
        assert result["mileage_in_km"].dtype == int


# ---------------------------------------------------------------------------
# _ml_model
# ---------------------------------------------------------------------------

class TestMlModel:
    def test_returns_three_values(self, loaded_data):
        result = _ml_model("VW Golf", loaded_data)
        assert len(result) == 3

    def test_r2_rounded_to_2_decimals(self, loaded_data):
        _, r2, _ = _ml_model("VW Golf", loaded_data)
        assert r2 == pytest.approx(0.75)

    def test_mape_rounded_as_percentage(self, loaded_data):
        # raw mape=0.123456 → multiplied by 100 and rounded → 12.35
        _, _, mape = _ml_model("VW Golf", loaded_data)
        assert mape == pytest.approx(12.35)

    def test_ml_model_object_returned(self, loaded_data):
        ml_model, _, _ = _ml_model("VW Golf", loaded_data)
        assert ml_model is loaded_data["models"]["VW Golf"]["ml_model"]


# ---------------------------------------------------------------------------
# _datapoint
# ---------------------------------------------------------------------------

class TestDatapoint:
    def test_returns_dataframe(self, user_input, loaded_data, end_date):
        model_data = _model_data("VW Golf", loaded_data)
        result = _datapoint(user_input, model_data, end_date)
        assert isinstance(result, pd.DataFrame)

    def test_single_row(self, user_input, loaded_data, end_date):
        model_data = _model_data("VW Golf", loaded_data)
        result = _datapoint(user_input, model_data, end_date)
        assert len(result) == 1

    def test_has_expected_columns(self, user_input, loaded_data, end_date):
        model_data = _model_data("VW Golf", loaded_data)
        result = _datapoint(user_input, model_data, end_date)
        expected_cols = {"power_ps", "transmission_type", "fuel_type", "mileage_in_km", "car_age"}
        assert expected_cols.issubset(set(result.columns))

    def test_values_match_user_input(self, user_input, loaded_data, end_date):
        model_data = _model_data("VW Golf", loaded_data)
        result = _datapoint(user_input, model_data, end_date)
        assert result["power_ps"].iloc[0] == user_input["power"]
        assert result["fuel_type"].iloc[0] == user_input["fuel_type"]
        assert result["transmission_type"].iloc[0] == user_input["transmission_type"]
        assert result["mileage_in_km"].iloc[0] == user_input["mileage"]

    def test_car_age_is_computed(self, user_input, loaded_data, end_date):
        model_data = _model_data("VW Golf", loaded_data)
        result = _datapoint(user_input, model_data, end_date)
        # registration 2018-06-30, end 2023-06-30 → exactly 5 years
        assert result["car_age"].iloc[0] == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _user_input_options
# ---------------------------------------------------------------------------

class TestUserInputOptions:
    def test_returns_expected_keys(self, loaded_data):
        result = _user_input_options(loaded_data, "VW Golf")
        assert {"model", "powers", "fuel_types", "transmission_types", "mileage_intervals"}.issubset(result)

    def test_powers_are_sorted(self, loaded_data):
        result = _user_input_options(loaded_data, "VW Golf")
        powers = result["powers"]
        assert list(powers) == sorted(powers)

    def test_fuel_types_are_sorted(self, loaded_data):
        result = _user_input_options(loaded_data, "VW Golf")
        fuel_types = result["fuel_types"]
        assert list(fuel_types) == sorted(fuel_types)

    def test_mileage_intervals_are_non_empty(self, loaded_data):
        result = _user_input_options(loaded_data, "VW Golf")
        assert len(result["mileage_intervals"]) > 0

    def test_mileage_intervals_are_monotonic(self, loaded_data):
        result = _user_input_options(loaded_data, "VW Golf")
        intervals = result["mileage_intervals"]
        assert all(intervals[i] <= intervals[i + 1] for i in range(len(intervals) - 1))


# ---------------------------------------------------------------------------
# _search_images
# ---------------------------------------------------------------------------

class TestSearchImages:
    def _make_search_response(self, links):
        """Build a mock response with the given image links."""
        items = [{"link": link} for link in links]
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"items": items}
        return mock_resp

    @patch("app.ger_cars_app.requests.get")
    def test_returns_requested_number_of_images(self, mock_get):
        links = [f"https://domain{i}.com/img.jpg" for i in range(10)]
        mock_get.return_value = self._make_search_response(links)
        result = _search_images("VW Golf", 3)
        assert len(result) == 3

    @patch("app.ger_cars_app.requests.get")
    def test_deduplicates_by_domain(self, mock_get):
        # Two images from the same domain → only the first should be kept
        links = [
            "https://same-domain.com/img1.jpg",
            "https://same-domain.com/img2.jpg",
            "https://other-domain.com/img.jpg",
        ]
        mock_get.return_value = self._make_search_response(links)
        result = _search_images("VW Golf", 5)
        domains = [url.split("/")[2] for url in result]
        assert len(domains) == len(set(domains))

    @patch("app.ger_cars_app.requests.get")
    def test_returns_none_on_http_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 403
        mock_get.return_value = mock_resp
        result = _search_images("VW Golf", 3)
        assert result is None
