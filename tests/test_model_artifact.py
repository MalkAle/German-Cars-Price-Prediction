"""
Structural tests for the trained model artifact (eda/complete_model.joblib).

These tests are skipped automatically when the file does not exist (it is
gitignored and must be generated locally by running ger_cars_model.py).
"""
from pathlib import Path

import pytest

JOBLIB_PATH = Path(__file__).resolve().parents[1] / "eda" / "complete_model.joblib"

pytestmark = pytest.mark.skipif(
    not JOBLIB_PATH.exists(),
    reason="complete_model.joblib not found — run `cd eda && python ger_cars_model.py` first",
)


@pytest.fixture(scope="module")
def complete_model():
    from joblib import load
    return load(JOBLIB_PATH)


class TestModelArtifactStructure:
    def test_top_level_key_exists(self, complete_model):
        assert "models" in complete_model

    def test_at_least_one_model_present(self, complete_model):
        assert len(complete_model["models"]) > 0

    def test_each_model_has_required_keys(self, complete_model):
        required = {"ml_model", "r2", "mape", "model_data"}
        for name, entry in complete_model["models"].items():
            assert required.issubset(entry.keys()), f"Missing keys for model '{name}'"

    def test_r2_above_threshold(self, complete_model):
        """All persisted models must have passed the R² ≥ 0.4 filter."""
        for name, entry in complete_model["models"].items():
            assert entry["r2"] >= 0.4, f"Model '{name}' has R²={entry['r2']:.3f} below threshold"

    def test_mape_is_positive(self, complete_model):
        for name, entry in complete_model["models"].items():
            assert entry["mape"] > 0, f"Model '{name}' has non-positive MAPE"

    def test_model_data_has_expected_columns(self, complete_model):
        expected_cols = {"price_in_euro", "power_ps", "mileage_in_km", "car_age",
                         "fuel_type", "transmission_type"}
        for name, entry in complete_model["models"].items():
            cols = set(entry["model_data"].columns)
            assert expected_cols.issubset(cols), f"Missing columns for model '{name}': {expected_cols - cols}"

    def test_model_data_non_empty(self, complete_model):
        for name, entry in complete_model["models"].items():
            assert len(entry["model_data"]) > 0, f"Empty model_data for '{name}'"

    def test_ml_model_can_predict(self, complete_model):
        """Smoke-test: each pipeline accepts a single row and returns a number."""
        import pandas as pd

        for name, entry in complete_model["models"].items():
            df = entry["model_data"]
            sample = df.drop(columns=["price_in_euro"]).iloc[[0]]
            prediction = entry["ml_model"].predict(sample)
            assert len(prediction) == 1
            assert prediction[0] > 0, f"Non-positive prediction for model '{name}'"
