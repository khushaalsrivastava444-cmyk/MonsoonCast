"""Integration tests for FastAPI endpoints."""

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_api_health():
    """Verify /health endpoint returns HTTP 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "MonsoonCast Research API"
    assert data["release_date"] == "2026-02-23"


def test_api_metadata():
    """Verify /metadata endpoint returns domain coordinates, resolution, and data provenance."""
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "domain" in data
    assert data["domain"]["name"] == "Central India Core Monsoon Zone"
    assert "resolution_deg" in data["domain"]
    assert "provenance" in data
    assert "is_synthetic" in data["provenance"]
    assert "disclaimer" in data["provenance"]
    assert "supported_models" in data
    assert "ensemble" in data["supported_models"]


def test_api_rejects_missing_required_features():
    """Verify API returns HTTP 422 when required meteorological predictors are missing."""
    invalid_payload = {
        "samples": [
            {
                "lat": 21.0,
                "lon": 78.0,
                "doy": 200,
                # Missing rain_current, rain_lag1, etc.
            }
        ],
        "model_name": "ensemble",
    }
    response = client.post("/predict/event-probability", json=invalid_payload)
    assert response.status_code == 422


def test_api_experiments():
    """Verify /experiments endpoint returns experiment list."""
    response = client.get("/experiments")
    assert response.status_code == 200
    assert "experiments" in response.json()


def test_api_predict_event_probability():
    """Verify /predict/event-probability returns calibrated probability and uncertainty."""
    sample_payload = {
        "samples": [
            {
                "lat": 21.0,
                "lon": 78.0,
                "doy": 200,
                "rain_current": 35.0,
                "rain_lag1": 20.0,
                "rain_lag2": 15.0,
                "rain_lag3": 10.0,
                "rain_lag7": 5.0,
                "rain_roll_sum3": 70.0,
                "rain_roll_sum7": 110.0,
                "rain_roll_max7": 35.0,
                "rain_roll_mean7": 15.7,
                "rain_spatial_mean3x3": 25.0,
                "rain_spatial_max3x3": 45.0,
                "climatology_mean": 12.0,
            }
        ],
        "model_name": "ensemble",
    }
    response = client.post("/predict/event-probability", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "calibrated_probability" in data
    assert len(data["calibrated_probability"]) == 1
    assert 0.0 <= data["calibrated_probability"][0] <= 1.0
    assert "epistemic_spread" in data
    assert "disagreement_entropy" in data


def test_api_predict_quantiles():
    """Verify /predict/rainfall-quantiles returns conditional quantiles."""
    sample_payload = {
        "samples": [
            {
                "lat": 21.0,
                "lon": 78.0,
                "doy": 200,
                "rain_current": 35.0,
                "rain_lag1": 20.0,
                "rain_lag2": 15.0,
                "rain_lag3": 10.0,
                "rain_lag7": 5.0,
                "rain_roll_sum3": 70.0,
                "rain_roll_sum7": 110.0,
                "rain_roll_max7": 35.0,
                "rain_roll_mean7": 15.7,
                "rain_spatial_mean3x3": 25.0,
                "rain_spatial_max3x3": 45.0,
                "climatology_mean": 12.0,
            }
        ]
    }
    response = client.post("/predict/rainfall-quantiles", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    assert "quantiles" in data
    assert "q_10" in data["quantiles"]
    assert "q_50" in data["quantiles"]
    assert "q_90" in data["quantiles"]
