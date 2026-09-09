"""Tests for the FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture
def client():
    return TestClient(app)


def test_api_health(client):
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_api_assess_risk(client):
    """Test assess risk endpoint."""
    payload = {
        "features": {
            "attempts": 5,
            "failed_attempts_24h": 3,
            "cpu_usage": 45.0,
            "network_connections": 10,
        },
        "source_type": "synthetic",
        "event_type": "auth",
        "cvss_score": 6.5,
        "asset_criticality": 70,
        "threat_score": 0.5,
        "incident_count": 1,
        "hours_since_event": 2.0,
    }
    response = client.post("/assess-risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "risk_score" in data
    assert "risk_level" in data
    assert 0.0 <= data["risk_score"] <= 1.0
    assert len(data["recommendations"]) > 0


def test_api_detect_anomalies(client):
    """Test anomaly detection endpoint with single event features."""
    features = {
        "cpu_usage": 85.5,
        "memory_usage": 90.0,
        "network_connections": 50,
    }
    response = client.post("/detect-anomalies", json=features)
    assert response.status_code == 200
    data = response.json()
    assert "anomalies" in data
    assert "isolation_forest" in data["anomalies"]


def test_api_recommend(client):
    """Test recommendation endpoint."""
    payload = {
        "features": {"attempts": 10},
        "source_type": "synthetic",
        "event_type": "auth",
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0


def test_api_models_and_rl_status(client):
    """Test model listing and RL agent status endpoints."""
    models_res = client.get("/models")
    assert models_res.status_code == 200
    models_data = models_res.json()
    assert "available_models" in models_data

    rl_res = client.get("/rl/status")
    assert rl_res.status_code == 200
    rl_data = rl_res.json()
    assert "trained" in rl_data
    assert "epsilon" in rl_data
