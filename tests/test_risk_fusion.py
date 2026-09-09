"""Tests for the risk fusion module."""

import numpy as np
import pytest

from src.risk_fusion.fusion import RiskFusionEngine


@pytest.fixture
def fusion_engine():
    return RiskFusionEngine()


def test_fuse_basic(fusion_engine):
    """Test basic risk fusion."""
    supervised = np.array([0.8])
    anomaly = np.array([0.6])
    contextual = np.array([0.4])

    risk = fusion_engine.fuse(supervised, anomaly, contextual)
    assert 0.0 <= risk[0] <= 1.0
    # Weighted average: 0.4*0.8 + 0.3*0.6 + 0.3*0.4 = 0.32 + 0.18 + 0.12 = 0.62
    assert abs(risk[0] - 0.62) < 1e-10


def test_fuse_clipping(fusion_engine):
    """Test that scores are clipped to [0, 1]."""
    supervised = np.array([1.5])
    anomaly = np.array([-0.5])
    contextual = np.array([2.0])

    risk = fusion_engine.fuse(supervised, anomaly, contextual)
    assert 0.0 <= risk[0] <= 1.0


def test_risk_levels(fusion_engine):
    """Test risk level classification."""
    scores = np.array([0.1, 0.4, 0.7, 0.85, 0.95])
    levels = fusion_engine.risk_level(scores)
    assert levels[0] == "minimal"
    assert levels[1] == "low"
    assert levels[2] == "medium"
    assert levels[3] == "high"
    assert levels[4] == "critical"


def test_explain_risk(fusion_engine):
    """Test risk explanation generation."""
    explanation = fusion_engine.explain_risk(0.7, 0.8, 0.6, 0.5)
    assert "risk_score" in explanation
    assert "risk_level" in explanation
    assert "contributions" in explanation
    assert "dominant_factor" in explanation
    assert "explanation" in explanation
