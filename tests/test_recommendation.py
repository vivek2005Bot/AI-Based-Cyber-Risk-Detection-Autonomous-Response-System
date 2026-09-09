"""Tests for recommendation engines."""

import pandas as pd
import pytest

from src.recommendation.collaborative import CollaborativeRecommender
from src.recommendation.content_based import ContentBasedRecommender
from src.recommendation.hybrid import HybridRecommender


def test_content_based_recommender():
    """Test content based recommendation."""
    rec = ContentBasedRecommender()
    event = {
        "event_type": "auth",
        "risk_level": "high",
        "attempts": 20,
        "failed_attempts_24h": 10,
    }
    recommendations = rec.recommend(event, top_k=3)
    assert len(recommendations) > 0
    assert "action" in recommendations[0]
    assert "score" in recommendations[0]
    assert recommendations[0]["score"] > 0


def test_collaborative_recommender():
    """Test collaborative filtering recommendation."""
    rec = CollaborativeRecommender(config={"n_factors": 5, "n_epochs": 5})
    interactions = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2", "u3"],
        "action": ["block_ip", "alert", "block_ip", "quarantine_host", "alert"],
        "rating": [5.0, 3.0, 4.0, 5.0, 4.0],
    })
    rec.fit(interactions)
    recs = rec.recommend(user_id="u1", top_k=2)
    assert len(recs) > 0
    assert "action" in recs[0]
    assert "score" in recs[0]


def test_hybrid_recommender():
    """Test hybrid recommendation combining content and collaborative signals."""
    hybrid = HybridRecommender()
    event = {
        "event_type": "auth",
        "risk_level": "critical",
        "attempts": 50,
    }
    recs = hybrid.recommend(event, user_id="user_001", top_k=3)
    assert len(recs) > 0
    assert "action" in recs[0]
    assert "score" in recs[0]
