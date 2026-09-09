"""Hybrid recommendation combining content-based and collaborative filtering."""

from typing import Any, Dict, List, Optional

import pandas as pd

from src.recommendation.collaborative import CollaborativeRecommender
from src.recommendation.content_based import ContentBasedRecommender
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class HybridRecommender:
    """Combines content-based and collaborative recommendations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("recommendation.hybrid")
        self.content_weight = self.config.get("content_weight", 0.6)
        self.collaborative_weight = self.config.get("collaborative_weight", 0.4)
        self.content_recommender = ContentBasedRecommender()
        self.collaborative_recommender = CollaborativeRecommender()

    def fit_collaborative(self, interactions: pd.DataFrame) -> None:
        """Fit the collaborative filtering component.

        Args:
            interactions: DataFrame with 'user_id', 'action', and 'rating' columns.
        """
        self.collaborative_recommender.fit(interactions)

    def recommend(
        self,
        event: Dict[str, Any],
        user_id: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Generate hybrid recommendations.

        Args:
            event: Dictionary describing the security event.
            user_id: Optional user ID for collaborative filtering.
            top_k: Number of recommendations.

        Returns:
            List of recommended actions with combined scores.
        """
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend(event, top_k=top_k * 2)

        # Get collaborative recommendations
        collab_recs = []
        if user_id is not None:
            collab_recs = self.collaborative_recommender.recommend(user_id, top_k=top_k * 2)

        # Merge recommendations
        merged: Dict[str, Dict[str, Any]] = {}

        for rec in content_recs:
            action = rec["action"]
            merged[action] = {
                "action": action,
                "description": rec.get("description", ""),
                "content_score": rec.get("score", 0),
                "collaborative_score": 0,
                "score": self.content_weight * rec.get("score", 0),
            }

        for rec in collab_recs:
            action = rec["action"]
            if action in merged:
                merged[action]["collaborative_score"] = rec.get("score", 0)
                merged[action]["score"] += self.collaborative_weight * rec.get("score", 0)
            else:
                merged[action] = {
                    "action": action,
                    "description": "",
                    "content_score": 0,
                    "collaborative_score": rec.get("score", 0),
                    "score": self.collaborative_weight * rec.get("score", 0),
                }

        # Sort by combined score and return top-k
        sorted_recs = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return sorted_recs[:top_k]