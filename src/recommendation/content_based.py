"""Content-based recommendation for security response actions."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default response catalog with descriptions and risk levels
DEFAULT_RESPONSES = [
    {
        "action": "monitor",
        "description": "Continue monitoring the event for further suspicious activity",
        "risk_levels": ["minimal", "low"],
        "event_types": ["auth", "behavior", "network"],
    },
    {
        "action": "alert",
        "description": "Send an alert to the security operations team for review",
        "risk_levels": ["low", "medium"],
        "event_types": ["auth", "endpoint", "behavior", "network"],
    },
    {
        "action": "block_ip",
        "description": "Block the source IP address at the firewall or network level",
        "risk_levels": ["medium", "high", "critical"],
        "event_types": ["auth", "network"],
    },
    {
        "action": "quarantine_host",
        "description": "Isolate the affected host from the network to prevent lateral movement",
        "risk_levels": ["high", "critical"],
        "event_types": ["endpoint", "network"],
    },
    {
        "action": "reset_credentials",
        "description": "Force password reset and revoke active sessions for compromised accounts",
        "risk_levels": ["medium", "high", "critical"],
        "event_types": ["auth", "behavior"],
    },
    {
        "action": "patch_vulnerability",
        "description": "Apply security patches to address known vulnerabilities",
        "risk_levels": ["medium", "high"],
        "event_types": ["endpoint", "network"],
    },
    {
        "action": "investigate",
        "description": "Conduct a detailed forensic investigation of the security event",
        "risk_levels": ["medium", "high", "critical"],
        "event_types": ["auth", "endpoint", "behavior", "network"],
    },
    {
        "action": "escalate",
        "description": "Escalate the incident to senior security management for immediate response",
        "risk_levels": ["high", "critical"],
        "event_types": ["auth", "endpoint", "behavior", "network"],
    },
]


class ContentBasedRecommender:
    """Recommends response actions based on content similarity."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("recommendation.content_based")
        self.similarity_metric = self.config.get("similarity_metric", "cosine")
        self.top_k = self.config.get("top_k", 5)
        self.responses = DEFAULT_RESPONSES
        self._vectorizer = TfidfVectorizer()
        self._response_vectors = None

    def _build_response_vectors(self) -> np.ndarray:
        """Build TF-IDF vectors for response descriptions."""
        descriptions = [
            f"{r['action'].replace('_', ' ')} {' '.join(r.get('risk_levels', []))} {' '.join(r.get('event_types', []))} {r['description']}"
            for r in self.responses
        ]
        self._response_vectors = self._vectorizer.fit_transform(descriptions)
        return self._response_vectors

    def _build_event_vector(self, event: Dict[str, Any]) -> np.ndarray:
        """Build a TF-IDF vector for an event description."""
        # Create a text description from event features
        text_parts = []

        if "event_type" in event:
            text_parts.append(str(event["event_type"]))
        if "risk_level" in event:
            text_parts.append(str(event["risk_level"]))
        if "attack_type" in event:
            text_parts.append(str(event["attack_type"]))
        if "description" in event:
            text_parts.append(str(event["description"]))

        # Add feature values as text
        for key, value in event.items():
            if key not in ["event_type", "risk_level", "attack_type", "description"]:
                if isinstance(value, (int, float)) and value > 0:
                    text_parts.append(f"{key}={value}")

        text = " ".join(text_parts)
        return self._vectorizer.transform([text])

    def recommend(
        self,
        event: Dict[str, Any],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Recommend response actions for a security event.

        Args:
            event: Dictionary describing the security event.
            top_k: Number of recommendations to return.

        Returns:
            List of recommended actions with scores.
        """
        if self._response_vectors is None:
            self._build_response_vectors()

        k = top_k or self.top_k
        event_vector = self._build_event_vector(event)

        if self.similarity_metric == "cosine":
            similarities = cosine_similarity(event_vector, self._response_vectors).flatten()
        else:
            logger.warning(f"Unknown similarity metric: {self.similarity_metric}. Using cosine.")
            similarities = cosine_similarity(event_vector, self._response_vectors).flatten()

        # Apply risk level filtering
        event_risk = event.get("risk_level", "medium")
        risk_priority = {"minimal": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
        event_risk_priority = risk_priority.get(event_risk, 2)

        adjusted_scores = similarities.copy()
        for i, response in enumerate(self.responses):
            response_risk_levels = response.get("risk_levels", [])
            response_risk_priorities = [risk_priority.get(rl, 2) for rl in response_risk_levels]

            # Penalize responses that don't match the event risk level
            if not any(rp >= event_risk_priority for rp in response_risk_priorities):
                adjusted_scores[i] *= 0.3

            # Penalize responses that don't match the event type
            event_type = event.get("event_type", "")
            if event_type and event_type not in response.get("event_types", []):
                adjusted_scores[i] *= 0.5

        # Get top-k recommendations
        top_indices = np.argsort(adjusted_scores)[::-1][:k]

        recommendations = []
        for idx in top_indices:
            if adjusted_scores[idx] <= 0:
                continue
            recommendations.append({
                "action": self.responses[idx]["action"],
                "description": self.responses[idx]["description"],
                "score": float(adjusted_scores[idx]),
                "similarity": float(similarities[idx]),
            })

        # Fallback if no text similarity matches occurred
        if not recommendations:
            event_type = event.get("event_type", "")
            candidate_indices = []
            for i, response in enumerate(self.responses):
                matches_risk = any(risk_priority.get(rl, 2) >= event_risk_priority for rl in response.get("risk_levels", []))
                matches_type = not event_type or event_type in response.get("event_types", [])
                score = (0.5 if matches_risk else 0.2) + (0.3 if matches_type else 0.1)
                candidate_indices.append((score, i))

            candidate_indices.sort(key=lambda x: x[0], reverse=True)
            for score, idx in candidate_indices[:k]:
                recommendations.append({
                    "action": self.responses[idx]["action"],
                    "description": self.responses[idx]["description"],
                    "score": float(score),
                    "similarity": 0.0,
                })

        return recommendations