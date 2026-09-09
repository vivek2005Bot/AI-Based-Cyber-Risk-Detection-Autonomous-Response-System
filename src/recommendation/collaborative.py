"""Collaborative filtering recommendation for security response actions."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class CollaborativeRecommender:
    """Recommends response actions using collaborative filtering."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("recommendation.collaborative")
        self.method = self.config.get("method", "matrix_factorization")
        self.n_factors = self.config.get("n_factors", 50)
        self.n_epochs = self.config.get("n_epochs", 20)
        self.lr = self.config.get("lr", 0.005)
        self.reg = self.config.get("reg", 0.02)
        self.random_state = get_config().get("preprocessing.random_state", 42)

        # User-item interaction matrix (users x actions)
        self._user_action_matrix: Optional[np.ndarray] = None
        self._user_factors: Optional[np.ndarray] = None
        self._action_factors: Optional[np.ndarray] = None
        self._user_ids: List[str] = []
        self._action_ids: List[str] = []

    def _build_interaction_matrix(
        self, interactions: pd.DataFrame
    ) -> np.ndarray:
        """Build a user-action interaction matrix from historical data.

        Args:
            interactions: DataFrame with 'user_id', 'action', and 'rating' columns.

        Returns:
            User-action interaction matrix.
        """
        self._user_ids = sorted(interactions["user_id"].unique())
        self._action_ids = sorted(interactions["action"].unique())

        n_users = len(self._user_ids)
        n_actions = len(self._action_ids)
        matrix = np.zeros((n_users, n_actions))

        user_idx = {u: i for i, u in enumerate(self._user_ids)}
        action_idx = {a: i for i, a in enumerate(self._action_ids)}

        for _, row in interactions.iterrows():
            u = user_idx.get(row["user_id"])
            a = action_idx.get(row["action"])
            if u is not None and a is not None:
                matrix[u, a] = row.get("rating", 1.0)

        return matrix

    def _matrix_factorization(
        self, matrix: np.ndarray
    ) -> None:
        """Perform matrix factorization using SGD.

        Args:
            matrix: User-action interaction matrix.
        """
        n_users, n_actions = matrix.shape
        rng = np.random.default_rng(self.random_state)

        self._user_factors = rng.normal(0, 0.1, (n_users, self.n_factors))
        self._action_factors = rng.normal(0, 0.1, (n_actions, self.n_factors))

        # Find non-zero entries
        user_indices, action_indices = np.nonzero(matrix)
        ratings = matrix[user_indices, action_indices]

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for u, a, r in zip(user_indices, action_indices, ratings):
                # Predict rating
                pred = np.dot(self._user_factors[u], self._action_factors[a])
                error = r - pred

                # Update factors
                user_grad = error * self._action_factors[a] - self.reg * self._user_factors[u]
                action_grad = error * self._user_factors[u] - self.reg * self._action_factors[a]

                self._user_factors[u] += self.lr * user_grad
                self._action_factors[a] += self.lr * action_grad

                total_loss += error ** 2

            if epoch % 5 == 0:
                logger.info(f"MF epoch {epoch}: loss = {total_loss:.4f}")

    def fit(self, interactions: pd.DataFrame) -> None:
        """Fit the collaborative filtering model.

        Args:
            interactions: DataFrame with 'user_id', 'action', and 'rating' columns.
        """
        if interactions.empty:
            logger.warning("No interaction data for collaborative filtering.")
            return

        matrix = self._build_interaction_matrix(interactions)

        if self.method == "matrix_factorization":
            self._matrix_factorization(matrix)
        else:
            # KNN-based approach: store the matrix directly
            self._user_action_matrix = matrix

        logger.info(f"Fitted collaborative recommender with {len(self._user_ids)} users and {len(self._action_ids)} actions")

    def recommend(
        self,
        user_id: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Recommend actions for a user based on collaborative filtering.

        Args:
            user_id: ID of the user.
            top_k: Number of recommendations.

        Returns:
            List of recommended actions with scores.
        """
        if user_id not in self._user_ids:
            logger.warning(f"User {user_id} not found in interaction data.")
            return []

        user_idx = self._user_ids.index(user_id)

        if self.method == "matrix_factorization" and self._user_factors is not None:
            scores = np.dot(self._user_factors[user_idx], self._action_factors.T)
        elif self._user_action_matrix is not None:
            # KNN approach: use interaction matrix directly
            scores = self._user_action_matrix[user_idx]
        else:
            logger.warning("Collaborative recommender not fitted.")
            return []

        # Get top-k actions
        top_indices = np.argsort(scores)[::-1][:top_k]

        recommendations = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            recommendations.append({
                "action": self._action_ids[idx],
                "score": float(scores[idx]),
            })

        return recommendations