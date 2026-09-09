"""Anomaly detection using Isolation Forest and Local Outlier Factor."""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """Detects anomalies using unsupervised methods."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("unsupervised_models")
        self.random_state = get_config().get("preprocessing.random_state", 42)

    def isolation_forest(self, X: np.ndarray, contamination: Optional[float] = None) -> Dict[str, Any]:
        """Apply Isolation Forest for anomaly detection.

        Args:
            X: Feature matrix.
            contamination: Expected proportion of outliers.

        Returns:
            Dictionary with predictions, scores, and model.
        """
        params = self.config.get("isolation_forest", {}).get("params", {})
        if contamination is not None:
            params["contamination"] = contamination
        params.setdefault("random_state", self.random_state)

        logger.info("Applying Isolation Forest...")
        model = IsolationForest(**params)
        model.fit(X)

        # -1 for anomalies, 1 for normal
        predictions = model.predict(X)
        # Higher score = more anomalous
        anomaly_scores = -model.score_samples(X)

        # Normalize scores to [0, 1]
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        if max_score > min_score:
            normalized_scores = (anomaly_scores - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(anomaly_scores)

        n_anomalies = int((predictions == -1).sum())
        logger.info(f"Isolation Forest detected {n_anomalies} anomalies ({n_anomalies / len(X) * 100:.2f}%)")

        return {
            "predictions": predictions,
            "anomaly_scores": normalized_scores,
            "raw_scores": anomaly_scores,
            "model": model,
            "n_anomalies": n_anomalies,
        }

    def local_outlier_factor(self, X: np.ndarray, contamination: Optional[float] = None) -> Dict[str, Any]:
        """Apply Local Outlier Factor for anomaly detection.

        Args:
            X: Feature matrix.
            contamination: Expected proportion of outliers.

        Returns:
            Dictionary with predictions, scores, and model.
        """
        if len(X) <= 1:
            return {
                "predictions": np.array([1]),
                "anomaly_scores": np.zeros(len(X)),
                "raw_scores": np.zeros(len(X)),
                "model": None,
                "n_anomalies": 0,
            }

        params = dict(self.config.get("local_outlier_factor", {}).get("params", {}))
        if contamination is not None:
            params["contamination"] = contamination
        if len(X) <= params.get("n_neighbors", 20):
            params["n_neighbors"] = max(1, len(X) - 1)

        logger.info("Applying Local Outlier Factor...")
        if len(X) > 5000:
            rng = np.random.default_rng(self.random_state)
            sample_indices = rng.choice(len(X), size=5000, replace=False)
            model = LocalOutlierFactor(novelty=True, **params)
            model.fit(X[sample_indices])
            predictions = model.predict(X)
            outlier_factors = -model.score_samples(X)
        else:
            model = LocalOutlierFactor(**params)
            predictions = model.fit_predict(X)
            outlier_factors = -model.negative_outlier_factor_

        # Normalize to [0, 1]
        min_score = outlier_factors.min()
        max_score = outlier_factors.max()
        if max_score > min_score:
            normalized_scores = (outlier_factors - min_score) / (max_score - min_score)
        else:
            normalized_scores = np.zeros_like(outlier_factors)

        n_anomalies = int((predictions == -1).sum())
        logger.info(f"LOF detected {n_anomalies} anomalies ({n_anomalies / len(X) * 100:.2f}%)")

        return {
            "predictions": predictions,
            "anomaly_scores": normalized_scores,
            "raw_scores": outlier_factors,
            "model": model,
            "n_anomalies": n_anomalies,
        }

    def run_all(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run all enabled anomaly detection algorithms.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary of algorithm name to results.
        """
        results: Dict[str, Dict[str, Any]] = {}

        if self.config.get("isolation_forest", {}).get("enabled", True):
            try:
                results["isolation_forest"] = self.isolation_forest(X)
            except Exception as e:
                logger.error(f"Isolation Forest failed: {e}")

        if self.config.get("local_outlier_factor", {}).get("enabled", True):
            try:
                results["local_outlier_factor"] = self.local_outlier_factor(X)
            except Exception as e:
                logger.error(f"LOF failed: {e}")

        return results

    def combined_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute a combined anomaly score from all enabled detectors.

        Args:
            X: Feature matrix.

        Returns:
            Array of combined anomaly scores in [0, 1].
        """
        results = self.run_all(X)
        if not results:
            return np.zeros(len(X))

        scores = []
        for name, result in results.items():
            scores.append(result["anomaly_scores"])

        # Average the normalized scores
        combined = np.mean(scores, axis=0)
        return combined