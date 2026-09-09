"""Unsupervised clustering techniques: K-Means, Hierarchical, and DBSCAN."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering

from src.utils.config import get_config
from src.utils.logging import get_logger
from src.utils.metrics import clustering_metrics

logger = get_logger(__name__)


class ClusteringAnalyzer:
    """Applies clustering algorithms to identify unusual patterns in data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("unsupervised_models")
        self.random_state = get_config().get("preprocessing.random_state", 42)

    def kmeans(self, X: np.ndarray, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Apply K-Means clustering.

        Args:
            X: Feature matrix.
            n_clusters: Number of clusters. Defaults to config value.

        Returns:
            Dictionary with labels, model, and metrics.
        """
        X_arr = np.asarray(X)
        params = self.config.get("kmeans", {}).get("params", {})
        if n_clusters is not None:
            params["n_clusters"] = n_clusters
        params.setdefault("random_state", self.random_state)

        logger.info(f"Applying K-Means with {params.get('n_clusters', 5)} clusters...")
        model = KMeans(**params)
        labels = model.fit_predict(X_arr)

        metrics = clustering_metrics(X_arr, labels)
        logger.info(f"K-Means metrics: {metrics}")
        return {"labels": labels, "model": model, "metrics": metrics}

    def hierarchical(self, X: np.ndarray, n_clusters: Optional[int] = None) -> Dict[str, Any]:
        """Apply Hierarchical (Agglomerative) clustering.

        Args:
            X: Feature matrix.
            n_clusters: Number of clusters. Defaults to config value.

        Returns:
            Dictionary with labels, model, and metrics.
        """
        X_arr = np.asarray(X)
        params = self.config.get("hierarchical_clustering", {}).get("params", {})
        if n_clusters is not None:
            params["n_clusters"] = n_clusters

        logger.info(f"Applying Hierarchical clustering with {params.get('n_clusters', 5)} clusters...")
        if len(X_arr) > 5000:
            rng = np.random.default_rng(self.random_state)
            sample_indices = rng.choice(len(X_arr), size=5000, replace=False)
            X_sample = X_arr[sample_indices]
            model = AgglomerativeClustering(**params)
            sample_labels = model.fit_predict(X_sample)
            from sklearn.neighbors import NearestCentroid
            clf = NearestCentroid()
            clf.fit(X_sample, sample_labels)
            labels = clf.predict(X_arr)
        else:
            model = AgglomerativeClustering(**params)
            labels = model.fit_predict(X_arr)

        metrics = clustering_metrics(X_arr, labels)
        logger.info(f"Hierarchical metrics: {metrics}")
        return {"labels": labels, "model": model, "metrics": metrics}

    def dbscan(self, X: np.ndarray, eps: Optional[float] = None, min_samples: Optional[int] = None) -> Dict[str, Any]:
        """Apply DBSCAN clustering.

        Args:
            X: Feature matrix.
            eps: Maximum distance between samples.
            min_samples: Minimum samples in a neighborhood.

        Returns:
            Dictionary with labels, model, and metrics.
        """
        X_arr = np.asarray(X)
        params = self.config.get("dbscan", {}).get("params", {})
        if eps is not None:
            params["eps"] = eps
        if min_samples is not None:
            params["min_samples"] = min_samples

        logger.info(f"Applying DBSCAN with eps={params.get('eps', 0.5)}...")
        model = DBSCAN(**params)
        labels = model.fit_predict(X_arr)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        logger.info(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")

        metrics = clustering_metrics(X_arr, labels)
        metrics["n_clusters"] = n_clusters
        metrics["n_noise"] = n_noise
        return {"labels": labels, "model": model, "metrics": metrics}

    def run_all(self, X: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """Run all enabled clustering algorithms.

        Args:
            X: Feature matrix.

        Returns:
            Dictionary of algorithm name to results.
        """
        results: Dict[str, Dict[str, Any]] = {}

        if self.config.get("kmeans", {}).get("enabled", True):
            try:
                results["kmeans"] = self.kmeans(X)
            except Exception as e:
                logger.error(f"K-Means failed: {e}")

        if self.config.get("hierarchical_clustering", {}).get("enabled", True):
            try:
                results["hierarchical"] = self.hierarchical(X)
            except Exception as e:
                logger.error(f"Hierarchical clustering failed: {e}")

        if self.config.get("dbscan", {}).get("enabled", True):
            try:
                results["dbscan"] = self.dbscan(X)
            except Exception as e:
                logger.error(f"DBSCAN failed: {e}")

        return results