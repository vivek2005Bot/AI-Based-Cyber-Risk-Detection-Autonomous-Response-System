"""Tests for the anomaly detection and clustering modules."""

import numpy as np
import pytest

from src.anomaly.clustering import ClusteringAnalyzer
from src.anomaly.detectors import AnomalyDetector


@pytest.fixture
def sample_features():
    np.random.seed(42)
    return np.random.randn(50, 5)


def test_isolation_forest(sample_features):
    """Test Isolation Forest detector."""
    detector = AnomalyDetector()
    result = detector.isolation_forest(sample_features)
    assert "predictions" in result
    assert "anomaly_scores" in result
    assert len(result["predictions"]) == len(sample_features)
    assert 0.0 <= result["anomaly_scores"].max() <= 1.0


def test_local_outlier_factor_multi(sample_features):
    """Test LOF detector with regular batch."""
    detector = AnomalyDetector()
    result = detector.local_outlier_factor(sample_features)
    assert "predictions" in result
    assert "anomaly_scores" in result
    assert len(result["predictions"]) == len(sample_features)


def test_local_outlier_factor_single_sample():
    """Test LOF detector with single sample edge case."""
    detector = AnomalyDetector()
    single = np.array([[1.0, 2.0, 3.0]])
    result = detector.local_outlier_factor(single)
    assert len(result["predictions"]) == 1
    assert result["n_anomalies"] == 0
    assert result["anomaly_scores"][0] == 0.0


def test_combined_anomaly_score(sample_features):
    """Test combined score computation."""
    detector = AnomalyDetector()
    scores = detector.combined_anomaly_score(sample_features)
    assert len(scores) == len(sample_features)
    assert 0.0 <= scores.min() <= scores.max() <= 1.0


def test_clustering_kmeans(sample_features):
    """Test K-Means clustering analyzer."""
    analyzer = ClusteringAnalyzer()
    result = analyzer.kmeans(sample_features, n_clusters=3)
    assert "labels" in result
    assert len(result["labels"]) == len(sample_features)
    assert "metrics" in result


def test_clustering_dbscan(sample_features):
    """Test DBSCAN clustering analyzer."""
    analyzer = ClusteringAnalyzer()
    result = analyzer.dbscan(sample_features, eps=1.5, min_samples=3)
    assert "labels" in result
    assert "metrics" in result
    assert "n_clusters" in result["metrics"]
