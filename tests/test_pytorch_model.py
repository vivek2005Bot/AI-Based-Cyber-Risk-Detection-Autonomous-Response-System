"""Tests for the PyTorch Deep Neural Network Classifier."""

import numpy as np
import pytest
from sklearn.base import clone, is_classifier
from sklearn.datasets import make_classification

from src.models.pytorch_net import PyTorchRiskClassifier


@pytest.fixture
def sample_binary_data():
    X, y = make_classification(
        n_samples=120,
        n_features=10,
        n_informative=6,
        n_classes=2,
        random_state=42,
    )
    return X, y


def test_pytorch_classifier_init():
    """Test default initialization and tag compliance."""
    clf = PyTorchRiskClassifier(epochs=5, hidden_layer_sizes=[32, 16])
    assert is_classifier(clf)
    assert clf.epochs == 5
    assert clf.hidden_layer_sizes == [32, 16]


def test_pytorch_classifier_fit_predict(sample_binary_data):
    """Test training and inference loop."""
    X, y = sample_binary_data
    clf = PyTorchRiskClassifier(epochs=10, batch_size=32, hidden_layer_sizes=[32, 16], random_state=42)
    clf.fit(X, y)

    assert clf.model_ is not None
    assert clf.classes_ is not None
    assert len(clf.classes_) == 2
    assert len(clf.training_losses_) > 0

    preds = clf.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset(set(np.unique(y)))

    probs = clf.predict_proba(X)
    assert probs.shape == (len(X), 2)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


def test_pytorch_classifier_clone(sample_binary_data):
    """Test scikit-learn clone compatibility."""
    clf = PyTorchRiskClassifier(epochs=5, hidden_layer_sizes=[64, 32])
    cloned_clf = clone(clf)
    assert cloned_clf.epochs == clf.epochs
    assert cloned_clf.hidden_layer_sizes == clf.hidden_layer_sizes
    assert cloned_clf.model_ is None
