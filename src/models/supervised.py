"""Supervised machine learning models for risk prediction."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.models.pytorch_net import PyTorchRiskClassifier
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SupervisedModelFactory:
    """Factory for creating supervised machine learning models."""

    MODEL_REGISTRY = {
        "logistic_regression": LogisticRegression,
        "naive_bayes": GaussianNB,
        "knn": KNeighborsClassifier,
        "svm": SVC,
        "decision_tree": DecisionTreeClassifier,
        "random_forest": RandomForestClassifier,
        "neural_network": MLPClassifier,
        "pytorch_dnn": PyTorchRiskClassifier,
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("supervised_models")

    def create_model(self, name: str) -> Any:
        """Create a model instance by name.

        Args:
            name: Model name (e.g., 'logistic_regression', 'random_forest').

        Returns:
            Configured model instance.
        """
        if name not in self.MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {name}. Available: {list(self.MODEL_REGISTRY.keys())}")

        model_class = self.MODEL_REGISTRY[name]
        params = self.config.get(name, {}).get("params", {})

        # Convert string params to appropriate types
        if name == "naive_bayes" and "var_smoothing" in params:
            params["var_smoothing"] = float(params["var_smoothing"])

        # Special handling for SVM to enable probability estimates
        if name == "svm":
            params.setdefault("probability", True)

        logger.info(f"Creating {name} model with params: {params}")
        return model_class(**params)

    def create_all_enabled(self) -> Dict[str, Any]:
        """Create all enabled models from configuration.

        Returns:
            Dictionary of model name to model instance.
        """
        models: Dict[str, Any] = {}
        for name, model_config in self.config.items():
            if isinstance(model_config, dict) and model_config.get("enabled", False):
                try:
                    models[name] = self.create_model(name)
                except Exception as e:
                    logger.error(f"Failed to create model {name}: {e}")
        return models


class ModelTrainer:
    """Trains and evaluates supervised models."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().data
        self.factory = SupervisedModelFactory(self.config.get("supervised_models", {}))
        self.random_state = self.config.get("preprocessing", {}).get("random_state", 42)

    def train(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> Any:
        """Train a model.

        Args:
            model: Model instance.
            X_train: Training features.
            y_train: Training labels.

        Returns:
            Trained model.
        """
        logger.info(f"Training {model.__class__.__name__}...")
        if model.__class__.__name__ in ["SVC", "NuSVC"] and len(X_train) > 15000:
            sample_idx = np.random.RandomState(self.random_state).choice(
                len(X_train), size=15000, replace=False
            )
            X_fit = X_train.iloc[sample_idx] if hasattr(X_train, "iloc") else X_train[sample_idx]
            y_fit = y_train.iloc[sample_idx] if hasattr(y_train, "iloc") else y_train[sample_idx]
            model.fit(X_fit, y_fit)
        else:
            model.fit(X_train, y_train)
        return model

    def predict(
        self, model: Any, X: np.ndarray
    ) -> np.ndarray:
        """Make predictions with a trained model.

        Args:
            model: Trained model.
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        return model.predict(X)

    def predict_proba(
        self, model: Any, X: np.ndarray
    ) -> Optional[np.ndarray]:
        """Get prediction probabilities if the model supports it.

        Args:
            model: Trained model.
            X: Feature matrix.

        Returns:
            Probability matrix or None if not supported.
        """
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)
        return None

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Train all enabled models.

        Args:
            X_train: Training features.
            y_train: Training labels.
            model_names: Optional list of specific models to train.

        Returns:
            Dictionary of model name to trained model.
        """
        models = self.factory.create_all_enabled()
        if model_names:
            models = {name: models[name] for name in model_names if name in models}

        trained: Dict[str, Any] = {}
        for name, model in models.items():
            try:
                trained[name] = self.train(model, X_train, y_train)
            except Exception as e:
                logger.error(f"Failed to train {name}: {e}")

        return trained