"""Model evaluation with cross-validation and comprehensive metrics."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score

from src.utils.config import get_config
from src.utils.logging import get_logger
from src.utils.metrics import classification_metrics

logger = get_logger(__name__)


class ModelEvaluator:
    """Evaluates model performance using cross-validation and multiple metrics."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("evaluation")
        self.random_state = get_config().get("preprocessing.random_state", 42)
        self.cv_enabled = self.config.get("cross_validation.enabled", True)
        self.n_folds = self.config.get("cross_validation.n_folds", 5)
        self.stratified = self.config.get("cross_validation.stratified", True)

    def evaluate(
        self,
        model: Any,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Evaluate a trained model on test data.

        Args:
            model: Trained model.
            X_test: Test features.
            y_test: Test labels.
            model_name: Name of the model for logging.

        Returns:
            Dictionary of evaluation metrics.
        """
        logger.info(f"Evaluating {model_name}...")
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = model.predict_proba(X_test)
            except Exception:
                y_prob = None

        metrics = classification_metrics(y_test, y_pred, y_prob)
        logger.info(
            f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, "
            f"F1: {metrics['f1']:.4f}, "
            f"Precision: {metrics['precision']:.4f}, "
            f"Recall: {metrics['recall']:.4f}"
        )
        return metrics

    def cross_validate(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str = "model",
    ) -> Dict[str, Any]:
        """Perform cross-validation evaluation.

        Args:
            model: Model instance (not yet trained).
            X: Feature matrix.
            y: Target labels.
            model_name: Name of the model for logging.

        Returns:
            Dictionary with cross-validation scores and metrics.
        """
        if not self.cv_enabled:
            logger.info("Cross-validation disabled. Skipping.")
            return {}

        cv = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        logger.info(f"Running {self.n_folds}-fold cross-validation for {model_name}...")

        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

        # Get cross-validated probabilities
        y_prob = None
        if hasattr(model, "predict_proba"):
            try:
                y_prob = cross_val_predict(model, X, y, cv=cv, n_jobs=-1, method="predict_proba")
            except Exception:
                y_prob = None

        metrics = classification_metrics(y, y_pred, y_prob)

        # Get per-fold scores for multiple metrics
        scoring_metrics = self.config.get("metrics", ["accuracy", "f1", "roc_auc"])

        for metric in scoring_metrics:
            if metric == "roc_auc" and y_prob is None:
                continue
            try:
                scores = cross_val_score(
                    model, X, y, cv=cv, scoring=metric, n_jobs=-1
                )
                metrics[f"cv_{metric}_mean"] = float(np.mean(scores))
                metrics[f"cv_{metric}_std"] = float(np.std(scores))
            except Exception as e:
                logger.warning(f"Could not compute CV score for {metric}: {e}")

        metrics["cv_folds"] = self.n_folds
        metrics["cv_predictions"] = y_pred.tolist()

        logger.info(
            f"{model_name} CV - Accuracy: {metrics.get('accuracy', 0):.4f}, "
            f"F1: {metrics.get('f1', 0):.4f}"
        )
        return metrics

    def evaluate_all(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> pd.DataFrame:
        """Evaluate multiple models and return a comparison DataFrame.

        Args:
            models: Dictionary of model name to trained model.
            X_test: Test features.
            y_test: Test labels.

        Returns:
            DataFrame with one row per model and metric columns.
        """
        results: Dict[str, Dict[str, Any]] = {}

        for name, model in models.items():
            try:
                metrics = self.evaluate(model, X_test, y_test, name)
                results[name] = {
                    k: v for k, v in metrics.items()
                    if k not in ["confusion_matrix", "classification_report", "cv_predictions"]
                }
            except Exception as e:
                logger.error(f"Failed to evaluate {name}: {e}")

        return pd.DataFrame(results).T