"""Evidence-based risk fusion mechanism combining multiple signals into a risk score."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.risk_fusion.contextual import ContextualIndicatorEngine
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class RiskFusionEngine:
    """Combines model predictions, anomaly scores, and contextual indicators into a risk score."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("risk_fusion")
        self.weights = self.config.get("weights", {
            "supervised_prediction": 0.4,
            "anomaly_score": 0.3,
            "contextual_indicators": 0.3,
        })
        self.thresholds = self.config.get("threshold", {
            "low": 0.3,
            "medium": 0.6,
            "high": 0.8,
            "critical": 0.9,
        })
        self.contextual_engine = ContextualIndicatorEngine()

    def _normalize_weights(self) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(self.weights.values())
        if total == 0:
            return {"supervised_prediction": 0.4, "anomaly_score": 0.3, "contextual_indicators": 0.3}
        return {k: v / total for k, v in self.weights.items()}

    def fuse(
        self,
        supervised_scores: np.ndarray,
        anomaly_scores: np.ndarray,
        contextual_scores: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Fuse multiple evidence sources into a single risk score.

        Args:
            supervised_scores: Model prediction probabilities (0-1).
            anomaly_scores: Anomaly detection scores (0-1).
            contextual_scores: Contextual indicator scores (0-1). If None, uses 0.

        Returns:
            Fused risk scores in [0, 1].
        """
        supervised_scores = np.asarray(supervised_scores, dtype=float)
        anomaly_scores = np.asarray(anomaly_scores, dtype=float)

        if contextual_scores is None:
            contextual_scores = np.zeros_like(supervised_scores)
        else:
            contextual_scores = np.asarray(contextual_scores, dtype=float)

        # Clip all inputs to [0, 1]
        supervised_scores = np.clip(supervised_scores, 0.0, 1.0)
        anomaly_scores = np.clip(anomaly_scores, 0.0, 1.0)
        contextual_scores = np.clip(contextual_scores, 0.0, 1.0)

        weights = self._normalize_weights()

        risk_scores = (
            weights["supervised_prediction"] * supervised_scores
            + weights["anomaly_score"] * anomaly_scores
            + weights["contextual_indicators"] * contextual_scores
        )

        return np.clip(risk_scores, 0.0, 1.0)

    def risk_level(self, risk_scores: np.ndarray) -> np.ndarray:
        """Map risk scores to categorical risk levels.

        Args:
            risk_scores: Fused risk scores in [0, 1].

        Returns:
            Array of risk level strings.
        """
        levels = []
        for score in risk_scores:
            if score >= self.thresholds.get("critical", 0.9):
                levels.append("critical")
            elif score >= self.thresholds.get("high", 0.8):
                levels.append("high")
            elif score >= self.thresholds.get("medium", 0.6):
                levels.append("medium")
            elif score >= self.thresholds.get("low", 0.3):
                levels.append("low")
            else:
                levels.append("minimal")
        return np.array(levels)

    def fuse_dataframe(
        self,
        df: pd.DataFrame,
        supervised_col: str = "supervised_score",
        anomaly_col: str = "anomaly_score",
        contextual_cols: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Fuse risk evidence from a DataFrame.

        Args:
            df: Input DataFrame with score columns.
            supervised_col: Column with supervised prediction scores.
            anomaly_col: Column with anomaly scores.
            contextual_cols: List of contextual indicator columns. If None, uses 'indicator_' prefixed columns.

        Returns:
            DataFrame with 'risk_score' and 'risk_level' columns added.
        """
        result = df.copy()

        if supervised_col not in result.columns:
            result[supervised_col] = 0.0
        if anomaly_col not in result.columns:
            result[anomaly_col] = 0.0

        if contextual_cols is None:
            contextual_cols = [c for c in result.columns if c.startswith("indicator_")]

        if contextual_cols:
            contextual_scores = result[contextual_cols].fillna(0).mean(axis=1).values
        else:
            contextual_scores = np.zeros(len(result))

        risk_scores = self.fuse(
            result[supervised_col].values,
            result[anomaly_col].values,
            contextual_scores,
        )

        result["risk_score"] = risk_scores
        result["risk_level"] = self.risk_level(risk_scores)
        return result

    def explain_risk(self, risk_score: float, supervised_score: float, anomaly_score: float, contextual_score: float) -> Dict[str, Any]:
        """Generate an interpretable explanation for a risk score.

        Args:
            risk_score: Final fused risk score.
            supervised_score: Supervised model score.
            anomaly_score: Anomaly detection score.
            contextual_score: Contextual indicator score.

        Returns:
            Dictionary with risk explanation details.
        """
        weights = self._normalize_weights()

        contributions = {
            "supervised_prediction": {
                "score": float(supervised_score),
                "weight": weights["supervised_prediction"],
                "contribution": float(supervised_score * weights["supervised_prediction"]),
            },
            "anomaly_score": {
                "score": float(anomaly_score),
                "weight": weights["anomaly_score"],
                "contribution": float(anomaly_score * weights["anomaly_score"]),
            },
            "contextual_indicators": {
                "score": float(contextual_score),
                "weight": weights["contextual_indicators"],
                "contribution": float(contextual_score * weights["contextual_indicators"]),
            },
        }

        # Determine dominant factor
        dominant = max(contributions, key=lambda k: contributions[k]["contribution"])

        return {
            "risk_score": float(risk_score),
            "risk_level": self.risk_level(np.array([risk_score]))[0],
            "contributions": contributions,
            "dominant_factor": dominant,
            "explanation": (
                f"Risk score {risk_score:.2f} ({self.risk_level(np.array([risk_score]))[0]}) "
                f"is primarily driven by {dominant.replace('_', ' ')} "
                f"contributing {contributions[dominant]['contribution']:.2f} to the total."
            ),
        }