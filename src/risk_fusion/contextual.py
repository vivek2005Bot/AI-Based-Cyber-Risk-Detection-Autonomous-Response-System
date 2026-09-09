"""Contextual security indicators for risk assessment."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ContextualIndicatorEngine:
    """Computes contextual security indicators that contribute to risk scoring."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("risk_fusion.contextual_indicators")
        self.enabled_indicators = {
            k: v for k, v in self.config.items() if isinstance(v, bool) and v
        }

    def vulnerability_severity(self, cvss_scores: np.ndarray) -> np.ndarray:
        """Convert CVSS scores to normalized risk contributions.

        Args:
            cvss_scores: Array of CVSS base scores (0-10).

        Returns:
            Normalized severity scores in [0, 1].
        """
        scores = np.asarray(cvss_scores, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0)
        # Normalize CVSS 0-10 to 0-1
        return np.clip(scores / 10.0, 0.0, 1.0)

    def asset_criticality(self, asset_values: np.ndarray, max_value: float = 100.0) -> np.ndarray:
        """Convert asset criticality values to normalized risk contributions.

        Args:
            asset_values: Array of asset criticality scores.
            max_value: Maximum possible criticality value.

        Returns:
            Normalized criticality scores in [0, 1].
        """
        values = np.asarray(asset_values, dtype=float)
        values = np.nan_to_num(values, nan=0.0)
        return np.clip(values / max_value, 0.0, 1.0)

    def threat_intelligence(self, threat_scores: np.ndarray) -> np.ndarray:
        """Convert threat intelligence scores to normalized risk contributions.

        Args:
            threat_scores: Array of threat intelligence scores.

        Returns:
            Normalized threat scores in [0, 1].
        """
        scores = np.asarray(threat_scores, dtype=float)
        scores = np.nan_to_num(scores, nan=0.0)
        return np.clip(scores, 0.0, 1.0)

    def historical_incidents(self, incident_counts: np.ndarray, max_count: int = 10) -> np.ndarray:
        """Convert historical incident counts to normalized risk contributions.

        Args:
            incident_counts: Array of historical incident counts.
            max_count: Count that maps to maximum risk.

        Returns:
            Normalized incident scores in [0, 1].
        """
        counts = np.asarray(incident_counts, dtype=float)
        counts = np.nan_to_num(counts, nan=0.0)
        return np.clip(counts / max_count, 0.0, 1.0)

    def time_sensitivity(self, hours_since_event: np.ndarray, decay_hours: float = 24.0) -> np.ndarray:
        """Compute time-based risk decay (recent events are riskier).

        Args:
            hours_since_event: Hours since the security event occurred.
            decay_hours: Hours after which risk decays to near zero.

        Returns:
            Time sensitivity scores in [0, 1].
        """
        hours = np.asarray(hours_since_event, dtype=float)
        hours = np.nan_to_num(hours, nan=0.0)
        # Exponential decay: recent events have higher risk
        return np.exp(-hours / decay_hours)

    def compute_indicators(
        self,
        df: pd.DataFrame,
        cvss_col: str = "cvss_score",
        asset_col: str = "asset_criticality",
        threat_col: str = "threat_score",
        incident_col: str = "incident_count",
        time_col: str = "hours_since_event",
    ) -> pd.DataFrame:
        """Compute all enabled contextual indicators for a DataFrame.

        Args:
            df: Input DataFrame with contextual indicator columns.
            cvss_col: Column name for CVSS scores.
            asset_col: Column name for asset criticality.
            threat_col: Column name for threat intelligence scores.
            incident_col: Column name for historical incident counts.
            time_col: Column name for hours since event.

        Returns:
            DataFrame with computed indicator columns.
        """
        result = df.copy()

        if "vulnerability_severity" in self.enabled_indicators and cvss_col in df.columns:
            result["indicator_vulnerability"] = self.vulnerability_severity(df[cvss_col].values)

        if "asset_criticality" in self.enabled_indicators and asset_col in df.columns:
            result["indicator_asset"] = self.asset_criticality(df[asset_col].values)

        if "threat_intelligence" in self.enabled_indicators and threat_col in df.columns:
            result["indicator_threat"] = self.threat_intelligence(df[threat_col].values)

        if "historical_incidents" in self.enabled_indicators and incident_col in df.columns:
            result["indicator_incidents"] = self.historical_incidents(df[incident_col].values)

        if "time_sensitivity" in self.enabled_indicators and time_col in df.columns:
            result["indicator_time"] = self.time_sensitivity(df[time_col].values)

        return result

    def combined_contextual_score(self, df: pd.DataFrame) -> np.ndarray:
        """Compute a combined contextual risk score from all enabled indicators.

        Args:
            df: DataFrame with indicator columns (prefixed with 'indicator_').

        Returns:
            Combined contextual scores in [0, 1].
        """
        indicator_cols = [c for c in df.columns if c.startswith("indicator_")]
        if not indicator_cols:
            return np.zeros(len(df))

        scores = df[indicator_cols].fillna(0).values
        return np.mean(scores, axis=1)