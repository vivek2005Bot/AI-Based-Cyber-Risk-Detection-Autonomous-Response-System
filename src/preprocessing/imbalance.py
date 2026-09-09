"""Class imbalance handling using resampling techniques."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ImbalanceHandler:
    """Handles class imbalance using various resampling techniques."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("preprocessing.class_imbalance")
        self.method = self.config.get("method", "smote")
        self.sampling_strategy = self.config.get("sampling_strategy", "auto")
        self.random_state = get_config().get("preprocessing.random_state", 42)

    def resample(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply the configured resampling technique.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            Resampled X and y.
        """
        # Check if resampling is needed
        unique, counts = np.unique(y, return_counts=True)
        if len(unique) < 2:
            logger.warning("Only one class present. Skipping resampling.")
            return X, y

        min_count = counts.min()
        max_count = counts.max()
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        if imbalance_ratio < 1.5:
            logger.info(f"Imbalance ratio {imbalance_ratio:.2f} is acceptable. Skipping resampling.")
            return X, y

        logger.info(f"Applying {self.method} resampling (imbalance ratio: {imbalance_ratio:.2f})")

        if self.method == "smote":
            sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
        elif self.method == "adasyn":
            sampler = ADASYN(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
        elif self.method == "borderline_smote":
            sampler = BorderlineSMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
        elif self.method == "random_oversample":
            sampler = RandomOverSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
        elif self.method == "random_undersample":
            sampler = RandomUnderSampler(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
        elif self.method == "smote_tomek":
            sampler = SMOTETomek(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )
        else:
            logger.warning(f"Unknown resampling method: {self.method}. Using SMOTE.")
            sampler = SMOTE(
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state,
            )

        X_resampled, y_resampled = sampler.fit_resample(X, y)
        logger.info(
            f"Resampled: {len(X)} -> {len(X_resampled)} samples. "
            f"Class distribution: {dict(zip(*np.unique(y_resampled, return_counts=True)))}"
        )
        return X_resampled, y_resampled

    def resample_dataframe(
        self, df: pd.DataFrame, target_col: str = "risk_label"
    ) -> pd.DataFrame:
        """Resample a DataFrame using the configured technique.

        Args:
            df: Input DataFrame.
            target_col: Name of the target column.

        Returns:
            Resampled DataFrame.
        """
        if df.empty or target_col not in df.columns:
            return df

        X = df.drop(columns=[target_col])
        y = df[target_col]

        # Encode target if needed
        y_encoded = y.astype("category").cat.codes
        X_resampled, y_resampled = self.resample(X.values, y_encoded.values)

        # Reconstruct DataFrame
        result = pd.DataFrame(X_resampled, columns=X.columns)
        result[target_col] = y_resampled
        return result