"""Feature selection techniques."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    SelectKBest,
    chi2,
    f_classif,
    mutual_info_classif,
    RFE,
)
from sklearn.linear_model import LogisticRegression

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FeatureSelector:
    """Selects the most relevant features for model training."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("preprocessing.feature_selection")
        self.method = self.config.get("method", "mutual_info")
        self.n_features = self.config.get("n_features", 50)
        self.random_state = get_config().get("preprocessing.random_state", 42)
        self.selected_features: List[str] = []

    def select(
        self, X: pd.DataFrame, y: pd.Series
    ) -> pd.DataFrame:
        """Select the most important features.

        Args:
            X: Feature DataFrame.
            y: Target series.

        Returns:
            DataFrame with only selected features.
        """
        if X.empty or len(y) == 0:
            return X

        # Ensure numeric data
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            logger.warning("No numeric features available for selection.")
            return X

        n_features = min(self.n_features, X_numeric.shape[1])

        if self.method == "mutual_info":
            # For large datasets, compute mutual info on a representative sample for speed
            if len(X_numeric) > 10000:
                sample_idx = np.random.RandomState(self.random_state).choice(
                    len(X_numeric), size=10000, replace=False
                )
                X_sub = X_numeric.iloc[sample_idx]
                y_sub = y.iloc[sample_idx] if hasattr(y, "iloc") else y[sample_idx]
                selector = SelectKBest(mutual_info_classif, k=n_features)
                selector.fit(X_sub, y_sub)
                self.selected_features = X_numeric.columns[selector.get_support()].tolist()
                logger.info(f"Selected {len(self.selected_features)} features using mutual_info (sampled)")
                return X_numeric[self.selected_features]
            selector = SelectKBest(mutual_info_classif, k=n_features)
        elif self.method == "chi2":
            # Chi2 requires non-negative values
            X_scaled = X_numeric - X_numeric.min()
            selector = SelectKBest(chi2, k=n_features)
            selector.fit(X_scaled, y)
            self.selected_features = X_numeric.columns[selector.get_support()].tolist()
            return X_numeric[self.selected_features]
        elif self.method == "f_classif":
            selector = SelectKBest(f_classif, k=n_features)
        elif self.method == "rfe":
            estimator = LogisticRegression(
                max_iter=1000,
                random_state=self.random_state,
            )
            selector = RFE(estimator, n_features_to_select=n_features)
        else:
            logger.warning(f"Unknown feature selection method: {self.method}. Using f_classif.")
            selector = SelectKBest(f_classif, k=n_features)

        selector.fit(X_numeric, y)
        self.selected_features = X_numeric.columns[selector.get_support()].tolist()
        logger.info(f"Selected {len(self.selected_features)} features using {self.method}")
        return X_numeric[self.selected_features]

    def get_feature_scores(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """Get feature importance scores.

        Args:
            X: Feature DataFrame.
            y: Target series.

        Returns:
            Series of feature scores.
        """
        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            return pd.Series(dtype=float)

        if self.method == "mutual_info":
            scores = mutual_info_classif(X_numeric, y, random_state=self.random_state)
        elif self.method == "f_classif":
            scores, _ = f_classif(X_numeric, y)
        elif self.method == "chi2":
            X_scaled = X_numeric - X_numeric.min()
            scores, _ = chi2(X_scaled, y)
        else:
            scores = mutual_info_classif(X_numeric, y, random_state=self.random_state)

        return pd.Series(scores, index=X_numeric.columns).sort_values(ascending=False)