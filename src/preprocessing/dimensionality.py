"""Dimensionality reduction techniques: PCA, LDA, and SVD."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DimensionalityReducer:
    """Reduces feature dimensionality using PCA, LDA, or SVD."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("preprocessing.dimensionality_reduction")
        self.method = self.config.get("method", "pca")
        self.n_components = self.config.get("n_components", 0.95)
        self.random_state = get_config().get("preprocessing.random_state", 42)
        self._reducer = None

    def fit_transform(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """Fit the dimensionality reducer and transform the data.

        Args:
            X: Feature DataFrame.
            y: Target series (required for LDA).

        Returns:
            Reduced DataFrame.
        """
        if X.empty:
            return X

        X_numeric = X.select_dtypes(include=[np.number])
        if X_numeric.empty:
            logger.warning("No numeric features available for dimensionality reduction.")
            return X

        n_samples, n_features = X_numeric.shape

        if self.method == "pca":
            self._reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
            )
        elif self.method == "lda":
            if y is None:
                logger.warning("LDA requires target labels. Using PCA instead.")
                self._reducer = PCA(
                    n_components=self.n_components,
                    random_state=self.random_state,
                )
            else:
                n_classes = y.nunique()
                n_components = min(n_classes - 1, n_features)
                self._reducer = LinearDiscriminantAnalysis(
                    n_components=n_components,
                    solver="svd",
                )
        elif self.method == "svd":
            n_components = self.n_components
            if isinstance(n_components, float) and n_components < 1:
                n_components = min(50, n_features)
            self._reducer = TruncatedSVD(
                n_components=int(n_components),
                algorithm="randomized",
                random_state=self.random_state,
            )
        else:
            logger.warning(f"Unknown dimensionality reduction method: {self.method}. Using PCA.")
            self._reducer = PCA(
                n_components=self.n_components,
                random_state=self.random_state,
            )

        if self.method == "lda" and y is not None:
            reduced = self._reducer.fit_transform(X_numeric, y)
        else:
            reduced = self._reducer.fit_transform(X_numeric)

        # Create column names
        n_components_actual = reduced.shape[1]
        columns = [f"PC{i+1}" for i in range(n_components_actual)]
        if self.method == "lda":
            columns = [f"LD{i+1}" for i in range(n_components_actual)]
        elif self.method == "svd":
            columns = [f"SVD{i+1}" for i in range(n_components_actual)]

        result = pd.DataFrame(reduced, columns=columns, index=X_numeric.index)
        logger.info(f"Reduced {n_features} features to {n_components_actual} using {self.method}")
        return result

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data using the fitted reducer.

        Args:
            X: Feature DataFrame.

        Returns:
            Reduced DataFrame.
        """
        if self._reducer is None:
            raise ValueError("Reducer not fitted. Call fit_transform first.")

        X_numeric = X.select_dtypes(include=[np.number])
        reduced = self._reducer.transform(X_numeric)

        n_components_actual = reduced.shape[1]
        columns = [f"PC{i+1}" for i in range(n_components_actual)]
        if self.method == "lda":
            columns = [f"LD{i+1}" for i in range(n_components_actual)]
        elif self.method == "svd":
            columns = [f"SVD{i+1}" for i in range(n_components_actual)]

        return pd.DataFrame(reduced, columns=columns, index=X_numeric.index)

    @property
    def explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get explained variance ratio (PCA only)."""
        if hasattr(self._reducer, "explained_variance_ratio_"):
            return self._reducer.explained_variance_ratio_
        return None