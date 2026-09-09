"""Data cleaning: missing value imputation and outlier detection."""

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataCleaner:
    """Cleans data by handling missing values and outliers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("preprocessing")
        self.missing_strategy = self.config.get("missing_value_strategy", "median")
        self.outlier_method = self.config.get("outlier_detection", "iqr")
        self.outlier_threshold = self.config.get("outlier_threshold", 1.5)

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using the configured strategy.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with missing values handled.
        """
        if df.empty:
            return df

        df = df.copy()
        missing_cols = df.columns[df.isnull().any()].tolist()

        if not missing_cols:
            return df

        logger.info(f"Handling missing values in columns: {missing_cols}")

        for col in missing_cols:
            if self.missing_strategy == "drop":
                df = df.dropna(subset=[col])
            elif pd.api.types.is_numeric_dtype(df[col]):
                if self.missing_strategy == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif self.missing_strategy == "median":
                    df[col] = df[col].fillna(df[col].median())
                else:
                    logger.warning(f"Strategy {self.missing_strategy} not applicable to numeric column {col}. Using median.")
                    df[col] = df[col].fillna(df[col].median())
            else:
                # Non-numeric columns: use mode
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

        return df

    def detect_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and optionally flag outliers in numeric columns.

        Args:
            df: Input DataFrame.

        Returns:
            DataFrame with an 'is_outlier' column added.
        """
        if df.empty:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if not numeric_cols:
            df["is_outlier"] = False
            return df

        X = df[numeric_cols].fillna(0)

        if self.outlier_method == "iqr":
            outlier_mask = np.zeros(len(df), dtype=bool)
            for col in numeric_cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
                outlier_mask |= (X[col] < lower) | (X[col] > upper)

        elif self.outlier_method == "zscore":
            z_scores = np.abs((X - X.mean()) / X.std().replace(0, 1))
            outlier_mask = (z_scores > self.outlier_threshold).any(axis=1).values

        elif self.outlier_method == "isolation_forest":
            contamination = self.config.get("outlier_contamination", 0.1)
            iso_forest = IsolationForest(
                contamination=contamination,
                random_state=self.config.get("random_state", 42),
                n_jobs=-1,
            )
            outlier_mask = iso_forest.fit_predict(X) == -1

        else:
            logger.warning(f"Unknown outlier detection method: {self.outlier_method}. Using IQR.")
            outlier_mask = np.zeros(len(df), dtype=bool)
            for col in numeric_cols:
                q1 = X[col].quantile(0.25)
                q3 = X[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - self.outlier_threshold * iqr
                upper = q3 + self.outlier_threshold * iqr
                outlier_mask |= (X[col] < lower) | (X[col] > upper)

        df["is_outlier"] = outlier_mask
        n_outliers = int(outlier_mask.sum())
        logger.info(f"Detected {n_outliers} outliers ({n_outliers / len(df) * 100:.2f}%)")
        return df

    def clean(self, df: pd.DataFrame, remove_outliers: bool = False) -> pd.DataFrame:
        """Full cleaning pipeline: missing values + optional outlier removal.

        Args:
            df: Input DataFrame.
            remove_outliers: If True, removes rows flagged as outliers.

        Returns:
            Cleaned DataFrame.
        """
        df = self.handle_missing_values(df)
        df = self.detect_outliers(df)

        if remove_outliers:
            n_before = len(df)
            df = df[~df["is_outlier"]].copy()
            logger.info(f"Removed {n_before - len(df)} outlier rows")

        return df