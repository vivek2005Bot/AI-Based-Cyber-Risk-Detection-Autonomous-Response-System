"""Data transformation: encoding and normalization."""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataTransformer:
    """Transforms data through encoding and normalization."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("preprocessing")
        self.normalization = self.config.get("normalization", "standard")
        self._encoders: Dict[str, LabelEncoder] = {}
        self._scaler = None

    def encode_categorical(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Encode categorical columns using label encoding.

        Args:
            df: Input DataFrame.
            columns: Columns to encode. If None, encodes all object columns.

        Returns:
            DataFrame with encoded columns.
        """
        if df.empty:
            return df

        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=["object", "category", "string", "str"]).columns.tolist()

        for col in columns:
            if col not in df.columns:
                continue
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]) or df[col].dtype in ["object", "category"]:
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                self._encoders[col] = encoder

        return df

    def normalize(self, df: pd.DataFrame, columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize numeric columns using the configured method.

        Args:
            df: Input DataFrame.
            columns: Columns to normalize. If None, normalizes all numeric columns.

        Returns:
            DataFrame with normalized columns.
        """
        if df.empty:
            return df

        df = df.copy()
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude boolean columns and label columns from normalization
        columns = [
            c for c in columns
            if c in df.columns
            and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]
            and c not in ["is_attack", "is_outlier", "risk_label"]
        ]

        if not columns:
            return df

        X = df[columns].fillna(0)

        if self.normalization == "minmax":
            self._scaler = MinMaxScaler()
        elif self.normalization == "robust":
            self._scaler = RobustScaler()
        else:  # standard
            self._scaler = StandardScaler()

        df[columns] = self._scaler.fit_transform(X)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Full transformation pipeline: encoding + normalization.

        Args:
            df: Input DataFrame.

        Returns:
            Transformed DataFrame.
        """
        df = self.encode_categorical(df)
        df = self.normalize(df)
        return df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform normalized columns back to original scale.

        Args:
            df: Normalized DataFrame.

        Returns:
            DataFrame with original scale values.
        """
        if df.empty or self._scaler is None:
            return df

        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [
            c for c in numeric_cols
            if c not in ["is_attack", "is_outlier", "risk_label"]
        ]

        if numeric_cols:
            df[numeric_cols] = self._scaler.inverse_transform(df[numeric_cols])

        return df