"""Tests for the preprocessing module."""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.cleaner import DataCleaner
from src.preprocessing.transformer import DataTransformer


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "feature_a": [1.0, 2.0, np.nan, 4.0, 5.0],
        "feature_b": [10, 20, 30, 40, 50],
        "category": ["x", "y", "x", "y", "x"],
        "is_attack": [0, 1, 0, 1, 0],
    })


def test_cleaner_missing_values(sample_df):
    """Test missing value handling."""
    cleaner = DataCleaner()
    result = cleaner.handle_missing_values(sample_df)
    assert result["feature_a"].isnull().sum() == 0
    assert result["feature_a"].iloc[2] == 3.0  # median of [1,2,4,5]


def test_cleaner_outliers(sample_df):
    """Test outlier detection."""
    cleaner = DataCleaner()
    result = cleaner.detect_outliers(sample_df)
    assert "is_outlier" in result.columns
    assert result["is_outlier"].dtype == bool


def test_transformer_encode(sample_df):
    """Test categorical encoding."""
    transformer = DataTransformer()
    result = transformer.encode_categorical(sample_df)
    assert result["category"].dtype in [np.int64, np.int32]


def test_transformer_normalize(sample_df):
    """Test normalization."""
    transformer = DataTransformer()
    result = transformer.normalize(sample_df)
    # Check that numeric columns are normalized (mean ~ 0, std ~ 1)
    assert abs(result["feature_a"].mean()) < 1e-10
    assert abs(result["feature_a"].std(ddof=0) - 1.0) < 1e-10