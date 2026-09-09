"""Tests for the synthetic data generator."""

import pandas as pd
import pytest

from src.data.synthetic import SyntheticDataGenerator


@pytest.fixture
def generator():
    return SyntheticDataGenerator()


def test_generate_authentication(generator):
    """Test authentication record generation."""
    df = generator.generate_authentication_records(n_records=100)
    assert len(df) == 100
    assert "user_id" in df.columns
    assert "is_attack" in df.columns
    assert "timestamp" in df.columns


def test_generate_endpoint(generator):
    """Test endpoint record generation."""
    df = generator.generate_endpoint_records(n_records=100)
    assert len(df) == 100
    assert "host_id" in df.columns
    assert "cpu_usage" in df.columns
    assert "is_attack" in df.columns


def test_generate_user_behavior(generator):
    """Test user behavior record generation."""
    df = generator.generate_user_behavior_records(n_records=100)
    assert len(df) == 100
    assert "user_id" in df.columns
    assert "files_accessed" in df.columns
    assert "is_attack" in df.columns


def test_generate_all(generator):
    """Test generating all datasets."""
    datasets = generator.generate_all()
    assert "authentication" in datasets
    assert "endpoint" in datasets
    assert "user_behavior" in datasets
    for df in datasets.values():
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0