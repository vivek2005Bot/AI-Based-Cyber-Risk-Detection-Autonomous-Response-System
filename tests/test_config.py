"""Tests for the configuration module."""

import pytest

from src.utils.config import Config


def test_config_loads():
    """Test that the config loads from the YAML file."""
    config = Config()
    assert config.data is not None
    assert "data_sources" in config.data


def test_config_get():
    """Test the get method with dot notation."""
    config = Config()
    assert config.get("data_sources.synthetic.enabled") is True
    assert config.get("preprocessing.random_state") == 42


def test_config_get_default():
    """Test the get method returns default for missing keys."""
    config = Config()
    assert config.get("nonexistent.key", "default") == "default"


def test_config_get_section():
    """Test getting an entire section."""
    config = Config()
    section = config.get_section("preprocessing")
    assert "missing_value_strategy" in section