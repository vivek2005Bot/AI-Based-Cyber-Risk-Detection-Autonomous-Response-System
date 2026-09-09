"""Configuration loader for the cyber risk detection system."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config:
    """Loads and provides access to the YAML configuration."""

    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._data: Dict[str, Any] = {}
        self.load()

    def load(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        with open(self.config_path, "r") as f:
            self._data = yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'data_sources.cicids2017.enabled')."""
        value: Any = self._data
        for part in key.split("."):
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire configuration section as a dictionary."""
        return self.get(section, {})

    @property
    def data(self) -> Dict[str, Any]:
        """Return the full configuration dictionary."""
        return self._data

    def __getitem__(self, key: str) -> Any:
        return self.get(key)


# Singleton instance for convenience
_config: Config | None = None


def get_config(config_path: str | Path | None = None) -> Config:
    """Return a singleton Config instance."""
    global _config
    if _config is None or config_path is not None:
        _config = Config(config_path)
    return _config