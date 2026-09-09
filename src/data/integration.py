"""Data integration module for combining multiple security data sources."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.data.loaders import AuthenticationLoader, CICIDS2017Loader, NVDCVELoader, UNSWNB15Loader
from src.data.synthetic import SyntheticDataGenerator
from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataIntegrator:
    """Integrates multiple security data sources into a unified representation."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config or get_config()
        self.cicids_loader = CICIDS2017Loader()
        self.unsw_loader = UNSWNB15Loader()
        self.auth_loader = AuthenticationLoader()
        self.nvd_loader = NVDCVELoader()
        self.synthetic_generator = SyntheticDataGenerator()

    def _is_enabled(self, key: str, default: bool = True) -> bool:
        """Check if a data source is enabled supporting dot-notation."""
        if hasattr(self.config, "get"):
            val = self.config.get(key, None)
            if val is not None:
                return bool(val)
        if isinstance(self.config, dict):
            curr = self.config
            for part in key.split("."):
                if isinstance(curr, dict) and part in curr:
                    curr = curr[part]
                else:
                    return default
            return bool(curr)
        return default

    def load_all(
        self,
        use_synthetic: bool = True,
        use_cicids: bool = True,
        use_unsw: bool = True,
        use_auth: bool = True,
        use_nvd: bool = False,
        sample_per_file: Optional[int] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load all configured data sources.

        Args:
            use_synthetic: Whether to generate synthetic data.
            use_cicids: Whether to load CICIDS2017 data.
            use_unsw: Whether to load UNSW-NB15 data.
            use_auth: Whether to load real authentication / BETH data.
            use_nvd: Whether to load NVD/CVE data.
            sample_per_file: Optional row limit per loaded file to manage memory.

        Returns:
            Dictionary of source name to DataFrame.
        """
        datasets: Dict[str, pd.DataFrame] = {}

        if use_synthetic and self._is_enabled("data_sources.synthetic.enabled", True):
            try:
                datasets["synthetic"] = self.synthetic_generator.generate_all()
            except Exception as e:
                logger.warning(f"Failed to generate synthetic data: {e}")

        if use_cicids and self._is_enabled("data_sources.cicids2017.enabled", True):
            try:
                datasets["cicids2017"] = self.cicids_loader.load(sample_per_file=sample_per_file)
            except Exception as e:
                logger.warning(f"Failed to load CICIDS2017: {e}")

        if use_unsw and self._is_enabled("data_sources.unsw_nb15.enabled", True):
            try:
                datasets["unsw_nb15"] = self.unsw_loader.load(sample_per_file=sample_per_file)
            except Exception as e:
                logger.warning(f"Failed to load UNSW-NB15: {e}")

        if use_auth and self._is_enabled("data_sources.authentication.enabled", True):
            try:
                datasets["authentication"] = self.auth_loader.load(sample_per_file=sample_per_file)
            except Exception as e:
                logger.warning(f"Failed to load Authentication data: {e}")

        if use_nvd and self._is_enabled("data_sources.nvd_cve.enabled", False):
            try:
                cached_cves = self.nvd_loader.load_local()
                if not cached_cves.empty:
                    datasets["nvd_cve"] = cached_cves
            except Exception as e:
                logger.warning(f"Failed to load NVD/CVE cache: {e}")

        return datasets

    def _normalize_synthetic(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Normalize synthetic datasets into a unified schema."""
        frames: List[pd.DataFrame] = []

        # Authentication records
        if "authentication" in datasets:
            auth = datasets["authentication"].copy()
            auth["source_type"] = "authentication"
            auth["event_type"] = "auth"
            if "is_attack" in auth.columns:
                auth["is_attack"] = auth["is_attack"].astype(int)
                auth["risk_label"] = auth["is_attack"].map({1: "suspicious", 0: "normal"})
            if "attack_type" not in auth.columns:
                auth["attack_type"] = auth["is_attack"].map({1: "auth_brute_force", 0: "normal"})
            frames.append(auth)

        # Endpoint records
        if "endpoint" in datasets:
            endpoint = datasets["endpoint"].copy()
            endpoint["source_type"] = "endpoint"
            endpoint["event_type"] = "endpoint"
            if "is_attack" in endpoint.columns:
                endpoint["is_attack"] = endpoint["is_attack"].astype(int)
                endpoint["risk_label"] = endpoint["is_attack"].map({1: "suspicious", 0: "normal"})
            if "attack_type" not in endpoint.columns:
                endpoint["attack_type"] = endpoint["is_attack"].map({1: "malware_execution", 0: "normal"})
            frames.append(endpoint)

        # User behavior records
        if "user_behavior" in datasets:
            behavior = datasets["user_behavior"].copy()
            behavior["source_type"] = "user_behavior"
            behavior["event_type"] = "behavior"
            if "is_attack" in behavior.columns:
                behavior["is_attack"] = behavior["is_attack"].astype(int)
                behavior["risk_label"] = behavior["is_attack"].map({1: "suspicious", 0: "normal"})
            if "attack_type" not in behavior.columns:
                behavior["attack_type"] = behavior["is_attack"].map({1: "anomalous_access", 0: "normal"})
            frames.append(behavior)

        if not frames:
            return pd.DataFrame()

        return pd.concat(frames, ignore_index=True)

    def _normalize_network(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """Normalize network dataset (CICIDS2017/UNSW-NB15) into unified schema."""
        normalized = df.copy()
        normalized["source_type"] = source
        normalized["event_type"] = "network"

        # Determine label column
        label_col = None
        for col in ["Label", "label", "attack_cat", "class"]:
            if col in normalized.columns:
                label_col = col
                break

        if label_col is None:
            normalized["is_attack"] = 0
            normalized["risk_label"] = "normal"
            normalized["attack_type"] = "normal"
        else:
            normal_values = {"BENIGN", "Normal", "normal", "0", 0, "benign"}
            normalized["is_attack"] = normalized[label_col].apply(
                lambda x: 0 if str(x).strip() in normal_values else 1
            )
            normalized["risk_label"] = normalized["is_attack"].map({1: "suspicious", 0: "normal"})
            normalized["attack_type"] = normalized[label_col].apply(
                lambda x: "normal" if str(x).strip() in normal_values else str(x).strip()
            )

        return normalized

    def _normalize_authentication(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize real authentication / host log dataset (e.g. BETH dataset)."""
        normalized = df.copy()
        normalized["source_type"] = "authentication"
        normalized["event_type"] = "auth"

        # BETH format: sus (1=suspicious), evil (1=malicious)
        if "evil" in normalized.columns or "sus" in normalized.columns:
            evil_series = normalized["evil"] if "evil" in normalized.columns else pd.Series(0, index=normalized.index)
            sus_series = normalized["sus"] if "sus" in normalized.columns else pd.Series(0, index=normalized.index)
            normalized["is_attack"] = ((evil_series == 1) | (sus_series == 1)).astype(int)
            normalized["attack_type"] = np.where(
                evil_series == 1,
                "host_malicious",
                np.where(sus_series == 1, "host_suspicious", "normal"),
            )
        elif "is_attack" in normalized.columns:
            normalized["is_attack"] = normalized["is_attack"].astype(int)
            normalized["attack_type"] = normalized.get("attack_type", "auth_event")
        elif "label" in normalized.columns:
            normalized["is_attack"] = (normalized["label"].astype(str) != "0").astype(int)
            normalized["attack_type"] = normalized["is_attack"].map({1: "auth_anomaly", 0: "normal"})
        else:
            normalized["is_attack"] = 0
            normalized["attack_type"] = "normal"

        normalized["risk_label"] = normalized["is_attack"].map({1: "suspicious", 0: "normal"})

        # Map identifier columns if present
        if "hostName" in normalized.columns and "host_id" not in normalized.columns:
            normalized["host_id"] = normalized["hostName"].astype(str)
        if "userId" in normalized.columns and "user_id" not in normalized.columns:
            normalized["user_id"] = normalized["userId"].astype(str)
        if "processName" in normalized.columns and "process_name" not in normalized.columns:
            normalized["process_name"] = normalized["processName"].astype(str)

        return normalized

    def integrate(
        self,
        datasets: Optional[Dict[str, pd.DataFrame]] = None,
        load_if_missing: bool = True,
        sample_per_source: Optional[int] = None,
        use_synthetic: bool = True,
    ) -> pd.DataFrame:
        """Integrate all data sources into a unified DataFrame.

        Args:
            datasets: Pre-loaded datasets. If None, loads them.
            load_if_missing: Whether to load missing datasets.
            sample_per_source: Optional row sampling per source to control memory footprint.
            use_synthetic: Whether to load synthetic telemetry data.

        Returns:
            Unified DataFrame with a common schema.
        """
        if datasets is None:
            if not load_if_missing:
                return pd.DataFrame()
            datasets = self.load_all(use_synthetic=use_synthetic, sample_per_file=sample_per_source)

        frames: List[pd.DataFrame] = []

        # Synthetic data
        if "synthetic" in datasets:
            synthetic = datasets["synthetic"]
            if isinstance(synthetic, dict):
                frames.append(self._normalize_synthetic(synthetic))
            elif isinstance(synthetic, pd.DataFrame):
                frames.append(synthetic)

        # Real Network datasets
        if "cicids2017" in datasets:
            frames.append(self._normalize_network(datasets["cicids2017"], "cicids2017"))

        if "unsw_nb15" in datasets:
            frames.append(self._normalize_network(datasets["unsw_nb15"], "unsw_nb15"))

        # Real Authentication datasets
        if "authentication" in datasets and not isinstance(datasets.get("synthetic"), dict):
            frames.append(self._normalize_authentication(datasets["authentication"]))

        if not frames:
            logger.warning("No data available for integration")
            return pd.DataFrame()

        integrated = pd.concat(frames, ignore_index=True)
        # Ensure infinity values are removed
        integrated = integrated.replace([np.inf, -np.inf], np.nan)
        logger.info(f"Integrated {len(integrated)} records from {len(frames)} sources")
        return integrated

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare a unified DataFrame for machine learning by selecting numeric features.

        Args:
            df: Unified DataFrame from integrate().

        Returns:
            DataFrame with only numeric feature columns and the risk label.
        """
        if df.empty:
            return df

        # Drop non-numeric metadata columns that aren't useful for ML
        drop_cols = [
            "timestamp", "user_id", "source_ip", "destination_host", "host_id",
            "process_name", "parent_process", "location", "auth_method",
            "attack_type", "source_type", "event_type", "cve_id", "description",
            "cwe_ids", "cvss_vector", "hostName", "processName", "eventName",
            "args", "stackAddresses", "mountNamespace", "id", "Label", "label",
            "attack_cat", "class", "is_outlier", "sus", "evil",
        ]
        drop_cols = [c for c in drop_cols if c in df.columns]

        feature_df = df.drop(columns=drop_cols, errors="ignore").copy()

        # Convert remaining object columns to categorical codes
        for col in feature_df.columns:
            if col not in ["is_attack", "risk_label"] and feature_df[col].dtype == "object":
                feature_df[col] = pd.Categorical(feature_df[col].astype(str)).codes

        # Ensure risk_label is present
        if "risk_label" not in feature_df.columns:
            if "is_attack" in feature_df.columns:
                feature_df["risk_label"] = feature_df["is_attack"].map({1: "suspicious", 0: "normal"})
            else:
                feature_df["risk_label"] = "unknown"

        # Replace non-finite values with 0
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan).fillna(0)

        return feature_df