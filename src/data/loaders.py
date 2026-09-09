"""Data loaders for CICIDS2017, UNSW-NB15, and NVD/CVE datasets."""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


import numpy as np


class CICIDS2017Loader:
    """Loader for the CICIDS2017 intrusion detection dataset."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("data_sources.cicids2017")
        self.local_path = Path(self.config.get("local_path", "data/raw/cicids2017"))

    def load(
        self,
        file_names: Optional[List[str]] = None,
        sample_per_file: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load CICIDS2017 CSV files into a single DataFrame."""
        files = file_names or self.config.get("files", [])
        candidate_files = []
        for file_name in files:
            file_path = self.local_path / file_name
            if file_path.exists():
                candidate_files.append(file_path)

        if not candidate_files and self.local_path.exists():
            candidate_files = sorted(list(self.local_path.glob("*.csv")))

        frames: List[pd.DataFrame] = []
        for file_path in candidate_files:
            logger.info(f"Loading CICIDS2017 file: {file_path.name}")
            try:
                if sample_per_file:
                    df = pd.read_csv(file_path, nrows=sample_per_file)
                else:
                    df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                for col in ["Label", "label"]:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.replace("\ufffd", "-").str.strip()
                df = df.replace([np.inf, -np.inf], np.nan)
                frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not frames:
            raise FileNotFoundError(
                f"No CICIDS2017 data files found in {self.local_path}."
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows from {len(frames)} CICIDS2017 files")
        return combined


class UNSWNB15Loader:
    """Loader for the UNSW-NB15 network intrusion dataset."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("data_sources.unsw_nb15")
        self.local_path = Path(self.config.get("local_path", "data/raw/unsw_nb15"))

    def load(
        self,
        file_names: Optional[List[str]] = None,
        sample_per_file: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load UNSW-NB15 CSV files into a single DataFrame."""
        files = file_names or self.config.get("files", [])
        candidate_files = []
        for file_name in files:
            file_path = self.local_path / file_name
            if file_path.exists():
                candidate_files.append(file_path)

        if not candidate_files and self.local_path.exists():
            candidate_files = sorted(
                [
                    f for f in self.local_path.glob("*.csv")
                    if "features" not in f.name.lower() and "events" not in f.name.lower()
                ]
            )

        frames: List[pd.DataFrame] = []
        for file_path in candidate_files:
            logger.info(f"Loading UNSW-NB15 file: {file_path.name}")
            try:
                if sample_per_file:
                    df = pd.read_csv(file_path, nrows=sample_per_file)
                else:
                    df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                df = df.replace([np.inf, -np.inf], np.nan)
                frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not frames:
            raise FileNotFoundError(
                f"No UNSW-NB15 data files found in {self.local_path}."
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows from {len(frames)} UNSW-NB15 files")
        return combined


class AuthenticationLoader:
    """Loader for real Authentication & Endpoint telemetry datasets (BETH, LANL, Sysmon)."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("data_sources.authentication")
        self.local_path = Path(self.config.get("local_path", "data/raw/authentication"))

    def load(
        self,
        file_names: Optional[List[str]] = None,
        sample_per_file: Optional[int] = None,
    ) -> pd.DataFrame:
        """Load Authentication CSV files into a single DataFrame."""
        files = file_names or self.config.get("files", [])
        candidate_files = []
        for file_name in files:
            file_path = self.local_path / file_name
            if file_path.exists():
                candidate_files.append(file_path)

        if not candidate_files and self.local_path.exists():
            candidate_files = sorted(list(self.local_path.glob("*.csv")))

        frames: List[pd.DataFrame] = []
        for file_path in candidate_files:
            logger.info(f"Loading Authentication file: {file_path.name}")
            try:
                if sample_per_file:
                    df = pd.read_csv(file_path, nrows=sample_per_file)
                else:
                    df = pd.read_csv(file_path)
                df.columns = df.columns.str.strip()
                df = df.replace([np.inf, -np.inf], np.nan)
                frames.append(df)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        if not frames:
            raise FileNotFoundError(
                f"No Authentication data files found in {self.local_path}."
            )

        combined = pd.concat(frames, ignore_index=True)
        logger.info(f"Loaded {len(combined)} rows from {len(frames)} Authentication files")
        return combined


class NVDCVELoader:
    """Loader for NVD/CVE vulnerability data from the NIST NVD API."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("data_sources.nvd_cve")
        self.api_url = self.config.get("api_url", "https://services.nvd.nist.gov/rest/json/cves/2.0")
        self.local_path = Path(self.config.get("local_path", "data/raw/nvd_cve"))
        self.local_path.mkdir(parents=True, exist_ok=True)

    def fetch(
        self,
        results_per_page: int = 2000,
        start_index: int = 0,
        max_records: Optional[int] = None,
        delay_seconds: float = 6.0,
    ) -> pd.DataFrame:
        """Fetch CVE data from the NVD API.

        Args:
            results_per_page: Number of results per API call (max 2000).
            start_index: Starting index for pagination.
            max_records: Maximum number of records to fetch. If None, fetches all.
            delay_seconds: Delay between API calls to respect rate limits.

        Returns:
            DataFrame of CVE records.
        """
        all_records: List[Dict[str, Any]] = []
        current_index = start_index
        total_results = None

        while True:
            params = {
                "resultsPerPage": results_per_page,
                "startIndex": current_index,
            }
            logger.info(f"Fetching NVD CVEs from index {current_index}")

            try:
                response = requests.get(self.api_url, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as e:
                logger.error(f"Failed to fetch NVD data: {e}")
                break

            if total_results is None:
                total_results = data.get("totalResults", 0)
                logger.info(f"Total CVE records available: {total_results}")

            vulnerabilities = data.get("vulnerabilities", [])
            if not vulnerabilities:
                break

            for vuln in vulnerabilities:
                cve = vuln.get("cve", {})
                metrics = cve.get("metrics", {})
                cvss_v31 = metrics.get("cvssMetricV31", [{}])
                cvss_v3 = metrics.get("cvssMetricV3", [{}])
                cvss_v2 = metrics.get("cvssMetricV2", [{}])

                cvss_data = None
                if cvss_v31:
                    cvss_data = cvss_v31[0].get("cvssData", {})
                elif cvss_v3:
                    cvss_data = cvss_v3[0].get("cvssData", {})
                elif cvss_v2:
                    cvss_data = cvss_v2[0].get("cvssData", {})

                record = {
                    "cve_id": cve.get("id"),
                    "source_identifier": cve.get("sourceIdentifier"),
                    "published": cve.get("published"),
                    "last_modified": cve.get("lastModified"),
                    "vuln_status": cve.get("vulnStatus"),
                    "description": (
                        cve.get("descriptions", [{}])[0].get("value", "")
                        if cve.get("descriptions")
                        else ""
                    ),
                    "cvss_score": cvss_data.get("baseScore") if cvss_data else None,
                    "cvss_severity": cvss_data.get("baseSeverity") if cvss_data else None,
                    "cvss_vector": cvss_data.get("vectorString") if cvss_data else None,
                    "cvss_version": cvss_data.get("version") if cvss_data else None,
                    "exploitability_score": (
                        cvss_v31[0].get("exploitabilityScore")
                        if cvss_v31
                        else (cvss_v3[0].get("exploitabilityScore") if cvss_v3 else None)
                    ),
                    "impact_score": (
                        cvss_v31[0].get("impactScore")
                        if cvss_v31
                        else (cvss_v3[0].get("impactScore") if cvss_v3 else None)
                    ),
                    "cwe_ids": [
                        weakness.get("description", [{}])[0].get("value")
                        for weakness in cve.get("weaknesses", [])
                        if weakness.get("description")
                    ],
                    "references_count": len(cve.get("references", [])),
                }
                all_records.append(record)

            current_index += len(vulnerabilities)

            if max_records is not None and len(all_records) >= max_records:
                all_records = all_records[:max_records]
                break

            if current_index >= total_results:
                break

            # Respect NVD rate limits
            time.sleep(delay_seconds)

        df = pd.DataFrame(all_records)
        logger.info(f"Fetched {len(df)} CVE records")
        return df

    def load_local(self) -> pd.DataFrame:
        """Load cached CVE data from local storage."""
        cache_file = self.local_path / "cves.json"
        if not cache_file.exists():
            logger.warning(f"No cached CVE data found at {cache_file}")
            return pd.DataFrame()
        with open(cache_file, "r") as f:
            data = json.load(f)
        return pd.DataFrame(data)

    def save_local(self, df: pd.DataFrame) -> None:
        """Save CVE data to local cache."""
        cache_file = self.local_path / "cves.json"
        with open(cache_file, "w") as f:
            json.dump(df.to_dict(orient="records"), f, indent=2)
        logger.info(f"Saved {len(df)} CVE records to {cache_file}")

    def load(self, use_cache: bool = True) -> pd.DataFrame:
        """Load CVE data, using cache if available and fresh.

        Args:
            use_cache: If True, use cached data if it exists.

        Returns:
            DataFrame of CVE records.
        """
        if use_cache:
            cached = self.load_local()
            if not cached.empty:
                return cached
        df = self.fetch()
        self.save_local(df)
        return df