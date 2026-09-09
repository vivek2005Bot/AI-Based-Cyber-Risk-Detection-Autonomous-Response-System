"""Synthetic data generator for authentication, endpoint, and user behavior records."""

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.config import get_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SyntheticDataGenerator:
    """Generates controlled synthetic security data for testing and development."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config().get_section("data_sources.synthetic")
        self.random_state = get_config().get("preprocessing.random_state", 42)
        self.rng = np.random.default_rng(self.random_state)

        # Common entities
        self.users = [f"user_{i:03d}" for i in range(1, 101)]
        self.hosts = [f"host_{i:03d}" for i in range(1, 51)]
        self.ips = [f"192.168.{rng.integers(0, 255)}.{rng.integers(1, 255)}" for rng in [self.rng] for _ in range(50)]
        self.applications = ["webmail", "vpn", "file_server", "database", "erp", "crm", "devops", "hr_system"]
        self.locations = ["office_a", "office_b", "remote_home", "remote_travel", "datacenter"]

    def _random_timestamp(self, start: datetime, end: datetime) -> datetime:
        """Generate a random timestamp between start and end."""
        delta = end - start
        random_seconds = self.rng.integers(0, int(delta.total_seconds()))
        return start + timedelta(seconds=int(random_seconds))

    def generate_authentication_records(self, n_records: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic authentication log records.

        Args:
            n_records: Number of records to generate. Defaults to config value.

        Returns:
            DataFrame of authentication records.
        """
        n = n_records or self.config.get("authentication_records", 10000)
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        records = []
        for _ in range(n):
            is_attack = self.rng.random() < 0.05  # 5% attack rate

            if is_attack:
                # Attack patterns: brute force, credential stuffing, privilege escalation
                attack_type = self.rng.choice(["brute_force", "credential_stuffing", "privilege_escalation", "replay"])
                user = self.rng.choice(self.users)
                success = self.rng.random() < 0.3
                attempts = int(self.rng.integers(5, 50))
                source_ip = f"{self.rng.integers(10, 200)}.{self.rng.integers(0, 255)}.{self.rng.integers(0, 255)}.{self.rng.integers(1, 255)}"
                location = "remote_travel" if self.rng.random() < 0.7 else "remote_home"
                auth_method = self.rng.choice(["password", "password", "password", "token"])
                unusual_hour = self.rng.random() < 0.8
            else:
                attack_type = "normal"
                user = self.rng.choice(self.users)
                success = self.rng.random() < 0.97
                attempts = int(self.rng.integers(1, 3))
                source_ip = self.rng.choice(self.ips)
                location = self.rng.choice(self.locations)
                auth_method = self.rng.choice(["password", "token", "biometric", "sso", "mfa"])
                unusual_hour = self.rng.random() < 0.05

            timestamp = self._random_timestamp(start, end)
            hour = timestamp.hour

            records.append({
                "timestamp": timestamp,
                "user_id": user,
                "source_ip": source_ip,
                "destination_host": self.rng.choice(self.hosts),
                "auth_method": auth_method,
                "success": success,
                "attempts": attempts,
                "location": location,
                "hour_of_day": hour,
                "is_weekend": timestamp.weekday() >= 5,
                "unusual_hour": unusual_hour,
                "failed_attempts_24h": int(self.rng.integers(0, 3)) if not is_attack else int(self.rng.integers(3, 100)),
                "attack_type": attack_type,
                "is_attack": is_attack,
            })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} authentication records")
        return df

    def generate_endpoint_records(self, n_records: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic endpoint telemetry records.

        Args:
            n_records: Number of records to generate. Defaults to config value.

        Returns:
            DataFrame of endpoint records.
        """
        n = n_records or self.config.get("endpoint_records", 10000)
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        processes = ["chrome.exe", "firefox.exe", "word.exe", "excel.exe", "powershell.exe", "cmd.exe", "python.exe", "svchost.exe", "explorer.exe", "outlook.exe", "winword.exe", "msedge.exe"]
        malicious_processes = ["mimikatz.exe", "powershell.exe", "cmd.exe", "rundll32.exe", "wscript.exe", "mshta.exe", "regsvr32.exe"]

        records = []
        for _ in range(n):
            is_attack = self.rng.random() < 0.05

            if is_attack:
                attack_type = self.rng.choice(["malware", "ransomware", "lateral_movement", "data_exfiltration", "crypto_mining"])
                process = self.rng.choice(malicious_processes)
                cpu_usage = float(self.rng.uniform(50, 100))
                memory_usage = float(self.rng.uniform(60, 100))
                network_connections = int(self.rng.integers(10, 100))
                file_operations = int(self.rng.integers(50, 500))
                registry_changes = int(self.rng.integers(10, 100))
                unusual_process = True
                high_privilege = self.rng.random() < 0.7
            else:
                attack_type = "normal"
                process = self.rng.choice(processes)
                cpu_usage = float(self.rng.uniform(0, 40))
                memory_usage = float(self.rng.uniform(5, 60))
                network_connections = int(self.rng.integers(0, 15))
                file_operations = int(self.rng.integers(0, 50))
                registry_changes = int(self.rng.integers(0, 10))
                unusual_process = self.rng.random() < 0.02
                high_privilege = self.rng.random() < 0.1

            records.append({
                "timestamp": self._random_timestamp(start, end),
                "host_id": self.rng.choice(self.hosts),
                "process_name": process,
                "process_id": int(self.rng.integers(1000, 99999)),
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "network_connections": network_connections,
                "file_operations": file_operations,
                "registry_changes": registry_changes,
                "unusual_process": unusual_process,
                "high_privilege": high_privilege,
                "parent_process": self.rng.choice(processes),
                "attack_type": attack_type,
                "is_attack": is_attack,
            })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} endpoint records")
        return df

    def generate_user_behavior_records(self, n_records: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic user behavior records.

        Args:
            n_records: Number of records to generate. Defaults to config value.

        Returns:
            DataFrame of user behavior records.
        """
        n = n_records or self.config.get("user_behavior_records", 10000)
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()

        records = []
        for _ in range(n):
            is_attack = self.rng.random() < 0.05

            if is_attack:
                attack_type = self.rng.choice(["data_exfiltration", "insider_threat", "account_takeover", "policy_violation"])
                files_accessed = int(self.rng.integers(20, 200))
                sensitive_files = int(self.rng.integers(5, 50))
                data_transferred_mb = float(self.rng.uniform(100, 5000))
                unusual_hours = self.rng.random() < 0.8
                new_device = self.rng.random() < 0.7
                new_location = self.rng.random() < 0.6
                apps_used = int(self.rng.integers(1, 3))
            else:
                attack_type = "normal"
                files_accessed = int(self.rng.integers(1, 30))
                sensitive_files = int(self.rng.integers(0, 3))
                data_transferred_mb = float(self.rng.uniform(0, 50))
                unusual_hours = self.rng.random() < 0.05
                new_device = self.rng.random() < 0.05
                new_location = self.rng.random() < 0.03
                apps_used = int(self.rng.integers(2, 6))

            records.append({
                "timestamp": self._random_timestamp(start, end),
                "user_id": self.rng.choice(self.users),
                "files_accessed": files_accessed,
                "sensitive_files_accessed": sensitive_files,
                "data_transferred_mb": data_transferred_mb,
                "applications_used": apps_used,
                "unusual_hours": unusual_hours,
                "new_device": new_device,
                "new_location": new_location,
                "login_frequency_7d": int(self.rng.integers(5, 50)) if not is_attack else int(self.rng.integers(1, 5)),
                "attack_type": attack_type,
                "is_attack": is_attack,
            })

        df = pd.DataFrame(records)
        logger.info(f"Generated {len(df)} user behavior records")
        return df

    def generate_all(self, n_records: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """Generate all synthetic datasets.

        Args:
            n_records: Optional number of records per dataset. If None, uses config defaults.

        Returns:
            Dictionary of dataset name to DataFrame.
        """
        return {
            "authentication": self.generate_authentication_records(n_records=n_records),
            "endpoint": self.generate_endpoint_records(n_records=n_records),
            "user_behavior": self.generate_user_behavior_records(n_records=n_records),
        }

    def save_all(self, output_dir: Optional[str | Path] = None) -> Dict[str, Path]:
        """Generate and save all synthetic datasets to CSV files.

        Args:
            output_dir: Directory to save files. Defaults to config local_path.

        Returns:
            Dictionary of dataset name to saved file path.
        """
        output_dir = Path(output_dir) if output_dir else Path(self.config.get("local_path", "data/raw/synthetic"))
        output_dir.mkdir(parents=True, exist_ok=True)

        datasets = self.generate_all()
        saved_paths: Dict[str, Path] = {}

        for name, df in datasets.items():
            path = output_dir / f"{name}.csv"
            df.to_csv(path, index=False)
            saved_paths[name] = path
            logger.info(f"Saved {name} data to {path}")

        return saved_paths