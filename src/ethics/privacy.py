"""Privacy preservation, data anonymization, and differential privacy utilities."""

import hashlib
import re
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class PrivacyEngine:
    """Provides PII/IP pseudonymization, K-anonymity auditing, and Differential Privacy."""

    def __init__(self, salt: str = "cyber_risk_privacy_salt_2026"):
        self.salt = salt

    def hash_identifier(self, identifier: str, prefix: str = "anon") -> str:
        """Cryptographically pseudonymize an identifier using salted SHA-256."""
        if not identifier or pd.isna(identifier):
            return f"{prefix}_unknown"
        salted = f"{self.salt}_{str(identifier)}".encode("utf-8")
        digest = hashlib.sha256(salted).hexdigest()[:12]
        return f"{prefix}_{digest}"

    def mask_ip(self, ip_address: str, mode: str = "subnet") -> str:
        """Anonymize IP address by subnet truncation (/24) or full hashing."""
        if not ip_address or pd.isna(ip_address):
            return "0.0.0.0"

        ip_str = str(ip_address).strip()
        if mode == "subnet":
            # Mask the last octet (e.g., 192.168.1.45 -> 192.168.1.0/24)
            parts = ip_str.split(".")
            if len(parts) == 4:
                return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
            return self.hash_identifier(ip_str, prefix="ip")
        elif mode == "hash":
            return self.hash_identifier(ip_str, prefix="ip")
        return ip_str

    def anonymize_dataframe(
        self,
        df: pd.DataFrame,
        ip_columns: Optional[List[str]] = None,
        user_columns: Optional[List[str]] = None,
        host_columns: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Produce an anonymized copy of a security telemetry DataFrame."""
        anonymized = df.copy()

        ip_cols = ip_columns or ["source_ip", "destination_ip", "dest_ip"]
        for col in ip_cols:
            if col in anonymized.columns:
                anonymized[col] = anonymized[col].apply(lambda x: self.mask_ip(x, mode="subnet"))

        user_cols = user_columns or ["user_id", "username", "account"]
        for col in user_cols:
            if col in anonymized.columns:
                anonymized[col] = anonymized[col].apply(lambda x: self.hash_identifier(str(x), prefix="usr"))

        host_cols = host_columns or ["host_id", "hostname", "destination_host"]
        for col in host_cols:
            if col in anonymized.columns:
                anonymized[col] = anonymized[col].apply(lambda x: self.hash_identifier(str(x), prefix="hst"))

        logger.info(f"Anonymized {len(anonymized)} records across identifiers.")
        return anonymized

    def audit_k_anonymity(
        self,
        df: pd.DataFrame,
        quasi_identifiers: List[str],
        target_k: int = 5,
    ) -> Dict[str, Any]:
        """Audit the k-anonymity score of a dataset given quasi-identifiers."""
        available_qi = [col for col in quasi_identifiers if col in df.columns]
        if not available_qi:
            return {
                "status": "ERROR",
                "message": "No specified quasi-identifiers found in DataFrame.",
                "k_value": 0,
            }

        grouped = df.groupby(available_qi, observed=False).size()
        min_group_size = int(grouped.min()) if len(grouped) > 0 else 0
        mean_group_size = float(grouped.mean()) if len(grouped) > 0 else 0.0

        # Percentage of records belonging to equivalence classes with size < target_k
        violating_groups = grouped[grouped < target_k]
        violating_records_count = int(violating_groups.sum()) if len(violating_groups) > 0 else 0
        total_records = len(df)
        reidentification_risk_rate = violating_records_count / total_records if total_records > 0 else 0.0

        return {
            "status": "COMPLIANT" if min_group_size >= target_k else "VULNERABLE",
            "k_achieved": min_group_size,
            "target_k": target_k,
            "mean_equivalence_class_size": round(mean_group_size, 2),
            "violating_records_count": violating_records_count,
            "reidentification_risk_rate": round(reidentification_risk_rate, 4),
            "quasi_identifiers_evaluated": available_qi,
        }

    def laplace_noise(self, scale: float, size: Optional[int] = None) -> Union[float, np.ndarray]:
        """Generate Laplace noise with given scale (b = sensitivity / epsilon)."""
        rng = np.random.default_rng()
        if size is None:
            u = rng.uniform(-0.5, 0.5)
            return -scale * np.sign(u) * np.log(1 - 2 * abs(u) + 1e-12)
        u = rng.uniform(-0.5, 0.5, size=size)
        return -scale * np.sign(u) * np.log(1 - 2 * np.abs(u) + 1e-12)

    def differential_privacy_metric(
        self,
        true_value: float,
        epsilon: float = 1.0,
        sensitivity: float = 1.0,
    ) -> float:
        """Apply the Laplace Mechanism to release a differentially private scalar query."""
        if epsilon <= 0:
            raise ValueError("Privacy parameter epsilon must be positive.")
        scale = sensitivity / epsilon
        noise = float(self.laplace_noise(scale=scale))
        return float(true_value + noise)

    def differentially_private_risk_report(
        self,
        risk_scores: np.ndarray,
        epsilon: float = 1.0,
    ) -> Dict[str, Any]:
        """Generate differentially private summary statistics for risk scores."""
        scores = np.asarray(risk_scores, dtype=float)
        n = len(scores)
        if n == 0:
            return {"noisy_mean_risk": 0.0, "noisy_critical_count": 0}

        true_mean = float(np.mean(scores))
        true_critical = int(np.sum(scores >= 0.8))

        # Sensitivity for mean risk is 1.0 / n (since risk score is bounded in [0, 1])
        sensitivity_mean = 1.0 / max(n, 1)
        noisy_mean = self.differential_privacy_metric(true_mean, epsilon=epsilon * 0.5, sensitivity=sensitivity_mean)

        # Sensitivity for count is 1.0
        noisy_critical = max(0, int(round(self.differential_privacy_metric(true_critical, epsilon=epsilon * 0.5, sensitivity=1.0))))

        return {
            "epsilon_budget": epsilon,
            "true_mean_risk": round(true_mean, 4),
            "differentially_private_mean_risk": round(float(np.clip(noisy_mean, 0.0, 1.0)), 4),
            "true_critical_incidents": true_critical,
            "differentially_private_critical_incidents": noisy_critical,
            "noise_level": round(sensitivity_mean / (epsilon * 0.5), 6),
        }
