"""Tests for the Ethical AI, Privacy, and Adversarial Security Engine."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.ethics.fairness import FairnessAuditor
from src.ethics.privacy import PrivacyEngine
from src.ethics.security import AdversarialRobustnessAuditor


def test_fairness_auditor():
    """Test algorithmic bias detection and disparate impact auditing."""
    auditor = FairnessAuditor(disparity_threshold=0.20)
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    y_pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 5)
    sensitive_attr = np.array(["dept_a", "dept_a", "dept_a", "dept_a", "dept_a",
                               "dept_b", "dept_b", "dept_b", "dept_b", "dept_b"] * 5)

    result = auditor.audit(y_true, y_pred, sensitive_attr)
    assert result["status"] in ["COMPLIANT", "WARNING"]
    assert "group_metrics" in result
    assert "dept_a" in result["group_metrics"]
    assert "dept_b" in result["group_metrics"]


def test_privacy_engine_pseudonymization():
    """Test identifier pseudonymization and IP masking."""
    privacy = PrivacyEngine(salt="test_salt_123")
    hashed_user = privacy.hash_identifier("john_doe", prefix="usr")
    assert hashed_user.startswith("usr_")
    assert hashed_user != "john_doe"

    masked_ip = privacy.mask_ip("192.168.1.55", mode="subnet")
    assert masked_ip == "192.168.1.0/24"


def test_privacy_engine_k_anonymity_and_dp():
    """Test k-anonymity scoring and Differential Privacy report."""
    privacy = PrivacyEngine()
    df = pd.DataFrame({
        "hour_of_day": [9, 9, 9, 9, 9, 14, 14, 14, 14, 14],
        "location": ["office", "office", "office", "office", "office",
                     "remote", "remote", "remote", "remote", "remote"],
    })
    k_res = privacy.audit_k_anonymity(df, quasi_identifiers=["hour_of_day", "location"], target_k=3)
    assert k_res["k_achieved"] >= 3
    assert k_res["status"] == "COMPLIANT"

    risk_scores = np.array([0.2, 0.4, 0.7, 0.85, 0.9])
    dp_rep = privacy.differentially_private_risk_report(risk_scores, epsilon=1.0)
    assert "differentially_private_mean_risk" in dp_rep
    assert 0.0 <= dp_rep["differentially_private_mean_risk"] <= 1.0


def test_adversarial_robustness_auditor():
    """Test model perturbation and evasion attack audit."""
    auditor = AdversarialRobustnessAuditor(random_state=42)
    X = np.random.normal(0, 1, size=(60, 5))
    y = np.array([1 if x[0] > 0 else 0 for x in X])

    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    res = auditor.evaluate_evasion_robustness(model, X, y, epsilons=[0.05, 0.15])
    assert "robustness_score" in res
    assert 0.0 <= res["robustness_score"] <= 100.0
    assert "perturbation_analysis" in res
