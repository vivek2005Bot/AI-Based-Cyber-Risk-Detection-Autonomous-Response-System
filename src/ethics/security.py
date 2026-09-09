"""Machine Learning Security & Adversarial Robustness Auditing.

Evaluates cybersecurity detection models against adversarial evasion attacks,
boundary jitter, and input manipulation.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class AdversarialRobustnessAuditor:
    """Audits ML model vulnerability to adversarial evasion attacks."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def evaluate_evasion_robustness(
        self,
        model: Any,
        X_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        epsilons: List[float] = [0.01, 0.05, 0.10, 0.20, 0.30],
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Simulate adversarial evasion attack on malicious samples.

        For each attack sample (y=1), adds calibrated adversarial perturbation
        in the feature space to observe if the model is tricked into predicting normal (y=0).

        Args:
            model: Trained classifier supporting .predict() or .predict_proba().
            X_test: Test features.
            y_test: Ground truth labels (1 = attack/suspicious, 0 = benign).
            epsilons: Normalized perturbation budgets to test.
            feature_names: Names of features for feature importance attribution.

        Returns:
            Dictionary with evasion rates, accuracy under attack, and robustness score.
        """
        if isinstance(X_test, pd.DataFrame):
            if feature_names is None:
                feature_names = X_test.columns.tolist()
            X_arr = X_test.values.astype(float)
        else:
            X_arr = np.asarray(X_test, dtype=float)

        y_arr = np.asarray(y_test).astype(int)

        # Baseline predictions
        base_preds = model.predict(X_arr).astype(int)
        baseline_acc = float(np.mean(base_preds == y_arr))

        # Focus evasion attack specifically on attack samples correctly identified
        attack_indices = np.where((y_arr == 1) & (base_preds == 1))[0]

        if len(attack_indices) == 0:
            return {
                "status": "NO_ATTACK_SAMPLES",
                "baseline_accuracy": round(baseline_acc, 4),
                "robustness_score": 100.0,
                "message": "No true positive attack samples detected to perturb.",
            }

        X_attack = X_arr[attack_indices]
        n_attacks = len(attack_indices)

        # Feature scale standard deviations
        feature_stds = np.std(X_arr, axis=0)
        feature_stds[feature_stds == 0] = 1.0

        perturbation_results: Dict[str, Dict[str, float]] = {}
        evasion_rates: List[float] = []

        for eps in epsilons:
            # Simulate gradient/boundary direction perturbation
            # Perturb each sample with direction matching negative feature weights or randomized gradient estimate
            noise = self.rng.normal(0, eps, size=X_attack.shape) * feature_stds
            X_perturbed = X_attack - np.abs(noise)  # pushing towards benign distribution

            # Predict on perturbed attack samples
            perturbed_preds = model.predict(X_perturbed).astype(int)
            evaded_count = int(np.sum(perturbed_preds == 0))
            evasion_rate = float(evaded_count / n_attacks)
            evasion_rates.append(evasion_rate)

            # Combined test set accuracy under this attack intensity
            X_full_perturbed = X_arr.copy()
            X_full_perturbed[attack_indices] = X_perturbed
            full_perturbed_preds = model.predict(X_full_perturbed).astype(int)
            perturbed_acc = float(np.mean(full_perturbed_preds == y_arr))

            perturbation_results[f"eps_{eps:.2f}"] = {
                "epsilon": eps,
                "evasion_success_rate": round(evasion_rate, 4),
                "accuracy_under_attack": round(perturbed_acc, 4),
                "evaded_count": evaded_count,
                "total_attacks_tested": n_attacks,
            }

        # Robustness score: 100 * (1 - average evasion rate across epsilons)
        avg_evasion = float(np.mean(evasion_rates))
        robustness_score = round(max(0.0, (1.0 - avg_evasion) * 100.0), 2)

        # Feature vulnerability ranking: which single feature causes highest evasion when perturbed
        feature_vulnerabilities: Dict[str, float] = {}
        if feature_names and len(feature_names) == X_arr.shape[1]:
            for i, fname in enumerate(feature_names[:15]):  # check top 15 features
                X_single = X_attack.copy()
                X_single[:, i] -= 0.15 * feature_stds[i]
                p_preds = model.predict(X_single).astype(int)
                single_evasion = float(np.mean(p_preds == 0))
                feature_vulnerabilities[fname] = round(single_evasion, 4)

            # Sort descending by vulnerability
            feature_vulnerabilities = dict(
                sorted(feature_vulnerabilities.items(), key=lambda item: item[1], reverse=True)
            )

        status = "ROBUST" if robustness_score >= 75.0 else ("MODERATE" if robustness_score >= 50.0 else "VULNERABLE")
        logger.info(f"Adversarial robustness audit completed. Score: {robustness_score}% ({status})")

        return {
            "status": status,
            "baseline_accuracy": round(baseline_acc, 4),
            "robustness_score": robustness_score,
            "mean_evasion_rate": round(avg_evasion, 4),
            "perturbation_analysis": perturbation_results,
            "top_vulnerable_features": feature_vulnerabilities,
        }
