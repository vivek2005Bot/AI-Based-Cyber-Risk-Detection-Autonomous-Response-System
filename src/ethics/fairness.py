"""Algorithmic fairness and bias auditing for cyber risk prediction.

Audits whether automated risk scores or attack predictions disproportionately
impact specific user groups, departments, geographic locations, or roles.
"""

from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

from src.utils.logging import get_logger

logger = get_logger(__name__)


class FairnessAuditor:
    """Audits ML models for demographic disparity, disparate impact, and equalized odds."""

    def __init__(self, disparity_threshold: float = 0.20, min_group_size: int = 10):
        self.disparity_threshold = disparity_threshold
        self.min_group_size = min_group_size

    def audit(
        self,
        y_true: Union[np.ndarray, pd.Series, List[int]],
        y_pred: Union[np.ndarray, pd.Series, List[int]],
        sensitive_attribute: Union[np.ndarray, pd.Series, List[Any]],
        privileged_group: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Audit predictions across sensitive attribute groups.

        Args:
            y_true: Ground truth binary risk labels (0 or 1).
            y_pred: Predicted binary risk labels (0 or 1).
            sensitive_attribute: Categorical attribute defining demographic/cohort groups.
            privileged_group: Baseline reference group (e.g., 'office_a', 'internal_user').
                              If None, the largest group is selected as reference.

        Returns:
            Dictionary containing group metrics, disparity metrics, and overall compliance status.
        """
        y_true_arr = np.asarray(y_true).astype(int)
        y_pred_arr = np.asarray(y_pred).astype(int)
        groups = np.asarray(sensitive_attribute)

        unique_groups, group_counts = np.unique(groups, return_counts=True)
        valid_mask = np.isin(groups, unique_groups[group_counts >= self.min_group_size])

        if not np.any(valid_mask):
            return {
                "status": "INSUFFICIENT_DATA",
                "message": f"No group has at least {self.min_group_size} samples for fairness analysis.",
            }

        if privileged_group is None or privileged_group not in unique_groups:
            # Default to the most populated group
            privileged_group = unique_groups[np.argmax(group_counts)]

        group_metrics: Dict[str, Dict[str, float]] = {}

        for grp in unique_groups:
            idx = np.where(groups == grp)[0]
            if len(idx) < 2:
                continue

            yt = y_true_arr[idx]
            yp = y_pred_arr[idx]

            selection_rate = float(np.mean(yp))
            base_rate = float(np.mean(yt))

            # Confusion matrix metrics
            tp = int(np.sum((yt == 1) & (yp == 1)))
            fp = int(np.sum((yt == 0) & (yp == 1)))
            tn = int(np.sum((yt == 0) & (yp == 0)))
            fn = int(np.sum((yt == 1) & (yp == 0)))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            accuracy = (tp + tn) / len(yt) if len(yt) > 0 else 0.0

            group_metrics[str(grp)] = {
                "count": int(len(idx)),
                "selection_rate": round(selection_rate, 4),
                "actual_rate": round(base_rate, 4),
                "tpr_recall": round(tpr, 4),
                "fpr": round(fpr, 4),
                "accuracy": round(accuracy, 4),
            }

        ref_metrics = group_metrics.get(str(privileged_group), {})
        ref_selection_rate = ref_metrics.get("selection_rate", 0.0)
        ref_tpr = ref_metrics.get("tpr_recall", 0.0)
        ref_fpr = ref_metrics.get("fpr", 0.0)

        disparities: Dict[str, Dict[str, Any]] = {}
        violations = []

        for grp_str, m in group_metrics.items():
            if grp_str == str(privileged_group):
                continue

            # Disparate Impact Ratio (Selection rate ratio)
            if ref_selection_rate > 0:
                dir_ratio = m["selection_rate"] / ref_selection_rate
            else:
                dir_ratio = 1.0 if m["selection_rate"] == 0 else 2.0

            demographic_parity_diff = abs(m["selection_rate"] - ref_selection_rate)
            equal_opp_diff = abs(m["tpr_recall"] - ref_tpr)
            fpr_parity_diff = abs(m["fpr"] - ref_fpr)

            # Four-fifths rule check (ratio should be between 0.8 and 1.25)
            is_dir_compliant = 0.80 <= dir_ratio <= 1.25
            is_parity_compliant = demographic_parity_diff <= self.disparity_threshold

            if not is_dir_compliant or not is_parity_compliant:
                violations.append(
                    f"Group '{grp_str}' flagged: Disparate Impact Ratio={dir_ratio:.2f}, "
                    f"Demographic Parity Diff={demographic_parity_diff:.2f}"
                )

            disparities[grp_str] = {
                "disparate_impact_ratio": round(dir_ratio, 4),
                "demographic_parity_difference": round(demographic_parity_diff, 4),
                "equal_opportunity_difference": round(equal_opp_diff, 4),
                "false_positive_rate_difference": round(fpr_parity_diff, 4),
                "compliant_four_fifths": bool(is_dir_compliant),
            }

        status = "COMPLIANT" if len(violations) == 0 else "WARNING"
        logger.info(f"Fairness audit completed with status: {status}")

        return {
            "status": status,
            "reference_group": str(privileged_group),
            "group_metrics": group_metrics,
            "disparity_analysis": disparities,
            "violations": violations,
            "summary": {
                "groups_evaluated": len(group_metrics),
                "total_violations": len(violations),
                "fairness_rating": "HIGH" if len(violations) == 0 else ("MODERATE" if len(violations) <= 2 else "LOW"),
            },
        }
