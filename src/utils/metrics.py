"""Evaluation metrics for classification, clustering, and reinforcement learning."""

from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    calinski_harabasz_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    labels: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities (for ROC-AUC / PR-AUC).
        labels: Optional label names.

    Returns:
        Dictionary of metric names to values.
    """
    metrics: Dict[str, Any] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "matthews_corrcoef": matthews_corrcoef(y_true, y_pred),
        "cohen_kappa": cohen_kappa_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if labels is not None:
        metrics["classification_report"] = classification_report(
            y_true, y_pred, target_names=labels, output_dict=True, zero_division=0
        )

    if y_prob is not None:
        # Binary classification
        if y_prob.ndim == 1 or y_prob.shape[1] == 2:
            prob_positive = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, prob_positive)
            except ValueError:
                metrics["roc_auc"] = None
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, prob_positive)
            _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
            metrics["precision_recall_auc"] = float(
                _trapz(recall_curve, precision_curve)
            ) if _trapz is not None else 0.0
        # Multiclass
        else:
            try:
                metrics["roc_auc"] = roc_auc_score(
                    y_true, y_prob, multi_class="ovr", average="weighted"
                )
            except ValueError:
                metrics["roc_auc"] = None

    return metrics


def clustering_metrics(
    X: np.ndarray, labels: np.ndarray
) -> Dict[str, float]:
    """Compute clustering quality metrics.

    Args:
        X: Feature matrix.
        labels: Cluster labels.

    Returns:
        Dictionary of clustering metric names to values.
    """
    metrics: Dict[str, float] = {}

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    if n_clusters < 2:
        return metrics

    try:
        sample_size = 5000 if len(X) > 5000 else None
        metrics["silhouette_score"] = float(silhouette_score(X, labels, sample_size=sample_size, random_state=42))
    except (ValueError, MemoryError):
        metrics["silhouette_score"] = None

    try:
        metrics["calinski_harabasz"] = float(calinski_harabasz_score(X, labels))
    except ValueError:
        metrics["calinski_harabasz"] = None

    try:
        metrics["davies_bouldin"] = float(davies_bouldin_score(X, labels))
    except ValueError:
        metrics["davies_bouldin"] = None

    return metrics


def rl_metrics(
    episode_rewards: List[float],
    episode_lengths: List[int],
    success_episodes: Optional[List[bool]] = None,
) -> Dict[str, float]:
    """Compute reinforcement learning evaluation metrics.

    Args:
        episode_rewards: Total reward per episode.
        episode_lengths: Number of steps per episode.
        success_episodes: Whether each episode ended in success.

    Returns:
        Dictionary of RL metric names to values.
    """
    rewards = np.array(episode_rewards, dtype=float)
    lengths = np.array(episode_lengths, dtype=float)

    metrics: Dict[str, float] = {
        "cumulative_reward": float(rewards.sum()),
        "avg_reward": float(rewards.mean()) if len(rewards) > 0 else 0.0,
        "max_reward": float(rewards.max()) if len(rewards) > 0 else 0.0,
        "min_reward": float(rewards.min()) if len(rewards) > 0 else 0.0,
        "avg_episode_length": float(lengths.mean()) if len(lengths) > 0 else 0.0,
    }

    if success_episodes is not None and len(success_episodes) > 0:
        metrics["win_rate"] = float(np.mean(success_episodes))

    # Convergence rate: fraction of last 10% episodes within 5% of best reward
    if len(rewards) >= 10:
        last_10pct = rewards[int(len(rewards) * 0.9):]
        best = rewards.max()
        if best > 0:
            metrics["convergence_rate"] = float(
                np.mean(np.abs(last_10pct - best) / best < 0.05)
            )
        else:
            metrics["convergence_rate"] = 0.0
    else:
        metrics["convergence_rate"] = 0.0

    return metrics