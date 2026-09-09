"""Visualization utilities using Matplotlib, Seaborn, and Plotly."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import plotly.express as px
    import plotly.graph_objects as go

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False


def save_figure(fig, path: str | Path, dpi: int = 150) -> None:
    """Save a matplotlib figure to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    if labels:
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_roc_curves(
    fprs: Dict[str, np.ndarray],
    tprs: Dict[str, np.ndarray],
    aucs: Dict[str, float],
    title: str = "ROC Curves",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot ROC curves for multiple models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in fprs:
        ax.plot(
            fprs[name],
            tprs[name],
            label=f"{name} (AUC = {aucs.get(name, 0):.3f})",
        )
    ax.plot([0, 1], [0, 1], "k--", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_model_comparison(
    results: pd.DataFrame,
    metric: str = "f1",
    title: str = "Model Comparison",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot a bar chart comparing model performance on a metric."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_results = results.sort_values(metric, ascending=False)
    bars = ax.bar(sorted_results.index, sorted_results[metric], color="steelblue")
    ax.set_ylabel(metric.upper())
    ax.set_title(title)
    ax.set_xticks(range(len(sorted_results.index)))
    ax.set_xticklabels(sorted_results.index, rotation=45, ha="right")
    for bar, val in zip(bars, sorted_results[metric]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_anomaly_scatter(
    X_2d: np.ndarray,
    labels: np.ndarray,
    anomalies: Optional[np.ndarray] = None,
    title: str = "Anomaly Detection Results",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot 2D scatter of data points colored by cluster/anomaly label."""
    fig, ax = plt.subplots(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        mask = labels == label
        if label == -1:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c="red",
                marker="x",
                s=50,
                label="Anomaly",
            )
        else:
            ax.scatter(
                X_2d[mask, 0],
                X_2d[mask, 1],
                c=[color],
                s=20,
                label=f"Cluster {label}",
                alpha=0.7,
            )

    if anomalies is not None:
        ax.scatter(
            X_2d[anomalies, 0],
            X_2d[anomalies, 1],
            facecolors="none",
            edgecolors="red",
            s=100,
            linewidths=1.5,
            label="Detected Anomalies",
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_rl_training(
    episode_rewards: List[float],
    episode_lengths: List[int],
    title: str = "RL Training Progress",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot RL training curves (rewards and episode lengths)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Rewards
    ax1.plot(episode_rewards, alpha=0.5, label="Episode Reward")
    window = min(50, max(1, len(episode_rewards) // 10))
    if len(episode_rewards) >= window:
        rolling_mean = pd.Series(episode_rewards).rolling(window).mean()
        ax1.plot(rolling_mean, color="red", label=f"Rolling Mean ({window})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.set_title("Rewards")
    ax1.legend()

    # Episode lengths
    ax2.plot(episode_lengths, alpha=0.5, label="Episode Length")
    if len(episode_lengths) >= window:
        rolling_mean = pd.Series(episode_lengths).rolling(window).mean()
        ax2.plot(rolling_mean, color="red", label=f"Rolling Mean ({window})")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Steps")
    ax2.set_title("Episode Lengths")
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_risk_distribution(
    risk_scores: np.ndarray,
    thresholds: Optional[Dict[str, float]] = None,
    title: str = "Risk Score Distribution",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot histogram of risk scores with threshold lines."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(risk_scores, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    ax.set_xlabel("Risk Score")
    ax.set_ylabel("Frequency")
    ax.set_title(title)

    if thresholds:
        colors = {"low": "green", "medium": "orange", "high": "red", "critical": "darkred"}
        for name, threshold in thresholds.items():
            ax.axvline(threshold, color=colors.get(name, "black"), linestyle="--", label=f"{name.capitalize()} ({threshold})")
        ax.legend()

    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_feature_importance(
    feature_names: List[str],
    importance: np.ndarray,
    top_n: int = 20,
    title: str = "Feature Importance",
    save_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Plot horizontal bar chart of feature importance."""
    indices = np.argsort(importance)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.4)))
    ax.barh(
        [feature_names[i] for i in indices][::-1],
        importance[indices][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        save_figure(fig, save_path)
    return fig


def plotly_risk_gauge(risk_score: float, title: str = "Risk Score") -> Any:
    """Create a Plotly gauge chart for a risk score."""
    if not _HAS_PLOTLY:
        return None
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=risk_score,
            title={"text": title},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 0.3], "color": "lightgreen"},
                    {"range": [0.3, 0.6], "color": "yellow"},
                    {"range": [0.6, 0.8], "color": "orange"},
                    {"range": [0.8, 1.0], "color": "red"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": risk_score,
                },
            },
        )
    )
    return fig