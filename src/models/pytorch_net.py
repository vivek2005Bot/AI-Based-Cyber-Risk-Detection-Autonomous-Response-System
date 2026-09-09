"""PyTorch Deep Learning Classifier for Cyber Risk Detection.

Provides a PyTorch-based Deep Neural Network that implements the
scikit-learn BaseEstimator and ClassifierMixin interfaces for seamless
integration with cross-validation, grid search, ensembles, and evaluation pipelines.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False

from src.utils.logging import get_logger

logger = get_logger(__name__)


if _HAS_TORCH:

    class PyTorchCyberRiskNet(nn.Module):
        """Configurable multi-layer perceptron for cybersecurity classification."""

        def __init__(
            self,
            input_dim: int,
            output_dim: int,
            hidden_layers: List[int] = [128, 64, 32],
            dropout_rate: float = 0.2,
            activation: str = "relu",
        ):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim

            act_fn = nn.ReLU if activation == "relu" else (nn.GELU if activation == "gelu" else nn.Tanh)

            layers: List[nn.Module] = []
            prev_dim = input_dim

            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(act_fn())
                if dropout_rate > 0:
                    layers.append(nn.Dropout(dropout_rate))
                prev_dim = hidden_dim

            layers.append(nn.Linear(prev_dim, output_dim))
            self.network = nn.Sequential(*layers)

            # Initialize weights
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0.0)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x)


class PyTorchRiskClassifier(ClassifierMixin, BaseEstimator):
    """Scikit-learn compatible wrapper around a PyTorch Deep Neural Network."""

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_layer_sizes: Union[List[int], Tuple[int, ...]] = (128, 64, 32),
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 40,
        dropout_rate: float = 0.2,
        activation: str = "relu",
        weight_decay: float = 1e-4,
        patience: int = 8,
        device: Optional[str] = None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.weight_decay = weight_decay
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_: Optional[nn.Module] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_in_: Optional[int] = None
        self.training_losses_: List[float] = []

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        if not _HAS_TORCH:
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def fit(self, X: Any, y: Any) -> "PyTorchRiskClassifier":
        """Fit the PyTorch model to training data."""
        if not _HAS_TORCH:
            raise ImportError("PyTorch is required. Install torch to use PyTorchRiskClassifier.")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # Convert inputs
        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y_arr = y.values.ravel()
        else:
            y_arr = np.asarray(y).ravel()

        self.classes_ = np.unique(y_arr)
        n_classes = len(self.classes_)
        self.n_features_in_ = X_arr.shape[1]

        # Map labels to 0..n_classes-1
        class_to_idx = {c: idx for idx, c in enumerate(self.classes_)}
        y_mapped = np.array([class_to_idx[val] for val in y_arr], dtype=np.int64)

        device = torch.device(self._resolve_device())
        output_dim = 1 if n_classes <= 2 else n_classes

        self.model_ = PyTorchCyberRiskNet(
            input_dim=self.n_features_in_,
            output_dim=output_dim,
            hidden_layers=list(self.hidden_layer_sizes),
            dropout_rate=self.dropout_rate,
            activation=self.activation,
        ).to(device)

        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        if n_classes <= 2:
            y_tensor = torch.tensor(y_mapped, dtype=torch.float32).unsqueeze(1)
            criterion = nn.BCEWithLogitsLoss()
        else:
            y_tensor = torch.tensor(y_mapped, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()

        dataset = TensorDataset(X_tensor, y_tensor)
        batch_size = max(self.batch_size, 256) if len(dataset) > 5000 else self.batch_size
        loader = DataLoader(
            dataset,
            batch_size=min(batch_size, len(dataset)),
            shuffle=True,
            drop_last=False,
        )

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        self.model_.train()
        self.training_losses_ = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                optimizer.zero_grad()
                logits = self.model_(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * batch_x.size(0)

            epoch_loss /= len(dataset)
            self.training_losses_.append(epoch_loss)

            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                if self.verbose:
                    logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities for X."""
        if not _HAS_TORCH or self.model_ is None:
            raise ValueError("Model must be fitted before calling predict_proba.")

        if isinstance(X, pd.DataFrame):
            X_arr = X.values.astype(np.float32)
        else:
            X_arr = np.asarray(X, dtype=np.float32)

        device = next(self.model_.parameters()).device
        self.model_.eval()

        X_tensor = torch.tensor(X_arr, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = self.model_(X_tensor)
            if len(self.classes_) <= 2:
                probs_pos = torch.sigmoid(logits).cpu().numpy().ravel()
                probs = np.vstack([1.0 - probs_pos, probs_pos]).T
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return probs

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for X."""
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]
