"""Deep Learning Model: PyTorch Multi-Layer Perceptron with Monte Carlo Dropout."""

from __future__ import annotations

import logging
from typing import Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:

    class MLPNetwork(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 64, dropout_rate: float = 0.2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)


class PyTorchMLPClassifier:
    """PyTorch Tabular MLP Classifier supporting Monte Carlo Dropout uncertainty."""

    def __init__(
        self,
        hidden_dim: int = 64,
        dropout_rate: float = 0.2,
        lr: float = 1e-3,
        epochs: int = 15,
        batch_size: int = 256,
        random_state: int = 42,
    ):
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: Any | None = None
        self.feature_cols: list[str] = []
        self.is_fitted = False

    def fit(
        self, X_train: pd.DataFrame, y_train: np.ndarray, feature_cols: list[str] | None = None
    ) -> PyTorchMLPClassifier:
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch is not installed. Install torch to use PyTorchMLPClassifier."
            )

        torch.manual_seed(self.random_state)
        self.feature_cols = feature_cols or list(X_train.columns)
        X_mat = self.scaler.fit_transform(X_train[self.feature_cols].values)
        y_arr = np.asarray(y_train, dtype=np.float32)

        # Calculate positive class weight to handle imbalance
        n_pos = np.sum(y_arr == 1.0)
        n_neg = len(y_arr) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], dtype=torch.float32)

        tensor_x = torch.tensor(X_mat, dtype=torch.float32)
        tensor_y = torch.tensor(y_arr, dtype=torch.float32)
        dataset = TensorDataset(tensor_x, tensor_y)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = MLPNetwork(
            input_dim=len(self.feature_cols),
            hidden_dim=self.hidden_dim,
            dropout_rate=self.dropout_rate,
        )
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

        self.is_fitted = True
        logger.info(f"Fitted PyTorch MLP Classifier ({self.epochs} epochs).")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_proba.")
        X_mat = self.scaler.transform(X[self.feature_cols].values)
        tensor_x = torch.tensor(X_mat, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor_x)
            probs = torch.sigmoid(logits).numpy()
        return np.clip(probs, 0.0, 1.0)

    def predict_mc_dropout(
        self, X: pd.DataFrame, n_samples: int = 20
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Perform Monte Carlo Dropout inference by keeping dropout active at test time."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict_mc_dropout.")
        X_mat = self.scaler.transform(X[self.feature_cols].values)
        tensor_x = torch.tensor(X_mat, dtype=torch.float32)

        # Keep dropout active during inference
        self.model.train()
        sample_probs = []
        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.model(tensor_x)
                probs = torch.sigmoid(logits).numpy()
                sample_probs.append(probs)

        stacked = np.stack(sample_probs, axis=0)  # (n_samples, N)
        mean_prob = np.mean(stacked, axis=0)
        std_prob = np.std(stacked, axis=0)
        return mean_prob, std_prob
