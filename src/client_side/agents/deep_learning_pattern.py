from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import ThinkingPattern


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepLearningThinkingPattern(ThinkingPattern):
    """Tabular deep learning thinking pattern implemented as a small MLP."""

    def __init__(self, epochs: int = 20, batch_size: int = 32, lr: float = 1e-3) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.scaler = StandardScaler()
        self.model: _MLP | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "deep_learning"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train).astype(np.float32)
        y = y_train.astype(np.float32)

        dataset = TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model = _MLP(input_dim=x_train.shape[1]).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        torch.autograd.set_detect_anomaly(True)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                optimizer.zero_grad()
                logits = self.model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeepLearningThinkingPattern must be fitted before prediction.")

        self.model.eval()
        x_scaled = self.scaler.transform(x).astype(np.float32)
        xt = torch.from_numpy(x_scaled).to(self.device)

        with torch.no_grad():
            logits = self.model(xt)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
        return np.clip(probs, 0.0, 1.0)
