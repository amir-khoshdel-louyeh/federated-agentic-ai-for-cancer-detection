from __future__ import annotations

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional

from .base import ThinkingPattern


class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=False),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeepLearningThinkingPattern(ThinkingPattern):
    def save_model(self, file_path: str) -> None:
        import torch
        import pickle
        if self.model is None:
            raise RuntimeError("Model must be trained before saving.")
        torch.save({'model_state_dict': self.model.state_dict()}, file_path + '.pt')
        with open(file_path + '_scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)

    def load_model(self, file_path: str) -> None:
        import torch
        import pickle
        if self.model is None:
            raise RuntimeError("Model instance must be created before loading.")
        checkpoint = torch.load(file_path + '.pt', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        with open(file_path + '_scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
    """Tabular deep learning thinking pattern implemented as a small MLP."""

    def __init__(
        self,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 1e-3,
        class_weight: str | dict[str, float] | float | None = "balanced",
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.class_weight = class_weight
        self.scaler = StandardScaler()
        self.model: _MLP | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def name(self) -> str:
        return "deep_learning"

    def _compute_pos_weight(self, y: torch.Tensor) -> Optional[float]:
        if self.class_weight is None:
            return None

        if isinstance(self.class_weight, (int, float)):
            return float(self.class_weight)

        if isinstance(self.class_weight, str):
            if self.class_weight == "balanced":
                labels = y.to(torch.int64).cpu().numpy().astype(np.int64).ravel()
                counts = np.bincount(labels, minlength=2)
                if counts[1] == 0:
                    return 1.0
                return float(counts[0]) / float(counts[1])
            raise ValueError(
                f"Unsupported class_weight string for DeepLearningThinkingPattern: {self.class_weight}. "
                "Use 'balanced' or a numeric scalar."
            )

        if isinstance(self.class_weight, dict):
            negative = float(self.class_weight.get(0, 1.0))
            positive = float(self.class_weight.get(1, 1.0))
            if negative <= 0 or positive <= 0:
                raise ValueError("class_weight dict values must be positive for DeepLearningThinkingPattern")
            return positive / negative

        return None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train).astype(np.float32)
        y = y_train.astype(np.float32)

        dataset = TensorDataset(torch.from_numpy(x_scaled), torch.from_numpy(y).unsqueeze(1))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if self.model is None or self.model.net[0].in_features != x_train.shape[1]:
            self.model = _MLP(input_dim=x_train.shape[1]).to(self.device)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        pos_weight = self._compute_pos_weight(y)
        if pos_weight is not None:
            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=self.device))
        else:
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
                # Ensure no in-place ops before backward
                loss = loss.clone()  # Defensive: avoid in-place modification
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

    def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DeepLearningThinkingPattern must be fitted before prediction.")

        # Monte Carlo Dropout: evaluate model in training mode multiple times.
        self.model.train()
        x_scaled = self.scaler.transform(x).astype(np.float32)
        xt = torch.from_numpy(x_scaled).to(self.device)

        preds = []
        for _ in range(max(1, n_samples)):
            logits = self.model(xt)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            preds.append(probs)

        preds = np.stack(preds, axis=0)
        uncertainty = preds.std(axis=0)
        uncertainty = np.clip(uncertainty, 0.0, 1.0)

        # return low certainty for stable predictions on deterministic model
        self.model.eval()
        return uncertainty
