from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import ThinkingPattern


class LogisticThinkingPattern(ThinkingPattern):
    """High-quality logistic regression thinking pattern with persistent model support."""

    def __init__(
        self,
        C: float = 1.0,
        penalty: str = "l2",
        class_weight: str | dict | None = "balanced",
        max_iter: int = 1000,
        random_state: int = 42,
        pretrained_path: str | None = None,
    ) -> None:
        self.C = C
        self.penalty = penalty
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = LogisticRegression(
            C=self.C,
            penalty=self.penalty,
            class_weight=self.class_weight,
            solver="lbfgs",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._is_fitted = False

        if pretrained_path is not None:
            self.load_model(pretrained_path)

    @property
    def name(self) -> str:
        return "logistic"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_scaled = self.scaler.fit_transform(x_train)
        self.model.fit(x_scaled, y_train)
        self._is_fitted = True

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("LogisticThinkingPattern must be fitted before prediction.")

        x_scaled = self.scaler.transform(x)
        return self.model.predict_proba(x_scaled)[:, 1]

    def save_model(self, file_path: str) -> None:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "is_fitted": self._is_fitted,
                "params": {
                    "C": self.C,
                    "penalty": self.penalty,
                    "max_iter": self.max_iter,
                    "random_state": self.random_state,
                },
            }, f)

    def load_model(self, file_path: str) -> None:
        with open(file_path + ".pkl", "rb") as f:
            state = pickle.load(f)

        self.model = state["model"]
        self.scaler = state["scaler"]
        self._is_fitted = state.get("is_fitted", hasattr(self.model, "coef_"))
