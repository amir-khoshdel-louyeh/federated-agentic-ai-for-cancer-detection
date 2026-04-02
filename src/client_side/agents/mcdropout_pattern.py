from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .base import ThinkingPattern


class MCDropoutThinkingPattern(ThinkingPattern):
    """Wrap an existing ThinkingPattern and do Monte Carlo dropout uncertainty."""

    def __init__(self, base_pattern: ThinkingPattern, n_samples: int = 30) -> None:
        self.base_pattern = base_pattern
        self.n_samples = n_samples

    @property
    def name(self) -> str:
        return f"{self.base_pattern.name}_mcdropout"

    def save_model(self, file_path: str) -> None:
        self.base_pattern.save_model(file_path)

    def load_model(self, file_path: str) -> None:
        self.base_pattern.load_model(file_path)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.base_pattern.fit(x_train, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.base_pattern.predict_proba(x)

    def predict_uncertainty(self, x: np.ndarray, n_samples: int | None = None) -> np.ndarray:
        if n_samples is None:
            n_samples = self.n_samples

        # if base supports its own uncertainty, use it.
        if hasattr(self.base_pattern, "predict_uncertainty"):
            return self.base_pattern.predict_uncertainty(x, n_samples=n_samples)

        # fallback: use repeated predict with stochasticity if possible
        probs = np.stack([self.predict_proba(x) for _ in range(max(1, n_samples))], axis=0)
        return np.clip(probs.std(axis=0), 0.0, 1.0)
