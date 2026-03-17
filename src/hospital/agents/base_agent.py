from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseAgent(ABC):
    """Shared interface for all local hospital agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name used in reports and meta-controller weighting."""

    @abstractmethod
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        """Train the agent on local hospital data."""

    @abstractmethod
    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """Return positive-class probabilities in shape (n_samples,)."""
