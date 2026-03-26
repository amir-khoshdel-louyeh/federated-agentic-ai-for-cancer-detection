from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import GaussianNB

from .base import ThinkingPattern


class BayesianThinkingPattern(ThinkingPattern):
    def save_model(self, file_path: str) -> None:
        import pickle
        with open(file_path + '.pkl', 'wb') as f:
            pickle.dump(self.model, f)

    def load_model(self, file_path: str) -> None:
        import pickle
        with open(file_path + '.pkl', 'rb') as f:
            self.model = pickle.load(f)
    """Probabilistic thinking pattern using Gaussian Naive Bayes."""

    def __init__(self) -> None:
        self.model = GaussianNB()

    @property
    def name(self) -> str:
        return "bayesian"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        self.model.fit(x_train, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)[:, 1]
