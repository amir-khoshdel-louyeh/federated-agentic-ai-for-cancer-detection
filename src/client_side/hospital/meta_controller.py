from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import roc_auc_score
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


@dataclass
class AgentScore:
    name: str
    auc: float
    weight: float


class LocalMetaController:
    """Assigns agent weights from validation AUC and builds ensemble probabilities."""

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.weights: dict[str, float] = {}

    def fit_weights(self, y_val: np.ndarray, val_predictions: dict[str, np.ndarray]) -> list[AgentScore]:
        aucs: dict[str, float] = {}
        for name, pred in val_predictions.items():
            aucs[name] = self._safe_auc(y_val, pred)

        score_sum = sum(aucs.values()) + self.epsilon
        self.weights = {name: auc / score_sum for name, auc in aucs.items()}

        return [AgentScore(name=name, auc=aucs[name], weight=self.weights[name]) for name in sorted(aucs)]

    def ensemble_predict(self, predictions: dict[str, np.ndarray]) -> np.ndarray:
        if not self.weights:
            raise RuntimeError("Call fit_weights before ensemble_predict.")

        ref_shape = next(iter(predictions.values())).shape
        combined = np.zeros(ref_shape, dtype=np.float32)

        for name, pred in predictions.items():
            combined += float(self.weights.get(name, 0.0)) * pred

        return np.clip(combined, 0.0, 1.0)

    @staticmethod
    def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        # Fallback prevents crash in rare constant-label validation splits.
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = 0.5
        return max(auc, 0.01)
