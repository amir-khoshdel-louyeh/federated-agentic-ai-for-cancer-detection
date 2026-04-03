from __future__ import annotations

import numpy as np

from .base import ThinkingPattern


class RuleBasedThinkingPattern(ThinkingPattern):
    def save_model(self, file_path: str) -> None:
        import json
        with open(file_path + '.json', 'w') as f:
            json.dump({
                'threshold': self.threshold,
                'weights': self.weights,
                'scale': self.scale,
            }, f)

    def load_model(self, file_path: str) -> None:
        import json
        with open(file_path + '.json', 'r') as f:
            data = json.load(f)
            self.threshold = data.get('threshold', self.threshold)
            self.weights = data.get('weights', self.weights)
            self.scale = data.get('scale', self.scale)
    """
    Deterministic baseline inspired by simple ABCD-style lesion heuristics.

    Expected normalized feature order:
    [asymmetry, border_irregularity, color_variegation, diameter_mm_scaled, age_scaled, ...]
    """

    def __init__(
        self,
        threshold: float = 0.58,
        weights: dict[str, float] | None = None,
        scale: float = 10.0,
    ) -> None:
        self.threshold = threshold
        self.weights = weights or {
            'asymmetry': 0.30,
            'border': 0.25,
            'color': 0.20,
            'diameter': 0.15,
            'age': 0.10,
        }
        self.scale = scale

    @property
    def name(self) -> str:
        return "rule_based"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # Rule parameters are fixed; no data-driven fitting required.
        _ = (x_train, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] < 5:
            raise ValueError("RuleBasedThinkingPattern requires at least 5 normalized tabular features.")

        asymmetry = x[:, 0]
        border = x[:, 1]
        color = x[:, 2]
        diameter = x[:, 3]
        age = x[:, 4]

        score = (
            self.weights.get('asymmetry', 0.0) * asymmetry
            + self.weights.get('border', 0.0) * border
            + self.weights.get('color', 0.0) * color
            + self.weights.get('diameter', 0.0) * diameter
            + self.weights.get('age', 0.0) * age
        )

        # Map rule score to a smooth probability while keeping deterministic behavior.
        prob = 1.0 / (1.0 + np.exp(-self.scale * (score - self.threshold)))
        return np.clip(prob, 0.0, 1.0)


class RuleBasedStrictThinkingPattern(RuleBasedThinkingPattern):
    """Stricter deterministic heuristic with a higher risk threshold."""

    def __init__(self, threshold: float = 0.68) -> None:
        super().__init__(threshold=threshold)

    @property
    def name(self) -> str:
        return "rule_based_strict"
