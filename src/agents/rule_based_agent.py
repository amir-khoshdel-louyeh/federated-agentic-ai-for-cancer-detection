from __future__ import annotations

import numpy as np

from .base_agent import BaseAgent


class RuleBasedAgent(BaseAgent):
    """
    Deterministic baseline inspired by simple ABCD-style lesion heuristics.

    Expected normalized feature order:
    [asymmetry, border_irregularity, color_variegation, diameter_mm_scaled, age_scaled, ...]
    """

    def __init__(self, threshold: float = 0.58) -> None:
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "rule_based"

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # Rule parameters are fixed; no data-driven fitting required.
        _ = (x_train, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] < 5:
            raise ValueError("RuleBasedAgent requires at least 5 normalized tabular features.")

        asymmetry = x[:, 0]
        border = x[:, 1]
        color = x[:, 2]
        diameter = x[:, 3]
        age = x[:, 4]

        score = (
            0.30 * asymmetry
            + 0.25 * border
            + 0.20 * color
            + 0.15 * diameter
            + 0.10 * age
        )

        # Map rule score to a smooth probability while keeping deterministic behavior.
        prob = 1.0 / (1.0 + np.exp(-10.0 * (score - self.threshold)))
        return np.clip(prob, 0.0, 1.0)
