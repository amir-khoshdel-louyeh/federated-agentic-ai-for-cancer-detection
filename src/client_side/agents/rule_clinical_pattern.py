from __future__ import annotations

import numpy as np

from .base import ThinkingPattern


class RuleClinicalThinkingPattern(ThinkingPattern):
    """Clinical safety filter based on age/sex/site and lesion features."""

    def __init__(
        self,
        age_threshold: int = 30,
        pediatric_penalty: float = 0.6,
        weights: dict[str, float] | None = None,
        scale: float = 10.0,
    ):
        self.age_threshold = age_threshold
        self.pediatric_penalty = pediatric_penalty
        self.weights = weights or {
            'asymmetry': 0.35,
            'border': 0.25,
            'color': 0.2,
            'diameter': 0.2,
        }
        self.scale = scale

    @property
    def name(self) -> str:
        return "rule_clinical"

    def save_model(self, file_path: str) -> None:
        import json
        with open(file_path + '.json', 'w') as f:
            json.dump({
                'age_threshold': self.age_threshold,
                'pediatric_penalty': self.pediatric_penalty,
            }, f)

    def load_model(self, file_path: str) -> None:
        import json
        with open(file_path + '.json', 'r') as f:
            data = json.load(f)
            self.age_threshold = data.get('age_threshold', self.age_threshold)
            self.pediatric_penalty = data.get('pediatric_penalty', self.pediatric_penalty)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # No learning; rule-based constants are used
        _ = (x_train, y_train)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] < 6:
            # required features: asymmetry, border, color, diameter, age, sex
            raise ValueError("RuleClinicalThinkingPattern expects at least 6 columns (asymmetry,border,color,diameter,age,sex)")

        asymmetry = x[:, 0]
        border = x[:, 1]
        color = x[:, 2]
        diameter = x[:, 3]
        age = x[:, 4]
        sex = x[:, 5]  # 0=F,1=M assumed

        base_score = (
            self.weights.get('asymmetry', 0.0) * asymmetry
            + self.weights.get('border', 0.0) * border
            + self.weights.get('color', 0.0) * color
            + self.weights.get('diameter', 0.0) * diameter
        )

        # age-based safety rule: reduce cancer probability for pediatric unless very high score.
        age_factor = np.where(age < self.age_threshold, self.pediatric_penalty, 1.0)
        score = base_score * age_factor

        # constrain to 0..1 via sigmoid
        prob = 1.0 / (1.0 + np.exp(-self.scale * (score - 0.5)))
        return np.clip(prob, 0.0, 1.0)

    def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
        # rule-based uncertainty based on proximity to threshold
        raw = self.predict_proba(x)
        return np.abs(raw - 0.5) * 2.0  # more uncertain near 0.5
