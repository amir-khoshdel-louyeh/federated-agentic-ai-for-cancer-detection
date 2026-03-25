from __future__ import annotations

from typing import Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..agents import (
    AKIECAgent,
    BCCAgent,
    MelanomaAgent,
    SCCAgent,
    SkinCancerAgent,
    ThinkingPattern,
)
from .pattern_factory import create_thinking_pattern

CANCER_TYPES = ("BCC", "SCC", "MELANOMA", "AKIEC")


class AgentPortfolio:
    """Hospital-local portfolio with exactly four fixed cancer agents."""

    def __init__(self, initial_patterns: Mapping[str, ThinkingPattern] | None = None) -> None:
        patterns = self._normalize_initial_patterns(initial_patterns)
        self._agents: dict[str, SkinCancerAgent] = {
            "BCC": BCCAgent(thinking_pattern=patterns["BCC"]),
            "SCC": SCCAgent(thinking_pattern=patterns["SCC"]),
            "MELANOMA": MelanomaAgent(thinking_pattern=patterns["MELANOMA"]),
            "AKIEC": AKIECAgent(thinking_pattern=patterns["AKIEC"]),
        }

    @property
    def cancer_types(self) -> tuple[str, str, str, str]:
        return CANCER_TYPES

    def get_agent(self, cancer_type: str) -> SkinCancerAgent:
        key = self._normalize_cancer_type(cancer_type)
        try:
            return self._agents[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported cancer type: {cancer_type}") from exc

    def set_pattern(self, cancer_type: str, pattern: ThinkingPattern) -> None:
        """Switch a fixed cancer-domain agent to another thinking pattern at runtime."""
        agent = self.get_agent(cancer_type)
        agent.set_thinking_pattern(pattern)

    def selected_patterns(self) -> dict[str, str]:
        return {cancer_type: agent.thinking_pattern_name for cancer_type, agent in self._agents.items()}

    def train_all(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        for cancer_type in CANCER_TYPES:
            self._agents[cancer_type].fit(x_train, y_train)

    def predict_all(self, x: np.ndarray) -> dict[str, np.ndarray]:
        predictions: dict[str, np.ndarray] = {}
        for cancer_type in CANCER_TYPES:
            agent = self._agents[cancer_type]
            predictions[agent.name] = agent.predict_proba(x)
        return predictions

    def evaluate_all(self, y_true: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
        report: dict[str, dict[str, float]] = {}
        for prediction_key, probs in predictions.items():
            report[prediction_key] = self._eval_probs(y_true, probs)
        return report

    @staticmethod
    def _normalize_cancer_type(cancer_type: str) -> str:
        key = cancer_type.strip().upper()
        if key not in CANCER_TYPES:
            raise ValueError(f"Unsupported cancer type: {cancer_type}")
        return key

    @staticmethod
    def _normalize_initial_patterns(
        initial_patterns: Mapping[str, ThinkingPattern] | None,
    ) -> dict[str, ThinkingPattern]:
        if initial_patterns is None:
            return {
                "BCC": create_thinking_pattern("rule_based"),
                "SCC": create_thinking_pattern("rule_based"),
                "MELANOMA": create_thinking_pattern("rule_based"),
                "AKIEC": create_thinking_pattern("rule_based"),
            }

        normalized: dict[str, ThinkingPattern] = {}
        for cancer_type, pattern in initial_patterns.items():
            key = cancer_type.strip().upper()
            if key not in CANCER_TYPES:
                raise ValueError(f"Unsupported cancer type in initial patterns: {cancer_type}")
            normalized[key] = pattern

        missing = [ct for ct in CANCER_TYPES if ct not in normalized]
        extra = [ct for ct in normalized if ct not in CANCER_TYPES]
        if missing or extra:
            raise ValueError(
                "Initial patterns must define exactly these cancer types: "
                f"{', '.join(CANCER_TYPES)}"
            )

        return normalized

    @staticmethod
    def _eval_probs(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
        preds = (probs >= 0.5).astype(int)
        metrics = {
            "accuracy": float(accuracy_score(y_true, preds)),
            "f1": float(f1_score(y_true, preds, zero_division=0)),
        }
        try:
            metrics["auc"] = float(roc_auc_score(y_true, probs))
        except ValueError:
            metrics["auc"] = 0.5
        return metrics
