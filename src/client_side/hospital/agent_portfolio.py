from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from ..agents import (
    AKIECAgent,
    BCCAgent,
    CancerDetectionAgent,
    MelanomaAgent,
    SCCAgent,
    SkinCancerAgent,
    ThinkingPattern,
)
from .config_helpers import get_cancer_types
from .pattern_factory import create_thinking_pattern


class AgentPortfolio:

    def __init__(
        self,
        initial_patterns: Mapping[str, ThinkingPattern | Sequence[ThinkingPattern]] | None = None,
        cancer_types: tuple[str, ...] | None = None,
    ) -> None:
        if cancer_types is None and initial_patterns is not None:
            cancer_types = tuple(sorted({str(key).strip().upper() for key in initial_patterns.keys() if str(key).strip()}))
        self._cancer_types = tuple(cancer_types) if cancer_types else get_cancer_types(None)
        self._validate_cancer_types(self._cancer_types)
        patterns = self._normalize_initial_patterns(initial_patterns)
        agent_class_map = {
            "CANCER": CancerDetectionAgent,
            "BCC": BCCAgent,
            "SCC": SCCAgent,
            "MELANOMA": MelanomaAgent,
            "AKIEC": AKIECAgent,
        }

        self._agents = {}
        for cancer_type in self._cancer_types:
            try:
                cls = agent_class_map[cancer_type]
            except KeyError as exc:
                raise ValueError(f"No agent class known for cancer type: {cancer_type}") from exc
            self._agents[cancer_type] = cls(thinking_pattern=patterns[cancer_type])

    def save_all_models(self, out_dir: str, hospital_id: str) -> None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        for cancer_type in self._cancer_types:
            agent = self._agents[cancer_type]
            patterns = getattr(agent, "thinking_patterns", [getattr(agent, "_thinking_pattern", None)])
            for pattern in patterns:
                if pattern is None:
                    continue
                file_base = os.path.join(out_dir, f"{hospital_id}_{cancer_type}_{pattern.name}")
                pattern.save_model(file_base)

    def load_all_models(self, out_dir: str, hospital_id: str) -> None:
        import os
        for cancer_type in self._cancer_types:
            agent = self._agents[cancer_type]
            patterns = getattr(agent, "thinking_patterns", [getattr(agent, "_thinking_pattern", None)])
            for pattern in patterns:
                if pattern is None:
                    continue
                file_base = os.path.join(out_dir, f"{hospital_id}_{cancer_type}_{pattern.name}")
                pattern.load_model(file_base)

    def _validate_cancer_types(self, cancer_types: tuple[str, ...]) -> None:
        if not cancer_types:
            raise ValueError("cancer_types must not be empty")
        if len(set(cancer_types)) != len(cancer_types):
            raise ValueError("cancer_types contains duplicate entries")

    @property
    def cancer_types(self) -> tuple[str, ...]:
        return self._cancer_types

    def get_agent(self, cancer_type: str) -> SkinCancerAgent:
        key = self._normalize_cancer_type(cancer_type)
        try:
            return self._agents[key]
        except KeyError as exc:
            raise ValueError(f"Unsupported cancer type: {cancer_type}") from exc

    def set_pattern(self, cancer_type: str, pattern: ThinkingPattern | Sequence[ThinkingPattern]) -> None:
        """Switch a fixed cancer-domain agent to one or more thinking patterns at runtime."""
        agent = self.get_agent(cancer_type)
        agent.set_thinking_pattern(pattern)

    def selected_patterns(self) -> dict[str, str]:
        return {cancer_type: agent.thinking_pattern_name for cancer_type, agent in self._agents.items()}

    def _normalize_cancer_type(self, cancer_type: str) -> str:
        key = cancer_type.strip().upper()
        if key not in self._cancer_types:
            raise ValueError(f"Unsupported cancer type: {cancer_type}")
        return key

    def _normalize_initial_patterns(
        self,
        initial_patterns: Mapping[str, ThinkingPattern | Sequence[ThinkingPattern]] | None,
    ) -> dict[str, list[ThinkingPattern]]:
        if initial_patterns is None:
            return {
                cancer_type: [create_thinking_pattern("ai_agent")]
                for cancer_type in self._cancer_types
            }

        normalized: dict[str, list[ThinkingPattern]] = {}
        for cancer_type, pattern in initial_patterns.items():
            key = cancer_type.strip().upper()
            if key not in self._cancer_types:
                raise ValueError(f"Unsupported cancer type in initial patterns: {cancer_type}")

            if isinstance(pattern, ThinkingPattern):
                normalized[key] = [pattern]
            elif isinstance(pattern, Sequence) and not isinstance(pattern, (str, bytes)):
                normalized[key] = [item for item in pattern]
            else:
                raise TypeError(
                    f"Pattern for {cancer_type} must be a ThinkingPattern or list of ThinkingPatterns."
                )

        missing = [ct for ct in self._cancer_types if ct not in normalized]
        extra = [ct for ct in normalized if ct not in self._cancer_types]
        if missing or extra:
            raise ValueError(
                "Initial patterns must define exactly these cancer types: "
                f"{', '.join(self._cancer_types)}"
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
