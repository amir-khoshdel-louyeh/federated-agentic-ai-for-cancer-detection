from __future__ import annotations

import re
from typing import Any

import numpy as np

from .base import LLMReasoner, ThinkingPattern


class AIThinkingPattern(ThinkingPattern):
    """Lightweight AI thinking pattern that uses a language model to produce probabilities."""

    def __init__(
        self,
        llm_reasoner: LLMReasoner | None = None,
        provider: str = "auto",
        model_name: str = "gpt-3.5-turbo",
        local_llm_config: dict[str, Any] | None = None,
        api_key: str | None = None,
        prompt_prefix: str | None = None,
    ) -> None:
        self.llm_reasoner = llm_reasoner or LLMReasoner(
            provider=provider,
            model_name=model_name,
            local_llm_config=local_llm_config,
            api_key=api_key,
        )
        self.prompt_prefix = (
            prompt_prefix
            or "Review the clinical lesion metadata and provide a malignancy probability between 0 and 1."
        )

    @property
    def name(self) -> str:
        return "ai_agent"

    def save_model(self, file_path: str) -> None:
        # No model weights to persist for the pure AI-agent pattern.
        return None

    def load_model(self, file_path: str) -> None:
        # No model weights to load for the pure AI-agent pattern.
        return None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # No training required for the pure language-agent reasoning pattern.
        # AIThinkingPattern relies on LLM inference at prediction time.
        return None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        probabilities: list[float] = []
        for row in x:
            feature_map = {
                f"feature_{index}": float(value)
                for index, value in enumerate(row, start=1)
            }
            prompt = self._build_prompt(row)
            response = self.llm_reasoner.generate_reasoning(
                "AI_AGENT",
                {
                    "patterns": [{"name": self.name, "details": prompt}],
                    "clinical_features": feature_map,
                },
            )
            probabilities.append(self._extract_probability(response))

        return np.array(probabilities, dtype=np.float32)

    def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
        # The AI agent pattern reports a fixed uncertainty baseline.
        probs = self.predict_proba(x)
        return np.full_like(probs, 0.2, dtype=np.float32)

    def _build_prompt(self, row: np.ndarray) -> str:
        feature_lines = []
        for index, value in enumerate(row, start=1):
            feature_lines.append(f"feature_{index}: {float(value):.4f}")

        return (
            f"{self.prompt_prefix}\n"
            "Use the following normalized clinical features for a lesion.\n"
            + "\n".join(feature_lines)
            + "\nProvide a numeric probability between 0 and 1 and a short explanation."
        )

    def _extract_probability(self, response: dict[str, Any]) -> float:
        json_data = response.get("json")
        if isinstance(json_data, dict):
            probability = json_data.get("probability")
            try:
                return float(max(0.0, min(1.0, float(probability))))
            except (TypeError, ValueError):
                pass

        text = response.get("text", "")
        if not text:
            return 0.5

        match = re.search(r"([01]?(?:\.\d+))", text)
        if match:
            try:
                value = float(match.group(1))
                return float(max(0.0, min(1.0, value)))
            except ValueError:
                return 0.5

        # Fallback to a conservative default probability on unparsable output.
        return 0.5
