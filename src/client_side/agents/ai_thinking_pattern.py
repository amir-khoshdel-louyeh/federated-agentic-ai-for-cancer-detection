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
            or "Review the clinical lesion metadata and provide a malignancy probability, uncertainty, and a brief clinical reasoning statement."
        )
        self._last_structured_input: np.ndarray | None = None
        self._last_structured_outputs: list[dict[str, Any]] | None = None

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
        structured = self.predict_structured(x)
        return np.array([float(item.get("probability", 0.5)) for item in structured], dtype=np.float32)

    def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
        structured = self.predict_structured(x, n_samples=n_samples)
        return np.array([float(item.get("uncertainty", 0.2)) for item in structured], dtype=np.float32)

    def predict_structured(self, x: np.ndarray, n_samples: int = 25) -> list[dict[str, Any]]:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        cached = self._get_cached_structured_outputs(x)
        if cached is not None:
            return cached

        structured_results: list[dict[str, Any]] = []
        for row in x:
            feature_map = {
                f"feature_{index}": float(value)
                for index, value in enumerate(row, start=1)
            }
            response = self.llm_reasoner.generate_reasoning(
                "AI_AGENT",
                {
                    "patterns": [{"name": self.name, "details": self._build_prompt(row)}],
                    "clinical_features": feature_map,
                },
            )
            structured_results.append(self._extract_structured_output(response))

        self._cache_structured_outputs(x, structured_results)
        return structured_results

    def _cache_structured_outputs(self, x: np.ndarray, outputs: list[dict[str, Any]]) -> None:
        self._last_structured_input = np.array(x, copy=True)
        self._last_structured_outputs = outputs

    def _get_cached_structured_outputs(self, x: np.ndarray) -> list[dict[str, Any]] | None:
        x = np.asarray(x)
        if self._last_structured_outputs is None or self._last_structured_input is None:
            return None
        if x.shape != self._last_structured_input.shape:
            return None
        if np.array_equal(x, self._last_structured_input):
            return self._last_structured_outputs
        return None

    def _extract_structured_output(self, response: dict[str, Any]) -> dict[str, Any]:
        json_data = response.get("json")
        if isinstance(json_data, dict):
            return {
                "probability": float(max(0.0, min(1.0, float(json_data.get("probability", 0.5))))),
                "uncertainty": float(max(0.0, min(1.0, float(json_data.get("uncertainty", 0.2))))),
                "clinical_reasoning": str(json_data.get("reasoning", "")),
                "details": str(json_data.get("details", "single-shot structured prediction")),
            }

        text = response.get("text", "")
        probability = 0.5
        uncertainty = 0.2
        reasoning = ""
        if text:
            prob_match = re.search(r"([01](?:\.\d+)?)", text)
            if prob_match:
                try:
                    probability = float(max(0.0, min(1.0, float(prob_match.group(1)))))
                except ValueError:
                    probability = 0.5
            unc_match = re.search(r"uncertainty\s*[:=]\s*([01](?:\.\d+)?)", text, re.I)
            if unc_match:
                try:
                    uncertainty = float(max(0.0, min(1.0, float(unc_match.group(1)))))
                except ValueError:
                    uncertainty = 0.2
            reasoning = text

        return {
            "probability": probability,
            "uncertainty": uncertainty,
            "clinical_reasoning": reasoning.strip(),
            "details": "parsed structured output fallback",
        }

    def _build_prompt(self, row: np.ndarray) -> str:
        feature_lines = []
        for index, value in enumerate(row, start=1):
            feature_lines.append(f"feature_{index}: {float(value):.4f}")

        return (
            f"{self.prompt_prefix}\n"
            "Use the following normalized clinical features for a lesion.\n"
            + "\n".join(feature_lines)
            + "\nProvide a numeric probability between 0 and 1, a numeric uncertainty between 0 and 1, and a short explanation."
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
