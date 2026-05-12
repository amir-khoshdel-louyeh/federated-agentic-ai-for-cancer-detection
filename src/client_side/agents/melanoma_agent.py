from __future__ import annotations

from typing import Any

import numpy as np

from .base import LLMReasoner, SkinCancerAgent, ThinkingPattern
from .tools import MedicalKnowledgeBaseTool, SearchTool, Tool, VisualAnalysisTool


class MelanomaAgent(SkinCancerAgent):
    """Fixed-domain agent dedicated to melanoma detection."""

    def __init__(
        self,
        thinking_pattern: ThinkingPattern | list[ThinkingPattern],
        llm_reasoner: LLMReasoner | None = None,
        tools: list[Tool] | None = None,
        uncertainty_threshold: float = 0.4,
    ) -> None:
        default_tools = tools or [SearchTool(), MedicalKnowledgeBaseTool(), VisualAnalysisTool()]
        super().__init__(thinking_patterns=thinking_pattern, llm_reasoner=llm_reasoner, tools=default_tools)
        self.uncertainty_threshold = uncertainty_threshold

    @property
    def cancer_type(self) -> str:
        return "MELANOMA"

    def predict_diagnoses(
        self,
        x: np.ndarray,
        patient_context: dict[str, Any] | None = None,
        n_samples: int = 25,
    ) -> list[dict[str, Any]]:
        if x.ndim == 1:
            x = x.reshape(1, -1)

        observations = self._build_observations(
            x,
            patient_context=patient_context,
            n_samples=n_samples,
            use_structured=True,
        )

        results: list[dict[str, Any]] = []
        for idx in range(x.shape[0]):
            obs = observations[idx]
            high_uncertainty = obs.get("uncertainty", 0.0) >= self.uncertainty_threshold
            if high_uncertainty:
                obs = self._invoke_tool_for_observation(obs, patient_context)
                reasoning_response = self._llm_reasoner.generate_reasoning(
                    self.cancer_type,
                    obs,
                    patient_context if isinstance(patient_context, dict) else None,
                )
                reasoning_text = reasoning_response.get("text", "")
            else:
                reasoning_text = obs.get("clinical_reasoning", "")

            probability = float(obs.get("probability", 0.0))
            uncertainty = float(obs.get("uncertainty", 0.0))
            results.append(
                {
                    "cancer_type": self.cancer_type,
                    "diagnosis": "cancer type found" if probability >= 0.5 else "not found",
                    "probability": probability,
                    "confidence": float(max(0.0, min(1.0, 1.0 - uncertainty))),
                    "uncertainty": uncertainty,
                    "clinical_reasoning": reasoning_text,
                    "observations": obs,
                }
            )

        return results
