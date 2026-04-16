from __future__ import annotations

import re
from typing import Any

from ..agents import LLMReasoner


class HospitalManagerAgent:
    """Hospital-level manager agent that selects the leading specialist for a patient case."""

    def __init__(
        self,
        llm_reasoner: LLMReasoner | None = None,
        provider: str = "auto",
        model_name: str = "gpt-3.5-turbo",
        local_model_path: str | None = None,
        local_llm_config: dict[str, Any] | None = None,
        api_key: str | None = None,
    ) -> None:
        self._llm_reasoner = llm_reasoner or LLMReasoner(
            provider=provider,
            model_name=model_name,
            local_model_path=local_model_path,
            local_llm_config=local_llm_config,
            api_key=api_key,
        )

    def recommend_lead_agent(
        self,
        patient_metadata: dict[str, Any],
        agent_performance: dict[str, dict[str, Any]],
        candidate_comparisons: dict[str, list[dict[str, Any]]] | None = None,
    ) -> dict[str, Any]:
        """Recommend a lead agent using cached metadata only."""
        best_agent = None
        best_score = float("-inf")
        rankings = self._rank_agents(agent_performance)

        for agent_name, performance in agent_performance.items():
            validation = performance.get("validation", {})
            test = performance.get("test", {})
            mean_val_uncertainty = float(performance.get("mean_validation_uncertainty", 1.0))
            mean_test_uncertainty = float(performance.get("mean_test_uncertainty", 1.0))

            score = (
                float(validation.get("auc", 0.0)) * 0.6
                + float(test.get("f1", 0.0)) * 0.3
                + (1.0 - mean_test_uncertainty) * 0.1
            )
            if score > best_score:
                best_score = score
                best_agent = agent_name

        if best_agent is None:
            return self._fallback_recommendation(patient_metadata, agent_performance, "No cached agent metadata available.")

        selected = agent_performance.get(best_agent, {})
        reasoning_text = (
            f"Selected lead agent {best_agent} based on cached validation AUC {selected.get('validation', {}).get('auc', 0.0):.3f}, "
            f"test F1 {selected.get('test', {}).get('f1', 0.0):.3f}, and mean test uncertainty {selected.get('mean_test_uncertainty', 1.0):.3f}."
        )
        return {
            "lead_agent": best_agent,
            "lead_pattern": selected.get("pattern", "unknown"),
            "clinical_reasoning": reasoning_text,
            "agent_rankings": rankings,
        }

    def _build_observations(
        self,
        agent_performance: dict[str, dict[str, Any]],
        candidate_comparisons: dict[str, list[dict[str, Any]]] | None,
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for cancer_type, performance in agent_performance.items():
            selected_pattern = performance.get("pattern", "unknown")
            validation = performance.get("validation", {})
            test = performance.get("test", {})
            observations.append(
                {
                    "name": f"{cancer_type}Agent",
                    "probability": float(validation.get("auc", 0.0)),
                    "uncertainty": float(test.get("f1", 0.0)),
                    "details": (
                        f"pattern={selected_pattern}; "
                        f"validation_auc={validation.get('auc', 0.0):.3f}; "
                        f"validation_f1={validation.get('f1', 0.0):.3f}; "
                        f"test_auc={test.get('auc', 0.0):.3f}; "
                        f"test_f1={test.get('f1', 0.0):.3f}"
                    ),
                }
            )

        if candidate_comparisons:
            comparisons: list[str] = []
            for cancer_type, candidates in candidate_comparisons.items():
                comparisons.append(
                    f"{cancer_type}: "
                    + "; ".join(
                        f"{item['pattern']} (rank={item.get('rank', '?')}, auc={item['auc']:.3f}, f1={item['f1']:.3f})"
                        for item in candidates
                    )
                )
            observations.append(
                {
                    "name": "candidate_comparison",
                    "probability": 0.0,
                    "uncertainty": 0.0,
                    "details": "Candidate pattern ranking by cancer type: " + " | ".join(comparisons),
                }
            )

        return observations

    def _extract_lead_agent(
        self,
        reasoning: str,
        agent_performance: dict[str, dict[str, Any]],
    ) -> str | None:
        candidates = list(agent_performance)
        lower_text = reasoning.lower()

        scores = {candidate: lower_text.count(candidate.lower()) for candidate in candidates}
        chosen = max(scores, key=scores.get)
        if scores[chosen] > 0:
            return chosen

        match = re.search(r"lead agent[:\s]*([A-Za-z0-9_]+)", reasoning, re.I)
        if match:
            candidate = match.group(1).strip()
            if candidate in candidates:
                return candidate

        return None

    def _rank_agents(self, agent_performance: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        ranking = []
        for cancer_type, performance in agent_performance.items():
            validation = performance.get("validation", {})
            score = 0.7 * float(validation.get("auc", 0.0)) + 0.3 * float(validation.get("f1", 0.0))
            ranking.append(
                {
                    "agent": f"{cancer_type}Agent",
                    "pattern": performance.get("pattern", "unknown"),
                    "auc": float(validation.get("auc", 0.0)),
                    "f1": float(validation.get("f1", 0.0)),
                    "score": score,
                }
            )
        return sorted(ranking, key=lambda item: item["score"], reverse=True)

    def _fallback_recommendation(
        self,
        patient_metadata: dict[str, Any],
        agent_performance: dict[str, dict[str, Any]],
        reasoning: str,
    ) -> dict[str, Any]:
        ranking = self._rank_agents(agent_performance)
        if not ranking:
            return {
                "lead_agent": "unknown",
                "lead_pattern": "unknown",
                "clinical_reasoning": (
                    "Unable to select a lead agent based on available metrics. "
                    "Please provide a valid agent performance dictionary."
                ),
                "agent_rankings": [],
            }

        top = ranking[0]
        return {
            "lead_agent": top["agent"].replace("Agent", ""),
            "lead_pattern": top["pattern"],
            "clinical_reasoning": (
                reasoning
                or "The hospital manager defaults to the highest-performing specialist based on validation AUC and F1."
            ),
            "agent_rankings": ranking,
        }
