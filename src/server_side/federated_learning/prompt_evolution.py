from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from configs.config_loader import save_config
from src.client_side.agents import LLMReasoner

DEFAULT_PROMPT_EVOLUTION_CONFIG = {
    "enabled": True,
    "top_k_hospitals": 1,
    "bottom_k_hospitals": 1,
    "min_hospitals": 2,
    "golden_prompt_rollback": True,
    "performance_delta": 0.0,
}


def _score_hospital_update(local_update: Mapping[str, Any]) -> float:
    selected_performance = local_update.get("metrics", {}).get("selected_performance", {})
    if not isinstance(selected_performance, Mapping) or not selected_performance:
        return 0.0

    scores: list[float] = []
    for performance in selected_performance.values():
        validation = performance.get("validation", {})
        if not isinstance(validation, Mapping):
            continue
        auc = float(validation.get("auc", 0.0))
        f1 = float(validation.get("f1", 0.0))
        scores.append(0.7 * auc + 0.3 * f1)

    return float(sum(scores) / len(scores)) if scores else 0.0


def _global_metrics_score(metrics: Mapping[str, Any] | None) -> float:
    if not isinstance(metrics, Mapping):
        return 0.0
    auc = float(metrics.get("auc", 0.0))
    f1 = float(metrics.get("f1", 0.0))
    return 0.7 * auc + 0.3 * f1


def _extract_reasoning_snippets(local_update: Mapping[str, Any], key: str) -> list[str]:
    reasoning = local_update.get("reasoning", {})
    if not isinstance(reasoning, Mapping):
        return []
    entries = []
    candidate = reasoning.get(key, {})
    if isinstance(candidate, Mapping):
        for values in candidate.values():
            if isinstance(values, list):
                entries.extend(str(item) for item in values if item is not None)
            elif values is not None:
                entries.append(str(values))
    elif isinstance(candidate, list):
        entries.extend(str(item) for item in candidate if item is not None)
    return entries


def _build_meta_prompt(
    *,
    top_hospitals: list[tuple[str, Mapping[str, Any]]],
    bottom_hospitals: list[tuple[str, Mapping[str, Any]]],
    current_prompt: str | None,
    round_index: int,
) -> dict[str, Any]:
    prompt_lines: list[str] = []
    prompt_lines.append(
        "You are a meta-agent that evolves the system prompt for clinical reasoning AI agents."
    )
    prompt_lines.append(
        "Analyze the following hospital reasoning examples and produce a better system prompt and a recommended decision threshold for the next run."
    )
    prompt_lines.append(
        "Return a single JSON object with the keys: system_prompt, decision_threshold, summary."
    )
    prompt_lines.append("Do not include any extra text outside the JSON object.")
    if current_prompt:
        prompt_lines.append("\nCurrent system prompt:")
        prompt_lines.append(current_prompt)

    if top_hospitals:
        prompt_lines.append("\nBest-performing hospitals:")
        for hid, update in top_hospitals:
            score = _score_hospital_update(update)
            prompt_lines.append(f"- Hospital {hid}: score={score:.3f}")
            reasoning = _extract_reasoning_snippets(update, "validation")
            if reasoning:
                prompt_lines.append("  validation reasoning samples:")
                for entry in reasoning[:3]:
                    prompt_lines.append(f"    * {entry}")

    if bottom_hospitals:
        prompt_lines.append("\nLower-performing hospitals:")
        for hid, update in bottom_hospitals:
            score = _score_hospital_update(update)
            prompt_lines.append(f"- Hospital {hid}: score={score:.3f}")
            reasoning = _extract_reasoning_snippets(update, "validation")
            if reasoning:
                prompt_lines.append("  validation reasoning samples:")
                for entry in reasoning[:3]:
                    prompt_lines.append(f"    * {entry}")

    prompt_lines.append(
        "\nFocus on the patterns that made the better hospitals more accurate and the pitfalls in the weaker hospitals' reasoning."
    )
    prompt_lines.append(
        "Rewrite the system prompt so that future AI agents produce clearer, more reliable malignancy probabilities, uncertainty estimates, and concise clinical reasoning."
    )
    prompt_lines.append(
        "The returned system prompt should mention the required JSON output format and should help the agent avoid ambiguity and overconfidence."
    )
    prompt_lines.append(f"Round index: {round_index}")

    return {
        "patterns": [
            {
                "name": "meta_prompt_evolution",
                "probability": 1.0,
                "uncertainty": 0.0,
                "details": "\n".join(prompt_lines),
            }
        ],
        "clinical_features": {
            "task": "evolve system prompt for clinical reasoning",
            "round_index": round_index,
        },
    }


def _select_best_and_worst_hospitals(
    local_updates: Mapping[str, Mapping[str, Any]],
    top_k: int,
    bottom_k: int,
) -> tuple[list[tuple[str, Mapping[str, Any]]], list[tuple[str, Mapping[str, Any]]]]:
    scored = [
        (hospital_id, local_update, _score_hospital_update(local_update))
        for hospital_id, local_update in local_updates.items()
    ]
    scored.sort(key=lambda item: item[2], reverse=True)
    best = [(hid, update) for hid, update, _ in scored[:top_k]]
    worst = [(hid, update) for hid, update, _ in scored[-bottom_k:]]
    return best, worst


def _extract_system_prompt(response: Mapping[str, Any]) -> str | None:
    if not isinstance(response, Mapping):
        return None
    json_payload = response.get("json")
    if isinstance(json_payload, Mapping):
        candidate = json_payload.get("system_prompt")
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    text = response.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _extract_decision_threshold(response: Mapping[str, Any]) -> float | None:
    if not isinstance(response, Mapping):
        return None
    json_payload = response.get("json")
    if isinstance(json_payload, Mapping):
        threshold = json_payload.get("decision_threshold")
        try:
            return float(threshold)
        except (TypeError, ValueError):
            pass
    text = response.get("text")
    if isinstance(text, str) and text.strip():
        match = re.search(r"decision_threshold\s*[:=]\s*([0-9]*\.?[0-9]+)", text, re.I)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _generated_config_path(config: Mapping[str, Any] | None) -> Path:
    if isinstance(config, Mapping):
        prompt_cfg = config.get("prompt_evolution")
        if isinstance(prompt_cfg, Mapping):
            custom_path = prompt_cfg.get("generated_config_path")
            if isinstance(custom_path, str) and custom_path.strip():
                return Path(custom_path)
        out_dir = config.get("out_dir", "outputs")
    else:
        out_dir = "outputs"
    return Path(str(out_dir)) / "system" / "ai_generated_config.yaml"


def _persist_prompt_update(
    system_prompt: str,
    decision_threshold: float | None,
    config: Mapping[str, Any] | None,
) -> None:
    generated_path = _generated_config_path(config)
    generated_path.parent.mkdir(parents=True, exist_ok=True)

    generated_config: dict[str, Any] = {
        "prompt_evolution": {"initial_system_prompt": system_prompt},
    }
    if decision_threshold is not None:
        generated_config["inference"] = {"decision_threshold": float(decision_threshold)}

    try:
        save_config(generated_config, str(generated_path))
        logging.info(
            "Persisted AI-generated config to %s",
            generated_path,
        )
    except Exception as exc:
        logging.warning("Failed to persist AI-generated config: %s", exc)


def evolve_prompt(
    *,
    local_updates: Mapping[str, Mapping[str, Any]],
    previous_global_state: Mapping[str, Any] | None = None,
    config: Mapping[str, Any] | None = None,
    round_index: int = 0,
    current_global_metrics: Mapping[str, Any] | None = None,
) -> Mapping[str, Any] | None:
    config = config or {}
    evolution_cfg = dict(DEFAULT_PROMPT_EVOLUTION_CONFIG)
    evolution_cfg.update(config.get("prompt_evolution", {}) if isinstance(config, Mapping) else {})

    if not evolution_cfg.get("enabled", False):
        return None

    if len(local_updates) < int(evolution_cfg.get("min_hospitals", 2)):
        return None

    top_k = max(1, int(evolution_cfg.get("top_k_hospitals", 1)))
    bottom_k = max(1, int(evolution_cfg.get("bottom_k_hospitals", 1)))
    best, worst = _select_best_and_worst_hospitals(local_updates, top_k=top_k, bottom_k=bottom_k)

    if not best:
        return None

    previous_prompt_evolution = previous_global_state.get("prompt_evolution", {}) if previous_global_state is not None else {}
    current_prompt = previous_prompt_evolution.get("system_prompt")
    golden_prompt = (
        previous_prompt_evolution.get("golden_system_prompt")
        or (config.get("prompt_evolution", {}).get("initial_system_prompt") if isinstance(config, Mapping) else None)
        or current_prompt
    )
    if not current_prompt:
        current_prompt = golden_prompt

    best_system_prompt = previous_prompt_evolution.get("best_system_prompt") or current_prompt
    best_global_metrics = previous_prompt_evolution.get("best_global_metrics") or current_global_metrics or {}

    current_score = _global_metrics_score(current_global_metrics)
    best_score = _global_metrics_score(best_global_metrics)
    performance_delta = float(evolution_cfg.get("performance_delta", 0.0))

    if (
        evolution_cfg.get("golden_prompt_rollback", False)
        and best_system_prompt
        and best_score > 0.0
        and current_score + performance_delta < best_score
    ):
        system_prompt = best_system_prompt
        prompt_state = {
            "system_prompt": system_prompt,
            "golden_system_prompt": golden_prompt,
            "previous_system_prompt": current_prompt,
            "best_system_prompt": best_system_prompt,
            "best_global_metrics": best_global_metrics,
            "top_hospitals": [hid for hid, _ in best],
            "bottom_hospitals": [hid for hid, _ in worst],
            "generated_at_utc": None,
            "reverted": True,
            "prompt_source": "fallback",
            "summary": (
                "Performance decreased relative to the best prior prompt, so the best known prompt is preserved "
                "instead of applying a new meta-agent rewrite."
            ),
        }
        _persist_prompt_update(system_prompt, None, config)
        return prompt_state

    observation = _build_meta_prompt(
        top_hospitals=best,
        bottom_hospitals=worst,
        current_prompt=current_prompt,
        round_index=round_index,
    )

    reasoner = LLMReasoner(
        provider=str(config.get("meta_agent", {}).get("provider", "local") if isinstance(config, Mapping) else "local"),
        model_name=str(config.get("meta_agent", {}).get("model_name", "gpt-3.5-turbo") if isinstance(config, Mapping) else "gpt-3.5-turbo"),
        local_llm_config=(config.get("meta_agent", {}).get("local_llm", {}) if isinstance(config, Mapping) else {}),
        api_key=(config.get("meta_agent", {}).get("api_key") if isinstance(config, Mapping) else None),
    )

    response = reasoner.generate_reasoning(
        "META_AGENT",
        observation,
        patient_context={"reasoning_task": "evolve prompt"},
    )

    system_prompt = _extract_system_prompt(response)
    if not system_prompt:
        logging.warning("Prompt evolution meta-agent returned no usable system prompt.")
        return None

    decision_threshold = _extract_decision_threshold(response)

    summary = None
    if isinstance(response.get("json"), Mapping):
        summary = response["json"].get("summary")
    if not isinstance(summary, str):
        summary = "Derived new system prompt from top-performing and lower-performing hospital reasoning."

    if current_score >= best_score:
        best_system_prompt = current_prompt
        best_global_metrics = current_global_metrics or best_global_metrics

    prompt_state = {
        "system_prompt": system_prompt,
        "decision_threshold": decision_threshold,
        "golden_system_prompt": golden_prompt,
        "previous_system_prompt": current_prompt,
        "best_system_prompt": best_system_prompt,
        "best_global_metrics": best_global_metrics,
        "summary": summary,
        "top_hospitals": [hid for hid, _ in best],
        "bottom_hospitals": [hid for hid, _ in worst],
        "generated_at_utc": response.get("generated_at_utc") if isinstance(response, Mapping) else None,
        "reverted": False,
        "prompt_source": "meta_agent",
    }
    _persist_prompt_update(system_prompt, decision_threshold, config)
    return prompt_state
