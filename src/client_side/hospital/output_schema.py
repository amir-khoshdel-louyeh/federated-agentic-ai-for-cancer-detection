from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


SCHEMA_VERSION = "1.0.0"


def build_hospital_output(
    *,
    hospital_id: str,
    lifecycle_state: str,
    selected_patterns: dict[str, str],
    evaluation: dict[str, Any],
    split_sizes: dict[str, int],
    training_warnings: dict[str, str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable, uniform hospital-level output payload for federation."""
    per_agent_metrics = evaluation.get("test", {})
    local_summary = _build_local_summary(per_agent_metrics)

    output = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "hospital": {
            "hospital_id": hospital_id,
            "lifecycle_state": lifecycle_state,
        },
        "metadata": {
            "split_sizes": split_sizes,
            "training_warnings": training_warnings or {},
            "extra": extra_metadata or {},
        },
        "selected_patterns": selected_patterns,
        "metrics": {
            "per_agent": per_agent_metrics,
            "selected_performance": evaluation.get("selected_performance", {}),
            "candidate_pattern_comparisons": evaluation.get("candidate_pattern_comparisons", {}),
        },
        "reasoning": evaluation.get("reasoning", {}),
        "local_summary": local_summary,
        "model_update_metadata": {
            "update_format": "hospital-local-metrics-with-weights",
            "has_model_weights": False,
        },
        "model_weights": None,
    }
    return output


def _build_local_summary(per_agent_metrics: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not per_agent_metrics:
        return {
            "num_agents": 0,
            "average_accuracy": 0.0,
            "average_f1": 0.0,
            "average_auc": 0.0,
            "best_agent_by_auc": None,
        }

    entries = list(per_agent_metrics.items())
    n = len(entries)

    avg_accuracy = sum(float(m.get("accuracy", 0.0)) for _, m in entries) / n
    avg_f1 = sum(float(m.get("f1", 0.0)) for _, m in entries) / n
    avg_auc = sum(float(m.get("auc", 0.0)) for _, m in entries) / n

    best_agent, best_metrics = max(entries, key=lambda item: float(item[1].get("auc", 0.0)))

    return {
        "num_agents": n,
        "average_accuracy": float(avg_accuracy),
        "average_f1": float(avg_f1),
        "average_auc": float(avg_auc),
        "best_agent_by_auc": {
            "name": best_agent,
            "auc": float(best_metrics.get("auc", 0.0)),
        },
    }
