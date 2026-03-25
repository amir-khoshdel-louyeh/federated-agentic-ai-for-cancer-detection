from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .contracts import LocalHospitalUpdatePayload


@dataclass(frozen=True)
class WeightComponents:
    """Per-hospital weighting components used by adaptive aggregation."""

    sample_size_weight: float
    quality_weight: float
    reliability_weight: float


def normalize_sample_size_weights(
    local_updates: Mapping[str, LocalHospitalUpdatePayload],
    *,
    split_key: str = "train",
) -> dict[str, float]:
    """Build normalized sample-size weights from `metadata.split_sizes[split_key]`."""
    raw_sizes: dict[str, float] = {}
    for hospital_id, payload in local_updates.items():
        split_sizes = payload["metadata"]["split_sizes"]
        raw_sizes[hospital_id] = float(split_sizes.get(split_key, 0))

    return _normalize_non_negative(raw_sizes)


def extract_quality_scores(
    local_updates: Mapping[str, LocalHospitalUpdatePayload],
    *,
    auc_weight: float = 0.7,
    f1_weight: float = 0.3,
) -> dict[str, float]:
    """Extract normalized quality score using local AUC/F1 evidence."""
    _validate_two_weight_sum(auc_weight, f1_weight, name_a="auc_weight", name_b="f1_weight")

    raw_scores: dict[str, float] = {}
    for hospital_id, payload in local_updates.items():
        local_summary = payload.get("local_summary", {})

        avg_auc = _safe_float(local_summary.get("average_auc"), default=None)
        avg_f1 = _safe_float(local_summary.get("average_f1"), default=None)

        # Fallback to per-agent averages if summary metrics are missing.
        if avg_auc is None or avg_f1 is None:
            avg_auc, avg_f1 = _derive_quality_from_per_agent(payload)

        score = (auc_weight * avg_auc) + (f1_weight * avg_f1)
        raw_scores[hospital_id] = max(0.0, min(1.0, score))

    return _normalize_non_negative(raw_scores)


def compute_reliability_scores(
    local_updates: Mapping[str, LocalHospitalUpdatePayload],
    *,
    expected_lifecycle_state: str = "evaluated",
    lifecycle_penalty: float = 0.35,
    warning_penalty_per_item: float = 0.05,
    min_score: float = 0.0,
) -> dict[str, float]:
    """Compute normalized reliability with lifecycle and warning penalties."""
    if lifecycle_penalty < 0.0 or warning_penalty_per_item < 0.0:
        raise ValueError("Penalties must be non-negative.")

    raw_scores: dict[str, float] = {}
    for hospital_id, payload in local_updates.items():
        reliability = 1.0
        lifecycle_state = str(payload["hospital"].get("lifecycle_state", "")).strip().lower()
        if lifecycle_state != expected_lifecycle_state.lower():
            reliability -= lifecycle_penalty

        training_warnings = payload["metadata"].get("training_warnings", {})
        warning_count = len(training_warnings) if isinstance(training_warnings, Mapping) else 0
        reliability -= float(warning_count) * warning_penalty_per_item

        reliability = max(min_score, min(1.0, reliability))
        raw_scores[hospital_id] = reliability

    return _normalize_non_negative(raw_scores)


def build_adaptive_weights(
    *,
    sample_size_weights: Mapping[str, float],
    quality_weights: Mapping[str, float],
    reliability_weights: Mapping[str, float],
    alpha: float = 0.4,
    beta: float = 0.4,
    gamma: float = 0.2,
) -> tuple[dict[str, float], dict[str, WeightComponents]]:
    """Combine component weights and return final normalized adaptive weights."""
    _validate_three_weight_sum(alpha=alpha, beta=beta, gamma=gamma)

    hospital_ids = _assert_same_hospital_ids(
        sample_size_weights=sample_size_weights,
        quality_weights=quality_weights,
        reliability_weights=reliability_weights,
    )

    raw: dict[str, float] = {}
    components: dict[str, WeightComponents] = {}
    for hospital_id in hospital_ids:
        sample_size_weight = _safe_float(sample_size_weights[hospital_id], default=0.0)
        quality_weight = _safe_float(quality_weights[hospital_id], default=0.0)
        reliability_weight = _safe_float(reliability_weights[hospital_id], default=0.0)

        blended = (
            (alpha * sample_size_weight)
            + (beta * quality_weight)
            + (gamma * reliability_weight)
        )
        raw[hospital_id] = max(0.0, blended)
        components[hospital_id] = WeightComponents(
            sample_size_weight=sample_size_weight,
            quality_weight=quality_weight,
            reliability_weight=reliability_weight,
        )

    return _normalize_non_negative(raw), components


def _derive_quality_from_per_agent(payload: LocalHospitalUpdatePayload) -> tuple[float, float]:
    per_agent = payload.get("metrics", {}).get("per_agent", {})
    if not isinstance(per_agent, Mapping) or not per_agent:
        return 0.0, 0.0

    auc_values: list[float] = []
    f1_values: list[float] = []
    for metrics in per_agent.values():
        if not isinstance(metrics, Mapping):
            continue
        auc_values.append(_clamp01(_safe_float(metrics.get("auc"), default=0.0)))
        f1_values.append(_clamp01(_safe_float(metrics.get("f1"), default=0.0)))

    if not auc_values or not f1_values:
        return 0.0, 0.0

    return (sum(auc_values) / len(auc_values), sum(f1_values) / len(f1_values))


def _normalize_non_negative(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}

    cleaned: dict[str, float] = {key: max(0.0, _safe_float(val, default=0.0)) for key, val in values.items()}
    total = sum(cleaned.values())

    if total > 0.0:
        return {key: value / total for key, value in cleaned.items()}

    uniform = 1.0 / float(len(cleaned))
    return {key: uniform for key in cleaned}


def _assert_same_hospital_ids(
    *,
    sample_size_weights: Mapping[str, float],
    quality_weights: Mapping[str, float],
    reliability_weights: Mapping[str, float],
) -> tuple[str, ...]:
    ids = tuple(sample_size_weights.keys())
    if set(ids) != set(quality_weights.keys()) or set(ids) != set(reliability_weights.keys()):
        raise ValueError(
            "Weight component dictionaries must have identical hospital IDs."
        )
    return ids


def _validate_two_weight_sum(weight_a: float, weight_b: float, *, name_a: str, name_b: str) -> None:
    total = float(weight_a) + float(weight_b)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"{name_a} + {name_b} must equal 1.0, got {total}.")


def _validate_three_weight_sum(*, alpha: float, beta: float, gamma: float) -> None:
    total = float(alpha) + float(beta) + float(gamma)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"alpha + beta + gamma must equal 1.0, got {total}.")


def _safe_float(value: object, *, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value)))
