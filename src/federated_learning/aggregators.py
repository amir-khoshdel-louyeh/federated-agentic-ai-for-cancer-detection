from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .contracts import AggregationOutput, BinaryMetrics, LocalHospitalUpdatePayload
from .weighting import normalize_sample_size_weights

METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)


@dataclass(frozen=True)
class FedAvgAggregator:
    """Federated averaging over hospital metric summaries."""

    split_key: str = "train"

    @property
    def name(self) -> str:
        return "fedavg"

    def aggregate(
        self,
        *,
        round_index: int,
        local_updates: Mapping[str, LocalHospitalUpdatePayload],
        previous_global_state: Mapping[str, Any] | None = None,
    ) -> AggregationOutput:
        del previous_global_state  # Not used in FedAvg.

        if not local_updates:
            raise ValueError("FedAvg aggregation requires at least one local update.")

        hospital_weights = normalize_sample_size_weights(local_updates, split_key=self.split_key)

        # Convert each hospital update into one metric vector before weighted fusion.
        per_hospital_metrics = {
            hospital_id: _compute_hospital_metric_mean(payload)
            for hospital_id, payload in local_updates.items()
        }

        global_metrics = _weighted_metric_mean(
            per_hospital_metrics=per_hospital_metrics,
            hospital_weights=hospital_weights,
        )

        return AggregationOutput(
            algorithm=self.name,
            round_index=round_index,
            global_metrics=global_metrics,
            hospital_weights=dict(hospital_weights),
            included_hospital_ids=tuple(local_updates.keys()),
            details={
                "weight_source": f"metadata.split_sizes.{self.split_key}",
                "num_hospitals": len(local_updates),
            },
        )


def _compute_hospital_metric_mean(payload: LocalHospitalUpdatePayload) -> BinaryMetrics:
    per_agent = payload.get("metrics", {}).get("per_agent", {})
    if not per_agent:
        # Validation should prevent this, but a deterministic fallback keeps the path safe.
        return _zero_metrics()

    aggregates = {metric: 0.0 for metric in METRIC_KEYS}
    count = 0

    for metric_bundle in per_agent.values():
        count += 1
        for metric in METRIC_KEYS:
            value = float(metric_bundle.get(metric, 0.0))
            aggregates[metric] += value

    if count == 0:
        return _zero_metrics()

    return {
        metric: aggregates[metric] / float(count)
        for metric in METRIC_KEYS
    }


def _weighted_metric_mean(
    *,
    per_hospital_metrics: Mapping[str, BinaryMetrics],
    hospital_weights: Mapping[str, float],
) -> BinaryMetrics:
    weighted = {metric: 0.0 for metric in METRIC_KEYS}

    for hospital_id, metrics in per_hospital_metrics.items():
        weight = float(hospital_weights.get(hospital_id, 0.0))
        for metric in METRIC_KEYS:
            weighted[metric] += weight * float(metrics[metric])

    return {
        metric: weighted[metric]
        for metric in METRIC_KEYS
    }


def _zero_metrics() -> BinaryMetrics:
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "auc": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
    }
