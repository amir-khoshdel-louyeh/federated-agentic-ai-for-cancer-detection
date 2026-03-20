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


@dataclass(frozen=True)
class FedProxAggregator:
    """FedProx-style metric aggregation with proximal stabilization."""

    split_key: str = "train"
    mu: float = 0.2

    @property
    def name(self) -> str:
        return "fedprox"

    def aggregate(
        self,
        *,
        round_index: int,
        local_updates: Mapping[str, LocalHospitalUpdatePayload],
        previous_global_state: Mapping[str, Any] | None = None,
    ) -> AggregationOutput:
        if not local_updates:
            raise ValueError("FedProx aggregation requires at least one local update.")
        if self.mu < 0.0 or self.mu > 1.0:
            raise ValueError(f"mu must be in [0.0, 1.0], got {self.mu}.")

        hospital_weights = normalize_sample_size_weights(local_updates, split_key=self.split_key)
        per_hospital_metrics = {
            hospital_id: _compute_hospital_metric_mean(payload)
            for hospital_id, payload in local_updates.items()
        }

        current_global_metrics = _weighted_metric_mean(
            per_hospital_metrics=per_hospital_metrics,
            hospital_weights=hospital_weights,
        )

        previous_metrics = _resolve_previous_global_metrics(previous_global_state)
        stabilized_metrics = _blend_metrics(
            current_metrics=current_global_metrics,
            previous_metrics=previous_metrics,
            mu=self.mu,
        )

        return AggregationOutput(
            algorithm=self.name,
            round_index=round_index,
            global_metrics=stabilized_metrics,
            hospital_weights=dict(hospital_weights),
            included_hospital_ids=tuple(local_updates.keys()),
            details={
                "weight_source": f"metadata.split_sizes.{self.split_key}",
                "num_hospitals": len(local_updates),
                "mu": float(self.mu),
                "raw_current_global_metrics": current_global_metrics,
                "used_previous_global": previous_metrics is not None,
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


def _blend_metrics(
    *,
    current_metrics: BinaryMetrics,
    previous_metrics: BinaryMetrics | None,
    mu: float,
) -> BinaryMetrics:
    if previous_metrics is None:
        return dict(current_metrics)

    one_minus_mu = 1.0 - float(mu)
    return {
        metric: (one_minus_mu * float(current_metrics[metric])) + (float(mu) * float(previous_metrics[metric]))
        for metric in METRIC_KEYS
    }


def _resolve_previous_global_metrics(previous_global_state: Mapping[str, Any] | None) -> BinaryMetrics | None:
    if previous_global_state is None:
        return None

    # Preferred shape: {"global_metrics": {...}}
    nested = previous_global_state.get("global_metrics")
    if isinstance(nested, Mapping) and all(metric in nested for metric in METRIC_KEYS):
        return {
            metric: float(nested[metric])
            for metric in METRIC_KEYS
        }

    # Fallback shape: metrics at the top level.
    if all(metric in previous_global_state for metric in METRIC_KEYS):
        return {
            metric: float(previous_global_state[metric])
            for metric in METRIC_KEYS
        }

    return None


def _zero_metrics() -> BinaryMetrics:
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "auc": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
    }
