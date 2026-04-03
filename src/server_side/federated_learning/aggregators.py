from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from src.client_side.hospital.config_helpers import get_cancer_types
from .contracts import AggregationOutput, BinaryMetrics, LocalHospitalUpdatePayload
from .weighting import (
    build_adaptive_weights,
    compute_reliability_scores,
    extract_quality_scores,
    normalize_sample_size_weights,
)

METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)

CANCER_TYPES: tuple[str, ...] = get_cancer_types(None)




@dataclass(frozen=True)
class NoOperationAggregator:
    """No-op aggregator for single-hospital mode: returns local update as global state."""

    @property
    def name(self) -> str:
        return "no_operation"

    def aggregate(
        self,
        *,
        round_index: int,
        local_updates: Mapping[str, LocalHospitalUpdatePayload],
        previous_global_state: Mapping[str, Any] | None = None,
    ) -> AggregationOutput:
        if len(local_updates) != 1:
            raise ValueError("NoOperationAggregator can only be used with a single hospital.")
        hospital_id, payload = next(iter(local_updates.items()))
        # Use the local hospital's metrics as the global metrics
        metrics = payload.get("metrics", {}).get("selected_performance", {})
        # Fallback to per_agent mean if selected_performance is missing
        if not metrics:
            from .aggregators import _compute_hospital_metric_mean
            metrics = _compute_hospital_metric_mean(payload)
        return AggregationOutput(
            algorithm=self.name,
            round_index=round_index,
            global_metrics=metrics,
            hospital_weights={hospital_id: 1.0},
            included_hospital_ids=(hospital_id,),
            details={"note": "No aggregation performed; single-hospital mode."},
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


@dataclass(frozen=True)
class FederatedPerAgentAggregator:
    """Federated per-agent model weight aggregator (for model parameter consolidation)."""

    @property
    def name(self) -> str:
        return "fedavg_per_agent"

    def aggregate(
        self,
        *,
        round_index: int,
        local_updates: Mapping[str, LocalHospitalUpdatePayload],
        previous_global_state: Mapping[str, Any] | None = None,
    ) -> AggregationOutput:
        del previous_global_state

        if not local_updates:
            raise ValueError("FederatedPerAgentAggregator requires at least one local update.")

        # combine metrics with existing fedavg logic
        hospital_weights = normalize_sample_size_weights(local_updates, split_key="train")
        per_hospital_metrics = {
            hospital_id: _compute_hospital_metric_mean(payload)
            for hospital_id, payload in local_updates.items()
        }
        global_metrics = _weighted_metric_mean(
            per_hospital_metrics=per_hospital_metrics,
            hospital_weights=hospital_weights,
        )

        # if model weights available, average per agent key across hospitals
        model_weights = {}
        weight_keys = None
        for payload in local_updates.values():
            mw = payload.get("model_weights")
            if mw is not None:
                weight_keys = mw.keys()
                break

        if weight_keys is not None:
            for agent_key in weight_keys:
                collected = [payload.get("model_weights", {}).get(agent_key) for payload in local_updates.values() if payload.get("model_weights", {}).get(agent_key) is not None]
                if not collected:
                    continue

                # If the payload is a wrapped framework dict
                first = collected[0]
                if isinstance(first, dict) and first.get("framework") == "torch":
                    avg_state = {}
                    for k in first["state_dict"].keys():
                        tensor_list = []
                        for w in collected:
                            val = w["state_dict"].get(k)
                            if isinstance(val, torch.Tensor):
                                tensor_list.append(val.float())
                            elif isinstance(val, np.ndarray):
                                tensor_list.append(torch.from_numpy(val).float())
                            else:
                                tensor_list.append(torch.tensor(float(val)))
                        stacked = torch.stack(tensor_list, dim=0)
                        avg_state[k] = stacked.mean(dim=0)
                    model_weights[agent_key] = {"framework": "torch", "state_dict": avg_state}
                elif isinstance(first, dict) and first.get("framework") == "sklearn":
                    # average coef and intercept for logistic models
                    coef_list = [torch.from_numpy(w["coef"]).float() if isinstance(w["coef"], np.ndarray) else torch.tensor(w["coef"]) for w in collected]
                    intercept_list = [torch.from_numpy(w["intercept"]).float() if isinstance(w["intercept"], np.ndarray) else torch.tensor(w["intercept"]) for w in collected]
                    avg_coef = torch.stack(coef_list, dim=0).mean(dim=0).numpy()
                    avg_intercept = torch.stack(intercept_list, dim=0).mean(dim=0).numpy()
                    model_weights[agent_key] = {
                        "framework": "sklearn",
                        "coef": avg_coef,
                        "intercept": avg_intercept,
                        "classes": first.get("classes", np.array([0, 1])),
                        "n_features_in": first.get("n_features_in", coef_list[0].shape[1] if coef_list else 0),
                    }
                else:
                    # fall back to pass-through for heterogeneous or unsupported representation
                    model_weights[agent_key] = first

        return AggregationOutput(
            algorithm=self.name,
            round_index=round_index,
            global_metrics=global_metrics,
            hospital_weights=dict(hospital_weights),
            included_hospital_ids=tuple(local_updates.keys()),
            details={
                "weight_source": "model-weights-federated-avg",
                "num_hospitals": len(local_updates),
                "model_weights_aggregated": bool(model_weights),
            },
            model_weights=model_weights or None,
        )


@dataclass(frozen=True)
class AdaptiveAggregator:
    """Adaptive FL aggregation using data size, quality, and reliability signals."""

    split_key: str = "train"
    alpha: float = 0.4
    beta: float = 0.4
    gamma: float = 0.2
    auc_weight: float = 0.7
    f1_weight: float = 0.3
    lifecycle_penalty: float = 0.35
    warning_penalty_per_item: float = 0.05
    min_reliability_score: float = 0.0
    cancer_types: tuple[str, ...] = CANCER_TYPES

    @property
    def name(self) -> str:
        return "adaptive"

    def aggregate(
        self,
        *,
        round_index: int,
        local_updates: Mapping[str, LocalHospitalUpdatePayload],
        previous_global_state: Mapping[str, Any] | None = None,
    ) -> AggregationOutput:
        del previous_global_state  # Adaptive here is single-round scoring only.

        if not local_updates:
            raise ValueError("Adaptive aggregation requires at least one local update.")

        sample_size_weights = normalize_sample_size_weights(local_updates, split_key=self.split_key)
        quality_weights = extract_quality_scores(
            local_updates,
            auc_weight=self.auc_weight,
            f1_weight=self.f1_weight,
        )
        reliability_weights = compute_reliability_scores(
            local_updates,
            lifecycle_penalty=self.lifecycle_penalty,
            warning_penalty_per_item=self.warning_penalty_per_item,
            min_score=self.min_reliability_score,
        )

        hospital_weights, component_weights = build_adaptive_weights(
            sample_size_weights=sample_size_weights,
            quality_weights=quality_weights,
            reliability_weights=reliability_weights,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
        )

        per_hospital_metrics = {
            hospital_id: _compute_hospital_metric_mean(payload)
            for hospital_id, payload in local_updates.items()
        }
        global_metrics = _weighted_metric_mean(
            per_hospital_metrics=per_hospital_metrics,
            hospital_weights=hospital_weights,
        )

        per_cancer_weights: dict[str, dict[str, float]] = {}
        per_cancer_global_metrics: dict[str, BinaryMetrics] = {}
        for cancer_type in self.cancer_types:
            cancer_quality_weights = _extract_cancer_quality_scores(
                local_updates,
                cancer_type=cancer_type,
                auc_weight=self.auc_weight,
                f1_weight=self.f1_weight,
            )
            cancer_weights, _ = build_adaptive_weights(
                sample_size_weights=sample_size_weights,
                quality_weights=cancer_quality_weights,
                reliability_weights=reliability_weights,
                alpha=self.alpha,
                beta=self.beta,
                gamma=self.gamma,
            )

            cancer_hospital_metrics = {
                hospital_id: _compute_hospital_metric_for_cancer(payload, cancer_type=cancer_type)
                for hospital_id, payload in local_updates.items()
            }
            per_cancer_weights[cancer_type] = cancer_weights
            per_cancer_global_metrics[cancer_type] = _weighted_metric_mean(
                per_hospital_metrics=cancer_hospital_metrics,
                hospital_weights=cancer_weights,
            )

        return AggregationOutput(
            algorithm=self.name,
            round_index=round_index,
            global_metrics=global_metrics,
            hospital_weights=dict(hospital_weights),
            included_hospital_ids=tuple(local_updates.keys()),
            details={
                "weight_source": {
                    "sample_size": f"metadata.split_sizes.{self.split_key}",
                    "quality": "auc_f1_blend",
                    "reliability": "lifecycle_and_training_warnings",
                },
                "coefficients": {
                    "alpha": float(self.alpha),
                    "beta": float(self.beta),
                    "gamma": float(self.gamma),
                    "auc_weight": float(self.auc_weight),
                    "f1_weight": float(self.f1_weight),
                },
                "num_hospitals": len(local_updates),
                "component_weights": {
                    hospital_id: {
                        "sample_size_weight": components.sample_size_weight,
                        "quality_weight": components.quality_weight,
                        "reliability_weight": components.reliability_weight,
                    }
                    for hospital_id, components in component_weights.items()
                },
                "per_cancer_weights": per_cancer_weights,
                "per_cancer_global_metrics": per_cancer_global_metrics,
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


def _compute_hospital_metric_for_cancer(
    payload: LocalHospitalUpdatePayload,
    *,
    cancer_type: str,
) -> BinaryMetrics:
    per_agent = payload.get("metrics", {}).get("per_agent", {})
    if not per_agent:
        return _zero_metrics()

    aggregates = {metric: 0.0 for metric in METRIC_KEYS}
    count = 0

    for prediction_key, metric_bundle in per_agent.items():
        key_cancer_type = prediction_key.split("::", maxsplit=1)[0].upper()
        if key_cancer_type != cancer_type.upper():
            continue
        count += 1
        for metric in METRIC_KEYS:
            value = float(metric_bundle.get(metric, 0.0))
            aggregates[metric] += value

    if count == 0:
        # Fallback to hospital-level mean if a cancer-specific key is unavailable.
        return _compute_hospital_metric_mean(payload)

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


def _extract_cancer_quality_scores(
    local_updates: Mapping[str, LocalHospitalUpdatePayload],
    *,
    cancer_type: str,
    auc_weight: float,
    f1_weight: float,
) -> dict[str, float]:
    raw_scores: dict[str, float] = {}

    for hospital_id, payload in local_updates.items():
        per_agent = payload.get("metrics", {}).get("per_agent", {})
        auc_values: list[float] = []
        f1_values: list[float] = []

        for prediction_key, metric_bundle in per_agent.items():
            key_cancer_type = prediction_key.split("::", maxsplit=1)[0].upper()
            if key_cancer_type != cancer_type.upper():
                continue
            auc_values.append(float(metric_bundle.get("auc", 0.0)))
            f1_values.append(float(metric_bundle.get("f1", 0.0)))

        if auc_values and f1_values:
            avg_auc = sum(auc_values) / float(len(auc_values))
            avg_f1 = sum(f1_values) / float(len(f1_values))
        else:
            local_summary = payload.get("local_summary", {})
            avg_auc = float(local_summary.get("average_auc", 0.0))
            avg_f1 = float(local_summary.get("average_f1", 0.0))

        raw_scores[hospital_id] = max(0.0, min(1.0, (auc_weight * avg_auc) + (f1_weight * avg_f1)))

    return _normalize_weight_map(raw_scores)


def _normalize_weight_map(values: Mapping[str, float]) -> dict[str, float]:
    if not values:
        return {}

    cleaned = {hospital_id: max(0.0, float(value)) for hospital_id, value in values.items()}
    total = sum(cleaned.values())
    if total > 0.0:
        return {hospital_id: value / total for hospital_id, value in cleaned.items()}

    uniform = 1.0 / float(len(cleaned))
    return {hospital_id: uniform for hospital_id in cleaned}


def _zero_metrics() -> BinaryMetrics:
    return {
        "accuracy": 0.0,
        "f1": 0.0,
        "auc": 0.0,
        "sensitivity": 0.0,
        "specificity": 0.0,
    }
