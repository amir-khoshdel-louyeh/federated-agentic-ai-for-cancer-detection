from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, TypedDict


class BinaryMetrics(TypedDict):
    """Standard binary classification metric bundle."""

    accuracy: float
    f1: float
    auc: float
    sensitivity: float
    specificity: float


class HospitalIdentityPayload(TypedDict):
    """Hospital identity and lifecycle information."""

    hospital_id: str
    lifecycle_state: str


class HospitalMetadataPayload(TypedDict):
    """Metadata attached to each local hospital update."""

    split_sizes: dict[str, int]
    training_warnings: dict[str, str]
    extra: dict[str, Any]


class HospitalMetricsPayload(TypedDict):
    """Metric section exported by each hospital node."""

    per_agent: dict[str, BinaryMetrics]
    selected_performance: dict[str, Any]
    candidate_pattern_comparisons: dict[str, Any]


class LocalSummaryBestAgent(TypedDict):
    """Best local agent snapshot in local summary."""

    name: str
    auc: float


class LocalSummaryPayload(TypedDict):
    """Compact local summary included in each update."""

    num_agents: int
    average_accuracy: float
    average_f1: float
    average_auc: float
    best_agent_by_auc: LocalSummaryBestAgent | None


class ModelUpdateMetadataPayload(TypedDict):
    """Model-update metadata for FL compatibility and upgrades."""

    update_format: str
    has_model_weights: bool


class LocalHospitalUpdatePayload(TypedDict):
    """Typed input payload consumed by federated aggregation."""

    schema_version: str
    generated_at_utc: str
    hospital: HospitalIdentityPayload
    metadata: HospitalMetadataPayload
    selected_patterns: dict[str, str]
    metrics: HospitalMetricsPayload
    local_summary: LocalSummaryPayload
    model_update_metadata: ModelUpdateMetadataPayload


@dataclass(frozen=True)
class AggregationOutput:
    """Canonical aggregator output for one federated round."""

    algorithm: str
    round_index: int
    global_metrics: BinaryMetrics
    hospital_weights: dict[str, float]
    included_hospital_ids: tuple[str, ...]
    dropped_hospitals: dict[str, str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrchestratorRoundOutput:
    """Output package returned after running one FL round."""

    round_index: int
    aggregator_name: str
    global_state: dict[str, Any]
    aggregation: AggregationOutput
    validation_report: dict[str, Any]


class AggregatorContract(Protocol):
    """Contract for all federation aggregation strategies."""

    @property
    def name(self) -> str:
        ...

    def aggregate(
        self,
        *,
        round_index: int,
        local_updates: Mapping[str, LocalHospitalUpdatePayload],
        previous_global_state: Mapping[str, Any] | None = None,
    ) -> AggregationOutput:
        ...


class FederatedOrchestratorContract(Protocol):
    """Contract for orchestrating federated rounds over hospital updates."""

    def run_round(
        self,
        *,
        round_index: int,
        local_updates: Mapping[str, LocalHospitalUpdatePayload],
        previous_global_state: Mapping[str, Any] | None = None,
    ) -> OrchestratorRoundOutput:
        ...

    def run(
        self,
        *,
        total_rounds: int,
        local_updates_by_round: Mapping[int, Mapping[str, LocalHospitalUpdatePayload]],
    ) -> list[OrchestratorRoundOutput]:
        ...
