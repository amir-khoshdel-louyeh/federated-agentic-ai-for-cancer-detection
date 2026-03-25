from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class HospitalDataBundle:
    """Standard local split contract used by each hospital node."""

    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


class AgentPortfolioContract(Protocol):
    """Contract for a hospital portfolio of fixed cancer-domain agents."""

    @property
    def cancer_types(self) -> tuple[str, str, str, str]:
        ...

    def get_agent(self, cancer_type: str) -> Any:
        ...

    def set_pattern(self, cancer_type: str, pattern: Any) -> None:
        ...

    def selected_patterns(self) -> dict[str, str]:
        ...

    def train_all(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        ...

    def predict_all(self, x: np.ndarray) -> dict[str, np.ndarray]:
        ...

    def evaluate_all(self, y_true: np.ndarray, predictions: Mapping[str, np.ndarray]) -> dict[str, dict[str, float]]:
        ...


class PatternPolicyContract(Protocol):
    """Contract for assigning active thinking patterns per cancer type."""

    def select_patterns(self) -> dict[str, str]:
        ...


@runtime_checkable
class AdaptivePatternPolicyContract(PatternPolicyContract, Protocol):
    """Optional contract for policies that support validation-driven adaptation."""

    def adapt_patterns(
        self,
        current_mapping: Mapping[str, str],
        validation_scores: Mapping[str, Mapping[str, float]],
    ) -> dict[str, str]:
        ...


@dataclass
class HospitalScope:
    """Declares what a hospital node owns and exports."""

    hospital_id: str
    data: HospitalDataBundle | None = None
    agent_portfolio: AgentPortfolioContract | None = None
    pattern_policy: PatternPolicyContract | None = None
    metrics_store: dict[str, Any] = field(default_factory=dict)
    report_output: dict[str, Any] = field(default_factory=dict)


class HospitalLifecycleContract(ABC):
    """Lifecycle contract for hospital node orchestration."""

    @abstractmethod
    def initialize(self) -> None:
        """Build local state: data, fixed agents, and initial pattern assignment."""

    @abstractmethod
    def train(self) -> None:
        """Train local agents using hospital-owned data."""

    @abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """Evaluate local predictions and return metric artifacts."""

    @abstractmethod
    def export_update(self) -> dict[str, Any]:
        """Export local results/updates using a stable schema for federation."""

    @abstractmethod
    def get_local_update(self) -> dict[str, Any]:
        """Return local update payload prepared for federated aggregation."""

    @abstractmethod
    def apply_global_update(self, global_state: Mapping[str, Any]) -> None:
        """Apply aggregator-provided global state (stub until model syncing is added)."""

    @abstractmethod
    def get_metadata_for_aggregation(self) -> dict[str, Any]:
        """Return compact metadata useful for aggregator weighting/routing logic."""
