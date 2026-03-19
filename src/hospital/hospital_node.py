from __future__ import annotations

from pathlib import Path
from typing import Any

from .contracts import (
    AgentPortfolioContract,
    HospitalDataBundle,
    HospitalLifecycleContract,
    HospitalScope,
    PatternPolicyContract,
)
from .agent_portfolio import AgentPortfolio
from .hospital_env import VirtualHospital
from .pattern_factory import create_thinking_pattern
from .pattern_policy import StaticPatternPolicy


class HospitalNode(HospitalLifecycleContract):
    """Central orchestrator for one hospital in the federated workflow."""

    def __init__(
        self,
        hospital_id: str,
        ham_metadata_csv: str | Path,
        isic_labels_csv: str | Path,
        dataset_handler: VirtualHospital | None = None,
        agent_portfolio: AgentPortfolioContract | None = None,
        pattern_policy: PatternPolicyContract | None = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.ham_metadata_csv = Path(ham_metadata_csv)
        self.isic_labels_csv = Path(isic_labels_csv)
        self.dataset_handler = dataset_handler or VirtualHospital(random_state=42)
        resolved_portfolio = agent_portfolio or AgentPortfolio()
        resolved_policy = pattern_policy or StaticPatternPolicy(hospital_id=hospital_id)

        self.scope = HospitalScope(
            hospital_id=hospital_id,
            data=None,
            agent_portfolio=resolved_portfolio,
            pattern_policy=resolved_policy,
        )

    @property
    def metrics_store(self) -> dict[str, Any]:
        return self.scope.metrics_store

    def initialize(self) -> None:
        """Load local dataset and set initial policy metadata."""
        splits = self.dataset_handler.load(
            ham_metadata_csv=self.ham_metadata_csv,
            isic_labels_csv=self.isic_labels_csv,
        )
        self.scope.data = HospitalDataBundle(
            x_train=splits.x_train,
            y_train=splits.y_train,
            x_val=splits.x_val,
            y_val=splits.y_val,
            x_test=splits.x_test,
            y_test=splits.y_test,
        )

        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before initialize().")

        if self.scope.pattern_policy is not None:
            selected_patterns = self.scope.pattern_policy.select_patterns()
            for cancer_type, pattern_name in selected_patterns.items():
                pattern = create_thinking_pattern(pattern_name)
                self.scope.agent_portfolio.set_pattern(cancer_type, pattern)

        self.metrics_store["selected_patterns"] = self.scope.agent_portfolio.selected_patterns()

        self.metrics_store["lifecycle_state"] = "initialized"

    def train(self) -> None:
        """Train all fixed cancer agents owned by this hospital node."""
        if self.scope.data is None:
            raise RuntimeError("Call initialize() before train().")
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before train().")

        self.scope.agent_portfolio.train_all(self.scope.data.x_train, self.scope.data.y_train)
        self.metrics_store["lifecycle_state"] = "trained"

    def evaluate(self) -> dict[str, Any]:
        """Run local predictions and return per-agent evaluation artifacts."""
        if self.scope.data is None:
            raise RuntimeError("Call initialize() before evaluate().")
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before evaluate().")

        predictions = self.scope.agent_portfolio.predict_all(self.scope.data.x_test)
        metrics = self.scope.agent_portfolio.evaluate_all(self.scope.data.y_test, predictions)

        self.metrics_store["evaluation"] = metrics
        self.scope.report_output = {
            "hospital_id": self.hospital_id,
            "metrics": metrics,
            "selected_patterns": self.metrics_store.get("selected_patterns", {}),
        }
        self.metrics_store["lifecycle_state"] = "evaluated"

        return self.scope.report_output

    def export_update(self) -> dict[str, Any]:
        """Export standardized local update payload for future federation."""
        output = {
            "hospital_id": self.hospital_id,
            "lifecycle_state": self.metrics_store.get("lifecycle_state", "created"),
            "selected_patterns": self.metrics_store.get("selected_patterns", {}),
            "metrics": self.metrics_store.get("evaluation", {}),
        }
        self.scope.report_output = output
        return output
