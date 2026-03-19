from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.agents import SkinCancerAgent

from .contracts import (
    AdaptivePatternPolicyContract,
    AgentPortfolioContract,
    HospitalLifecycleContract,
    HospitalScope,
    PatternPolicyContract,
)
from .agent_portfolio import AgentPortfolio
from .data_pipeline import LocalDataPipeline, LocalHospitalData
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
        data_pipeline: LocalDataPipeline | None = None,
        agent_portfolio: AgentPortfolioContract | None = None,
        pattern_policy: PatternPolicyContract | None = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.ham_metadata_csv = Path(ham_metadata_csv)
        self.isic_labels_csv = Path(isic_labels_csv)
        self.dataset_handler = dataset_handler or VirtualHospital(random_state=42)
        self.data_pipeline = data_pipeline or LocalDataPipeline(dataset_handler=self.dataset_handler)
        self.local_data: LocalHospitalData | None = None
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
        self.local_data = self.data_pipeline.load(
            ham_metadata_csv=self.ham_metadata_csv,
            isic_labels_csv=self.isic_labels_csv,
        )
        self.scope.data = self.local_data.bundle

        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before initialize().")

        if self.scope.pattern_policy is not None:
            selected_patterns = self.scope.pattern_policy.select_patterns()
            for cancer_type, pattern_name in selected_patterns.items():
                pattern = create_thinking_pattern(pattern_name)
                self.scope.agent_portfolio.set_pattern(cancer_type, pattern)

        self.metrics_store["selected_patterns"] = self.scope.agent_portfolio.selected_patterns()
        self.metrics_store["split_sizes"] = {
            "train": int(self.scope.data.x_train.shape[0]),
            "val": int(self.scope.data.x_val.shape[0]),
            "test": int(self.scope.data.x_test.shape[0]),
        }

        self.metrics_store["lifecycle_state"] = "initialized"

    def get_cancer_filtered_split(
        self,
        cancer_type: str,
        split: Literal["train", "val", "test"] = "train",
        positive_only: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Expose per-cancer local filtering for downstream policy/evaluation logic."""
        if self.local_data is None:
            raise RuntimeError("Call initialize() before requesting cancer-filtered splits.")
        return self.local_data.filter_for_cancer(cancer_type=cancer_type, split=split, positive_only=positive_only)

    def train(self) -> None:
        """Train all fixed cancer agents owned by this hospital node."""
        if self.scope.data is None:
            raise RuntimeError("Call initialize() before train().")
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before train().")
        if self.local_data is None:
            raise RuntimeError("Call initialize() before train().")

        val_predictions: dict[str, np.ndarray] = {}
        test_predictions: dict[str, np.ndarray] = {}
        training_warnings: dict[str, str] = {}

        for cancer_type in self.scope.agent_portfolio.cancer_types:
            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            if not isinstance(agent, SkinCancerAgent):
                raise TypeError(f"Portfolio agent for {cancer_type} must be a SkinCancerAgent.")

            x_train, y_train = self.get_cancer_filtered_split(cancer_type=cancer_type, split="train")
            x_val, _ = self.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            x_test, _ = self.get_cancer_filtered_split(cancer_type=cancer_type, split="test")

            # Some local hospital splits can miss positives for a cancer subtype.
            if np.unique(y_train).size < 2:
                x_train = self.scope.data.x_train
                y_train = self.scope.data.y_train
                training_warnings[cancer_type] = (
                    "Insufficient one-vs-rest label diversity in train split; "
                    "used malignant binary labels fallback."
                )

            agent.fit(x_train, y_train)
            val_predictions[agent.name] = agent.predict_proba(x_val)
            test_predictions[agent.name] = agent.predict_proba(x_test)

        self.metrics_store["predictions"] = {
            "val": val_predictions,
            "test": test_predictions,
        }
        if training_warnings:
            self.metrics_store["training_warnings"] = training_warnings

        self.metrics_store["lifecycle_state"] = "trained"

    def apply_adaptive_pattern_policy(
        self,
        validation_scores: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        """Apply policy-driven pattern replacements after validation comparison."""
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before adaptive policy application.")
        if self.scope.pattern_policy is None:
            return self.scope.agent_portfolio.selected_patterns()

        policy = self.scope.pattern_policy
        if not isinstance(policy, AdaptivePatternPolicyContract):
            return self.scope.agent_portfolio.selected_patterns()

        current_patterns = self.scope.agent_portfolio.selected_patterns()
        updated_patterns = policy.adapt_patterns(current_patterns, validation_scores)

        for cancer_type, pattern_name in updated_patterns.items():
            if current_patterns.get(cancer_type) == pattern_name:
                continue
            self.scope.agent_portfolio.set_pattern(cancer_type, create_thinking_pattern(pattern_name))

        selected = self.scope.agent_portfolio.selected_patterns()
        self.metrics_store["selected_patterns"] = selected
        self.metrics_store["adaptive_policy_applied"] = True
        return selected

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
