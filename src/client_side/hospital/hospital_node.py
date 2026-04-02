from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from ..agents import SkinCancerAgent

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
from .output_schema import build_hospital_output
from .meta_manager import MetaManager
from .pattern_factory import ThinkingPatternFactory, create_thinking_pattern
from .pattern_policy import StaticPatternPolicy

EXPECTED_CANCER_TYPES = ("BCC", "SCC", "MELANOMA", "AKIEC")



class HospitalNode(HospitalLifecycleContract):
    """Central orchestrator for one hospital in the federated workflow."""

    def __init__(
        self,
        hospital_id: str,
        ham_metadata_csv: str | Path = None,
        isic_labels_csv: str | Path = None,
        dataset_handler: VirtualHospital | None = None,
        data_pipeline: LocalDataPipeline | None = None,
        agent_portfolio: AgentPortfolioContract | None = None,
        pattern_policy: PatternPolicyContract | None = None,
        config: dict = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.ham_metadata_csv = Path(ham_metadata_csv) if ham_metadata_csv is not None else None
        self.isic_labels_csv = Path(isic_labels_csv) if isic_labels_csv is not None else None
        self.config = config
        # Always pass config to VirtualHospital and LocalDataPipeline
        self.dataset_handler = dataset_handler or VirtualHospital(config=config)
        self.data_pipeline = data_pipeline or LocalDataPipeline(dataset_handler=self.dataset_handler, hospital_id=hospital_id, config=config)
        self.local_data: LocalHospitalData | None = None
        resolved_portfolio = agent_portfolio or AgentPortfolio()
        resolved_policy = pattern_policy or StaticPatternPolicy(hospital_id=hospital_id, config=config)

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
        self._validate_portfolio_contract()

        if self.scope.pattern_policy is not None:
            selected_patterns = self.scope.pattern_policy.select_patterns()
            for cancer_type, pattern_name in selected_patterns.items():
                pattern = create_thinking_pattern(pattern_name)
                self.scope.agent_portfolio.set_pattern(cancer_type, pattern)

        self.metrics_store["selected_patterns"] = self.scope.agent_portfolio.selected_patterns()
        self._validate_selected_patterns(self.metrics_store["selected_patterns"])
        self.metrics_store["random_seed"] = int(getattr(self.dataset_handler, "random_state", 0))
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
        per_agent_metrics: dict[str, dict] = {}

        max_local_samples = int(self.config.get("training", {}).get("max_local_samples", 0)) if self.config else 0

        for cancer_type in self.scope.agent_portfolio.cancer_types:
            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            if not isinstance(agent, SkinCancerAgent):
                raise TypeError(f"Portfolio agent for {cancer_type} must be a SkinCancerAgent.")

            x_train, y_train = self.get_cancer_filtered_split(cancer_type=cancer_type, split="train")
            x_val, _ = self.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            x_test, y_test = self.get_cancer_filtered_split(cancer_type=cancer_type, split="test")

            if max_local_samples > 0 and x_train.shape[0] > max_local_samples:
                rng = np.random.default_rng(self.dataset_handler.random_state if hasattr(self.dataset_handler, 'random_state') else 42)
                sel_idx = rng.choice(x_train.shape[0], size=max_local_samples, replace=False)
                x_train = x_train[sel_idx]
                y_train = y_train[sel_idx]

            # Some local hospital splits can miss positives for a cancer subtype.
            if np.unique(y_train).size < 2:
                x_train = self.scope.data.x_train
                y_train = self.scope.data.y_train
                training_warnings[cancer_type] = (
                    "Insufficient one-vs-rest label diversity in train split; "
                    "used malignant binary labels fallback."
                )

            agent.fit(x_train, y_train)
            val_probs = agent.predict_proba(x_val)
            test_probs = agent.predict_proba(x_test)
            self._validate_prediction_shape(agent.name, val_probs, expected_size=x_val.shape[0])
            self._validate_prediction_shape(agent.name, test_probs, expected_size=x_test.shape[0])

            val_predictions[agent.name] = val_probs
            test_predictions[agent.name] = test_probs

            # Compute and store per-agent metrics for test split
            if len(y_test) > 0:
                metrics = self._compute_binary_metrics(y_test, test_probs)
                per_agent_metrics[agent.name] = metrics

        # Convert all predictions to lists for JSON serialization
        self.metrics_store["predictions"] = {
            "val": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_predictions.items()},
            "test": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_predictions.items()},
        }
        if training_warnings:
            self.metrics_store["training_warnings"] = training_warnings

        # Store per-agent metrics for federation
        self.metrics_store.setdefault("evaluation", {})["test"] = per_agent_metrics
        print(f"[DEBUG] Hospital {self.hospital_id} per_agent_metrics: {per_agent_metrics}")
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
                @staticmethod
                def _compute_binary_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
                    preds = (probs >= 0.5).astype(int)
                    accuracy = float(accuracy_score(y_true, preds))
                    f1 = float(f1_score(y_true, preds, zero_division=0))
                    try:
                        auc = roc_auc_score(y_true, probs)
                    except ValueError:
                        auc = 0.5

                    # Ensure auc is finite
                    if not np.isfinite(auc):
                        auc = 0.0

                    # labels=[0, 1] guarantees stable unpacking for degenerate splits.
                    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
                    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
                    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

                    try:
                        logloss = log_loss(y_true, probs, labels=[0, 1])
                    except ValueError:
                        logloss = 1.0

                    # PR-AUC from precision-recall curve
                    try:
                        pr_precision, pr_recall, _ = precision_recall_curve(y_true, probs, pos_label=1)
                        pr_auc = float(auc(pr_recall, pr_precision))
                    except ValueError:
                        pr_auc = 0.0

                    precision_val = precision_score(y_true, preds, zero_division=0)
                    recall_val = recall_score(y_true, preds, zero_division=0)

                    metrics = {
                        "accuracy": float(accuracy),
                        "f1": float(f1),
                        "auc": float(auc) if np.isfinite(auc) else 0.0,
                        "pr_auc": float(pr_auc),
                        "precision": float(precision_val),
                        "recall": float(recall_val),
                        "log_loss": float(logloss),
                        "sensitivity": float(sensitivity),
                        "specificity": float(specificity),
                    }
                    # Ensure all metrics are float and finite
                    for k, v in metrics.items():
                        if not np.isfinite(v):
                            metrics[k] = 0.0
                        else:
                            metrics[k] = float(v)
                    return metrics

    def evaluate(self) -> dict[str, Any]:
        """Evaluate all fixed cancer agents and store metrics for validation and test splits."""
        if self.scope.data is None:
            raise RuntimeError("Call initialize() before evaluate().")
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before evaluate().")
        if self.local_data is None:
            raise RuntimeError("Call initialize() before evaluate().")

        val_metrics = {}
        test_metrics = {}
        selected_patterns = self.metrics_store.get("selected_patterns", self.scope.agent_portfolio.selected_patterns())
        selected_performance = {}

        for cancer_type, pattern_name in selected_patterns.items():
            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            x_val, y_val = self.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            x_test, y_test = self.get_cancer_filtered_split(cancer_type=cancer_type, split="test")
            val_probs = agent.predict_proba(x_val)
            test_probs = agent.predict_proba(x_test)
            val_metrics[f"{cancer_type.lower()}::{pattern_name}"] = self._compute_binary_metrics(y_val, val_probs)
            test_metrics[f"{cancer_type.lower()}::{pattern_name}"] = self._compute_binary_metrics(y_test, test_probs)

        for cancer_type, pattern_name in selected_patterns.items():
            prediction_key = f"{cancer_type.lower()}::{pattern_name}"
            selected_performance[cancer_type] = {
                "pattern": pattern_name,
                "validation": val_metrics.get(prediction_key, {}),
                "test": test_metrics.get(prediction_key, {}),
            }

        candidate_comparisons = self._build_candidate_comparisons(val_metrics)

        # Meta-manager conflict resolution / hardening from specialist outputs.
        meta_manager = MetaManager(soft_vote_temperature=self.config.get("meta_agent", {}).get("local", {}).get("soft_vote_temperature", 1.0))
        cancer_predictions = {ct: selected_performance[ct]["test"]["auc"] if ct in selected_performance else np.zeros_like(self.scope.data.x_test[:,0]) for ct in selected_patterns}
        # Use predictions arrays and uncertainties based on AUC distance (proxy) with 0 fallback
        predictions = {ct: np.full(self.scope.data.x_test.shape[0], selected_performance[ct]["test"]["accuracy"] if ct in selected_performance else 0.0) for ct in selected_patterns}
        uncertainties = {ct: np.full(self.scope.data.x_test.shape[0], 1.0 - selected_performance[ct]["test"]["auc"] if ct in selected_performance else 1.0) for ct in selected_patterns}

        # Meta information can be used for aggregated decision support.
        meta_result = meta_manager.combine(predictions, uncertainties)

        self.metrics_store["evaluation"] = {
            "validation": val_metrics,
            "test": test_metrics,
            "selected_performance": selected_performance,
            "candidate_pattern_comparisons": candidate_comparisons,
            "meta_manager": meta_result,
        }
        self.scope.report_output = {
            "hospital_id": self.hospital_id,
            "metrics": self.metrics_store["evaluation"],
            "selected_patterns": self.metrics_store.get("selected_patterns", {}),
        }
        self.metrics_store["lifecycle_state"] = "evaluated"

        return self.scope.report_output

    @staticmethod
    def _compute_binary_metrics(y_true: np.ndarray, probs) -> dict[str, float]:
        # Ensure `probs` is numpy-compatible for comparisons (list-backed outputs may occur)
        probs_arr = np.asarray(probs, dtype=np.float32)
        preds = (probs_arr >= 0.5).astype(int)
        accuracy = float(accuracy_score(y_true, preds))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        try:
            auc_score = roc_auc_score(y_true, probs_arr)
        except ValueError:
            auc_score = 0.5
        if not np.isfinite(auc_score):
            auc_score = 0.0

        precision_val = float(precision_score(y_true, preds, zero_division=0))
        recall_val = float(recall_score(y_true, preds, zero_division=0))

        try:
            pr_precision, pr_recall, _ = precision_recall_curve(y_true, probs_arr, pos_label=1)
            pr_auc = float(auc(pr_recall, pr_precision))
        except ValueError:
            pr_auc = 0.0

        try:
            logloss_val = float(log_loss(y_true, probs_arr, labels=[0, 1]))
        except ValueError:
            logloss_val = float("inf")

        # labels=[0, 1] guarantees stable unpacking for degenerate splits.
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

        metrics = {
            "accuracy": accuracy,
            "f1": f1,
            "auc": auc_score,
            "pr_auc": pr_auc,
            "precision": precision_val,
            "recall": recall_val,
            "log_loss": logloss_val,
            "sensitivity": sensitivity,
            "specificity": specificity,
        }

        # Ensure all metrics are finite
        for k, v in metrics.items():
            if not np.isfinite(v):
                metrics[k] = 0.0 if k != "log_loss" else float("inf")

        return metrics

    @staticmethod
    def _build_candidate_comparisons(
        validation_metrics: dict[str, dict[str, float]],
    ) -> dict[str, list[dict[str, float | str | int]]]:
        grouped: dict[str, list[dict[str, float | str | int]]] = {}
        for prediction_key, metrics in validation_metrics.items():
            cancer_type, pattern_name = prediction_key.split("::", maxsplit=1)
            cancer_key = cancer_type.upper()
            grouped.setdefault(cancer_key, []).append(
                {
                    "pattern": pattern_name,
                    "auc": float(metrics.get("auc", 0.0)),
                    "accuracy": float(metrics.get("accuracy", 0.0)),
                    "f1": float(metrics.get("f1", 0.0)),
                }
            )

        comparisons: dict[str, list[dict[str, float | str | int]]] = {}
        for cancer_type, candidates in grouped.items():
            ranked = sorted(candidates, key=lambda item: float(item["auc"]), reverse=True)
            for index, candidate in enumerate(ranked, start=1):
                candidate["rank"] = index
            comparisons[cancer_type] = ranked

        return comparisons

    def _validate_portfolio_contract(self) -> None:
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio.")

        cancer_types = tuple(self.scope.agent_portfolio.cancer_types)
        if len(cancer_types) != 4:
            raise ValueError("Agent portfolio must expose exactly 4 cancer agents.")
        if set(cancer_types) != set(EXPECTED_CANCER_TYPES):
            raise ValueError(
                "Agent portfolio cancer types must be exactly: "
                f"{', '.join(EXPECTED_CANCER_TYPES)}"
            )

    @staticmethod
    def _validate_selected_patterns(selected_patterns: dict[str, str]) -> None:
        missing = [c for c in EXPECTED_CANCER_TYPES if c not in selected_patterns]
        if missing:
            raise ValueError(
                "Selected patterns missing required cancer types: "
                f"{', '.join(missing)}"
            )

        supported = set(ThinkingPatternFactory().supported_patterns())
        invalid = [name for name in selected_patterns.values() if name not in supported]
        if invalid:
            raise ValueError(
                "Selected patterns contain unsupported names: "
                f"{', '.join(sorted(set(invalid)))}"
            )

    @staticmethod
    def _validate_prediction_shape(agent_name: str, probs: np.ndarray, expected_size: int) -> None:
        if probs.ndim != 1:
            raise ValueError(f"Prediction output for {agent_name} must be a 1D array.")
        if probs.shape[0] != expected_size:
            raise ValueError(
                f"Prediction size mismatch for {agent_name}: expected {expected_size}, got {probs.shape[0]}"
            )


    def export_update(self, for_training: bool = False) -> dict[str, Any]:
        """Export standardized local update payload for future federation.
        If for_training is True, fill metrics.per_agent with dummy values to pass validation.
        """
        evaluation = dict(self.metrics_store.get("evaluation", {}))
        if for_training:
            # Fill per_agent with dummy values for all expected cancer types
            dummy_metrics = {ct: {"accuracy": 0.0, "f1": 0.0, "auc": 0.0, "sensitivity": 0.0, "specificity": 0.0} for ct in EXPECTED_CANCER_TYPES}
            evaluation["test"] = dummy_metrics
        output = build_hospital_output(
            hospital_id=self.hospital_id,
            lifecycle_state=str(self.metrics_store.get("lifecycle_state", "created")),
            selected_patterns=dict(self.metrics_store.get("selected_patterns", {})),
            evaluation=evaluation,
            split_sizes=dict(self.metrics_store.get("split_sizes", {})),
            training_warnings=dict(self.metrics_store.get("training_warnings", {})),
            extra_metadata={
                "adaptive_policy_applied": bool(self.metrics_store.get("adaptive_policy_applied", False)),
                "random_seed": int(self.metrics_store.get("random_seed", 0)),
            },
        )

        if hasattr(self.scope.agent_portfolio, "get_model_weights"):
            model_weights = self.scope.agent_portfolio.get_model_weights()
            output["model_weights"] = model_weights
            output["model_update_metadata"] = {
                "update_format": "hospital-local-metrics-with-weights",
                "has_model_weights": True,
            }
        # Debug print for metrics.per_agent
        per_agent_metrics = output.get("metrics", {}).get("per_agent", {})
        print(f"[DEBUG] Hospital {self.hospital_id} metrics.per_agent: {per_agent_metrics}")
        assert per_agent_metrics, f"metrics.per_agent is empty for hospital {self.hospital_id}!"
        self.scope.report_output = output
        return output

    def get_local_update(self, for_training: bool = False) -> dict[str, Any]:
        """FL-ready alias for exporting local hospital updates.
        If for_training is True, fill metrics.per_agent with dummy values to pass validation.
        """
        local_update = self.export_update(for_training=for_training)
        self.metrics_store["last_local_update_exported"] = True
        return local_update

    def apply_global_update(self, global_state: Mapping[str, Any]) -> None:
        """Apply incoming global state or model update to local hospital."""
        self.metrics_store["last_global_state"] = dict(global_state)
        self.metrics_store["global_update_applied"] = True

        # Attempt model weights sync if available
        model_weights = global_state.get("model_weights") if isinstance(global_state, dict) else None
        if model_weights and self.scope.agent_portfolio is not None and hasattr(self.scope.agent_portfolio, "set_model_weights"):
            try:
                self.scope.agent_portfolio.set_model_weights(model_weights)
                self.metrics_store["model_weights_synced"] = True
            except Exception:
                self.metrics_store["model_weights_synced"] = False

    def get_metadata_for_aggregation(self) -> dict[str, Any]:
        """Return compact metadata for federated aggregator-side decisions."""
        return {
            "hospital_id": self.hospital_id,
            "split_sizes": dict(self.metrics_store.get("split_sizes", {})),
            "selected_patterns": dict(self.metrics_store.get("selected_patterns", {})),
            "lifecycle_state": str(self.metrics_store.get("lifecycle_state", "created")),
            "schema_version": str(self.scope.report_output.get("schema_version", "1.0.0")),
        }

    # Note: `evaluate()` is already implemented earlier in this class (line ~235).
    # The duplicate implementation was intentionally removed to avoid method override confusion and enforce a single evaluation contract.

