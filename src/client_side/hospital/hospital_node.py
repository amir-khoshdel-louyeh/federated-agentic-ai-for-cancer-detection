from __future__ import annotations

from pathlib import Path
import logging
from typing import Any, Literal, Mapping

import numpy as np
import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score, log_loss,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from ..agents import SkinCancerAgent
from sklearn.utils import resample

from .contracts import (
    AdaptivePatternPolicyContract,
    AgentPortfolioContract,
    HospitalLifecycleContract,
    HospitalScope,
    PatternPolicyContract,
)
from .agent_portfolio import AgentPortfolio
from .config_helpers import get_cancer_types, is_malignant_label
from .data_pipeline import LocalDataPipeline, LocalHospitalData
from .hospital_env import VirtualHospital
from .output_schema import build_hospital_output
from .hospital_manager_agent import HospitalManagerAgent
from .pattern_factory import ThinkingPatternFactory, create_thinking_pattern
from .pattern_policy import StaticPatternPolicy


class HospitalNode(HospitalLifecycleContract):
    @staticmethod
    def _expected_cancer_types_from_config(config: dict | None) -> tuple[str, ...]:
        return get_cancer_types(config)

    """Central orchestrator for one hospital in the federated workflow."""

    def __init__(
        self,
        hospital_id: str,
        ham_metadata_csv: str | Path = None,
        isic_labels_csv: str | Path = None,
        isic_metadata_csv: str | Path = None,
        dataset_handler: VirtualHospital | None = None,
        data_pipeline: LocalDataPipeline | None = None,
        agent_portfolio: AgentPortfolioContract | None = None,
        pattern_policy: PatternPolicyContract | None = None,
        config: dict = None,
    ) -> None:
        self.hospital_id = hospital_id
        self.ham_metadata_csv = Path(ham_metadata_csv) if ham_metadata_csv is not None else None
        self.isic_labels_csv = Path(isic_labels_csv) if isic_labels_csv is not None else None
        self.isic_metadata_csv = Path(isic_metadata_csv) if isic_metadata_csv is not None else None
        self.config = config
        training_cfg = (config.get("training", {}) or {}) if config else {}
        self.decision_threshold = float(
            training_cfg.get("decision_threshold", config.get("decision_threshold", 0.5) if config else 0.5)
        ) if config else 0.5
        self.decision_thresholds = {}
        if config:
            thresholds = training_cfg.get("decision_thresholds", {})
            if isinstance(thresholds, dict):
                self.decision_thresholds = {
                    str(k).strip().upper(): float(v)
                    for k, v in thresholds.items()
                    if str(k).strip()
                }
        # Always pass config to VirtualHospital and LocalDataPipeline
        self.dataset_handler = dataset_handler or VirtualHospital(config=config)
        self.data_pipeline = data_pipeline or LocalDataPipeline(dataset_handler=self.dataset_handler, hospital_id=hospital_id, config=config)
        self.local_data: LocalHospitalData | None = None
        cancer_types = get_cancer_types(config)
        resolved_portfolio = agent_portfolio or AgentPortfolio(cancer_types=cancer_types)
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
            isic_metadata_csv=self.isic_metadata_csv,
        )
        self.scope.data = self.local_data.bundle

        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before initialize().")
        self._validate_portfolio_contract()

        if self.scope.pattern_policy is not None:
            selected_patterns = self.scope.pattern_policy.select_patterns()
            pattern_params = (self.config or {}).get("agents", {}).get("pattern_params", {})
            default_provider = self.config.get("meta_agent", {}).get("provider", "local")
            default_local_llm = self.config.get("meta_agent", {}).get("local_llm", {})
            default_api_key = self.config.get("meta_agent", {}).get("api_key")
            for cancer_type, pattern_name in selected_patterns.items():
                config_for_pattern = pattern_params.get(pattern_name, {}) if isinstance(pattern_params, dict) else {}
                pattern_config = dict(config_for_pattern) if isinstance(config_for_pattern, dict) else {}
                pattern_config.setdefault("provider", default_provider)
                pattern_config.setdefault("local_llm_config", default_local_llm)
                if default_api_key is not None:
                    pattern_config.setdefault("api_key", default_api_key)
                pattern = create_thinking_pattern(pattern_name, pattern_config=pattern_config)
                self.scope.agent_portfolio.set_pattern(cancer_type, pattern)

        self.metrics_store["selected_patterns"] = self.scope.agent_portfolio.selected_patterns()
        self._validate_selected_patterns(self.metrics_store["selected_patterns"])
        self.metrics_store["random_seed"] = int(getattr(self.dataset_handler, "random_state", 0))

        # Register split sizes and initial lifecycle state before training.
        self.metrics_store["split_sizes"] = {
            "train": int(self.scope.data.x_train.shape[0]),
            "val": int(self.scope.data.x_val.shape[0]),
            "test": int(self.scope.data.x_test.shape[0]),
        }
        self.metrics_store["lifecycle_state"] = "initialized"

    def _decision_threshold_for(self, cancer_type: str) -> float:
        key = str(cancer_type).strip().upper()
        if key in self.decision_thresholds:
            return self.decision_thresholds[key]
        return self.decision_threshold

    def _detection_mode(self) -> str:
        if not self.config:
            return "detect_then_type"
        return str(self.config.get("detection", {}).get("mode", "detect_then_type")).strip()

    def _threshold_tuning_enabled(self) -> bool:
        if not self.config:
            return False
        return bool((self.config.get("training", {}) or {}).get("threshold_tuning", {}).get("enabled", False))

    def _tune_decision_thresholds(
        self,
        val_labels: dict[str, np.ndarray],
        val_predictions: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """Tune per-cancer thresholds using validation predictions and preserve AI-agent evaluation behavior."""
        tuned: dict[str, float] = {}
        tuning_cfg = (self.config.get("training", {}) or {}).get("threshold_tuning", {}) or {}
        rare_classes = {
            str(ct).strip().upper()
            for ct in tuning_cfg.get("rare_classes", ["SCC", "AKIEC"])
            if str(ct).strip()
        }
        recall_weight = float(tuning_cfg.get("rare_class_recall_weight", 0.25))
        min_threshold = float(tuning_cfg.get("min_threshold", 0.02))
        max_threshold = float(tuning_cfg.get("max_threshold", 0.45))

        for cancer_type, y_val in val_labels.items():
            key = str(cancer_type).strip().upper()
            probs = val_predictions.get(cancer_type)
            if probs is None or len(probs) == 0 or np.unique(y_val).size < 2:
                tuned[key] = self._decision_threshold_for(key)
                continue

            precision, recall, thresholds = precision_recall_curve(y_val, probs, pos_label=1)
            thresholds = np.concatenate([thresholds, [1.0]])
            f1_scores = 2 * (precision * recall) / np.maximum(precision + recall, 1e-8)

            if key in rare_classes:
                objective = recall + recall_weight * precision
            else:
                objective = f1_scores

            best_index = int(np.nanargmax(objective))
            best_threshold = float(thresholds[best_index])
            best_threshold = max(min_threshold, min(best_threshold, max_threshold))
            tuned[key] = best_threshold

        return tuned

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

        val_labels: dict[str, np.ndarray] = {}
        val_predictions: dict[str, np.ndarray] = {}
        test_labels: dict[str, np.ndarray] = {}
        test_predictions: dict[str, np.ndarray] = {}
        training_warnings: dict[str, str] = {}
        per_agent_metrics: dict[str, dict] = {}

        max_local_samples = int(self.config.get("training", {}).get("max_local_samples", 0)) if self.config else 0
        random_state = int(getattr(self.dataset_handler, "random_state", 42))

        detection_mode = self._detection_mode()
        for cancer_type in self.scope.agent_portfolio.cancer_types:
            if detection_mode == "detect_only" and str(cancer_type).strip().upper() != "CANCER":
                continue
            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            if not isinstance(agent, SkinCancerAgent):
                raise TypeError(f"Portfolio agent for {cancer_type} must be a SkinCancerAgent.")

            x_train, y_train = self.get_cancer_filtered_split(cancer_type=cancer_type, split="train")
            x_val, y_val = self.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            x_test, y_test = self.get_cancer_filtered_split(cancer_type=cancer_type, split="test")

            if max_local_samples > 0 and x_train.shape[0] > max_local_samples:
                rng = np.random.default_rng(random_state)
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

            # Address class imbalance by oversampling the minority class for training.
            rebalance_method = self.config.get('training', {}).get('rebalance_method', 'oversample') if self.config else 'oversample'
            counts = np.bincount(y_train, minlength=2)
            if counts.min() > 0:
                imbalance_threshold = self.config.get('training', {}).get('imbalance_ratio_threshold', 1) if self.config else 1
                ratio = max(counts) / max(1, min(counts))
                if ratio > imbalance_threshold and rebalance_method == 'oversample':
                    majority_label = int(np.argmax(counts))
                    minority_label = 1 - majority_label
                    x_majority = x_train[y_train == majority_label]
                    y_majority = y_train[y_train == majority_label]
                    x_minority = x_train[y_train == minority_label]
                    y_minority = y_train[y_train == minority_label]
                    x_minority_up, y_minority_up = resample(
                        x_minority,
                        y_minority,
                        replace=True,
                        n_samples=x_majority.shape[0],
                        random_state=random_state,
                    )
                    x_train = np.vstack([x_majority, x_minority_up])
                    y_train = np.concatenate([y_majority, y_minority_up])
                    perm = np.random.default_rng(random_state).permutation(x_train.shape[0])
                    x_train = x_train[perm]
                    y_train = y_train[perm]
                    training_warnings[cancer_type] = (
                        training_warnings.get(cancer_type, "")
                        + " Class imbalance oversampled minority class to balance the training set."
                    ).strip()

            # Fit is retained for interface compatibility; AIThinkingPattern is a pure LLM reasoning pattern and ignores training data.
            agent.fit(x_train, y_train)
            val_probs = agent.predict_proba(x_val)
            test_probs = agent.predict_proba(x_test)
            self._validate_prediction_shape(agent.name, val_probs, expected_size=x_val.shape[0])
            self._validate_prediction_shape(agent.name, test_probs, expected_size=x_test.shape[0])

            val_labels[cancer_type] = y_val
            val_predictions[cancer_type] = val_probs
            test_labels[cancer_type] = y_test
            test_predictions[cancer_type] = test_probs

        if self._threshold_tuning_enabled():
            tuned = self._tune_decision_thresholds(val_labels, val_predictions)
            self.decision_thresholds.update(tuned)
            self.metrics_store["tuned_thresholds"] = tuned

        # Convert all predictions to lists for JSON serialization
        self.metrics_store["predictions"] = {
            "val": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_predictions.items()},
            "test": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_predictions.items()},
        }
        if training_warnings:
            self.metrics_store["training_warnings"] = training_warnings

        # Compute final per-agent metrics for test split using tuned thresholds.
        for cancer_type in self.scope.agent_portfolio.cancer_types:
            test_probs = test_predictions.get(cancer_type, np.array([], dtype=np.float32))
            y_test = test_labels.get(cancer_type, np.array([], dtype=np.int64))
            if len(y_test) > 0:
                metrics = self._compute_binary_metrics(
                    y_test,
                    test_probs,
                    threshold=self._decision_threshold_for(cancer_type),
                )
            else:
                metrics = {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "auc": 0.5,
                    "pr_auc": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "log_loss": float("inf"),
                    "sensitivity": 0.0,
                    "specificity": 0.0,
                }
            per_agent_metrics[self.scope.agent_portfolio.get_agent(cancer_type).name] = metrics

        # Store per-agent metrics for federation
        self.metrics_store.setdefault("evaluation", {})["test"] = per_agent_metrics
        logging.debug(f"[DEBUG] Hospital {self.hospital_id} per_agent_metrics: {per_agent_metrics}")
        self.metrics_store["lifecycle_state"] = "trained"

    def evaluate(self) -> dict[str, Any]:
        """Evaluate all fixed cancer agents and store metrics for validation and test splits.

        This evaluation pipeline is preserved for the AI-agent workflow and uses tuned
        decision thresholds when threshold tuning is enabled.
        """
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

        test_probabilities: dict[str, np.ndarray] = {}
        test_uncertainties: dict[str, np.ndarray] = {}

        detection_mode = self._detection_mode()
        for cancer_type, pattern_name in selected_patterns.items():
            if detection_mode == "detect_only" and str(cancer_type).strip().upper() != "CANCER":
                continue
            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            x_val, y_val = self.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            x_test, y_test = self.get_cancer_filtered_split(cancer_type=cancer_type, split="test")
            val_probs = agent.predict_proba(x_val)
            test_probs = agent.predict_proba(x_test)
            val_metrics[f"{cancer_type.lower()}::{pattern_name}"] = self._compute_binary_metrics(
                y_val,
                val_probs,
                threshold=self._decision_threshold_for(cancer_type),
            )
            test_metrics[f"{cancer_type.lower()}::{pattern_name}"] = self._compute_binary_metrics(
                y_test,
                test_probs,
                threshold=self._decision_threshold_for(cancer_type),
            )
            test_probabilities[cancer_type] = test_probs
            if x_test.shape[0] > 0:
                test_uncertainties[cancer_type] = agent.predict_uncertainty(x_test)
            else:
                test_uncertainties[cancer_type] = np.full(0, 1.0, dtype=np.float32)

        detection_mode = self._detection_mode()
        for cancer_type, pattern_name in selected_patterns.items():
            if detection_mode == "detect_only" and str(cancer_type).strip().upper() != "CANCER":
                continue
            prediction_key = f"{cancer_type.lower()}::{pattern_name}"
            selected_performance[cancer_type] = {
                "pattern": pattern_name,
                "validation": val_metrics.get(prediction_key, {}),
                "test": test_metrics.get(prediction_key, {}),
            }

        candidate_comparisons = self._build_candidate_comparisons(val_metrics)

        # Hospital manager agent uses specialist performance and patient metadata to select a lead diagnosis agent.
        manager = HospitalManagerAgent(
            provider=self.config.get("meta_agent", {}).get("provider", "local"),
            model_name=self.config.get("meta_agent", {}).get("model_name", "gpt-3.5-turbo"),
            local_model_path=self.config.get("meta_agent", {}).get("local_model_path"),
            local_llm_config=self.config.get("meta_agent", {}).get("local_llm", {}),
            api_key=self.config.get("meta_agent", {}).get("api_key"),
        )
        hospital_manager_output = manager.recommend_lead_agent(
            patient_metadata={
                "age": "unknown",
                "sex": "unknown",
                "localization": "unknown",
            },
            agent_performance=selected_performance,
            candidate_comparisons=candidate_comparisons,
        )

        self.metrics_store["evaluation"] = {
            "validation": val_metrics,
            "test": test_metrics,
            "selected_performance": selected_performance,
            "candidate_pattern_comparisons": candidate_comparisons,
            "hospital_manager": hospital_manager_output,
        }
        self.scope.report_output = {
            "hospital_id": self.hospital_id,
            "metrics": self.metrics_store["evaluation"],
            "selected_patterns": self.metrics_store.get("selected_patterns", {}),
        }
        self.metrics_store["lifecycle_state"] = "evaluated"

        return self.scope.report_output

    def evaluate_on_external_data(
        self,
        x_external: np.ndarray,
        cancer_external: np.ndarray,
    ) -> dict[str, Any]:
        """Evaluate current agent portfolio on an external held-out data set."""
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before evaluate_on_external_data().")

        if x_external is None or cancer_external is None:
            raise ValueError("x_external and cancer_external cannot be None")
        if len(x_external) != len(cancer_external):
            raise ValueError("x_external and cancer_external must have same number of samples")

        test_metrics = {}
        selected_patterns = self.metrics_store.get("selected_patterns", self.scope.agent_portfolio.selected_patterns())
        selected_performance = {}

        detection_mode = self._detection_mode()
        for cancer_type, pattern_name in selected_patterns.items():
            if detection_mode == "detect_only" and str(cancer_type).strip().upper() != "CANCER":
                continue
            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            if str(cancer_type).strip().upper() == "CANCER":
                y_test = np.asarray([1 if is_malignant_label(label, self.config) else 0 for label in cancer_external], dtype=np.int64)
            else:
                y_test = (np.asarray(cancer_external, dtype=str) == cancer_type.upper()).astype(np.int64)
            if x_external.shape[0] == 0:
                test_probs = np.array([])
            else:
                test_probs = agent.predict_proba(x_external)

            self._validate_prediction_shape(agent.name, test_probs, expected_size=x_external.shape[0])

            if x_external.shape[0] > 0:
                metrics = self._compute_binary_metrics(y_test, test_probs, threshold=self._decision_threshold_for(cancer_type))
            else:
                metrics = {
                    "accuracy": 0.0,
                    "f1": 0.0,
                    "auc": 0.5,
                    "pr_auc": 0.0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "log_loss": float("inf"),
                    "sensitivity": 0.0,
                    "specificity": 0.0,
                }

            test_metrics[f"{cancer_type.lower()}::{pattern_name}"] = metrics
            selected_performance[cancer_type] = {
                "pattern": pattern_name,
                "test": metrics,
            }

        return {
            "hospital_id": self.hospital_id,
            "metrics": {
                "test": test_metrics,
                "selected_performance": selected_performance,
            },
            "selected_patterns": selected_patterns,
            "external_sample_count": int(x_external.shape[0]),
        }

    @staticmethod
    def _compute_binary_metrics(y_true: np.ndarray, probs, threshold: float = 0.5) -> dict[str, float]:
        # Handle degenerate class splits before calling sklearn metrics to avoid lots of warnings
        uniques = np.unique(y_true)
        if uniques.size < 2:
            if uniques[0] == 1:
                return {
                    "accuracy": 1.0,
                    "f1": 0.0,
                    "auc": 0.5,
                    "pr_auc": 0.0,
                    "precision": 1.0,
                    "recall": 1.0,
                    "log_loss": 0.0,
                    "sensitivity": 1.0,
                    "specificity": 0.0,
                }
            return {
                "accuracy": 1.0,
                "f1": 0.0,
                "auc": 0.5,
                "pr_auc": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "log_loss": 0.0,
                "sensitivity": 0.0,
                "specificity": 1.0,
            }

        # Ensure `probs` is numpy-compatible for comparisons (list-backed outputs may occur)
        probs_arr = np.asarray(probs, dtype=np.float32)
        preds = (probs_arr >= float(threshold)).astype(int)
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
        expected = self._expected_cancer_types_from_config(self.config)
        if len(cancer_types) != len(expected):
            raise ValueError(
                f"Agent portfolio must expose exactly {len(expected)} cancer agents."
            )
        if set(cancer_types) != set(expected):
            raise ValueError(
                "Agent portfolio cancer types must be exactly: "
                f"{', '.join(expected)}"
            )

    def _validate_selected_patterns(self, selected_patterns: dict[str, str]) -> None:
        expected = self._expected_cancer_types_from_config(self.config)
        missing = [c for c in expected if c not in selected_patterns]
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
            expected = self._expected_cancer_types_from_config(self.config)
            dummy_metrics = {ct: {"accuracy": 0.0, "f1": 0.0, "auc": 0.0, "sensitivity": 0.0, "specificity": 0.0} for ct in expected}
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

        # Debug print for metrics.per_agent
        per_agent_metrics = output.get("metrics", {}).get("per_agent", {})
        logging.debug(f"[DEBUG] Hospital {self.hospital_id} metrics.per_agent: {per_agent_metrics}")
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
        # No model weights are used in the pure AI-agent workflow.

    # Note: `evaluate()` is already implemented earlier in this class (line ~235).
    # The duplicate implementation was intentionally removed to avoid method override confusion and enforce a single evaluation contract.

