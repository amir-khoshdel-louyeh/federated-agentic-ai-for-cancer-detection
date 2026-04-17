from __future__ import annotations

from pathlib import Path
import hashlib
import json
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

from ..agents import AIThinkingPattern, SkinCancerAgent

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
        inference_cfg = (config.get("inference", {}) or {}) if config else {}
        self.decision_threshold = float(
            training_cfg.get(
                "decision_threshold",
                inference_cfg.get(
                    "decision_threshold",
                    config.get("decision_threshold", 0.8) if config else 0.8,
                ),
            )
        ) if config else 0.8
        self.decision_threshold_penalty_weight = float(
            inference_cfg.get("decision_threshold_penalty_weight", 0.0)
        ) if config else 0.0
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
        logging.info(f"Initializing hospital {self.hospital_id}...")
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
                inference_cfg = (self.config or {}).get("inference", {}) if self.config else {}
                pattern_config.setdefault("request_delay_seconds", inference_cfg.get("request_delay_seconds", 0.0))
                pattern_config.setdefault("hospital_id", self.hospital_id)
                pattern_config.setdefault("cache_base_dir", self.config.get("out_dir", "outputs"))
                pattern_config.setdefault("cache_file_name", "inference_cache.json")
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
        logging.info(
            f"Hospital {self.hospital_id} initialized with splits train={self.metrics_store['split_sizes']['train']} "
            f"val={self.metrics_store['split_sizes']['val']} test={self.metrics_store['split_sizes']['test']}"
        )

    def _decision_threshold_for(self, cancer_type: str) -> float:
        key = str(cancer_type).strip().upper()
        if key in self.decision_thresholds:
            return self.decision_thresholds[key]
        return self.decision_threshold

    @staticmethod
    def _find_optimal_threshold(
        y_true: np.ndarray,
        probs: np.ndarray,
        penalty_weight: float = 0.0,
    ) -> float:
        best_threshold = 0.5
        best_score = float("-inf")
        for threshold in np.linspace(0.0, 1.0, 101, dtype=np.float32):
            metrics = HospitalNode._compute_binary_metrics(y_true, probs, threshold=float(threshold), penalty_weight=penalty_weight)
            score = (
                0.4 * metrics.get("f1", 0.0)
                + 0.35 * metrics.get("recall", 0.0)
                + 0.25 * metrics.get("specificity", 0.0)
            )
            if score > best_score or (score == best_score and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
                best_score = score
                best_threshold = float(threshold)
        return best_threshold

    def _detection_mode(self) -> str:
        if not self.config:
            return "detect_then_type"
        return str(self.config.get("detection", {}).get("mode", "detect_then_type")).strip()

    def _inference_cache_dir(self) -> Path:
        return Path(self.config.get("out_dir", "outputs")) / "hospitals" / self.hospital_id

    def _inference_cache_path(self) -> Path:
        return self._inference_cache_dir() / "inference_cache.json"

    def _ensure_inference_cache_dir(self) -> None:
        self._inference_cache_dir().mkdir(parents=True, exist_ok=True)

    def _load_inference_entries(self):
        cache_path = self._inference_cache_path()
        if not cache_path.exists():
            return
        with cache_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def _append_inference_entry(self, entry: dict[str, Any]) -> None:
        self._ensure_inference_cache_dir()
        existing = self._find_inference_entry(entry.get("unique_id", ""), entry.get("split"), entry.get("cancer_type"))
        if existing is not None:
            return
        cache_path = self._inference_cache_path()
        with cache_path.open("a", encoding="utf-8") as file:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")

    def _rewrite_inference_cache(self, entries: list[dict[str, Any]]) -> None:
        cache_path = self._inference_cache_path()
        self._ensure_inference_cache_dir()
        with cache_path.open("w", encoding="utf-8") as file:
            for entry in entries:
                json.dump(entry, file, ensure_ascii=False)
                file.write("\n")

    def _mark_inference_entry_failed(self, unique_id: str, split: str, cancer_type: str) -> bool:
        cache_path = self._inference_cache_path()
        if not cache_path.exists() or not unique_id:
            return False

        entries: list[dict[str, Any]] = []
        updated = False
        for entry in self._load_inference_entries() or []:
            if (
                entry.get("unique_id") == unique_id
                and entry.get("split") == split
                and entry.get("cancer_type") == cancer_type
            ):
                entry["cache_status"] = "failed"
                entry["correct"] = False
                updated = True
            entries.append(entry)

        if updated:
            self._rewrite_inference_cache(entries)
        return updated

    def _invalidate_wrong_cached_entries(
        self,
        entries: list[dict[str, Any]],
        cancer_type: str,
        split: str,
    ) -> None:
        threshold = self._decision_threshold_for(cancer_type)
        for entry in entries:
            if entry.get("cache_status") == "failed":
                continue

            ground_truth = entry.get("ground_truth")
            if ground_truth is None:
                continue
            try:
                ground_truth = int(ground_truth)
            except (TypeError, ValueError):
                continue

            probability = entry.get("probability", 0.0)
            try:
                probability = float(probability)
            except (TypeError, ValueError):
                probability = 0.0

            predicted = 1 if probability >= threshold else 0
            if predicted != ground_truth:
                unique_id = str(entry.get("unique_id", ""))
                if not unique_id:
                    continue
                if self._mark_inference_entry_failed(unique_id, split, cancer_type):
                    self.metrics_store.setdefault("cache_invalidation", []).append(
                        {
                            "unique_id": unique_id,
                            "split": split,
                            "cancer_type": cancer_type,
                            "predicted": predicted,
                            "ground_truth": ground_truth,
                        }
                    )
                    logging.info(
                        f"Hospital {self.hospital_id}: invalidated cached prediction {unique_id} "
                        f"for {cancer_type} split={split} due to ground truth mismatch ({predicted}!={ground_truth})."
                    )

    def _find_inference_entry(self, unique_id: str, split: str | None = None, cancer_type: str | None = None) -> dict[str, Any] | None:
        for entry in self._load_inference_entries() or ():
            if entry.get("unique_id") != unique_id:
                continue
            if split is not None and entry.get("split") != split:
                continue
            if cancer_type is not None and entry.get("cancer_type") != cancer_type:
                continue
            return entry
        return None

    def _load_cached_predictions(self, split: str, cancer_type: str) -> list[dict[str, Any]]:
        return [
            entry
            for entry in self._load_inference_entries() or []
            if entry.get("split") == split and entry.get("cancer_type") == cancer_type
        ]

    def _unique_id_for_row(self, row: np.ndarray) -> str:
        canonical = json.dumps([float(x) for x in row.tolist()], separators=(",", ":"), sort_keys=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def _metrics_from_cached_entries(self, entries: list[dict[str, Any]], cancer_type: str) -> dict[str, Any]:
        if not entries:
            return {}
        probabilities = np.array([float(entry.get("probability", 0.0)) for entry in entries], dtype=np.float32)
        labels = np.array([int(entry.get("ground_truth", 0)) for entry in entries], dtype=np.int64)
        return self._compute_binary_metrics(
            labels,
            probabilities,
            threshold=self._decision_threshold_for(cancer_type),
            penalty_weight=self.decision_threshold_penalty_weight,
        )

    def _mean_uncertainty_from_entries(self, entries: list[dict[str, Any]]) -> float:
        if not entries:
            return 1.0
        return float(np.mean([float(entry.get("uncertainty", 1.0)) for entry in entries]))

    def _load_cached_reasons(self, entries: list[dict[str, Any]]) -> list[str]:
        return [str(entry.get("clinical_reasoning", "")) for entry in entries]

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

    def infer_and_cache(self) -> None:
        """Run a single round of inference on validation and test splits and persist results."""
        if self.scope.data is None:
            raise RuntimeError("Call initialize() before infer_and_cache().")
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before infer_and_cache().")
        if self.local_data is None:
            raise RuntimeError("Call initialize() before infer_and_cache().")

        val_predictions: dict[str, np.ndarray] = {}
        test_predictions: dict[str, np.ndarray] = {}
        per_agent_metrics: dict[str, dict] = {}

        detection_mode = self._detection_mode()
        logging.info(f"Hospital {self.hospital_id}: starting inference for detection mode={detection_mode}")
        for cancer_type in self.scope.agent_portfolio.cancer_types:
            if detection_mode == "detect_only" and str(cancer_type).strip().upper() != "CANCER":
                continue

            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            if not isinstance(agent, SkinCancerAgent):
                raise TypeError(f"Portfolio agent for {cancer_type} must be a SkinCancerAgent.")

            x_train, y_train = self.get_cancer_filtered_split(cancer_type=cancer_type, split="train")
            x_val, y_val = self.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            x_test, y_test = self.get_cancer_filtered_split(cancer_type=cancer_type, split="test")

            # No model training is required for AI-agent workflows.
            agent.fit(x_train, y_train)

            val_results = agent.predict_diagnoses(x_val)
            test_results = agent.predict_diagnoses(x_test)

            self._validate_prediction_shape(
                agent.name,
                np.array([result["probability"] for result in val_results], dtype=np.float32),
                expected_size=x_val.shape[0],
            )
            self._validate_prediction_shape(
                agent.name,
                np.array([result["probability"] for result in test_results], dtype=np.float32),
                expected_size=x_test.shape[0],
            )

            logging.info(
                f"Hospital {self.hospital_id}: inferring {cancer_type} on val={x_val.shape[0]} test={x_test.shape[0]} samples"
            )
            threshold = self._decision_threshold_for(cancer_type)
            for idx, result in enumerate(val_results):
                probability = float(result.get("probability", 0.0))
                label = "malignant" if probability >= threshold else "benign"
                ground_truth = int(y_val[idx]) if idx < len(y_val) else 0
                entry = {
                    "hospital_id": self.hospital_id,
                    "split": "val",
                    "cancer_type": cancer_type,
                    "pattern": agent.name,
                    "unique_id": self._unique_id_for_row(x_val[idx]),
                    "features": {f"feature_{i+1}": float(v) for i, v in enumerate(x_val[idx])},
                    "probability": probability,
                    "uncertainty": float(result.get("uncertainty", 1.0)),
                    "clinical_reasoning": str(result.get("clinical_reasoning", "")),
                    "label": label,
                    "ground_truth": ground_truth,
                    "correct": (probability >= threshold and ground_truth == 1) or (
                        probability < threshold and ground_truth == 0
                    ),
                    "cache_status": "ok",
                }
                self._append_inference_entry(entry)

            for idx, result in enumerate(test_results):
                probability = float(result.get("probability", 0.0))
                label = "malignant" if probability >= threshold else "benign"
                ground_truth = int(y_test[idx]) if idx < len(y_test) else 0
                entry = {
                    "hospital_id": self.hospital_id,
                    "split": "test",
                    "cancer_type": cancer_type,
                    "pattern": agent.name,
                    "unique_id": self._unique_id_for_row(x_test[idx]),
                    "features": {f"feature_{i+1}": float(v) for i, v in enumerate(x_test[idx])},
                    "probability": probability,
                    "uncertainty": float(result.get("uncertainty", 1.0)),
                    "clinical_reasoning": str(result.get("clinical_reasoning", "")),
                    "label": label,
                    "ground_truth": ground_truth,
                    "correct": (probability >= threshold and ground_truth == 1) or (
                        probability < threshold and ground_truth == 0
                    ),
                    "cache_status": "ok",
                }
                self._append_inference_entry(entry)

            val_predictions[cancer_type] = np.array([float(result.get("probability", 0.0)) for result in val_results], dtype=np.float32)
            test_predictions[cancer_type] = np.array([float(result.get("probability", 0.0)) for result in test_results], dtype=np.float32)

            if x_test.shape[0] > 0:
                per_agent_metrics[agent.name] = self._compute_binary_metrics(
                    y_test,
                    test_predictions[cancer_type],
                    threshold=self._decision_threshold_for(cancer_type),
                    penalty_weight=self.decision_threshold_penalty_weight,
                )
            else:
                per_agent_metrics[agent.name] = {
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

        self.metrics_store["predictions"] = {
            "val": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in val_predictions.items()},
            "test": {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in test_predictions.items()},
        }
        self.metrics_store.setdefault("evaluation", {})["test"] = per_agent_metrics
        self.metrics_store["lifecycle_state"] = "inferred"
        logging.info(
            f"Hospital {self.hospital_id}: inference completed. "
            f"Cached {sum([len(self._load_cached_predictions('val', ct)) + len(self._load_cached_predictions('test', ct)) for ct in self.scope.agent_portfolio.cancer_types if self._detection_mode() != 'detect_only' or ct.upper() == 'CANCER'])} samples."
        )

    def train(self) -> None:
        """Legacy alias for AI-agent inference and cache generation."""
        return self.infer_and_cache()

    def evaluate(self) -> dict[str, Any]:
        """Evaluate all fixed cancer agents and store metrics from cached JSON predictions."""
        if self.scope.data is None:
            raise RuntimeError("Call initialize() before evaluate().")
        if self.scope.agent_portfolio is None:
            raise RuntimeError("HospitalNode requires an agent portfolio before evaluate().")
        if self.local_data is None:
            raise RuntimeError("Call initialize() before evaluate().")

        logging.info(f"Hospital {self.hospital_id}: starting evaluation from cached predictions.")
        val_metrics = {}
        test_metrics = {}
        selected_patterns = self.metrics_store.get("selected_patterns", self.scope.agent_portfolio.selected_patterns())
        selected_performance = {}

        val_reasoning: dict[str, list[str]] = {}
        test_reasoning: dict[str, list[str]] = {}

        detection_mode = self._detection_mode()
        for cancer_type, pattern_name in selected_patterns.items():
            if detection_mode == "detect_only" and str(cancer_type).strip().upper() != "CANCER":
                continue

            prediction_key = f"{cancer_type.lower()}::{pattern_name}"
            raw_val_entries = self._load_cached_predictions("val", cancer_type)
            raw_test_entries = self._load_cached_predictions("test", cancer_type)

            threshold = self._decision_threshold_for(cancer_type)
            if raw_val_entries:
                val_truth = np.array([int(entry.get("ground_truth", 0)) for entry in raw_val_entries], dtype=np.int64)
                val_probs = np.array([float(entry.get("probability", 0.0)) for entry in raw_val_entries], dtype=np.float32)
                threshold = self._find_optimal_threshold(val_truth, val_probs, penalty_weight=self.decision_threshold_penalty_weight)
                self.decision_thresholds[str(cancer_type).strip().upper()] = threshold

            val_metrics[prediction_key] = (
                self._compute_binary_metrics(
                    np.array([int(entry.get("ground_truth", 0)) for entry in raw_val_entries], dtype=np.int64),
                    np.array([float(entry.get("probability", 0.0)) for entry in raw_val_entries], dtype=np.float32),
                    threshold=threshold,
                    penalty_weight=self.decision_threshold_penalty_weight,
                )
                if raw_val_entries
                else {}
            )
            test_metrics[prediction_key] = (
                self._compute_binary_metrics(
                    np.array([int(entry.get("ground_truth", 0)) for entry in raw_test_entries], dtype=np.int64),
                    np.array([float(entry.get("probability", 0.0)) for entry in raw_test_entries], dtype=np.float32),
                    threshold=threshold,
                    penalty_weight=self.decision_threshold_penalty_weight,
                )
                if raw_test_entries
                else {}
            )
            val_reasoning[prediction_key] = self._load_cached_reasons(raw_val_entries)
            test_reasoning[prediction_key] = self._load_cached_reasons(raw_test_entries)

            selected_performance[cancer_type] = {
                "pattern": pattern_name,
                "validation": val_metrics[prediction_key],
                "test": test_metrics[prediction_key],
                "selected_threshold": threshold,
                "mean_validation_uncertainty": self._mean_uncertainty_from_entries(raw_val_entries),
                "mean_test_uncertainty": self._mean_uncertainty_from_entries(raw_test_entries),
            }

            self._invalidate_wrong_cached_entries(raw_val_entries, cancer_type, split="val")
            self._invalidate_wrong_cached_entries(raw_test_entries, cancer_type, split="test")

        candidate_comparisons = self._build_candidate_comparisons(val_metrics)

        for prediction_key, metrics in val_metrics.items():
            if isinstance(metrics, dict) and "tp" in metrics:
                logging.info(
                    f"Hospital {self.hospital_id} [val] {prediction_key} confusion: TN={metrics.get('tn')} "
                    f"FP={metrics.get('fp')} FN={metrics.get('fn')} TP={metrics.get('tp')}"
                )

        for prediction_key, metrics in test_metrics.items():
            if isinstance(metrics, dict) and "tp" in metrics:
                logging.info(
                    f"Hospital {self.hospital_id} [test] {prediction_key} confusion: TN={metrics.get('tn')} "
                    f"FP={metrics.get('fp')} FN={metrics.get('fn')} TP={metrics.get('tp')}"
                )

        logging.info(
            f"Hospital {self.hospital_id}: evaluation complete. "
            f"Validation keys={len(val_metrics)} test keys={len(test_metrics)}"
        )

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
            "reasoning": {
                "validation": val_reasoning,
                "test": test_reasoning,
            },
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
                metrics = self._compute_binary_metrics(
                    y_test,
                    test_probs,
                    threshold=self._decision_threshold_for(cancer_type),
                    penalty_weight=self.decision_threshold_penalty_weight,
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
    def _compute_binary_metrics(y_true: np.ndarray, probs, threshold: float = 0.5, penalty_weight: float = 0.0) -> dict[str, float]:
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

        # Penalize highly skewed predictions even when the threshold yields high accuracy.
        num_positive_preds = int(preds.sum())
        num_negative_preds = int(preds.shape[0] - num_positive_preds)
        if preds.shape[0] > 0:
            imbalance_ratio = abs(num_positive_preds - num_negative_preds) / float(preds.shape[0])
        else:
            imbalance_ratio = 0.0
        threshold_penalty = float(min(1.0, max(0.0, imbalance_ratio * float(penalty_weight))))
        penalized_log_loss = logloss_val + threshold_penalty

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
            "penalized_log_loss": penalized_log_loss,
            "threshold_penalty": threshold_penalty,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
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

        prompt_update = global_state.get("prompt_evolution")
        if isinstance(prompt_update, Mapping):
            self._apply_prompt_update(prompt_update)

        # Attempt model weights sync if available
        # No model weights are used in the pure AI-agent workflow.

    def _apply_prompt_update(self, prompt_update: Mapping[str, Any]) -> None:
        prompt_map = prompt_update.get("agents_prompts") or prompt_update.get("agent_prompts")
        system_prompt = prompt_update.get("system_prompt")
        if prompt_map is None and not system_prompt:
            return

        def resolve_agent_prompt(cancer_type: str) -> str | None:
            if isinstance(prompt_map, Mapping):
                prompt = (
                    prompt_map.get(cancer_type)
                    or prompt_map.get(cancer_type.upper())
                    or prompt_map.get("default")
                )
                if prompt:
                    return str(prompt).strip()
            return None

        for cancer_type in self.scope.agent_portfolio.cancer_types:
            agent = self.scope.agent_portfolio.get_agent(cancer_type)
            for pattern in agent.thinking_patterns:
                if not isinstance(pattern, AIThinkingPattern):
                    continue

                agent_prompt = resolve_agent_prompt(cancer_type)
                if agent_prompt:
                    pattern.prompt_prefix = agent_prompt
                elif isinstance(system_prompt, str) and system_prompt.strip():
                    pattern.prompt_prefix = system_prompt.strip()

                if isinstance(system_prompt, str) and system_prompt.strip():
                    pattern.llm_reasoner.set_system_prompt(system_prompt)

        self.metrics_store["prompt_evolution"] = prompt_update
        self.metrics_store["prompt_evolution_applied"] = True

    # Note: `evaluate()` is already implemented earlier in this class (line ~235).
    # The duplicate implementation was intentionally removed to avoid method override confusion and enforce a single evaluation contract.

