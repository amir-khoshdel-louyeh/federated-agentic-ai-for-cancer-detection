from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .base import LLMReasoner, ThinkingPattern


class AIThinkingPattern(ThinkingPattern):
    """Lightweight AI thinking pattern that uses a language model to produce probabilities."""

    def __init__(
        self,
        llm_reasoner: LLMReasoner | None = None,
        provider: str = "auto",
        model_name: str = "gpt-3.5-turbo",
        local_llm_config: dict[str, Any] | None = None,
        api_key: str | None = None,
        prompt_prefix: str | None = None,
        system_prompt: str | None = None,
        hospital_id: str | None = None,
        cache_base_dir: str | None = None,
        cache_file_name: str = "inference_cache.json",
        max_retries: int = 2,
        calibration_temperature: float = 0.1,
        request_delay_seconds: float = 0.0,
    ) -> None:
        self.calibration_temperature = float(calibration_temperature)
        self.request_delay_seconds = float(request_delay_seconds)
        if local_llm_config is None:
            local_llm_config = {}
        if "temperature" not in local_llm_config:
            local_llm_config["temperature"] = self.calibration_temperature

        self.llm_reasoner = llm_reasoner or LLMReasoner(
            provider=provider,
            model_name=model_name,
            local_llm_config=local_llm_config,
            api_key=api_key,
            system_prompt=system_prompt,
        )
        if llm_reasoner is not None:
            self.llm_reasoner.temperature = self.calibration_temperature
        if llm_reasoner is not None and system_prompt:
            self.llm_reasoner.set_system_prompt(system_prompt)
        self.prompt_prefix = (
            prompt_prefix
            or (
                "You are a strict pathologist. Your duty is accurate diagnosis, not alarm. "
                "Avoid defaulting to all-positive or all-negative conclusions. "
                "If evidence is weak, report a moderate probability and explain the uncertainty clearly. "
                "Treat false negatives and false positives as equally important, and base the final classification on evidence strength."
            )
        )
        self.hospital_id = hospital_id
        self.cache_base_dir = cache_base_dir or "outputs"
        self.cache_file_name = cache_file_name
        self.max_retries = int(max_retries)
        self._last_structured_input: np.ndarray | None = None
        self._last_structured_outputs: list[dict[str, Any]] | None = None

    @property
    def name(self) -> str:
        return "ai_agent"

    def save_model(self, file_path: str) -> None:
        # No model weights to persist for the pure AI-agent pattern.
        return None

    def load_model(self, file_path: str) -> None:
        # No model weights to load for the pure AI-agent pattern.
        return None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        # No training required for the pure language-agent reasoning pattern.
        # AIThinkingPattern relies on LLM inference at prediction time.
        return None

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        structured = self.predict_structured(x)
        return np.array([float(item.get("probability", 0.5)) for item in structured], dtype=np.float32)

    def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
        structured = self.predict_structured(x, n_samples=n_samples)
        return np.array([float(item.get("uncertainty", 0.2)) for item in structured], dtype=np.float32)

    def _cache_dir(self) -> Path:
        base = Path(self.cache_base_dir or "outputs")
        if self.hospital_id is None:
            return base / "hospitals" / "unknown"
        return base / "hospitals" / str(self.hospital_id)

    def _cache_path(self) -> Path:
        return self._cache_dir() / self.cache_file_name

    def _ensure_cache_directory(self) -> None:
        self._cache_dir().mkdir(parents=True, exist_ok=True)

    def _load_local_cache(self):
        cache_path = self._cache_path()
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

    def _find_cached_entry(self, unique_id: str) -> dict[str, Any] | None:
        for entry in self._load_local_cache() or ():
            if entry.get("unique_id") != unique_id:
                continue
            if entry.get("cache_status") == "failed":
                continue
            return entry
        return None

    def _find_cache_entry_by_unique_id(self, unique_id: str) -> dict[str, Any] | None:
        for entry in self._load_local_cache() or ():
            if entry.get("unique_id") == unique_id:
                return entry
        return None

    def _rewrite_local_cache(self, entries: list[dict[str, Any]]) -> None:
        cache_path = self._cache_path()
        self._ensure_cache_directory()
        with cache_path.open("w", encoding="utf-8") as file:
            for entry in entries:
                json.dump(entry, file, ensure_ascii=False)
                file.write("\n")

    def _replace_cache_entry(self, entry: dict[str, Any]) -> None:
        unique_id = entry.get("unique_id", "")
        if not unique_id:
            self._append_cache_entry(entry)
            return

        entries: list[dict[str, Any]] = []
        replaced = False
        for existing in self._load_local_cache() or []:
            if existing.get("unique_id") == unique_id:
                entries.append(entry)
                replaced = True
            else:
                entries.append(existing)

        if not replaced:
            entries.append(entry)
        self._rewrite_local_cache(entries)

    def _append_cache_entry(self, entry: dict[str, Any]) -> None:
        if entry.get("cache_status") is None:
            entry["cache_status"] = "ok"
        existing = self._find_cached_entry(entry.get("unique_id", ""))
        if existing is not None:
            return

        # If there is a previous failed entry for this sample, overwrite it.
        failed_entry = None
        for existing in self._load_local_cache() or []:
            if existing.get("unique_id") == entry.get("unique_id") and existing.get("cache_status") == "failed":
                failed_entry = existing
                break
        if failed_entry is not None:
            self._replace_cache_entry(entry)
            return

        self._ensure_cache_directory()
        cache_path = self._cache_path()
        with cache_path.open("a", encoding="utf-8") as file:
            json.dump(entry, file, ensure_ascii=False)
            file.write("\n")

    def _load_experience_entries(self, cancer_type: str | None = None) -> list[dict[str, Any]]:
        experiences: list[dict[str, Any]] = []
        for entry in self._load_local_cache() or []:
            if cancer_type is not None and entry.get("cancer_type") != cancer_type:
                continue
            if not isinstance(entry.get("features"), Mapping):
                continue
            experiences.append(entry)
        return experiences

    def _feature_distance(self, features_a: dict[str, Any], features_b: dict[str, Any]) -> float:
        shared_keys = set(features_a) & set(features_b)
        if not shared_keys:
            return float("inf")
        squared_sum = 0.0
        for key in shared_keys:
            try:
                a = float(features_a.get(key, 0.0))
                b = float(features_b.get(key, 0.0))
            except (TypeError, ValueError):
                a = 0.0
                b = 0.0
            squared_sum += (a - b) ** 2
        return math.sqrt(squared_sum)

    def _find_similar_experiences(
        self,
        feature_map: dict[str, float],
        cancer_type: str,
        top_k: int = 2,
        max_distance: float = 1.0,
    ) -> list[dict[str, Any]]:
        neighbors: list[tuple[float, dict[str, Any]]] = []
        for entry in self._load_experience_entries(cancer_type=cancer_type):
            distance = self._feature_distance(feature_map, entry.get("features", {}))
            if distance <= max_distance:
                neighbors.append((distance, entry))

        neighbors.sort(key=lambda item: item[0])
        return [entry for _, entry in neighbors[:top_k]]

    def _build_experience_context(self, feature_map: dict[str, float], cancer_type: str) -> str:
        neighbors = self._find_similar_experiences(feature_map, cancer_type=cancer_type, top_k=2, max_distance=1.0)
        if not neighbors:
            return ""

        lines: list[str] = [
            "Similar past cases found:",
        ]
        failed_warning = False
        for entry in neighbors:
            outcome = "correct" if entry.get("correct") else "misclassified"
            prediction = str(entry.get("label", "unknown"))
            ground_truth = entry.get("ground_truth", "unknown")
            reasoning = str(entry.get("clinical_reasoning", "")).replace("\n", " ")
            lines.append(
                f"- Known case: predicted {prediction}, ground_truth={ground_truth}, outcome={outcome}. Reasoning: {reasoning}"
            )
            if not entry.get("correct"):
                failed_warning = True

        if failed_warning:
            lines.insert(
                1,
                "Warning: similar past cases were previously misclassified. Pay extra attention to avoid repeating the same mistake.",
            )

        return "\n".join(lines)

    def _make_unique_id(self, feature_map: dict[str, float]) -> str:
        canonical = json.dumps(feature_map, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    def predict_structured(self, x: np.ndarray, n_samples: int = 25) -> list[dict[str, Any]]:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        cached = self._get_cached_structured_outputs(x)
        if cached is not None:
            return cached

        structured_results: list[dict[str, Any]] = []
        for row in x:
            feature_map = {
                f"feature_{index}": float(value)
                for index, value in enumerate(row, start=1)
            }
            unique_id = self._make_unique_id(feature_map)
            cached_entry = self._find_cached_entry(unique_id)
            if cached_entry is not None:
                logging.info(f"AIThinkingPattern cache hit for {self.name} unique_id={unique_id}")
                structured_results.append(cached_entry)
                continue

            failed_entry = self._find_cache_entry_by_unique_id(unique_id)
            if failed_entry is not None and failed_entry.get("cache_status") == "failed":
                retry_count = int(failed_entry.get("retry_count", 0))
                if retry_count >= self.max_retries:
                    logging.info(
                        f"AIThinkingPattern max retries reached for {self.name} unique_id={unique_id}; reusing failed cache entry"
                    )
                    structured_results.append(failed_entry)
                    continue
                logging.info(
                    f"AIThinkingPattern retrying failed sample for {self.name} unique_id={unique_id}; retry_count={retry_count}"
                )

            logging.info(f"AIThinkingPattern cache miss for {self.name} unique_id={unique_id}; calling LLM")
            experience_context = self._build_experience_context(feature_map, self.name)
            response = self.llm_reasoner.generate_reasoning(
                "AI_AGENT",
                {
                    "patterns": [{"name": self.name, "details": self._build_prompt(row, experience_context)}],
                    "clinical_features": feature_map,
                },
            )
            if self.request_delay_seconds > 0.0:
                time.sleep(self.request_delay_seconds)
            output = self._extract_structured_output(response)
            output["unique_id"] = unique_id
            output["features"] = feature_map
            output["label"] = "malignant" if output["probability"] >= 0.5 else "benign"
            output["cancer_type"] = self.name
            output["cache_status"] = "ok"
            output["retry_count"] = int(failed_entry.get("retry_count", 0)) + 1 if failed_entry is not None else 0

            self._replace_cache_entry(output)
            structured_results.append(output)

        self._cache_structured_outputs(x, structured_results)
        return structured_results

    def _cache_structured_outputs(self, x: np.ndarray, outputs: list[dict[str, Any]]) -> None:
        self._last_structured_input = np.array(x, copy=True)
        self._last_structured_outputs = outputs

    def _get_cached_structured_outputs(self, x: np.ndarray) -> list[dict[str, Any]] | None:
        x = np.asarray(x)
        if self._last_structured_outputs is None or self._last_structured_input is None:
            return None
        if x.shape != self._last_structured_input.shape:
            return None
        if np.array_equal(x, self._last_structured_input):
            return self._last_structured_outputs
        return None

    def _extract_structured_output(self, response: dict[str, Any]) -> dict[str, Any]:
        json_data = response.get("json")
        if isinstance(json_data, dict):
            probability = json_data.get("probability")
            if probability is None:
                probability = json_data.get("Probability")
            probability = float(max(0.0, min(1.0, float(probability)))) if probability is not None else 0.5

            uncertainty = json_data.get("uncertainty")
            if uncertainty is None:
                uncertainty = json_data.get("uncertainty")
            if uncertainty is None:
                confidence = json_data.get("confidence") or json_data.get("Confidence")
                try:
                    confidence = float(confidence)
                    uncertainty = 1.0 - max(0.0, min(1.0, confidence))
                except (TypeError, ValueError):
                    uncertainty = None
            uncertainty = float(max(0.0, min(1.0, float(uncertainty)))) if uncertainty is not None else 0.2

            reasoning = json_data.get("reasoning") or json_data.get("Reasoning") or json_data.get("analysis") or ""
            return {
                "probability": probability,
                "uncertainty": uncertainty,
                "clinical_reasoning": str(reasoning),
                "details": str(json_data.get("details", "single-shot structured prediction")),
            }

        text = response.get("text", "")
        probability = 0.5
        uncertainty = 0.2
        reasoning = ""
        if text:
            prob_match = re.search(r"Probability\s*[:=]\s*([01](?:\.\d+)?)", text, re.I)
            if not prob_match:
                prob_match = re.search(r"([01](?:\.\d+)?)", text)
            if prob_match:
                try:
                    probability = float(max(0.0, min(1.0, float(prob_match.group(1)))))
                except ValueError:
                    probability = 0.5

            conf_match = re.search(r"Confidence\s*[:=]\s*([01](?:\.\d+)?)", text, re.I)
            if conf_match:
                try:
                    confidence = float(max(0.0, min(1.0, float(conf_match.group(1)))))
                    uncertainty = 1.0 - confidence
                except ValueError:
                    pass
            else:
                unc_match = re.search(r"uncertainty\s*[:=]\s*([01](?:\.\d+)?)", text, re.I)
                if unc_match:
                    try:
                        uncertainty = float(max(0.0, min(1.0, float(unc_match.group(1)))))
                    except ValueError:
                        uncertainty = 0.2
            reasoning = text

        label = "malignant" if probability >= 0.5 else "benign"
        return {
            "probability": probability,
            "uncertainty": uncertainty,
            "clinical_reasoning": reasoning.strip(),
            "label": label,
            "details": "parsed structured output fallback",
        }

    def _build_prompt(self, row: np.ndarray, experience_context: str | None = None) -> str:
        feature_lines = []
        for index, value in enumerate(row, start=1):
            feature_lines.append(
                f"feature_{index}: {float(value):.4f} "
                "(normalized clinical signal; interpret values as scaled risk indicators rather than raw measurements)"
            )

        prompt = (
            f"{self.prompt_prefix}\n"
            "Review the following normalized clinical features carefully. "
            "List evidence that supports malignancy and evidence that supports benignity before you decide. "
            "Then reflect internally: could this lesion be benign, and if you have overestimated malignancy, reduce the probability. "
            "If you refer to prior cases, keep benign and malignant examples balanced and avoid over-weighting malignant cases. "
            "If the evidence is weak, avoid extreme probabilities and preserve a balanced confidence estimate.\n"
            "Use the following normalized clinical features for a lesion.\n"
            + "\n".join(feature_lines)
        )

        if experience_context:
            prompt += "\n\n" + experience_context

        prompt += (
            "\n\nProvide a short structured reasoning summary with positive and negative evidence, then return a JSON object with keys: probability, uncertainty, reasoning. "
            "The reasoning text should include a confidence check and a final reflective step."
        )
        return prompt

    def _extract_probability(self, response: dict[str, Any]) -> float:
        json_data = response.get("json")
        if isinstance(json_data, dict):
            probability = json_data.get("probability")
            try:
                return float(max(0.0, min(1.0, float(probability))))
            except (TypeError, ValueError):
                pass

        text = response.get("text", "")
        if not text:
            return 0.5

        match = re.search(r"(?:probability|prob)\s*[:=]\s*([01](?:\.\d+)?)", text, re.I)
        if match:
            try:
                value = float(match.group(1))
                return float(max(0.0, min(1.0, value)))
            except ValueError:
                return 0.5

        match = re.search(r"([01](?:\.\d+))", text)
        if match:
            try:
                value = float(match.group(1))
                return float(max(0.0, min(1.0, value)))
            except ValueError:
                return 0.5

        # Fallback to a conservative default probability on unparsable output.
        return 0.5
