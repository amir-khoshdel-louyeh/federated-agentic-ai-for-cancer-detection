from __future__ import annotations

from abc import ABC, abstractmethod
import json
import logging
import os
import re
from typing import Any

import numpy as np
from .tools import Tool


class ThinkingPattern(ABC):
	@abstractmethod
	def save_model(self, file_path: str) -> None:
		"""Save model parameters to file."""

	@abstractmethod
	def load_model(self, file_path: str) -> None:
		"""Load model parameters from file."""

	@property
	@abstractmethod
	def name(self) -> str:
		"""Unique algorithm name used in reports and weighting."""

	@abstractmethod
	def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
		"""Train the algorithm on local hospital data."""

	@abstractmethod
	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		"""Return positive-class probabilities in shape (n_samples,)."""

	def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
		"""Return per-sample uncertainty score (0..1)."""
		probs = self.predict_proba(x)
		return np.full_like(probs, 0.05, dtype=np.float32)


class LLMReasoner:
	"""Lightweight LLM reasoner adapter for clinical rationale generation."""

	def __init__(
		self,
		provider: str = "auto",
		model_name: str = "gpt-3.5-turbo",
		local_model_path: str | None = None,
		local_llm_config: dict[str, Any] | None = None,
		api_key: str | None = None,
	) -> None:
		self.provider = provider.strip().lower()
		self.local_model_path = local_model_path
		self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
		self.local_llm_config = local_llm_config or {}
		self.local_llm_base_url = str(self.local_llm_config.get("base_url", "")).strip()
		self.model_name = str(self.local_llm_config.get("model_name", model_name)).strip()
		self.temperature = float(self.local_llm_config.get("temperature", 0.0))
		self._openai_available = self._check_openai_available()
		self._local_dependencies_available = self._check_local_dependencies_available()
		self._warnings_emitted: set[str] = set()

	def _check_openai_available(self) -> bool:
		try:
			import openai  # noqa: F401
			return True
		except ImportError:
			return False

	def _check_local_dependencies_available(self) -> bool:
		try:
			import transformers  # noqa: F401
			import torch  # noqa: F401
			return True
		except ImportError:
			return False

	def _log_backend_warning(self, message: str, key: str) -> None:
		if key not in self._warnings_emitted:
			logging.warning(message)
			self._warnings_emitted.add(key)

	def _openai_backend_usable(self) -> bool:
		return self._openai_available and bool(self.api_key or self.local_llm_base_url)

	def _ollama_backend_usable(self) -> bool:
		return self._openai_available and bool(self.local_llm_base_url)

	def _local_backend_usable(self) -> bool:
		return self.local_model_path is not None and self._local_dependencies_available

	def generate_reasoning(
		self,
		cancer_type: str,
		observations: dict[str, Any],
		patient_context: dict[str, Any] | None = None,
		functions: list[dict[str, Any]] | None = None,
		function_call: str | dict[str, Any] | None = None,
	) -> dict[str, Any]:
		prompt = self._build_prompt(cancer_type, observations, patient_context, functions=functions)

		if self.provider in {"openai", "api"} and not self._openai_available:
			self._log_backend_warning(
				"OpenAI provider selected but the `openai` package is not installed. "
				"Install it with `pip install openai` or configure local_llm in config.",
				"openai_not_installed",
			)

		if self.local_llm_base_url and not self._openai_available:
			self._log_backend_warning(
				"local_llm is configured for Ollama, but the `openai` Python package is not installed. "
				"Install `openai` so the Ollama endpoint can be used.",
				"ollama_openai_missing",
			)

		if self.provider in {"openai", "api", "auto"} and self.api_key is None and not self._ollama_backend_usable():
			self._log_backend_warning(
				"OpenAI provider selected but OPENAI_API_KEY is not configured. "
				"If you intend to use Ollama locally, configure `local_llm.base_url` in config.",
				"openai_api_key_missing",
			)

		if self.provider == "local" and not self._ollama_backend_usable() and self.local_model_path is None:
			self._log_backend_warning(
				"Local provider selected but no local Ollama or local model path is configured. "
				"Provide `local_llm` settings for Ollama or `local_model_path` for a local Transformers model.",
				"local_model_missing",
			)

		if self.provider == "local" and self.local_model_path is not None and not self._local_dependencies_available:
			self._log_backend_warning(
				"Local provider selected but required packages are not installed. "
				"Install `transformers` and `torch` for local model support.",
				"local_deps_missing",
			)

		if self.provider == "auto" and not self._openai_backend_usable() and not self._local_backend_usable():
			self._log_backend_warning(
				"Auto provider could not find a usable model backend. "
				"OpenAI is unavailable and no supported local model is configured. "
				"Falling back to deterministic reasoning outputs.",
				"auto_backend_unavailable",
			)

		use_openai = self._openai_available and (
			self._ollama_backend_usable() or self.api_key is not None
		)
		if use_openai:
			try:
				import openai

				if self.local_llm_base_url:
					openai.api_base = self.local_llm_base_url
				# Ollama does not require an API key by default for local access.
				openai.api_key = self.api_key or ""

				model_name = self.model_name
				kwargs: dict[str, Any] = {
					"model": model_name,
					"messages": [
						{"role": "system", "content": self._system_prompt()},
						{
							"role": "user",
							"content": prompt,
						},
					],
					"temperature": self.temperature,
					"max_tokens": 250,
				}
				if functions is not None:
					kwargs["functions"] = functions
				if function_call is not None:
					kwargs["function_call"] = function_call
				response = openai.ChatCompletion.create(**kwargs)
				message = response.choices[0].message
				content = message.content.strip() if message.content else ""
				json_data = self._parse_json_response(content)
				function_call_data = None
				if hasattr(message, "function_call") and message.function_call is not None:
					function_call_data = {
						"name": message.function_call.name,
						"arguments": message.function_call.arguments,
					}
				return {
					"text": content,
					"json": json_data,
					"function_call": function_call_data,
				}
			except Exception as exc:
				if self.local_llm_base_url:
					logging.error(
						"Ollama local endpoint unreachable at %s: %s",
						self.local_llm_base_url,
						repr(exc),
					)
				else:
					logging.error(
						"OpenAI backend request failed: %s",
						repr(exc),
					)
				return {
					"text": self._fallback_reasoning(cancer_type, observations, patient_context),
					"json": None,
					"function_call": None,
				}

		if self.provider == "local" and self.local_model_path is not None:
			try:
				from transformers import AutoModelForCausalLM, AutoTokenizer
				import torch

				tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
				model = AutoModelForCausalLM.from_pretrained(self.local_model_path)
				inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
				outputs = model.generate(**inputs, max_new_tokens=200)
				content = tokenizer.decode(outputs[0], skip_special_tokens=True)
				json_data = self._parse_json_response(content)
				return {
					"text": content,
					"json": json_data,
					"function_call": None,
				}
			except Exception as exc:
				logging.error(
					"Local transformers model failed: %s",
					repr(exc),
				)
				return {
					"text": self._fallback_reasoning(cancer_type, observations, patient_context),
					"json": None,
					"function_call": None,
				}

		return {
			"text": self._fallback_reasoning(cancer_type, observations, patient_context),
			"json": None,
			"function_call": None,
		}

	def _build_prompt(
		self,
		cancer_type: str,
		observations: dict[str, Any],
		patient_context: dict[str, Any] | None,
		functions: list[dict[str, Any]] | None = None,
	) -> str:
		context_lines = []
		if patient_context:
			context_lines.append("Patient context:")
			for key, value in patient_context.items():
				context_lines.append(f"- {key}: {value}")

		feature_lines = []
		if isinstance(observations.get("clinical_features"), dict):
			feature_lines.append("Clinical features:")
			for key, value in observations.get("clinical_features", {}).items():
				feature_lines.append(f"- {key}: {value}")

		pattern_lines = ["Observations from predictive patterns:"]
		for pattern in observations.get("patterns", []):
			pattern_lines.append(
				"- {name}: probability={prob:.3f}, uncertainty={unc:.3f}"
				.format(
					name=pattern.get("name", "unknown"),
					prob=pattern.get("probability", 0.0),
					unc=pattern.get("uncertainty", 0.0),
				)
			)
			pattern_lines.append(f"  details: {pattern.get('details', '')}")

		prompt = (
			"You are a clinical assistant. Based on the following pattern outputs, "
			"provide a concise diagnostic probability and a clinical reasoning statement. "
			"Use the cancer type and available patient context to explain why the final probability is chosen.\n\n"
			f"Cancer type: {cancer_type}\n"
			+ "\n".join(context_lines)
			+ "\n\n"
			+ "\n".join(feature_lines)
			+ "\n\n"
			+ "\n".join(pattern_lines)
		)
		if observation := observations.get("tool_action") if isinstance(observations, dict) else None:
			prompt += (
				"\n\nA tool was executed to reduce uncertainty. "
				f"Tool result: {observation.get('result', {})}\n"
			)
		if functions is not None:
			prompt += (
				"\n\nIf a tool can improve the diagnosis in cases of high uncertainty, "
				"select and call the appropriate function using the provided schema. "
				"Do not make up arguments.\n"
			)
		prompt += "\n\nAnswer with a short clinical reasoning summary and a probability estimate."
		return prompt

	def _system_prompt(self) -> str:
		return (
			"You are a clinical reasoning assistant for skin cancer diagnosis. "
			"Respond with a single valid JSON object only. "
			"The JSON object must contain exactly these keys: "
			"`probability` (a number between 0 and 1) and `reasoning` (a short string). "
			"Do not include any extra text, markdown, or explanation outside the JSON object. "
			"Example: {\"probability\": 0.23, \"reasoning\": \"Short clinical conclusion.\"}"
		)

	def _extract_json(self, text: str) -> str | None:
		if not text:
			return None
		match = re.search(r"\{.*\}", text, re.DOTALL)
		if not match:
			return None
		return match.group(0)

	def _parse_json_response(self, text: str) -> dict[str, Any] | None:
		json_text = self._extract_json(text)
		if not json_text:
			logging.warning("LLM response could not be parsed as JSON: %s", text)
			return None
		try:
			return json.loads(json_text)
		except json.JSONDecodeError as exc:
			logging.warning("Failed to decode JSON from LLM response: %s", exc)
			return None

	def _fallback_reasoning(
		self,
		cancer_type: str,
		observations: dict[str, Any],
		patient_context: dict[str, Any] | None,
	) -> str:
		lines = [
			f"Clinical reasoning for {cancer_type}:",
			"The final probability is derived by aggregating multiple predictive patterns.",
		]
		for pattern in observations.get("patterns", []):
			lines.append(
				"- {name} predicted probability {prob:.3f} with uncertainty {unc:.3f}."
				.format(
					name=pattern.get("name", "unknown"),
					prob=pattern.get("probability", 0.0),
					unc=pattern.get("uncertainty", 0.0),
				)
			)
		if patient_context:
			lines.append("Patient context considered:")
			for key, value in patient_context.items():
				lines.append(f"- {key}: {value}")
		return " ".join(lines)


class SkinCancerAgent(ABC):
	"""
	Fixed cancer-domain agent with runtime-switchable thinking patterns.

	Agents can aggregate multiple ThinkingPatterns and use an LLM reasoner for final output.
	"""

	def __init__(
		self,
		thinking_patterns: ThinkingPattern | list[ThinkingPattern],
		llm_reasoner: LLMReasoner | None = None,
		tools: list[Tool] | None = None,
	) -> None:
		if isinstance(thinking_patterns, ThinkingPattern):
			self._thinking_patterns = [thinking_patterns]
		else:
			self._thinking_patterns = list(thinking_patterns)
		if not self._thinking_patterns:
			raise ValueError("At least one ThinkingPattern is required.")
		self._llm_reasoner = llm_reasoner or LLMReasoner()
		self._tools = tools or []
		self._tool_registry = {tool.name: tool for tool in self._tools}

	@property
	@abstractmethod
	def cancer_type(self) -> str:
		"""Fixed skin cancer domain handled by this agent."""

	@property
	def thinking_pattern_name(self) -> str:
		return "+".join(pattern.name for pattern in self._thinking_patterns)

	@property
	def thinking_patterns(self) -> tuple[ThinkingPattern, ...]:
		return tuple(self._thinking_patterns)

	@property
	def name(self) -> str:
		return f"{self.cancer_type.lower()}::{self.thinking_pattern_name}"

	def set_thinking_pattern(self, thinking_pattern: ThinkingPattern | list[ThinkingPattern]) -> None:
		if isinstance(thinking_pattern, ThinkingPattern):
			self._thinking_patterns = [thinking_pattern]
		else:
			self._thinking_patterns = list(thinking_pattern)
		if not self._thinking_patterns:
			raise ValueError("At least one ThinkingPattern is required.")

	def set_llm_reasoner(self, llm_reasoner: LLMReasoner) -> None:
		self._llm_reasoner = llm_reasoner

	def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
		for pattern in self._thinking_patterns:
			pattern.fit(x_train, y_train)

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		probabilities = [pattern.predict_proba(x) for pattern in self._thinking_patterns]
		return np.mean(np.stack(probabilities, axis=0), axis=0)

	def predict_uncertainty(self, x: np.ndarray, n_samples: int = 25) -> np.ndarray:
		uncertainties = [pattern.predict_uncertainty(x, n_samples=n_samples) for pattern in self._thinking_patterns]
		return np.clip(np.mean(np.stack(uncertainties, axis=0), axis=0), 0.0, 1.0)

	def _tool_function_schemas(self) -> list[dict[str, Any]]:
		return [tool.function_schema() for tool in self._tools]

	def _execute_tool(self, function_call: dict[str, Any]) -> dict[str, Any] | None:
		if function_call is None:
			return None
		tool_name = function_call.get("name")
		arguments = function_call.get("arguments")
		if not tool_name or arguments is None:
			return None
		tool = self._tool_registry.get(tool_name)
		if tool is None:
			return None
		try:
			parsed = json.loads(arguments) if isinstance(arguments, str) else arguments
			return tool.execute(**parsed)
		except Exception:
			return None

	def _invoke_tool_for_observation(
		self,
		observation: dict[str, Any],
		patient_context: dict[str, Any] | None = None,
	) -> dict[str, Any]:
		if not self._tools:
			return observation
		functions = self._tool_function_schemas()
		response = self._llm_reasoner.generate_reasoning(
			self.cancer_type,
			observation,
			patient_context,
			functions=functions,
			function_call="auto",
		)
		function_call_data = response.get("function_call")
		if function_call_data is None:
			return observation
		tool_result = self._execute_tool(function_call_data)
		observation["tool_action"] = {
			"selected_tool": function_call_data.get("name"),
			"arguments": json.loads(function_call_data.get("arguments", "{}")) if isinstance(function_call_data.get("arguments"), str) else function_call_data.get("arguments"),
			"result": tool_result,
		}
		return observation

	def _build_observations(
		self,
		x: np.ndarray,
		patient_context: dict[str, Any] | None = None,
		n_samples: int = 25,
	) -> list[dict[str, Any]]:
		observations: list[dict[str, Any]] = []
		for idx in range(x.shape[0]):
			pattern_entries: list[dict[str, Any]] = []
			for pattern in self._thinking_patterns:
				prob = float(pattern.predict_proba(x[idx:idx+1])[0])
				unc = float(pattern.predict_uncertainty(x[idx:idx+1], n_samples=n_samples)[0])
				pattern_entries.append(
					{
						"name": pattern.name,
						"probability": prob,
						"uncertainty": unc,
						"details": "model output for a single patient sample",
					}
				)
			observations.append(
				{
					"patterns": pattern_entries,
					"patient_context": patient_context or {},
				}
			)
		return observations

	def predict_diagnoses(
		self,
		x: np.ndarray,
		patient_context: dict[str, Any] | None = None,
		n_samples: int = 25,
		use_tools: bool = False,
	) -> list[dict[str, Any]]:
		if x.ndim == 1:
			x = x.reshape(1, -1)

		probs = self.predict_proba(x)
		uncertainties = self.predict_uncertainty(x, n_samples=n_samples)
		observations = self._build_observations(x, patient_context=patient_context, n_samples=n_samples)

		results: list[dict[str, Any]] = []
		for idx in range(x.shape[0]):
			obs = observations[idx]
			if use_tools:
				obs = self._invoke_tool_for_observation(obs, patient_context)
			reasoning_response = self._llm_reasoner.generate_reasoning(
				self.cancer_type,
				obs,
				patient_context if isinstance(patient_context, dict) else None,
			)
			results.append(
				{
					"probability": float(probs[idx]),
					"uncertainty": float(uncertainties[idx]),
					"clinical_reasoning": reasoning_response.get("text", ""),
					"observations": obs,
				}
			)

		return results
