from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class ThinkingPattern(ABC):
	@abstractmethod
	def save_model(self, file_path: str) -> None:
		"""Save model parameters to file."""

	@abstractmethod
	def load_model(self, file_path: str) -> None:
		"""Load model parameters from file."""
	"""Interchangeable decision-making algorithm used by a cancer agent."""

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
		# Default: deterministic patterns emit constant low uncertainty.
		probs = self.predict_proba(x)
		return np.full_like(probs, 0.05, dtype=np.float32)


class SkinCancerAgent(ABC):
	"""
	Fixed cancer-domain agent with runtime-switchable thinking pattern.

	The cancer type remains constant per agent instance, while the thinking pattern can
	be replaced to compare performance across algorithms.
	"""

	def __init__(self, thinking_pattern: ThinkingPattern) -> None:
		self._thinking_pattern = thinking_pattern

	@property
	@abstractmethod
	def cancer_type(self) -> str:
		"""Fixed skin cancer domain handled by this agent."""

	@property
	def thinking_pattern_name(self) -> str:
		return self._thinking_pattern.name

	@property
	def name(self) -> str:
		return f"{self.cancer_type.lower()}::{self.thinking_pattern_name}"

	def set_thinking_pattern(self, thinking_pattern: ThinkingPattern) -> None:
		self._thinking_pattern = thinking_pattern

	def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
		self._thinking_pattern.fit(x_train, y_train)

	def predict_proba(self, x: np.ndarray) -> np.ndarray:
		return self._thinking_pattern.predict_proba(x)
