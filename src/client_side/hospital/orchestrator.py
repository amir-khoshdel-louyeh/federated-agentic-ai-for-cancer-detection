from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Mapping, Protocol

from src.server_side.federated_learning.contracts import (
	AggregatorContract,
	FederatedOrchestratorContract,
	LocalHospitalUpdatePayload,
	OrchestratorRoundOutput,
)
from src.server_side.federated_learning.factory import build_aggregator
from src.server_side.federated_learning.prompt_evolution import evolve_prompt
from src.server_side.federated_learning.validators import ValidationReport, validate_local_updates


class HospitalFederatedClient(Protocol):
	"""Minimal hospital-side interface required by the orchestrator."""

	def get_local_update(self) -> dict[str, Any]:
		...

	def apply_global_update(self, global_state: Mapping[str, Any]) -> None:
		...


@dataclass
class FederatedRoundOrchestrator(FederatedOrchestratorContract):
	"""Coordinates federated rounds: collect, validate, aggregate, broadcast."""

	aggregator: AggregatorContract
	required_schema_version: str | None = None
	current_global_state: dict[str, Any] | None = None
	round_history: list[OrchestratorRoundOutput] = field(default_factory=list)
	prompt_evolution_config: dict[str, Any] | None = None

	@classmethod
	def from_algorithm(
		cls,
		*,
		name: str,
		required_schema_version: str | None = None,
		prompt_evolution_config: dict[str, Any] | None = None,
		**kwargs: Any,
	) -> "FederatedRoundOrchestrator":
		"""Build orchestrator from a supported runtime aggregator name."""
		return cls(
			aggregator=build_aggregator(name, **kwargs),
			required_schema_version=required_schema_version,
			prompt_evolution_config=prompt_evolution_config,
		)

	def collect_local_updates(
		self,
		hospitals: Mapping[str, HospitalFederatedClient],
	) -> dict[str, LocalHospitalUpdatePayload]:
		"""Collect local updates from provided hospital clients."""
		updates: dict[str, LocalHospitalUpdatePayload] = {}
		for hospital_id, hospital in hospitals.items():
			update = hospital.get_local_update()
			updates[hospital_id] = update
		return updates

	def run_round(
		self,
		*,
		round_index: int,
		local_updates: Mapping[str, LocalHospitalUpdatePayload],
		previous_global_state: Mapping[str, Any] | None = None,
	) -> OrchestratorRoundOutput:
		"""Validate updates, run aggregation, and generate one global round state."""
		validation = validate_local_updates(
			local_updates,
			required_schema_version=self.required_schema_version,
		)

		prior_state = previous_global_state if previous_global_state is not None else self.current_global_state
		aggregation = self.aggregator.aggregate(
			round_index=round_index,
			local_updates=local_updates,
			previous_global_state=prior_state,
		)

		global_state = self._build_global_state(
			round_index=round_index,
			validation=validation,
			aggregation=aggregation,
			previous_global_state=prior_state,
		)

		prompt_update = self._maybe_evolve_prompt(
			local_updates=local_updates,
			previous_global_state=prior_state,
			round_index=round_index,
		)
		if prompt_update is not None:
			global_state["prompt_evolution"] = prompt_update

		round_output = OrchestratorRoundOutput(
			round_index=round_index,
			aggregator_name=self.aggregator.name,
			global_state=global_state,
			aggregation=aggregation,
			validation_report={
				"schema_version": validation.schema_version,
				"validated_hospitals": list(validation.validated_hospitals),
				"count": validation.count,
			},
		)

		self.current_global_state = global_state
		self.round_history.append(round_output)
		return round_output

	def _maybe_evolve_prompt(
		self,
		*,
		local_updates: Mapping[str, LocalHospitalUpdatePayload],
		previous_global_state: Mapping[str, Any] | None,
		round_index: int,
	) -> dict[str, Any] | None:
		"""Optionally evolve the system prompt after a federated round."""
		if not self.prompt_evolution_config:
			return None
		try:
			return evolve_prompt(
				local_updates=local_updates,
				previous_global_state=previous_global_state,
				config=self.prompt_evolution_config,
				round_index=round_index,
			)
		except Exception as exc:
			logging.warning("Prompt evolution failed: %s", exc)
			return None

# run_round_from_hospitals and run are deprecated in this minimal pipeline and removed for cleanup.

	def run_with_early_stopping(
		self,
		*,
		hospitals: Mapping[str, HospitalFederatedClient],
		num_epochs: int,
		num_rounds: int,
		validation_fn: Callable[[Mapping[str, HospitalFederatedClient]], dict[str, Any]],
		monitor_metric: str = "f1",
		patience: int = 2,
		min_delta: float = 1e-4,
	) -> dict[str, Any]:
		"""Run training rounds with early stopping based on validation metric.

		validation_fn returns a dict with keys:
		- 'validation_reports': per-hospital validation metrics
		- 'should_stop': optional hack flag
		"""
		if num_epochs < 1 or num_rounds < 1:
			raise ValueError("num_epochs and num_rounds must be positive integers")

		if monitor_metric not in {"f1", "auc", "loss"}:
			raise ValueError("monitor_metric must be 'f1', 'auc', or 'loss'")

		if monitor_metric == "loss":
			best_metric = float("inf")
			better_than = lambda current, best: current < best - min_delta
		else:
			best_metric = float("-inf")
			better_than = lambda current, best: current > best + min_delta

		best_epoch = 0
		stale = 0
		history = []

		for epoch in range(1, num_epochs + 1):
			for round_idx in range(1, num_rounds + 1):
				current_round = (epoch - 1) * num_rounds + round_idx
				# Train hospitals before collecting an update for this round.
				for hospital in hospitals.values():
					hospital.train()
				local_updates = self.collect_local_updates(hospitals)
				round_output = self.run_round(
					round_index=current_round,
					local_updates=local_updates,
					previous_global_state=self.current_global_state,
				)
				self.broadcast_global_state(hospitals, round_output.global_state)
				history.append(round_output)

			validation_result = validation_fn(hospitals)
			validation_reports = validation_result.get("validation_reports") if isinstance(validation_result, dict) else {}
			should_stop = bool(validation_result.get("should_stop", False)) if isinstance(validation_result, dict) else False

			# compute average metric over all hospitals and cancer agents
			values = []
			for hid, hospital_report in (validation_reports or {}).items():
				for cancer_result in hospital_report.values():
					m = cancer_result.get("metrics", {}).get(monitor_metric)
					if m is not None:
						values.append(float(m))

			avg_metric = float(sum(values) / len(values)) if values else (float("inf") if monitor_metric == "loss" else 0.0)
			if better_than(avg_metric, best_metric):
				best_metric = avg_metric
				best_epoch = epoch
				stale = 0
			else:
				stale += 1

			if should_stop or (patience > 0 and stale >= patience):
				return {
					"rounds": history,
					"best_epoch": best_epoch,
					"best_metric": best_metric,
					"stopped_early": True,
					"validation_reports": validation_reports,
				}

		return {
			"rounds": history,
			"best_epoch": best_epoch,
			"best_metric": best_metric,
			"stopped_early": False,
			"validation_reports": validation_reports,
		}

	@staticmethod
	def _build_global_state(
		*,
		round_index: int,
		validation: ValidationReport,
		aggregation: Any,
		previous_global_state: Mapping[str, Any] | None,
	) -> dict[str, Any]:
		return {
			"schema_version": validation.schema_version,
			"generated_at_utc": datetime.now(timezone.utc).isoformat(),
			"round_index": int(round_index),
			"aggregator": str(aggregation.algorithm),
			"global_metrics": dict(aggregation.global_metrics),
			"hospital_weights": dict(aggregation.hospital_weights),
			"included_hospital_ids": list(aggregation.included_hospital_ids),
			"dropped_hospitals": dict(aggregation.dropped_hospitals),
			"aggregation_details": dict(aggregation.details),
			"model_weights": aggregation.model_weights,
			"validation": {
				"count": validation.count,
				"validated_hospitals": list(validation.validated_hospitals),
			},
			"previous_global_state_available": previous_global_state is not None,
		}

	def broadcast_global_state(
		self,
		hospitals: Mapping[str, HospitalFederatedClient],
		global_state: Mapping[str, Any],
	) -> None:
		"""Broadcast global state to all participating hospitals."""
		for hospital in hospitals.values():
			hospital.apply_global_update(global_state)

