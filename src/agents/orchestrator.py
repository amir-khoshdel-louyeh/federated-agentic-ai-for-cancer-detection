from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Protocol

from src.federated_learning.contracts import (
	AggregatorContract,
	FederatedOrchestratorContract,
	LocalHospitalUpdatePayload,
	OrchestratorRoundOutput,
)
from src.federated_learning.factory import build_aggregator
from src.federated_learning.validators import ValidationReport, validate_local_updates


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

	@classmethod
	def from_algorithm(
		cls,
		*,
		name: str,
		required_schema_version: str | None = None,
		**kwargs: Any,
	) -> "FederatedRoundOrchestrator":
		"""Build orchestrator from a supported runtime aggregator name."""
		return cls(
			aggregator=build_aggregator(name, **kwargs),
			required_schema_version=required_schema_version,
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

	def run_round_from_hospitals(
		self,
		*,
		round_index: int,
		hospitals: Mapping[str, HospitalFederatedClient],
		previous_global_state: Mapping[str, Any] | None = None,
	) -> OrchestratorRoundOutput:
		"""Execute a full round including collection and broadcast."""
		local_updates = self.collect_local_updates(hospitals)
		round_output = self.run_round(
			round_index=round_index,
			local_updates=local_updates,
			previous_global_state=previous_global_state,
		)
		self.broadcast_global_state(hospitals, round_output.global_state)
		return round_output

	def run(
		self,
		*,
		total_rounds: int,
		local_updates_by_round: Mapping[int, Mapping[str, LocalHospitalUpdatePayload]],
	) -> list[OrchestratorRoundOutput]:
		"""Run multiple rounds using explicit per-round local updates."""
		if total_rounds <= 0:
			raise ValueError("`total_rounds` must be > 0.")

		outputs: list[OrchestratorRoundOutput] = []
		for round_index in range(1, total_rounds + 1):
			if round_index not in local_updates_by_round:
				raise ValueError(
					f"Missing local updates for round {round_index}. "
					"Provide updates for every round index from 1..total_rounds."
				)

			output = self.run_round(
				round_index=round_index,
				local_updates=local_updates_by_round[round_index],
				previous_global_state=self.current_global_state,
			)
			outputs.append(output)

		return outputs

	def broadcast_global_state(
		self,
		hospitals: Mapping[str, HospitalFederatedClient],
		global_state: Mapping[str, Any],
	) -> None:
		"""Broadcast global state to all participating hospitals."""
		for hospital in hospitals.values():
			hospital.apply_global_update(global_state)

	def get_round_history(self) -> list[OrchestratorRoundOutput]:
		"""Return a copy of the in-memory round history."""
		return list(self.round_history)

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
			"validation": {
				"count": validation.count,
				"validated_hospitals": list(validation.validated_hospitals),
			},
			"previous_global_state_available": previous_global_state is not None,
		}
