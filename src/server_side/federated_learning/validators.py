from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import LocalHospitalUpdatePayload

REQUIRED_TOP_LEVEL_KEYS: tuple[str, ...] = (
    "schema_version",
    "generated_at_utc",
    "hospital",
    "metadata",
    "selected_patterns",
    "metrics",
    "local_summary",
    "model_update_metadata",
)

REQUIRED_METRIC_KEYS: tuple[str, ...] = (
    "accuracy",
    "f1",
    "auc",
    "sensitivity",
    "specificity",
)

REQUIRED_SPLIT_KEYS: tuple[str, ...] = ("train", "val", "test")


class UpdateValidationError(ValueError):
    """Raised when local updates fail structural or semantic validation."""


@dataclass(frozen=True)
class ValidationReport:
    """Summary emitted after validating local hospital updates."""

    schema_version: str
    validated_hospitals: tuple[str, ...]

    @property
    def count(self) -> int:
        return len(self.validated_hospitals)


def validate_local_updates(
    local_updates: Mapping[str, LocalHospitalUpdatePayload],
    *,
    required_schema_version: str | None = None,
) -> ValidationReport:
    """Validate all local hospital updates before aggregation starts."""
    if not local_updates:
        raise UpdateValidationError("No local updates were provided for aggregation.")

    validated_hospital_ids: list[str] = []
    observed_schema_versions: set[str] = set()

    for source_hospital_id, payload in local_updates.items():
        validate_local_update(
            payload,
            required_schema_version=required_schema_version,
            source_hospital_id=source_hospital_id,
        )
        payload_hospital_id = payload["hospital"]["hospital_id"]
        validated_hospital_ids.append(payload_hospital_id)
        observed_schema_versions.add(payload["schema_version"])

    _ensure_unique_hospital_ids(validated_hospital_ids)

    if len(observed_schema_versions) != 1:
        raise UpdateValidationError(
            "Schema version mismatch across local updates: "
            f"found {sorted(observed_schema_versions)}"
        )

    schema_version = next(iter(observed_schema_versions))
    return ValidationReport(
        schema_version=schema_version,
        validated_hospitals=tuple(validated_hospital_ids),
    )


def validate_local_update(
    payload: LocalHospitalUpdatePayload,
    *,
    required_schema_version: str | None = None,
    source_hospital_id: str | None = None,
) -> None:
    """Validate one local hospital update payload."""
    _ensure_required_top_level_keys(payload)

    schema_version = payload["schema_version"]
    if not isinstance(schema_version, str) or not schema_version.strip():
        raise UpdateValidationError("`schema_version` must be a non-empty string.")
    if required_schema_version is not None and schema_version != required_schema_version:
        raise UpdateValidationError(
            "Schema version mismatch: "
            f"expected `{required_schema_version}`, got `{schema_version}`"
        )

    hospital = payload["hospital"]
    if not isinstance(hospital, Mapping):
        raise UpdateValidationError("`hospital` must be a mapping object.")

    hospital_id = hospital.get("hospital_id")
    lifecycle_state = hospital.get("lifecycle_state")
    if not isinstance(hospital_id, str) or not hospital_id.strip():
        raise UpdateValidationError("`hospital.hospital_id` must be a non-empty string.")
    if source_hospital_id is not None and hospital_id != source_hospital_id:
        raise UpdateValidationError(
            "Hospital ID mismatch between source key and payload: "
            f"source=`{source_hospital_id}`, payload=`{hospital_id}`"
        )
    if not isinstance(lifecycle_state, str) or not lifecycle_state.strip():
        raise UpdateValidationError("`hospital.lifecycle_state` must be a non-empty string.")

    _validate_split_sizes(payload)
    _validate_metrics(payload)
    _validate_selected_patterns(payload)


def _ensure_required_top_level_keys(payload: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in payload]
    if missing:
        raise UpdateValidationError(
            "Missing required top-level keys in local update: " + ", ".join(missing)
        )


def _validate_split_sizes(payload: Mapping[str, Any]) -> None:
    metadata = payload["metadata"]
    if not isinstance(metadata, Mapping):
        raise UpdateValidationError("`metadata` must be a mapping object.")

    split_sizes = metadata.get("split_sizes")
    if not isinstance(split_sizes, Mapping):
        raise UpdateValidationError("`metadata.split_sizes` must be a mapping object.")

    missing_split_keys = [key for key in REQUIRED_SPLIT_KEYS if key not in split_sizes]
    if missing_split_keys:
        raise UpdateValidationError(
            "Missing required split size keys: " + ", ".join(missing_split_keys)
        )

    for split_name in REQUIRED_SPLIT_KEYS:
        value = split_sizes[split_name]
        if not isinstance(value, int):
            raise UpdateValidationError(
                f"`metadata.split_sizes.{split_name}` must be an integer, got {type(value).__name__}."
            )
        if value < 0:
            raise UpdateValidationError(
                f"`metadata.split_sizes.{split_name}` must be >= 0, got {value}."
            )


def _validate_metrics(payload: Mapping[str, Any]) -> None:
    metrics = payload["metrics"]
    if not isinstance(metrics, Mapping):
        raise UpdateValidationError("`metrics` must be a mapping object.")

    per_agent = metrics.get("per_agent")
    if not isinstance(per_agent, Mapping):
        raise UpdateValidationError("`metrics.per_agent` must be a mapping object.")
    if not per_agent:
        raise UpdateValidationError("`metrics.per_agent` cannot be empty.")

    for agent_name, metric_bundle in per_agent.items():
        if not isinstance(metric_bundle, Mapping):
            raise UpdateValidationError(
                f"`metrics.per_agent.{agent_name}` must be a mapping object."
            )
        missing_metric_keys = [key for key in REQUIRED_METRIC_KEYS if key not in metric_bundle]
        if missing_metric_keys:
            raise UpdateValidationError(
                f"`metrics.per_agent.{agent_name}` is missing keys: "
                + ", ".join(missing_metric_keys)
            )

        for key in REQUIRED_METRIC_KEYS:
            _ensure_probability_like_metric(
                value=metric_bundle[key],
                path=f"metrics.per_agent.{agent_name}.{key}",
            )


def _validate_selected_patterns(payload: Mapping[str, Any]) -> None:
    selected_patterns = payload["selected_patterns"]
    if not isinstance(selected_patterns, Mapping):
        raise UpdateValidationError("`selected_patterns` must be a mapping object.")
    if not selected_patterns:
        raise UpdateValidationError("`selected_patterns` cannot be empty.")

    for cancer_type, pattern_name in selected_patterns.items():
        if not isinstance(cancer_type, str) or not cancer_type.strip():
            raise UpdateValidationError("`selected_patterns` contains an empty cancer type key.")
        if not isinstance(pattern_name, str) or not pattern_name.strip():
            raise UpdateValidationError(
                f"`selected_patterns.{cancer_type}` must be a non-empty string."
            )


def _ensure_probability_like_metric(*, value: Any, path: str) -> None:
    if not isinstance(value, (int, float)):
        raise UpdateValidationError(f"`{path}` must be numeric, got {type(value).__name__}.")

    numeric = float(value)
    if not math.isfinite(numeric):
        raise UpdateValidationError(f"`{path}` must be finite, got {value!r}.")
    if numeric < 0.0 or numeric > 1.0:
        raise UpdateValidationError(f"`{path}` must be in [0.0, 1.0], got {numeric}.")


def _ensure_unique_hospital_ids(hospital_ids: Iterable[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()

    for hospital_id in hospital_ids:
        if hospital_id in seen:
            duplicates.add(hospital_id)
        seen.add(hospital_id)

    if duplicates:
        raise UpdateValidationError(
            "Duplicate hospital IDs detected in local updates: " + ", ".join(sorted(duplicates))
        )
