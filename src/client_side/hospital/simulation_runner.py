from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .hospital_env import VirtualHospital
from .hospital_node import HospitalNode
from .pattern_policy import StaticPatternPolicy


@dataclass(frozen=True)
class MultiHospitalSimulationResult:
    hospital_outputs: dict[str, dict[str, Any]]
    consistency_report: dict[str, Any]


def simulate_multi_hospital(
    *,
    ham_metadata_csv: str | Path,
    isic_labels_csv: str | Path,
    hospital_ids: Iterable[str],
) -> MultiHospitalSimulationResult:
    """Run independent HospitalNode instances and verify uniform output contracts."""
    ids = [hospital_id.strip() for hospital_id in hospital_ids if hospital_id.strip()]
    if len(ids) < 2:
        raise ValueError("simulate_multi_hospital requires at least 2 hospital IDs.")

    outputs: dict[str, dict[str, Any]] = {}
    for index, hospital_id in enumerate(ids):
        # Distinct random_seeds and pattern mappings emulate heterogeneous local centers.
        dataset_handler = VirtualHospital(random_state=42 + index)
        policy = StaticPatternPolicy(
            hospital_id=hospital_id,
            hospital_overrides={hospital_id: _hospital_override_mapping(index)},
        )

        node = HospitalNode(
            hospital_id=hospital_id,
            ham_metadata_csv=ham_metadata_csv,
            isic_labels_csv=isic_labels_csv,
            dataset_handler=dataset_handler,
            pattern_policy=policy,
        )
        node.initialize()
        node.train()
        node.evaluate()
        outputs[hospital_id] = node.get_local_update()

    consistency = _validate_uniform_contracts(outputs)
    return MultiHospitalSimulationResult(hospital_outputs=outputs, consistency_report=consistency)


def _hospital_override_mapping(index: int) -> dict[str, str]:
    mappings = [
        {"BCC": "rule_based", "SCC": "bayesian", "MELANOMA": "deep_learning", "AKIEC": "rule_based"},
        {"BCC": "bayesian", "SCC": "rule_based", "MELANOMA": "deep_learning", "AKIEC": "rule_based_strict"},
        {"BCC": "rule_based_strict", "SCC": "bayesian", "MELANOMA": "rule_based", "AKIEC": "deep_learning"},
    ]
    return mappings[index % len(mappings)]


def _validate_uniform_contracts(hospital_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if not hospital_outputs:
        raise ValueError("No hospital outputs were produced during simulation.")

    output_list = list(hospital_outputs.values())
    top_level_keys = set(output_list[0].keys())
    schema_version = output_list[0].get("schema_version")

    for hospital_id, output in hospital_outputs.items():
        if set(output.keys()) != top_level_keys:
            raise ValueError(f"Output schema key mismatch for hospital {hospital_id}.")
        if output.get("schema_version") != schema_version:
            raise ValueError(f"Schema version mismatch for hospital {hospital_id}.")

    per_hospital_agent_counts: dict[str, int] = {}
    for hospital_id, output in hospital_outputs.items():
        per_agent = output.get("metrics", {}).get("per_agent", {})
        per_hospital_agent_counts[hospital_id] = len(per_agent)

    return {
        "num_hospitals": len(hospital_outputs),
        "schema_version": schema_version,
        "top_level_keys": sorted(top_level_keys),
        "agent_count_by_hospital": per_hospital_agent_counts,
        "uniform_contract": True,
    }
