from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.federated_learning import FederatedRoundOrchestrator, supported_aggregator_names

from .artifacts import save_hospital_artifacts
from .hospital_env import VirtualHospital
from .hospital_node import HospitalNode
from .pattern_policy import StaticPatternPolicy
from .simulation_runner import simulate_multi_hospital

LOGGER = logging.getLogger("hospital_runner")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HospitalNode thin runner")
    parser.add_argument("--ham-csv", required=True, help="Path to HAM10000 metadata CSV")
    parser.add_argument("--isic-csv", required=True, help="Path to ISIC 2019 labels CSV")
    parser.add_argument("--out-dir", default="outputs", help="Output directory")
    parser.add_argument(
        "--hospital-ids",
        default="hospital_1,hospital_2,hospital_3",
        help="Comma-separated hospital IDs",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--simulate-multi",
        action="store_true",
        help="Run multi-hospital simulation and validate uniform contracts",
    )
    parser.add_argument(
        "--config",
        default="configs/hospital_local.yaml",
        help="Path to YAML config file for runner defaults",
    )
    parser.add_argument(
        "--aggregation",
        default=None,
        help="Federation algorithm override: fedavg|fedprox|adaptive",
    )
    parser.add_argument(
        "--compare-all",
        action="store_true",
        help="Run all aggregation algorithms on identical local updates and compare outputs",
    )
    return parser.parse_args()


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def _set_reproducibility_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        LOGGER.warning("Torch is not installed; skipped torch seed setup.")


def _parse_hospital_ids(raw_hospital_ids: str) -> list[str]:
    ids = [item.strip() for item in raw_hospital_ids.split(",") if item.strip()]
    if not ids:
        raise ValueError("At least one hospital ID is required.")
    return ids


def _load_runner_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        LOGGER.warning("Config file %s not found; using defaults/CLI values.", config_path)
        return {}

    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {config_path} must contain a top-level mapping.")
    return data


def _resolve_aggregation_algorithm(args: argparse.Namespace, config: dict[str, Any]) -> str:
    cli_value = str(args.aggregation).strip().lower() if args.aggregation is not None else ""
    federation_config = config.get("federation", {})
    config_value = ""
    if isinstance(federation_config, dict):
        config_value = str(federation_config.get("aggregation_algorithm", "")).strip().lower()

    selected = cli_value or config_value or "fedavg"
    allowed = set(supported_aggregator_names())
    if selected not in allowed:
        raise ValueError(
            "Invalid aggregation algorithm `{}`; allowed values: {}".format(
                selected,
                ", ".join(sorted(allowed)),
            )
        )
    return selected


def _infer_schema_version(local_updates: dict[str, dict[str, Any]]) -> str | None:
    for payload in local_updates.values():
        schema_version = payload.get("schema_version")
        if isinstance(schema_version, str) and schema_version.strip():
            return schema_version
    return None


def _run_federation_comparison(
    *,
    local_updates: dict[str, dict[str, Any]],
    selected_algorithm: str,
    compare_all: bool,
) -> dict[str, Any]:
    algorithms = list(supported_aggregator_names()) if compare_all else [selected_algorithm]
    schema_version = _infer_schema_version(local_updates)

    results: dict[str, Any] = {}
    for algorithm in algorithms:
        orchestrator = FederatedRoundOrchestrator.from_algorithm(
            name=algorithm,
            required_schema_version=schema_version,
        )
        round_output = orchestrator.run_round(round_index=1, local_updates=local_updates)
        results[algorithm] = {
            "global_metrics": dict(round_output.aggregation.global_metrics),
            "hospital_weights": dict(round_output.aggregation.hospital_weights),
            "details": dict(round_output.aggregation.details),
            "validation_report": dict(round_output.validation_report),
            "global_state": dict(round_output.global_state),
        }

    return {
        "mode": "compare_all" if compare_all else "single",
        "selected_algorithm": selected_algorithm,
        "algorithms_executed": algorithms,
        "results": results,
    }


def _run_single_hospital(
    *,
    hospital_id: str,
    ham_csv: str,
    isic_csv: str,
    seed: int,
) -> dict[str, object]:
    hospital = HospitalNode(
        hospital_id=hospital_id,
        ham_metadata_csv=ham_csv,
        isic_labels_csv=isic_csv,
        dataset_handler=VirtualHospital(random_state=seed),
        pattern_policy=StaticPatternPolicy(hospital_id=hospital_id),
    )
    hospital.initialize()
    hospital.train()
    hospital.evaluate()
    return hospital.get_local_update()


def main() -> None:
    _configure_logging()
    args = parse_args()
    runner_config = _load_runner_config(Path(args.config))
    _set_reproducibility_seed(args.seed)
    selected_algorithm = _resolve_aggregation_algorithm(args, runner_config)

    hospital_ids = _parse_hospital_ids(args.hospital_ids)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.simulate_multi and len(hospital_ids) < 2:
        raise ValueError("--simulate-multi requires at least 2 hospital IDs.")

    if args.simulate_multi or len(hospital_ids) > 1:
        result = simulate_multi_hospital(
            ham_metadata_csv=args.ham_csv,
            isic_labels_csv=args.isic_csv,
            hospital_ids=hospital_ids,
        )
        outputs = result.hospital_outputs
        consistency_report = result.consistency_report
    else:
        output = _run_single_hospital(
            hospital_id=hospital_ids[0],
            ham_csv=args.ham_csv,
            isic_csv=args.isic_csv,
            seed=args.seed,
        )
        outputs = {hospital_ids[0]: output}
        consistency_report = {
            "num_hospitals": 1,
            "uniform_contract": True,
            "schema_version": output.get("schema_version"),
        }

    artifact_manifest: dict[str, dict[str, str]] = {}
    for hospital_id, output in outputs.items():
        saved_paths = save_hospital_artifacts(hospital_output=output, out_dir=out_dir)
        artifact_manifest[hospital_id] = saved_paths

        metadata = output.get("metadata", {})
        LOGGER.info("Hospital %s seed=%s", hospital_id, metadata.get("extra", {}).get("random_seed"))
        LOGGER.info("Hospital %s selected_patterns=%s", hospital_id, output.get("selected_patterns", {}))
        LOGGER.info("Hospital %s summary=%s", hospital_id, output.get("local_summary", {}))

    federation_report = _run_federation_comparison(
        local_updates=outputs,
        selected_algorithm=selected_algorithm,
        compare_all=bool(args.compare_all),
    )
    federation_report_path = out_dir / "federation_comparison.json"
    federation_report_path.write_text(json.dumps(federation_report, indent=2), encoding="utf-8")

    LOGGER.info(
        "Federation mode=%s selected=%s algorithms=%s",
        federation_report["mode"],
        federation_report["selected_algorithm"],
        federation_report["algorithms_executed"],
    )

    run_report = {
        "consistency_report": consistency_report,
        "artifacts": artifact_manifest,
        "hospitals": outputs,
        "federation": federation_report,
    }
    (out_dir / "multi_hospital_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "consistency_report": consistency_report,
                "artifacts": artifact_manifest,
                "federation_mode": federation_report["mode"],
                "algorithms_executed": federation_report["algorithms_executed"],
                "federation_report": str(federation_report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
