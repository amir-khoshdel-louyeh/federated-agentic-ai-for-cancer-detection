from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np

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
    _set_reproducibility_seed(args.seed)

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

    run_report = {
        "consistency_report": consistency_report,
        "artifacts": artifact_manifest,
        "hospitals": outputs,
    }
    (out_dir / "multi_hospital_report.json").write_text(json.dumps(run_report, indent=2), encoding="utf-8")

    print(json.dumps({"consistency_report": consistency_report, "artifacts": artifact_manifest}, indent=2))


if __name__ == "__main__":
    main()
