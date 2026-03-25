from __future__ import annotations

import json
from pathlib import Path
from typing import Any


from src.client_side.hospital.orchestrator import FederatedRoundOrchestrator
from .validators import validate_local_updates
# Import config loader from configs package
from configs.config_loader import load_config



# Load config once at module level
_CONFIG = load_config()

def _build_local_update(hospital_id: str, random_seed_shift: int) -> dict[str, Any]:
    # Use config-driven values
    agents_cfg = _CONFIG.get("agents", {})
    patterns_cfg = agents_cfg.get("patterns", {})
    default_mapping = patterns_cfg.get("default_mapping", {
        "BCC": "rule_based",
        "SCC": "bayesian",
        "MELANOMA": "deep_learning",
        "AKIEC": "rule_based_strict",
    })
    agent_types = agents_cfg.get("types", ["akiec_agent", "bcc_agent", "melanoma_agent", "scc_agent"])
    num_agents = _CONFIG.get("num_agents_per_hospital", 4)
    random_seed_base = _CONFIG.get("sampling", {}).get("random_seed", 42)
    # Compose metrics keys from mapping
    pattern_keys = {
        f"{cancer_type.lower()}::{pattern}": cancer_type for cancer_type, pattern in default_mapping.items()
    }
    base = 0.62 + 0.03 * random_seed_shift
    metrics = {}
    for key in pattern_keys:
        # Just for demonstration, use base values as before
        if "bcc" in key:
            metrics[key] = {
                "accuracy": base,
                "f1": base - 0.05,
                "auc": base + 0.07,
                "sensitivity": base - 0.03,
                "specificity": base + 0.02,
            }
        elif "scc" in key:
            metrics[key] = {
                "accuracy": base + 0.01,
                "f1": base - 0.02,
                "auc": base + 0.05,
                "sensitivity": base - 0.01,
                "specificity": base + 0.01,
            }
        elif "melanoma" in key:
            metrics[key] = {
                "accuracy": base + 0.03,
                "f1": base + 0.01,
                "auc": base + 0.09,
                "sensitivity": base + 0.02,
                "specificity": base + 0.02,
            }
        elif "akiec" in key:
            metrics[key] = {
                "accuracy": base - 0.01,
                "f1": base - 0.04,
                "auc": base + 0.03,
                "sensitivity": base - 0.02,
                "specificity": base,
            }

    return {
        "schema_version": "1.0.0",
        "generated_at_utc": "2026-03-18T00:00:00+00:00",
        "hospital": {
            "hospital_id": hospital_id,
            "lifecycle_state": "evaluated",
        },
        "metadata": {
            "split_sizes": {
                "train": 400 + (random_seed_shift * 50),
                "val": 100 + (random_seed_shift * 10),
                "test": 120 + (random_seed_shift * 10),
            },
            "training_warnings": {} if random_seed_shift != 1 else {"SCC": "class imbalance fallback"},
            "extra": {"random_seed": random_seed_base + random_seed_shift},
        },
        "selected_patterns": dict(default_mapping),
        "metrics": {
            "per_agent": metrics,
            "selected_performance": {},
            "candidate_pattern_comparisons": {},
        },
        "local_summary": {
            "num_agents": num_agents,
            "average_accuracy": sum(v["accuracy"] for v in metrics.values()) / len(metrics) if metrics else 0.0,
            "average_f1": sum(v["f1"] for v in metrics.values()) / len(metrics) if metrics else 0.0,
            "average_auc": sum(v["auc"] for v in metrics.values()) / len(metrics) if metrics else 0.0,
            "best_agent_by_auc": {
                "name": next(iter(metrics)),
                "auc": metrics[next(iter(metrics))]["auc"] if metrics else 0.0,
            },
        },
        "model_update_metadata": {
            "update_format": "hospital-local-metrics-only",
            "has_model_weights": False,
        },
    }


def _run_once(local_updates: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, dict[str, Any]] = {}
    previous_global_state: dict[str, Any] | None = None

    for algorithm in ("fedavg", "fedprox", "adaptive"):
        orchestrator = FederatedRoundOrchestrator.from_algorithm(
            name=algorithm,
            required_schema_version="1.0.0",
        )
        round_output = orchestrator.run_round(
            round_index=1,
            local_updates=local_updates,
            previous_global_state=previous_global_state,
        )
        outputs[algorithm] = {
            "global_metrics": dict(round_output.aggregation.global_metrics),
            "hospital_weights": dict(round_output.aggregation.hospital_weights),
            "schema_version": str(round_output.global_state.get("schema_version")),
        }
        previous_global_state = dict(round_output.global_state)

    return outputs


def run_deterministic_validation(out_dir: str | Path = "outputs/deterministic_validation") -> dict[str, Any]:
    local_updates = {
        "hospital_1": _build_local_update("hospital_1", 0),
        "hospital_2": _build_local_update("hospital_2", 1),
        "hospital_3": _build_local_update("hospital_3", 2),
    }

    validation_report = validate_local_updates(
        local_updates,
        required_schema_version="1.0.0",
    )

    run1 = _run_once(local_updates)
    run2 = _run_once(local_updates)

    report = {
        "validated_hospitals": list(validation_report.validated_hospitals),
        "algorithms": ["fedavg", "fedprox", "adaptive"],
        "run1": run1,
        "run2": run2,
        "reproducible": run1 == run2,
        "schema_consistent": all(v.get("schema_version") == "1.0.0" for v in run1.values()),
    }

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "step12_validation_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    report["report_path"] = str(output_path)
    return report


def main() -> None:
    report = run_deterministic_validation()
    print(
        json.dumps(
            {
                "reproducible": report["reproducible"],
                "schema_consistent": report["schema_consistent"],
                "report_path": report["report_path"],
                "final_global_metrics": {
                    key: value["global_metrics"]
                    for key, value in report["run1"].items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
