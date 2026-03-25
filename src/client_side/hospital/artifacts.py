from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_hospital_artifacts(
    *,
    hospital_output: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, str]:
    """Persist hospital output artifacts with hospital-specific filenames."""
    hospital_id = str(hospital_output.get("hospital", {}).get("hospital_id", "unknown_hospital"))
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    full_output_path = output_dir / f"{hospital_id}_output.json"
    summary_path = output_dir / f"{hospital_id}_summary.json"

    summary_payload = _build_summary_payload(hospital_output)

    full_output_path.write_text(json.dumps(hospital_output, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    return {
        "full_output_path": str(full_output_path),
        "summary_path": str(summary_path),
    }


def _build_summary_payload(hospital_output: dict[str, Any]) -> dict[str, Any]:
    metadata = hospital_output.get("metadata", {})
    selected_patterns = hospital_output.get("selected_patterns", {})
    local_summary = hospital_output.get("local_summary", {})

    return {
        "schema_version": hospital_output.get("schema_version"),
        "hospital": hospital_output.get("hospital", {}),
        "random_seed": metadata.get("extra", {}).get("random_seed"),
        "split_sizes": metadata.get("split_sizes", {}),
        "selected_patterns": selected_patterns,
        "local_summary": local_summary,
    }
