from __future__ import annotations

from collections.abc import Sequence

DEFAULT_CANCER_TYPES: tuple[str, ...] = ("BCC", "SCC", "MELANOMA", "AKIEC")
DEFAULT_MALIGNANT_HAM: set[str] = {"mel", "bcc", "akiec", "scc"}
DEFAULT_MALIGNANT_ISIC: set[str] = {"MEL", "BCC", "AK", "SCC"}


def _normalize_name(name: str) -> str:
    return name.strip().upper()


def _sequence_to_tuple(names: Sequence[str]) -> tuple[str, ...]:
    normalized = [_normalize_name(str(name)) for name in names if str(name).strip()]
    return tuple(sorted(dict.fromkeys(normalized)))


def get_cancer_types(config: dict | None = None) -> tuple[str, ...]:
    if not config:
        return DEFAULT_CANCER_TYPES

    # Explicit top-level section takes precedence.
    c_types = config.get("cancer_types")
    if isinstance(c_types, (list, tuple, set)) and c_types:
        return _sequence_to_tuple(c_types)

    # Fallback to agents.types list if present.
    types_cfg = config.get("agents", {}).get("types")
    if isinstance(types_cfg, (list, tuple, set)) and types_cfg:
        return _sequence_to_tuple(types_cfg)

    # Fallback to pattern default mapping keys.
    default_mapping = config.get("agents", {}).get("patterns", {}).get("default_mapping")
    if isinstance(default_mapping, dict) and default_mapping:
        return _sequence_to_tuple(default_mapping.keys())

    return DEFAULT_CANCER_TYPES


def get_malignant_ham(config: dict | None = None) -> set[str]:
    if not config:
        return DEFAULT_MALIGNANT_HAM
    ham = config.get("data_split", {}).get("malignant_ham")
    if isinstance(ham, (list, tuple, set)) and ham:
        return {str(x).strip().lower() for x in ham if str(x).strip()}
    return DEFAULT_MALIGNANT_HAM


def get_malignant_isic(config: dict | None = None) -> set[str]:
    if not config:
        return DEFAULT_MALIGNANT_ISIC
    isic = config.get("data_split", {}).get("malignant_isic")
    if isinstance(isic, (list, tuple, set)) and isic:
        return {str(x).strip().upper() for x in isic if str(x).strip()}
    return DEFAULT_MALIGNANT_ISIC
