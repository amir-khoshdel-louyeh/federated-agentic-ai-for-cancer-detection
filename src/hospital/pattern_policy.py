from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .pattern_factory import ThinkingPatternFactory

CANCER_TYPES = ("BCC", "SCC", "MELANOMA", "AKIEC")


@dataclass
class StaticPatternPolicy:
    """Deterministic, configurable pattern policy used for initial hospital rollout."""

    hospital_id: str
    default_mapping: Mapping[str, str] = field(
        default_factory=lambda: {
            "BCC": "rule_based",
            "SCC": "bayesian",
            "MELANOMA": "deep_learning",
            "AKIEC": "rule_based_strict",
        }
    )
    hospital_overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def select_patterns(self) -> dict[str, str]:
        resolved = {key.upper(): value.strip().lower() for key, value in self.default_mapping.items()}

        override = self.hospital_overrides.get(self.hospital_id, {})
        for cancer_type, pattern_name in override.items():
            resolved[cancer_type.strip().upper()] = pattern_name.strip().lower()

        self._validate_mapping(resolved)
        return dict(resolved)

    @staticmethod
    def _validate_mapping(mapping: Mapping[str, str]) -> None:
        missing = [cancer_type for cancer_type in CANCER_TYPES if cancer_type not in mapping]
        extra = [cancer_type for cancer_type in mapping if cancer_type not in CANCER_TYPES]
        if missing or extra:
            raise ValueError(
                "Pattern policy must define exactly these cancer types: "
                f"{', '.join(CANCER_TYPES)}"
            )

        supported = set(ThinkingPatternFactory().supported_patterns())
        invalid = [name for name in mapping.values() if name not in supported]
        if invalid:
            raise ValueError(
                "Pattern policy includes unsupported pattern names: "
                f"{', '.join(sorted(set(invalid)))}"
            )
