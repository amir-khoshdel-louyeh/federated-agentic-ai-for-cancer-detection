from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from .pattern_factory import ThinkingPatternFactory

CANCER_TYPES = ("BCC", "SCC", "MELANOMA", "AKIEC")
ValidationScoreByPattern = Mapping[str, Mapping[str, float]]


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


@dataclass
class AdaptivePatternPolicy(StaticPatternPolicy):
    """Adaptive hook that can switch patterns from validation leaderboard scores."""

    min_improvement: float = 0.0

    def adapt_patterns(
        self,
        current_mapping: Mapping[str, str],
        validation_scores: ValidationScoreByPattern,
    ) -> dict[str, str]:
        # Keep adaptation strictly in policy layer so orchestration stays simple.
        resolved = {key.strip().upper(): value.strip().lower() for key, value in current_mapping.items()}
        self._validate_mapping(resolved)

        for cancer_type in CANCER_TYPES:
            candidates = validation_scores.get(cancer_type, {})
            if not candidates:
                continue

            normalized_candidates = {
                pattern_name.strip().lower(): float(score)
                for pattern_name, score in candidates.items()
            }

            supported = set(ThinkingPatternFactory().supported_patterns())
            normalized_candidates = {
                pattern_name: score
                for pattern_name, score in normalized_candidates.items()
                if pattern_name in supported
            }
            if not normalized_candidates:
                continue

            current_pattern = resolved[cancer_type]
            current_score = normalized_candidates.get(current_pattern, float("-inf"))

            best_pattern = max(normalized_candidates, key=normalized_candidates.get)
            best_score = normalized_candidates[best_pattern]

            if best_score > current_score + self.min_improvement:
                resolved[cancer_type] = best_pattern

        self._validate_mapping(resolved)
        return resolved
