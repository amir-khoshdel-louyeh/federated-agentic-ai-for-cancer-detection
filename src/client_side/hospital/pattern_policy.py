from __future__ import annotations


from dataclasses import dataclass, field
from typing import Mapping, Optional

from .config_helpers import get_cancer_types
from .pattern_factory import ThinkingPatternFactory

ValidationScoreByPattern = Mapping[str, Mapping[str, float]]

def _cancer_types_from_mapping(mapping: Mapping[str, str] | None) -> tuple[str, ...]:
    if not mapping:
        return get_cancer_types(None)
    return tuple(sorted({key.strip().upper() for key in mapping.keys()}))


def _extract_default_mapping_from_config(config: Optional[dict]) -> dict:
    # Try to extract from config['agents']['patterns']['default_mapping']
    if config is not None:
        try:
            mapping = dict(config["agents"]["patterns"]["default_mapping"])
            if mapping:
                return mapping
        except Exception:
            pass

    cancer_types = get_cancer_types(config)
    # fallback to ai_agent for each configured cancer type
    return {ct: "ai_agent" for ct in cancer_types}

@dataclass
class StaticPatternPolicy:
    """Deterministic, configurable pattern policy used for initial hospital rollout."""

    hospital_id: str
    config: Optional[dict] = None
    default_mapping: Mapping[str, str] = field(init=False)
    hospital_overrides: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def __post_init__(self):
        self.cancer_types = get_cancer_types(self.config)
        self.default_mapping = _extract_default_mapping_from_config(self.config)

    def select_patterns(self) -> dict[str, str]:
        resolved = {key.upper(): value.strip().lower() for key, value in self.default_mapping.items()}

        override = self.hospital_overrides.get(self.hospital_id, {})
        for cancer_type, pattern_name in override.items():
            resolved[cancer_type.strip().upper()] = pattern_name.strip().lower()

        self._validate_mapping(resolved, expected_cancer_types=tuple(self.cancer_types))
        return dict(resolved)

    @staticmethod
    def _validate_mapping(mapping: Mapping[str, str], expected_cancer_types: tuple[str, ...] | None = None) -> None:
        if expected_cancer_types is None:
            expected_cancer_types = _cancer_types_from_mapping(mapping)

        missing = [cancer_type for cancer_type in expected_cancer_types if cancer_type not in mapping]
        extra = [cancer_type for cancer_type in mapping if cancer_type not in expected_cancer_types]
        if missing or extra:
            raise ValueError(
                "Pattern policy must define exactly these cancer types: "
                f"{', '.join(expected_cancer_types)}"
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

        dynamic_cancer_types = _cancer_types_from_mapping(resolved)
        for cancer_type in dynamic_cancer_types:
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
