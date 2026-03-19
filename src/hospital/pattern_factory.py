from __future__ import annotations

from collections.abc import Callable

from src.agents import (
    BayesianThinkingPattern,
    DeepLearningThinkingPattern,
    RuleBasedStrictThinkingPattern,
    RuleBasedThinkingPattern,
    ThinkingPattern,
)


class ThinkingPatternFactory:
    """Centralized builder for all supported thinking-pattern implementations."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[], ThinkingPattern]] = {
            "rule_based": RuleBasedThinkingPattern,
            "rule_based_strict": RuleBasedStrictThinkingPattern,
            "bayesian": BayesianThinkingPattern,
            "deep_learning": lambda: DeepLearningThinkingPattern(epochs=20, batch_size=64, lr=1e-3),
        }

    def create(self, name: str) -> ThinkingPattern:
        key = name.strip().lower()
        try:
            return self._builders[key]()
        except KeyError as exc:
            supported = ", ".join(sorted(self._builders))
            raise ValueError(f"Unsupported thinking pattern: {name}. Supported: {supported}") from exc

    def supported_patterns(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))


def create_thinking_pattern(name: str) -> ThinkingPattern:
    """Convenience function for one-off pattern construction."""
    return ThinkingPatternFactory().create(name)
