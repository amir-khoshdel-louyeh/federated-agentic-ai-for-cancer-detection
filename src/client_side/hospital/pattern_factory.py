from __future__ import annotations

from collections.abc import Callable

from ..agents import (
    BayesianThinkingPattern,
    DeepLearningThinkingPattern,
    LogisticThinkingPattern,
    MCDropoutThinkingPattern,
    RuleBasedStrictThinkingPattern,
    RuleBasedThinkingPattern,
    RuleClinicalThinkingPattern,
    ThinkingPattern,
)


class ThinkingPatternFactory:
    """Centralized builder for all supported thinking-pattern implementations."""

    def __init__(self, deep_learning_epochs: int = 20, deep_learning_batch_size: int = 64, deep_learning_lr: float = 1e-3) -> None:
        self._builders: dict[str, Callable[[], ThinkingPattern]] = {
            "rule_based": RuleBasedThinkingPattern,
            "rule_based_strict": RuleBasedStrictThinkingPattern,
            "rule_clinical": RuleClinicalThinkingPattern,
            "bayesian": BayesianThinkingPattern,
            "deep_learning": lambda: DeepLearningThinkingPattern(epochs=deep_learning_epochs, batch_size=deep_learning_batch_size, lr=deep_learning_lr),
            "deep_learning_mc": lambda: MCDropoutThinkingPattern(DeepLearningThinkingPattern(epochs=deep_learning_epochs, batch_size=deep_learning_batch_size, lr=deep_learning_lr)),
            "pretrained_library": PretrainedLibraryThinkingPattern,
            "logistic": LogisticThinkingPattern,
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


def create_thinking_pattern(name: str, deep_learning_epochs: int = 20, deep_learning_batch_size: int = 64, deep_learning_lr: float = 1e-3) -> ThinkingPattern:
    """Convenience function for one-off pattern construction."""
    return ThinkingPatternFactory(
        deep_learning_epochs=deep_learning_epochs,
        deep_learning_batch_size=deep_learning_batch_size,
        deep_learning_lr=deep_learning_lr,
    ).create(name)
