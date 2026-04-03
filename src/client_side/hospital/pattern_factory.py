from __future__ import annotations

from collections.abc import Callable

from ..agents import (
    BayesianThinkingPattern,
    DeepLearningThinkingPattern,
    LogisticThinkingPattern,
    MCDropoutThinkingPattern,
    PretrainedLibraryThinkingPattern,
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

    def create(self, name: str, pattern_config: dict | None = None) -> ThinkingPattern:
        key = name.strip().lower()
        try:
            builder = self._builders[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._builders))
            raise ValueError(f"Unsupported thinking pattern: {name}. Supported: {supported}") from exc

        if key == "pretrained_library":
            return PretrainedLibraryThinkingPattern.from_config(pattern_config)

        if key == "deep_learning" and pattern_config:
            # allow override of training hyperparameters per pattern
            return DeepLearningThinkingPattern(
                epochs=int(pattern_config.get("epochs", 20)),
                batch_size=int(pattern_config.get("batch_size", 64)),
                lr=float(pattern_config.get("lr", 1e-3)),
            )

        if key == "rule_based":
            threshold = float(pattern_config.get("threshold", 0.58)) if pattern_config else 0.58
            return RuleBasedThinkingPattern(threshold=threshold)

        if key == "rule_based_strict":
            threshold = float(pattern_config.get("threshold", 0.68)) if pattern_config else 0.68
            return RuleBasedStrictThinkingPattern(threshold=threshold)

        if key == "rule_clinical":
            if pattern_config:
                age_threshold = int(pattern_config.get("age_threshold", 30))
                pediatric_penalty = float(pattern_config.get("pediatric_penalty", 0.6))
            else:
                age_threshold = 30
                pediatric_penalty = 0.6
            return RuleClinicalThinkingPattern(age_threshold=age_threshold, pediatric_penalty=pediatric_penalty)

        if key == "logistic":
            if pattern_config:
                return LogisticThinkingPattern(
                    C=float(pattern_config.get("C", 1.0)),
                    penalty=str(pattern_config.get("penalty", "l2")),
                    class_weight=pattern_config.get("class_weight", "balanced"),
                    max_iter=int(pattern_config.get("max_iter", 1000)),
                    random_state=int(pattern_config.get("random_state", 42)),
                )
            return LogisticThinkingPattern()

        # default behavior for simple constructors
        return builder()

    def supported_patterns(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))


def create_thinking_pattern(
    name: str,
    pattern_config: dict | None = None,
    deep_learning_epochs: int = 20,
    deep_learning_batch_size: int = 64,
    deep_learning_lr: float = 1e-3,
) -> ThinkingPattern:
    """Convenience function for one-off pattern construction."""
    factory = ThinkingPatternFactory(
        deep_learning_epochs=deep_learning_epochs,
        deep_learning_batch_size=deep_learning_batch_size,
        deep_learning_lr=deep_learning_lr,
    )
    return factory.create(name, pattern_config=pattern_config)
