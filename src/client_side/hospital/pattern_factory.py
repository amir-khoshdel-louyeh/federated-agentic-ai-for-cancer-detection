from __future__ import annotations

from collections.abc import Callable

from ..agents import (
    AIThinkingPattern,
    ThinkingPattern,
)


class ThinkingPatternFactory:
    """Centralized builder for all supported thinking-pattern implementations."""

    def __init__(self) -> None:
        self._builders: dict[str, Callable[[], ThinkingPattern]] = {
            "ai_agent": AIThinkingPattern,
        }

    def create(self, name: str, pattern_config: dict | None = None) -> ThinkingPattern:
        key = name.strip().lower()
        try:
            builder = self._builders[key]
        except KeyError as exc:
            supported = ", ".join(sorted(self._builders))
            raise ValueError(f"Unsupported thinking pattern: {name}. Supported: {supported}") from exc

        if pattern_config is None:
            return builder()
        if isinstance(pattern_config, dict):
            return builder(**pattern_config)
        return builder()

    def supported_patterns(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders.keys()))


def create_thinking_pattern(
    name: str,
    pattern_config: dict | None = None,
) -> ThinkingPattern:
    """Convenience function for one-off pattern construction."""
    factory = ThinkingPatternFactory()
    return factory.create(name, pattern_config=pattern_config)
