from __future__ import annotations

from .base_pattern import SkinCancerAgent, ThinkingPattern


class SCCAgent(SkinCancerAgent):
    """Fixed-domain agent dedicated to squamous cell carcinoma detection."""

    def __init__(self, thinking_pattern: ThinkingPattern) -> None:
        super().__init__(thinking_pattern=thinking_pattern)

    @property
    def cancer_type(self) -> str:
        return "SCC"
