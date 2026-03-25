from __future__ import annotations

from .base import SkinCancerAgent, ThinkingPattern


class BCCAgent(SkinCancerAgent):
    """Fixed-domain agent dedicated to basal cell carcinoma detection."""

    def __init__(self, thinking_pattern: ThinkingPattern) -> None:
        super().__init__(thinking_pattern=thinking_pattern)

    @property
    def cancer_type(self) -> str:
        return "BCC"
