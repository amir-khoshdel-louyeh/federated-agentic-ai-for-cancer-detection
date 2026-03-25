from __future__ import annotations

from .base_pattern import SkinCancerAgent, ThinkingPattern


class AKIECAgent(SkinCancerAgent):
    """Fixed-domain agent dedicated to actinic keratosis/intraepithelial carcinoma."""

    def __init__(self, thinking_pattern: ThinkingPattern) -> None:
        super().__init__(thinking_pattern=thinking_pattern)

    @property
    def cancer_type(self) -> str:
        return "AKIEC"
