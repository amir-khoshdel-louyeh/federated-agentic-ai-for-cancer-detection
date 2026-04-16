from __future__ import annotations

from __future__ import annotations

from .base import SkinCancerAgent, ThinkingPattern
from .tools import SearchTool, VisualAnalysisTool, Tool


class AKIECAgent(SkinCancerAgent):
    """Fixed-domain agent dedicated to actinic keratosis/intraepithelial carcinoma."""

    def __init__(
        self,
        thinking_pattern: ThinkingPattern | list[ThinkingPattern],
        tools: list[Tool] | None = None,
    ) -> None:
        default_tools = tools or [SearchTool(), VisualAnalysisTool()]
        super().__init__(thinking_patterns=thinking_pattern, tools=default_tools)

    @property
    def cancer_type(self) -> str:
        return "AKIEC"
