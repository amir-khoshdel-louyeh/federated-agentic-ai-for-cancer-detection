from __future__ import annotations

from .base import SkinCancerAgent, ThinkingPattern
from .tools import SearchTool, VisualAnalysisTool, Tool


class SCCAgent(SkinCancerAgent):
    """Fixed-domain agent dedicated to squamous cell carcinoma detection."""

    def __init__(
        self,
        thinking_pattern: ThinkingPattern | list[ThinkingPattern],
        tools: list[Tool] | None = None,
    ) -> None:
        default_tools = tools or [SearchTool(), VisualAnalysisTool()]
        super().__init__(thinking_patterns=thinking_pattern, tools=default_tools)

    @property
    def cancer_type(self) -> str:
        return "SCC"
