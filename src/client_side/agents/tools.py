from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        raise NotImplementedError

    def function_schema(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": [k for k, v in self.parameters.items() if v.get("required", False)],
            },
        }

    @abstractmethod
    def execute(self, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError


class SearchTool(Tool):
    @property
    def name(self) -> str:
        return "search_tool"

    @property
    def description(self) -> str:
        return "Look up the latest clinical guidelines for a melanoma lesion type."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "lesion_type": {
                "type": "string",
                "description": "The lesion type or melanoma subtype to look up.",
                "required": True,
            },
            "query": {
                "type": "string",
                "description": "Optional search query to narrow down guidelines.",
                "required": False,
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        lesion_type = kwargs.get("lesion_type", "melanoma")
        query = kwargs.get("query")
        summary = (
            f"Retrieved latest clinical guideline summary for {lesion_type}. "
            f"Focus on staging, dermoscopic criteria, and follow-up recommendations."
        )
        if query:
            summary += f" Search query: {query}."
        return {
            "tool": self.name,
            "lesion_type": lesion_type,
            "query": query,
            "result": summary,
        }


class VisualAnalysisTool(Tool):
    @property
    def name(self) -> str:
        return "visual_analysis_tool"

    @property
    def description(self) -> str:
        return "Re-process a lesion image using an alternate reasoning model or configuration."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "lesion_type": {
                "type": "string",
                "description": "The lesion type being analyzed.",
                "required": True,
            },
            "image_id": {
                "type": "string",
                "description": "Identifier for the image to reprocess.",
                "required": False,
            },
            "library_model": {
                "type": "string",
                "description": "The name of the alternate reasoning model to use.",
                "required": False,
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        lesion_type = kwargs.get("lesion_type", "melanoma")
        image_id = kwargs.get("image_id")
        library_model = kwargs.get("library_model", "alternate_reasoning")
        summary = (
            f"Reprocessed the lesion with {library_model}. "
            f"The alternate reasoning configuration suggests a second opinion for {lesion_type}."
        )
        if image_id:
            summary += f" Image id: {image_id}."
        return {
            "tool": self.name,
            "lesion_type": lesion_type,
            "image_id": image_id,
            "library_model": library_model,
            "result": summary,
        }
