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
        return "Look up the latest clinical guidelines for a lesion type."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "lesion_type": {
                "type": "string",
                "description": "The lesion type or cancer subtype to look up.",
                "required": True,
            },
            "query": {
                "type": "string",
                "description": "Optional search query to narrow down guidelines.",
                "required": False,
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        lesion_type = kwargs.get("lesion_type", "cancer")
        query = kwargs.get("query")
        summary = (
            f"Retrieved latest clinical guideline summary for {lesion_type}. "
            f"Focus on staging, diagnostic criteria, and supporting evidence."
        )
        if query:
            summary += f" Search query: {query}."
        return {
            "tool": self.name,
            "lesion_type": lesion_type,
            "query": query,
            "result": summary,
        }


class MedicalKnowledgeBaseTool(Tool):
    @property
    def name(self) -> str:
        return "medical_knowledge_base"

    @property
    def description(self) -> str:
        return "Consult a medical knowledge base for evidence and diagnostic criteria relevant to the cancer type."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "cancer_type": {
                "type": "string",
                "description": "The cancer type being evaluated.",
                "required": True,
            },
            "feature_summary": {
                "type": "string",
                "description": "A brief summary of the clinical features or evidence being considered.",
                "required": True,
            },
            "query": {
                "type": "string",
                "description": "Optional targeted query for additional domain evidence.",
                "required": False,
            },
        }

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        cancer_type = kwargs.get("cancer_type", "cancer")
        feature_summary = kwargs.get("feature_summary", "clinical features not provided")
        query = kwargs.get("query")
        evidence = (
            f"Medical knowledge base indicates that {cancer_type} diagnosis should be based on patterns such as asymmetry, border irregularity, color variation, and lesion history. "
            f"The current evidence summary is: {feature_summary}."
        )
        if query:
            evidence += f" Focused knowledge query: {query}."
        return {
            "tool": self.name,
            "cancer_type": cancer_type,
            "feature_summary": feature_summary,
            "query": query,
            "result": evidence,
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
