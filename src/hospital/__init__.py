"""Single-hospital multi-agent package for initial implementation."""

from .agent_portfolio import AgentPortfolio
from .contracts import (
	AdaptivePatternPolicyContract,
	HospitalDataBundle,
	HospitalLifecycleContract,
	HospitalScope,
)
from .hospital_node import HospitalNode
from .pattern_factory import ThinkingPatternFactory, create_thinking_pattern
from .pattern_policy import AdaptivePatternPolicy, StaticPatternPolicy

__all__ = [
	"AgentPortfolio",
	"HospitalDataBundle",
	"HospitalScope",
	"HospitalLifecycleContract",
	"AdaptivePatternPolicyContract",
	"HospitalNode",
	"ThinkingPatternFactory",
	"create_thinking_pattern",
	"StaticPatternPolicy",
	"AdaptivePatternPolicy",
]
