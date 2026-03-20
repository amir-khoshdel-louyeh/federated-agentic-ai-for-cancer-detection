"""Single-hospital multi-agent package for initial implementation."""

from .agent_portfolio import AgentPortfolio
from .artifacts import save_hospital_artifacts
from .contracts import (
	AdaptivePatternPolicyContract,
	HospitalDataBundle,
	HospitalLifecycleContract,
	HospitalScope,
)
from .data_pipeline import LocalDataPipeline, LocalHospitalData
from .hospital_node import HospitalNode
from .output_schema import SCHEMA_VERSION, build_hospital_output
from .pattern_factory import ThinkingPatternFactory, create_thinking_pattern
from .pattern_policy import AdaptivePatternPolicy, StaticPatternPolicy
from .simulation_runner import MultiHospitalSimulationResult, simulate_multi_hospital

__all__ = [
	"AgentPortfolio",
	"save_hospital_artifacts",
	"HospitalDataBundle",
	"HospitalScope",
	"HospitalLifecycleContract",
	"AdaptivePatternPolicyContract",
	"LocalDataPipeline",
	"LocalHospitalData",
	"HospitalNode",
	"SCHEMA_VERSION",
	"build_hospital_output",
	"ThinkingPatternFactory",
	"create_thinking_pattern",
	"StaticPatternPolicy",
	"AdaptivePatternPolicy",
	"MultiHospitalSimulationResult",
	"simulate_multi_hospital",
]
