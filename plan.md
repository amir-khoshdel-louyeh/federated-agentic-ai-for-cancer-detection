# Project Plan for Required Changes

## Objective
Implement the professor's requested changes by focusing on:
- a cancer-specific AI agent skill,
- server-side LLM reasoning and self-correcting loop,
- local update timing and federated model update flow.

---

## Goals and Code Paths

### 1. Develop a skill on a specific cancer behavior
The agent must correctly identify whether the target cancer type is present or not.

Path references:
- `src/client_side/agents/` — core agent classes and reasoning patterns
- `src/client_side/agents/base.py` — `LLMReasoner`, reasoning configuration, local LLM support
- `src/client_side/agents/ai_thinking_pattern.py` — `AIThinkingPattern`, the pattern used for LLM-based inference
- `src/client_side/agents/cancer_detection_agent.py` — cancer-specific agent wrapper and prediction logic
- `src/client_side/agents/melanoma_agent.py`, `akiec_agent.py`, `bcc_agent.py`, `scc_agent.py` — per-cancer agents that should specialize in their cancer type

Implementation notes:
- Ensure each cancer agent has a clear decision boundary for presence/absence of its cancer type.
- Treat each agent's "Skill" as a combination of Prompt + Logic + Tools, not just a prediction function.
- Use a Medical Knowledge Base as a tool to support binary decisions, so the agent reasons from evidence instead of guessing.
- Keep the output focused on: "cancer type found" vs "not found" plus confidence.
- Validate the agent skill using the existing structured reasoning and confidence scoring.

### 2. Use server-side LLM reasoning and self-correcting autonomous loop
The LLM reasoning should be executed at the server side and manage the loop that evaluates and corrects itself.

Path references:
- `src/server_side/federated_learning/prompt_evolution.py` — server-side prompt evolution and meta-reasoning
- `src/client_side/hospital/orchestrator.py` — federated round orchestration using server-side evolution
- `src/server_side/federated_learning/aggregators.py` — how local updates are aggregated into a global model
- `src/server_side/federated_learning/factory.py` — builds aggregator algorithms used in federated rounds
- `src/server_side/federated_learning/validators.py` — validates local updates before aggregation

Implementation notes:
- Use `prompt_evolution.py` as the LLM-managed self-correcting loop component.
- The server should act as a supervisor and perform Meta-Analysis: reading hospital reports, comparing local summaries, and explaining in natural language why the model failed or succeeded in different regions.
- The server should evaluate global performance and adjust prompts/system behavior across rounds.
- The loop should monitor local update quality and apply corrective changes via the meta-agent.

### 3. Understand when local update should be done and how to update the whole model
Define the local update trigger conditions and the federated global update flow.

Path references:
- `src/client_side/hospital/hospital_node.py` — hospital node behavior and local model state
- `src/client_side/hospital/hospital_manager_agent.py` — local manager agent generating updates
- `src/client_side/hospital/orchestrator.py` — federated orchestration and round control
- `src/client_side/hospital/contracts.py` — local update contract and payload structure
- `src/server_side/federated_learning/validators.py` — local update validation rules and timing
- `src/server_side/federated_learning/aggregators.py` — global model update logic after validation
- `src/simulator/controller.py` — execution driver for federated rounds and update cycles

Implementation notes:
- Local updates should be produced after a hospital evaluates its agents and computes local summaries.
- The server-side orchestrator should accept validated local updates, aggregate them, and return a global state.
- The global state should then be redistributed to hospitals for the next round.
- If using OpenClaw later, this same flow should map to OpenClaw hospital agents and server meta-controller.

---

## Recommended Next Steps
1. Review `src/client_side/agents/ai_thinking_pattern.py` and `src/client_side/agents/base.py` for current local LLM reasoning behavior.
2. Update the cancer-specific agent classes so they explicitly classify the target cancer type presence/absence.
3. Enhance `src/server_side/federated_learning/prompt_evolution.py` to manage the self-correcting loop and server-side prompt updates.
4. Verify the local update contract in `src/client_side/hospital/contracts.py` and the aggregator logic in `src/server_side/federated_learning/aggregators.py`.
5. Add any required configuration values in `configs/config.yaml` to support server-side prompt evolution and local update scheduling.

---

## Notes
- The project already supports local LLM inference via Ollama and OpenAI-compatible APIs.
- The current design appears to already separate local agent reasoning from server aggregation, which is the primary structure needed for this professor request.
- If the project must be ported to OpenClaw later, the file-level plan still applies: server-side prompt evolution becomes the OpenClaw meta-agent controller, and local hospital agents become OpenClaw hospital nodes.

---

## OpenClaw-Compatible Component Mapping

If the project transitions into OpenClaw, map the current architecture like this:

- OpenClaw hospital node = `src/client_side/hospital/hospital_node.py`
- OpenClaw hospital agent portfolio = `src/client_side/agents/` per-cancer agents
- OpenClaw agent reasoning skill = `src/client_side/agents/cancer_detection_agent.py` and `AIThinkingPattern`
- OpenClaw meta-controller = `src/server_side/federated_learning/prompt_evolution.py`
- OpenClaw federated aggregator = `src/server_side/federated_learning/aggregators.py`
- OpenClaw update validation = `src/server_side/federated_learning/validators.py`
- OpenClaw global update flow = `src/client_side/hospital/orchestrator.py` and `src/simulator/controller.py`

This keeps the same responsibilities while aligning with OpenClaw's agentic AI pattern.
