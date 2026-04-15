# Hybrid Agentic AI for Collaborative Cancer Detection

## Overview

This project proposes a next-generation AI-agent system for cancer detection that combines:

- Agentic AI (multi-agent systems)
- LLM reasoning over structured clinical metadata
- Privacy-preserving techniques

The goal is to enable multiple hospitals to compare AI-agent evaluations without sharing sensitive patient data.

## Installation

1. Activate the project virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   pip install openai
   ```
3. Install and run Ollama for local LLM inference:
   - Install Ollama from the official source
   - Pull the model:
     ```bash
     ollama pull llama3.1:8b
     ```
   - Start the Ollama server:
     ```bash
     ollama serve
     ```
4. Confirm the local model is available:
   ```bash
   curl http://localhost:11434/v1/models
   ```

> The system now uses local Ollama access for Llama 3.1 8B and does not require an external OpenAI API key.

## Problem Statement

Cancer diagnosis requires:

- High accuracy
- Large and diverse datasets
- Strong privacy guarantees

However, real-world medical AI faces several constraints:

- Medical data is distributed across hospitals
- Data sharing is restricted due to privacy regulations (GDPR, HIPAA)
- Single-site models often overfit to local datasets

## Proposed Solution

We design an **AI-agent-first system** where each hospital uses a shared LLM reasoning pattern to make lesion-level predictions locally.

- Each hospital runs an `ai_agent` that reasons over clinical metadata
- Raw data stays on-site
- A central policy layer aggregates evaluation metrics and reweights specialists
- A Meta-Agent Controller selects the best reasoning strategy per hospital

## System Architecture

### 1. Local Agentic AI (Per Hospital)

Each hospital runs a lightweight LLM-based AI agent using local Ollama or OpenAI inference.

This is an LLM reasoning-based workflow rather than a classical machine learning training pipeline; `ai_agent` uses `predict_proba()` to query the LLM and `fit()` is a no-op.

- **AI Agent**: uses the configured LLM to reason over normalized clinical features and emit a probability plus reasoning.

Each agent outputs:

- Prediction probability
- Confidence / uncertainty estimate
- Structured clinical reasoning

### 2. Performance Aggregation Layer

- Hospitals share only evaluation metrics, not raw patient data
- A central aggregator uses these metrics to compare agent performance
- The Meta-Agent Controller assigns weights and selects lead strategies

### 3. Meta-Agent Controller

- Monitors agent performance
- Assigns dynamic weights to agents
- Selects optimal strategies per hospital

## Workflow

1. Patient metadata stays inside each hospital.
2. Each `ai_agent` reasons locally and outputs probability and rationale.
3. Hospitals share evaluation summaries, not raw data.
4. The global policy layer updates agent weighting and selection.
5. The selected agents are used for downstream evaluation.

## Aggregation Algorithms

- **No-operation**: single-hospital evaluation only
- **Adaptive Aggregation**: uses agent confidence/performance to weight hospital evaluations

## Security and Privacy

- **No raw data sharing** across institutions
- **Privacy-preserving evaluation reporting** for distributed AI agents

Compliance targets:

- GDPR
- HIPAA

## Evaluation Metrics

- Accuracy
- Sensitivity / Specificity
- F1-Score
- AUC-ROC
- Convergence Rate
- Agent Contribution Score

## Experimental Setup

- **Hospitals**: 2-5 distributed nodes
- **Agents**: LLM-based `ai_agent` using local Ollama inference
- **Workflow**: local evaluation and adaptive aggregation; `ai_agent` is not trained with classical model weights
- **Evaluation**: local and global validation datasets

## Implementation Roadmap

| Phase | Description |
| --- | --- |
| Phase 1 | Single hospital, multi-agent AI-agent evaluation |
| Phase 2 | Multi-hospital evaluation with adaptive aggregation |
| Phase 3 | Meta-Agent selects best-performing agents |
| Phase 4 | Add privacy-preserving evaluation reporting |
| Phase 5 | Scale system across hospitals |

## Expected Results

- Higher accuracy than single-hospital models
- Reduced overfitting
- Strong privacy guarantees
- Scalable architecture across institutions

## Challenges

- Data heterogeneity across hospitals
- Agent disagreement/conflicts
- High computational cost
- Communication overhead in FL

## Future Work

- Multi-modal data (imaging + genomics + clinical)
- Adaptive agent evolution
- Extension to other diseases
- Edge deployment for real-time predictions

## Suggested Tech Stack

- Python
- `openai` Python client (for local Ollama access)
- Ollama + Llama 3.1 8B
- NumPy / Pandas
- Scikit-learn

## Supervision

- **Professor**: Giovanni Finocchio
- **Project Lead**: Amir Khoshdel Louyeh

## License

This project is for academic and research purposes and distributed under the **MIT License**.

## Contribution

Contributions, ideas, and improvements are welcome.

Feel free to fork the project and experiment with new agent strategies, privacy mechanisms, or aggregation methods.

## Single Hospital Multi-Agent Implementation

The initial implementation is intentionally simple and local-only (no federated server yet).

### Project Paths

- `src/hospital/hospital_env.py`: virtual hospital loader for HAM10000 + ISIC 2019 metadata
- `src/hospital/agents/rule_based_agent.py`: deterministic ABCD-inspired rule engine
- `src/hospital/agents/bayesian_agent.py`: probabilistic agent (Gaussian Naive Bayes)
- `src/hospital/agents/deep_learning_agent.py`: compact PyTorch MLP agent
- `src/hospital/meta_controller.py`: adaptive weighted ensemble controller
- `src/hospital/main.py`: end-to-end train/evaluate runner
- `configs/hospital_local.yaml`: editable local paths and parameters

### Dataset Inputs

Expected files:

- HAM10000 metadata CSV (for example `HAM10000_metadata.csv`)
- ISIC 2019 ground truth CSV (for example `ISIC_2019_Training_GroundTruth.csv`)


### Run

Run the main simulation using the project configuration:

```bash
python3 main.py
```

The system loads `configs/config.yaml` by default and uses the local Ollama LLM setup.

## Configuration Reference (New)

The pipeline now supports a centralized, configurable setup from `configs/config.yaml`, including local Ollama LLM access:

- `cancer_types`: dynamic set of types in use (default: BCC, SCC, MELANOMA, AKIEC)
- `data_split`: includes `holdout_test`, `k_folds`, `current_fold`, plus optional:
  - `malignant_ham` (HAM10000 malignant class labels, lowercase)
  - `malignant_isic` (ISIC malignant labels, uppercase)
- `agents.types`: active agent subtype list
- `agents.patterns.available`: list of enabled reasoning patterns. In the current workflow this is `ai_agent` only.
- `agents.patterns.default_mapping`: maps each cancer type to a thinking pattern; in the AI-agent workflow this should be `ai_agent` for every enabled cancer type.
- `agents.patterns.pattern_params`: per-pattern hyperparameters, including:
  - `ai_agent`: local LLM prompt parameters, if needed

- `meta_agent.local_llm`: local LLM configuration for Ollama
  - `base_url`: local Ollama endpoint (e.g. `http://localhost:11434/v1`)
  - `model_name`: local Ollama model name (e.g. `llama3.1:8b`)
  - `temperature`: sampling temperature for the LLM
- `federation`: includes `aggregation_algorithm` and
  - `fedprox.mu`
  - `adaptive.alpha`, `beta`, `gamma`, `auc_weight`, `f1_weight`, `lifecycle_penalty`, `warning_penalty_per_item`, `min_reliability_score`

These settings drive behavior in:

- `src/client_side/hospital/agent_portfolio.py` (dynamic cancer type portfolio)
- `src/client_side/hospital/hospital_node.py` (policy and thresholds)
- `src/client_side/hospital/hospital_env.py` (data labels and splitting)
- `src/client_side/hospital/pattern_policy.py` (pattern mapping and fallback)
- `src/client_side/hospital/pattern_factory.py` (pattern construction from config)
- `src/server_side/federated_learning/aggregators.py` (federation weights and penalty configuration)
- `src/simulator/cli.py` (library-mode mapping uses configured cancer type set)

This will load all parameters from `configs/config.yaml` and run the CLI interface defined in `src/simulator/cli.py`.

You can still run the original hospital main for local-only experiments:

```bash
python -m src.hospital.main \
  --ham-csv data/raw/ham10000/HAM10000_metadata.csv \
  --isic-csv data/raw/isic2019/ISIC_2019_Training_GroundTruth.csv \
  --out-dir outputs
```

### Outputs

- `outputs/metrics.json`: per-agent and ensemble metrics (accuracy, f1, auc, weights)
- `outputs/predictions.csv`: test predictions and probabilities per agent
## AI-agent default workflow

This repository now defaults to `ai_agent` for all cancer types.
- Configure via `configs/config.yaml` under `agents.patterns.available` and `agents.patterns.default_mapping`.
- `ai_agent` relies on `AIThinkingPattern.predict_proba()` and local/OpenAI reasoning.
- `fit()` remains a no-op for `AIThinkingPattern`; no classical model weights are required.

Legacy `pretrained_library` support is not part of the default AI-agent workflow.