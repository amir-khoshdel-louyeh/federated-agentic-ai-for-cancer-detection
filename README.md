# Hybrid Agentic AI with Federated Learning for Collaborative Cancer Detection

## Overview

This project proposes a next-generation AI system for cancer detection that combines:

- Agentic AI (multi-agent systems)
- Federated Learning (FL)
- Privacy-preserving techniques

The goal is to enable multiple hospitals to collaboratively train high-quality cancer detection models without sharing sensitive patient data.

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

We design a **Hybrid Agentic AI system with Federated Learning**, where:

- Each hospital trains local AI agents
- Only model updates (not raw data) are shared
- A global model is built collaboratively
- A Meta-Agent Controller dynamically selects the best-performing agents

## System Architecture

### 1. Local Agentic AI (Per Hospital)

Each hospital runs multiple intelligent agents:

- **Rule-Based Agent**: deterministic medical logic
- **Bayesian Agent**: probabilistic reasoning
- **Deep Learning Agent**: CNN/Transformer for medical imaging

Each agent outputs:

- Prediction
- Confidence score

### 2. Federated Learning Layer

- Local models are trained on private hospital data
- Only weights/gradients are shared
- A central aggregator builds the global model

### 3. Meta-Agent Controller

- Monitors agent performance
- Assigns dynamic weights to agents
- Selects optimal strategies per hospital

### 4. Global Model

- Combines knowledge from all hospitals
- Improves generalization across distributed datasets
- Continuously updated through FL rounds

## Workflow

1. Patient data remains inside each hospital.
2. Local agents train and make predictions.
3. Model updates are sent to the federated aggregator.
4. The global model is updated.
5. The improved global model is redistributed to hospitals.

## Aggregation Algorithms

- **FedAvg**: standard weighted averaging
- **FedProx**: handles heterogeneous client data
- **Adaptive Aggregation**: uses agent confidence/performance
- **Secure Aggregation**: protects model updates during aggregation

## Security and Privacy

- **Differential Privacy (DP)**: adds noise to updates
- **Secure Aggregation**: helps prevent reconstruction of local data
- **No raw data sharing** across institutions

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
- **Agents**: Rule-Based, Bayesian, Deep Learning
- **Training**: local epochs + federated rounds
- **Evaluation**: local and global validation datasets

## Implementation Roadmap

| Phase | Description |
| --- | --- |
| Phase 1 | Single hospital, multi-agent system |
| Phase 2 | Multi-hospital FL with FedAvg |
| Phase 3 | Adaptive aggregation + Meta-Agent |
| Phase 4 | Add privacy (DP + Secure Aggregation) |
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
- PyTorch / TensorFlow
- Flower / PySyft (Federated Learning)
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


### Run (CLI Mode)

You can now use a command-line interface to run the system using your config.yaml:

```bash
python main.py cli --config configs/config.yaml
```

## Configuration Reference (New)

The pipeline now supports a centralized, configurable setup from `configs/config.yaml`:

- `cancer_types`: dynamic set of types in use (default: BCC, SCC, MELANOMA, AKIEC)
- `data_split`: includes `holdout_test`, `k_folds`, `current_fold`, plus optional:
  - `malignant_ham` (HAM10000 malignant class labels, lowercase)
  - `malignant_isic` (ISIC malignant labels, uppercase)
- `agents.types`: active agent subtype list
- `agents.patterns.default_mapping`: maps each cancer type to a thinking pattern (e.g. `BCC: pretrained_library`)
- `agents.patterns.pattern_params`: per-pattern hyperparameters, including:
  - `rule_based`: `threshold`, `weights`, `scale`
  - `rule_based_strict`: `threshold`
  - `rule_clinical`: `age_threshold`, `pediatric_penalty`, `weights`, `scale`
  - `logistic`: `C`, `penalty`, `class_weight`, `max_iter`, `random_state`
  - `pretrained_library`: `max_iter`, `learning_rate`, `max_depth`, `class_weight`, `random_state`
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
## Pre-trained and high-quality agents

This repository now includes `LogisticThinkingPattern` in `src/client_side/agents/logistic_pattern.py`.
- Configure via `configs/config.yaml` under `agents.patterns.default_mapping` (e.g., `BCC: logistic`).
- Save a trained model via `AgentPortfolio.save_all_models()`.
- Load a pre-trained model via `ThinkingPattern.load_model()` (used from `LogisticThinkingPattern`).

To use pre-trained weights in simulations:
1. Train one hospital once with data and persist the model path.
2. Set `pretrained_path` in your agent factory or pattern setup (future extension).
3. Use `logistic` as a thinking pattern for stronger baseline performance.