# Federated Agentic AI for Cancer Detection

## Overview

A research-oriented Python project for cancer detection using distributed AI agents and local LLM inference.

The system combines:

- LLM-based reasoning over structured clinical metadata
- Hospital-local inference with Ollama or OpenAI-compatible APIs
- Privacy-preserving evaluation across distributed nodes
- Validation threshold calibration and balanced performance selection

## Requirements

- Python 3.11+ recommended
- Local or remote LLM inference capabilities
- `pip install -r requirements.txt`

## Installation

1. Activate the project virtual environment:
   ```bash
   source .venv/bin/activate
   ```
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Ollama and pull the local model if using local inference:
   ```bash
   ollama pull llama3.1:8b
   ```
4. Start Ollama to expose a local API:
   ```bash
   ollama serve
   ```
5. Verify the endpoint is reachable:
   ```bash
   curl http://localhost:11434/v1/models
   ```

## Local AI Support

This project supports local Ollama inference and OpenAI-compatible endpoints. Configure the active LLM in `configs/config.yaml`.

Example local configuration:

```yaml
meta_agent:
  local_llm:
    enabled: true
    base_url: http://localhost:11434/v1
    model_name: llama3.1:8b
    temperature: 0.25
```

`src/client_side/agents/base.py` selects between local and remote inference methods and parses structured JSON-like responses.

## Running the Project

Execute the main simulation:

```bash
python main.py
```

The default configuration is loaded from `configs/config.yaml`.

## Project Layout

- `main.py` — primary entrypoint
- `configs/config.yaml` — runtime settings and local LLM configuration
- `src/client_side/agents/` — LLM reasoning patterns and agent wrappers
- `src/client_side/hospital/` — hospital orchestration, evaluation, and threshold search
- `src/client_side/pre_processing/` — metadata normalization and feature preprocessing
- `src/server_side/federated_learning/` — federated aggregation and validation helpers
- `src/simulator/` — simulation controller and k-fold runner
- `tests/` — unit tests for calibration, prompt parsing, and evaluation logic

## Key Features

- AI agent probability prediction with calibrated temperature
- Prompt and CoT reasoning improvements for more stable inference
- JSON-safe output parsing with free-text fallback
- Balanced validation threshold search and confusion matrix evaluation
- Request delay support for smoother local LLM inference

## Configuration

Use `configs/config.yaml` to tune:

- local LLM settings and model endpoint
- reasoning prompt temperature and thresholds
- hospital splits and evaluation parameters
- validation/test split behavior

## Outputs

The project writes evaluation outputs to `outputs/`, including metrics and cached predictions.

## Notes

- The default workflow uses `AIThinkingPattern` for agent inference.
- `fit()` is effectively a no-op for the LLM-based AI agent pattern.
- Raw patient metadata is intended to remain on the local hospital node.

## Contact

Refer to the source code in `src/` and update `configs/config.yaml` to experiment with different AI reasoning and federated evaluation settings.
