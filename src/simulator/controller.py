from configs.config_loader import load_config
from pathlib import Path
import logging
from src.client_side.hospital.agent_portfolio import AgentPortfolio
from src.client_side.hospital.hospital_node import HospitalNode
from src.client_side.hospital.data_pipeline import LocalDataPipeline
from src.client_side.hospital.orchestrator import FederatedRoundOrchestrator
from src.client_side.hospital.pattern_factory import create_thinking_pattern




def make_hospitals(config, data_pipeline):
    """Instantiate hospitals and their agent portfolios."""
    hospital_ids = [h.strip() for h in config["hospital_ids"].split(",")]
    agent_patterns = config["agents"]["patterns"]["default_mapping"]
    enabled_datasets = config.get("enabled_datasets", [])
    hospitals = {}
    for hid in hospital_ids:
        patterns = {ct: create_thinking_pattern(agent_patterns[ct]) for ct in agent_patterns}
        portfolio = AgentPortfolio(initial_patterns=patterns)
        # Each hospital gets its own pipeline with correct ids
        pipeline = LocalDataPipeline(
            hospital_id=hid,
            hospital_ids=hospital_ids,
            config=config
        )
        hospital_kwargs = dict(
            hospital_id=hid,
            data_pipeline=pipeline,
            agent_portfolio=portfolio,
            config=config
        )
        if "HAM10000" in enabled_datasets:
            hospital_kwargs["ham_metadata_csv"] = config["ham_csv"]
        if "ISIC2019" in enabled_datasets:
            hospital_kwargs["isic_labels_csv"] = config["isic_csv"]
        hospitals[hid] = HospitalNode(**hospital_kwargs)
    return hospitals

def initialize_system(config):
    """Initialize config, output dirs, logging, data pipeline, and hospitals."""
    ensure_output_dirs(config)
    logging.basicConfig(
        filename=Path(config.get("tracking", {}).get("log_dir", "outputs/logs")) / config.get("tracking", {}).get("log_file_name", "simulation.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    data_pipeline = LocalDataPipeline()
    hospitals = make_hospitals(config, data_pipeline)
    allocate_data(hospitals)
    logging.info("System initialized.")
    return hospitals

def train_system(config, hospitals, save_history=False, save_models=False):
    """Run federated training rounds.

    Args:
        config (dict): configuration object.
        hospitals (dict): hospital_id -> HospitalNode.
        save_history (bool): if True, writes per-round metrics and decision files.
        save_models (bool): if True, saves local models at end.

    Returns:
        FederatedRoundOrchestrator
    """
    import json
    from pathlib import Path
    aggregation_name = config["federation"]["aggregation_algorithm"]
    num_hospitals = len(hospitals)
    if num_hospitals == 1:
        if aggregation_name != "no_operation":
            logging.warning("Single-hospital mode detected. Overriding aggregation algorithm to 'no_operation'.")
        aggregation_name = "no_operation"
    elif aggregation_name == "no_operation":
        raise ValueError("'no_operation' aggregation can only be used with a single hospital.")
    orchestrator = FederatedRoundOrchestrator.from_algorithm(name=aggregation_name)
    num_rounds = config["simulation"]["num_rounds"]

    # Optionally prepare output directory and log paths
    out_dir = Path(config.get("output", {}).get("history_dir", "outputs/history"))
    hospital_logs = {hid: [] for hid in hospitals}
    log_paths = {hid: out_dir / f"{hid}_metrics.json" for hid in hospitals}
    federated_decision_log = []
    federated_decision_path = out_dir / "federated_decision.json"
    if save_history:
        out_dir.mkdir(parents=True, exist_ok=True)

    for round_idx in range(1, num_rounds + 1):
        for hospital in hospitals.values():
            hospital.train()
        local_updates = {hid: h.get_local_update() for hid, h in hospitals.items()}
        round_output = orchestrator.run_round(
            round_index=round_idx,
            local_updates=local_updates,
        )
        orchestrator.broadcast_global_state(hospitals, round_output.global_state)
        logging.info(f"Completed round {round_idx}")

        # Collect and append metrics/state for each hospital in-memory
        for hid, hospital in hospitals.items():
            entry = {
                "round": round_idx,
                "metrics": getattr(hospital, "metrics_store", {}),
                "local_update": local_updates.get(hid, {}),
                "global_state": getattr(orchestrator, "global_state", {}),
            }
            hospital_logs[hid].append(entry)
            if save_history:
                with open(log_paths[hid], "w") as f:
                    json.dump(hospital_logs[hid], f, indent=2)

        # Collect and optionally write federated decision for this round
        federated_decision_log.append({
            "round": round_idx,
            "aggregator_name": round_output.aggregator_name,
            "global_state": round_output.global_state,
            "aggregation": {
                "algorithm": round_output.aggregation.algorithm,
                "global_metrics": round_output.aggregation.global_metrics,
                "hospital_weights": round_output.aggregation.hospital_weights,
                "included_hospital_ids": round_output.aggregation.included_hospital_ids,
                "dropped_hospitals": round_output.aggregation.dropped_hospitals,
                "details": round_output.aggregation.details,
            },
            "validation_report": round_output.validation_report,
        })
        if save_history:
            with open(federated_decision_path, "w") as f:
                json.dump(federated_decision_log, f, indent=2)

    # After training, optionally save all models for each hospital to outputs/system
    if save_models:
        system_dir = Path("outputs/system")
        system_dir.mkdir(parents=True, exist_ok=True)
        for hid, hospital in hospitals.items():
            if hasattr(hospital, "scope") and hasattr(hospital.scope, "agent_portfolio"):
                portfolio = hospital.scope.agent_portfolio
                if hasattr(portfolio, "save_all_models"):
                    portfolio.save_all_models(str(system_dir), hid)
    return orchestrator

def test_system(hospitals):
    """Evaluate all hospitals' agents on their test data."""
    results = {}
    import traceback
    for hid, hospital in hospitals.items():
        try:
            results[hid] = hospital.evaluate()
            logging.info(f"Tested hospital {hid}")
        except Exception as e:
            logging.error(f"Testing failed for hospital {hid}: {e}\nTraceback:\n{traceback.format_exc()}")
            results[hid] = None
    return results

def show_results(results):
    """Display or save evaluation results."""
    print("\n===== Evaluation Results =====")
    for hid, res in results.items():
        print(f"Hospital: {hid}")
        if res is None:
            print("  Evaluation failed.")
        else:
            for k, v in res.items():
                print(f"  {k}: {v}")
    print("============================\n")

def show_log_location(config):
    log_dir = Path(config.get("tracking", {}).get("log_dir", "outputs/logs"))
    log_file = config.get("tracking", {}).get("log_file_name", "simulation.log")
    print(f"Logs saved at: {log_dir / log_file}")

def allocate_data(hospitals):
    """Load data and initialize each hospital."""
    for hospital in hospitals.values():
        hospital.initialize()

def run_federated_learning(config, hospitals):
    """Run the federated learning rounds."""
    aggregation_name = config["federation"]["aggregation_algorithm"]
    orchestrator = FederatedRoundOrchestrator.from_algorithm(name=aggregation_name)
    num_rounds = config["simulation"]["num_rounds"]
    for round_idx in range(1, num_rounds + 1):
        for hospital in hospitals.values():
            hospital.train()
        local_updates = {hid: h.get_local_update() for hid, h in hospitals.items()}
        round_output = orchestrator.run_round(
            round_index=round_idx,
            local_updates=local_updates,
        )
        orchestrator.broadcast_global_state(hospitals, round_output.global_state)
        logging.info(f"Completed round {round_idx}")

def ensure_output_dirs(config):
    """Ensure output and log directories exist."""
    out_dir = Path(config.get("out_dir", "outputs"))
    log_dir = Path(config.get("tracking", {}).get("log_dir", out_dir / "logs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)


# --- New function for GUI: run a single federated training round ---
def run_one_training_round(orchestrator, hospitals, round_idx, for_training=True):
    """
    Run a single federated training round and return round output and agent metrics for GUI.
    Args:
        orchestrator: FederatedRoundOrchestrator instance
        hospitals: dict of hospital_id -> HospitalNode
        round_idx: int, current round index (1-based)
        for_training: bool, whether to get local updates for training
    Returns:
        round_output: output from orchestrator.run_round
        agent_metrics: dict of hospital_id -> agent metrics (if available)
    """
    # Train all hospitals for this round
    for hospital in hospitals.values():
        hospital.train()
    # Collect local updates
    local_updates = {hid: h.get_local_update(for_training=for_training) for hid, h in hospitals.items()}
    # Run federated round
    round_output = orchestrator.run_round(
        round_index=round_idx,
        local_updates=local_updates,
    )
    orchestrator.broadcast_global_state(hospitals, round_output.global_state)
    # Optionally, collect agent metrics for GUI display
    agent_metrics = {}
    for hid, hospital in hospitals.items():
        if hasattr(hospital, "scope") and hasattr(hospital.scope, "agent_portfolio"):
            metrics = {}
            for cancer_type in hospital.scope.agent_portfolio.cancer_types:
                agent = hospital.scope.agent_portfolio.get_agent(cancer_type)
                metrics[cancer_type] = getattr(agent, "thinking_pattern_name", None)
            agent_metrics[hid] = metrics
    return round_output, agent_metrics


def validation_system(hospitals, output_dir=None, early_stop_threshold=None, save_to_disk=False):
    import json
    from pathlib import Path

    if output_dir is None:
        output_dir = Path("outputs/history")
    else:
        output_dir = Path(output_dir)

    if save_to_disk:
        output_dir.mkdir(parents=True, exist_ok=True)

    validation_reports = {}
    for hid, hospital in hospitals.items():
        if hospital.local_data is None:
            raise RuntimeError(f"Hospital {hid} must be initialized before validation_system().")
        if hospital.scope is None or hospital.scope.agent_portfolio is None:
            raise RuntimeError(f"Hospital {hid} must have an agent portfolio before validation_system().")

        hospital_validation = {}
        for cancer_type in hospital.scope.agent_portfolio.cancer_types:
            agent = hospital.scope.agent_portfolio.get_agent(cancer_type)
            x_val, y_val = hospital.get_cancer_filtered_split(cancer_type=cancer_type, split="val")
            if x_val is None or len(x_val) == 0:
                hospital_validation[cancer_type] = {
                    "agent": agent.name,
                    "metrics": {},
                    "warning": "No validation samples for this cancer type.",
                }
                continue

            val_probs = agent.predict_proba(x_val)
            metrics = hospital._compute_binary_metrics(y_val, val_probs)
            hospital_validation[cancer_type] = {
                "agent": agent.name,
                "metrics": metrics,
            }

        hospital.metrics_store.setdefault("validation", {})["results"] = hospital_validation
        hospital.metrics_store["lifecycle_state"] = "validated"

        validation_reports[hid] = hospital_validation

        if save_to_disk:
            with open(output_dir / f"{hid}_validation_metrics.json", "w") as f:
                json.dump({"hospital_id": hid, "validation": hospital_validation}, f, indent=2)

            logging.info(f"Saved validation metrics for hospital {hid} to {output_dir / f'{hid}_validation_metrics.json'}")
        else:
            logging.info(f"Validation metrics computed for hospital {hid}; disk save is disabled.")

    should_stop = False
    if early_stop_threshold is not None:
        # simple early stopping rule: stop if average validation f1 drops below threshold.
        all_f1_scores = []
        for hid, hospital_validation in validation_reports.items():
            for cancer_cfg in hospital_validation.values():
                score = cancer_cfg.get("metrics", {}).get("f1")
                if score is not None:
                    all_f1_scores.append(score)

        if len(all_f1_scores) > 0:
            avg_f1 = float(sum(all_f1_scores) / len(all_f1_scores))
            should_stop = avg_f1 < float(early_stop_threshold)
            logging.info(f"Validation avg_f1={avg_f1:.4f}, early_stop_threshold={early_stop_threshold}, should_stop={should_stop}")

    if early_stop_threshold is not None:
        return validation_reports, should_stop
    return validation_reports


# For GUI integration: call these functions in the desired order.
# Example usage in GUI:
#   config = load_config()
#   hospitals = initialize_system(config)
#   orchestrator = train_system(config, hospitals)
#   results = test_system(hospitals)
#   show_results(results)
#   show_log_location(config)
