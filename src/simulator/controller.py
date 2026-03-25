import yaml
import logging
import sys
from pathlib import Path
from src.client_side.hospital.agent_portfolio import AgentPortfolio
from src.client_side.hospital.hospital_node import HospitalNode
from src.client_side.hospital.data_pipeline import LocalDataPipeline
from src.client_side.hospital.orchestrator import FederatedRoundOrchestrator
from src.client_side.hospital.pattern_factory import create_thinking_pattern

def load_config(config_path="configs/config.yaml"):
    """Load experiment configuration from YAML file."""
    try:
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        import traceback
        logging.error(f"Failed to load config: {e}\nTraceback:\n{traceback.format_exc()}")
        sys.exit(1)


def make_hospitals(config, data_pipeline):
    """Instantiate hospitals and their agent portfolios."""
    hospital_ids = [h.strip() for h in config["hospital_ids"].split(",")]
    agent_patterns = config["agents"]["patterns"]["default_mapping"]
    enabled_datasets = config.get("enabled_datasets", [])
    hospitals = {}
    for hid in hospital_ids:
        patterns = {ct: create_thinking_pattern(agent_patterns[ct]) for ct in agent_patterns}
        portfolio = AgentPortfolio(initial_patterns=patterns)
        hospital_kwargs = dict(
            hospital_id=hid,
            data_pipeline=data_pipeline,
            agent_portfolio=portfolio,
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

def train_system(config, hospitals):
    """Run federated training rounds."""
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



# For GUI integration: call these functions in the desired order.
# Example usage in GUI:
#   config = load_config()
#   hospitals = initialize_system(config)
#   orchestrator = train_system(config, hospitals)
#   results = test_system(hospitals)
#   show_results(results)
#   show_log_location(config)
