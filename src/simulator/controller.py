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
        logging.error(f"Failed to load config: {e}")
        sys.exit(1)

def make_hospitals(config, data_pipeline):
    """Instantiate hospitals and their agent portfolios."""
    hospital_ids = [h.strip() for h in config["hospital_ids"].split(",")]
    agent_patterns = config["agents"]["patterns"]["default_mapping"]
    hospitals = {}
    for hid in hospital_ids:
        patterns = {ct: create_thinking_pattern(agent_patterns[ct]) for ct in agent_patterns}
        portfolio = AgentPortfolio(initial_patterns=patterns)
        hospitals[hid] = HospitalNode(
            hospital_id=hid,
            ham_metadata_csv=config["ham_csv"],
            isic_labels_csv=config["isic_csv"],
            data_pipeline=data_pipeline,
            agent_portfolio=portfolio,
        )
    return hospitals

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

def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config.yaml"
    config = load_config(config_path)
    ensure_output_dirs(config)
    logging.basicConfig(
        filename=Path(config.get("tracking", {}).get("log_dir", "outputs/logs")) / config.get("tracking", {}).get("log_file_name", "simulation.log"),
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )
    data_pipeline = LocalDataPipeline()
    hospitals = make_hospitals(config, data_pipeline)
    allocate_data(hospitals)
    run_federated_learning(config, hospitals)

if __name__ == "__main__":
    main()
