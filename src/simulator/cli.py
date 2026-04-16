import logging
import os
import shutil
from pathlib import Path

from configs.config_loader import load_config
from src.simulator.controller import (
	configure_logging,
	initialize_system,
	federated_evaluation_round,
	test_system,
	show_results,
	show_log_location,
	validation_system
)


def delete_existing_artifacts() -> None:
	outputs_base = "outputs"
	if os.path.exists(outputs_base):
		try:
			shutil.rmtree(outputs_base)
			logging.info(f"Removed previous output directory: {outputs_base}")
		except Exception as e:
			logging.error(f"Failed to remove {outputs_base}: {e}")

	# Recreate core directory structure for the upcoming run
	os.makedirs(os.path.join(outputs_base, "logs"), exist_ok=True)
	os.makedirs(os.path.join(outputs_base, "history"), exist_ok=True)
	os.makedirs(os.path.join(outputs_base, "system"), exist_ok=True)
	logging.info("Initialized fresh output directories.")


def prepare_output_environment(config: dict) -> None:
	"""Prepare output directories using config-driven behavior."""
	clear_on_start = bool(config.get("tracking", {}).get("clear_output_on_start", True))
	if clear_on_start:
		delete_existing_artifacts()
		print("Starting with a clean output directory.")
	else:
		outputs_base = config.get("out_dir", "outputs")
		os.makedirs(os.path.join(outputs_base, "logs"), exist_ok=True)
		os.makedirs(os.path.join(outputs_base, "history"), exist_ok=True)
		os.makedirs(os.path.join(outputs_base, "system"), exist_ok=True)
		logging.info("Preserving existing output files and created missing directories.")
		print("Preserving existing outputs and continuing.")


def main():
	# Step 1: Load config.yaml
	config = load_config()
	configure_logging(config)
	prepare_output_environment(config)
	detection_mode = str(config.get("detection", {}).get("mode", "detect_then_type"))
	config.setdefault("detection", {})["mode"] = detection_mode

	from src.client_side.hospital.config_helpers import get_cancer_types
	config.setdefault("agents", {}).setdefault("patterns", {})
	cancer_types = get_cancer_types(config)
	config["agents"]["patterns"]["default_mapping"] = {
		cancer_type: "ai_agent" for cancer_type in cancer_types
	}
	logging.info("Using AI-agent mode for all specialists.")
	logging.info(f"Using detection mode from config: {detection_mode}")
	print(f"Using detection mode from config: {detection_mode}")


	# Step 2: Initialize system
	hospitals = initialize_system(config)


	# Log split sizes for each hospital
	logging.info("=== Data Split Sizes Per Hospital ===")
	print("=== Data Split Sizes Per Hospital ===")
	all_test_ids = {}
	for hid, hospital in hospitals.items():
		split_sizes = hospital.metrics_store.get("split_sizes", {})
		logging.info(f"Hospital {hid}: {split_sizes}")
		print(f"Hospital {hid}: {split_sizes}")
		# Try to get test_ids from local_data
		test_ids = None
		if hasattr(hospital, "local_data") and hospital.local_data is not None:
			test_ids = getattr(hospital.local_data, "test_ids", None)
		if test_ids is not None:
			all_test_ids[hid] = set(test_ids.tolist() if hasattr(test_ids, 'tolist') else list(test_ids))
	logging.info("====================================")
	print("====================================")

	# Resolve log path according to config
	log_dir = config.get("tracking", {}).get("log_dir", config.get("out_dir", "outputs") + "/logs")
	log_file = config.get("tracking", {}).get("log_file_name", "simulation.log")
	log_path = Path(log_dir) / log_file
	logging.info(f"Using log file: {log_path.resolve()}")

	# Check uniqueness of test_ids between hospitals
	is_unique = True
	if len(all_test_ids) > 1:
		all_ids = list(all_test_ids.values())
		for i in range(len(all_ids)):
			for j in range(i+1, len(all_ids)):
				if set(all_ids[i]) & set(all_ids[j]):
					is_unique = False
	if all_test_ids:
		logging.info(f"Data between hospitals is {'UNIQUE' if is_unique else 'NOT UNIQUE'} (based on test_ids).")
		print(f"Data between hospitals is {'UNIQUE' if is_unique else 'NOT UNIQUE'} (based on test_ids).")
	# choose k-fold vs single pipeline
	k_folds = int(config.get("data_split", {}).get("k_folds", 1))
	if k_folds > 1:
		from src.simulator.k_fold_runner import run_k_fold_experiment
		logging.info(f"Running k-fold cross-validation: {k_folds} folds")
		run_k_fold_experiment(config)
	else:
		logging.info("Starting federated evaluation round...")
		print("Starting federated evaluation round...")
		federated_evaluation_round(config, hospitals)
		logging.info("Federated evaluation round complete.")
		print("Federated evaluation round complete.")
		validation_system(hospitals)
		logging.info("Running evaluation on test data...")
		print("Running evaluation on test data...")
		results = test_system(hospitals)
		show_results(results)
		show_log_location(config)
		

if __name__ == "__main__":
	main()
