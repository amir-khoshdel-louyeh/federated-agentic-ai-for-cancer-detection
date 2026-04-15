import argparse
import logging
import os
import shutil
from pathlib import Path

from configs.config_loader import load_config
from src.simulator.controller import (
	configure_logging,
	initialize_system,
	train_system,
	test_system,
	show_results,
	show_log_location,
	validation_system
)


def confirm_previous_history() -> None:
	"""Ask user whether to keep or delete previous logs/output."""
	keep_old = input("Do you want to keep existing logs/output? (y/N): ").strip().lower()
	if keep_old in {"y", "yes"}:
		print("Please make a backup of logs/output before continuing.")
		continue_after_backup = input("Continue and delete all previous logs/output after backup? (y/N): ").strip().lower()
		if continue_after_backup in {"y", "yes"}:
			delete_existing_artifacts()
			print("Existing logs/output removed. Continuing with clean slate.")
		else:
			print("Keeping existing logs/output. Continuing without deletion.")
	else:
		delete_existing_artifacts()
		print("Existing logs/output removed. Continuing with clean slate.")


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

def choose_agent_mode() -> str:
	logging.info("Agent mode forced to AI-agent only.")
	print("=== Agent mode forced to AI-agent only ===")
	return "ai_agent"


def choose_detection_mode() -> str:
	print("=== Choose cancer detection flow ===")
	print("1) Only cancer detector")
	print("2) Cancer detector + other agents")
	selection = input("Select mode [1/2]: ").strip().lower()
	if selection in {"2", "2)", "other", "both", "cancer detector + other agents"}:
		print("Using cancer detector + other agents.")
		return "detect_then_type"
	print("Using cancer detector only.")
	return "detect_only"


def main():
	# Step 1: Load config.yaml
	config = load_config()
	configure_logging(config)
	confirm_previous_history()
	mode = choose_agent_mode()
	detection_mode = choose_detection_mode()
	config.setdefault("detection", {})["mode"] = detection_mode

	from src.client_side.hospital.config_helpers import get_cancer_types
	config.setdefault("agents", {}).setdefault("patterns", {})
	cancer_types = get_cancer_types(config)
	config["agents"]["patterns"]["default_mapping"] = {
		cancer_type: "ai_agent" for cancer_type in cancer_types
	}
	logging.info("Using AI-agent mode for all specialists.")


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
		logging.info("Starting federated AI-agent evaluation...")
		print("Starting federated AI-agent evaluation...")
		train_system(config, hospitals)
		logging.info("AI-agent evaluation complete.")
		print("AI-agent evaluation complete.")
		validation_system(hospitals)
		logging.info("Running evaluation on test data...")
		print("Running evaluation on test data...")
		results = test_system(hospitals)
		show_results(results)
		show_log_location(config)
		

if __name__ == "__main__":
	main()
