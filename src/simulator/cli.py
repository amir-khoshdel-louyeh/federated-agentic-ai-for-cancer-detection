import argparse
from pathlib import Path

from configs.config_loader import load_config
from src.simulator.controller import (
	initialize_system,
	train_system,
	test_system,
	show_results,
	show_log_location,
	validation_system
)

def choose_agent_mode() -> str:
	print("\n=== Agent mode selection ===")
	print("1) Existing logic-based AI agents (rule/deep/logistic/bayesian)")
	print("2) Library-based powerful AI agents (pretrained ensemble mode)")
	choice = input("Enter mode (1 or 2) [1]: ").strip() or "1"
	if choice not in {"1", "2"}:
		print("Invalid choice, defaulting to 1")
		choice = "1"
	return "built_in" if choice == "1" else "library"


def main():
	# Step 1: Load config.yaml
	config = load_config()
	mode = choose_agent_mode()

	if mode == "library":
		config.setdefault("agents", {}).setdefault("patterns", {})
		config["agents"]["patterns"]["default_mapping"] = {
			"BCC": "pretrained_library",
			"SCC": "pretrained_library",
			"MELANOMA": "pretrained_library",
			"AKIEC": "pretrained_library",
		}
		print("Using pretrained library agent strategy for all specialists.\n")
	else:
		print("Using existing logic-based agents.\n")


	# Step 2: Initialize system
	hospitals = initialize_system(config)


	# Print split sizes for each hospital
	print("\n=== Data Split Sizes Per Hospital ===")
	all_test_ids = {}
	for hid, hospital in hospitals.items():
		split_sizes = hospital.metrics_store.get("split_sizes", {})
		print(f"Hospital {hid}: {split_sizes}")
		# Try to get test_ids from local_data
		test_ids = None
		if hasattr(hospital, "local_data") and hospital.local_data is not None:
			test_ids = getattr(hospital.local_data, "test_ids", None)
		if test_ids is not None:
			all_test_ids[hid] = set(test_ids.tolist() if hasattr(test_ids, 'tolist') else list(test_ids))
	print("====================================\n")

	# Resolve and print log path according to config
	log_dir = config.get("tracking", {}).get("log_dir", config.get("out_dir", "outputs") + "/logs")
	log_file = config.get("tracking", {}).get("log_file_name", "simulation.log")
	log_path = Path(log_dir) / log_file
	print(f"Using log file: {log_path.resolve()}")

	# Check uniqueness of test_ids between hospitals
	is_unique = True
	if len(all_test_ids) > 1:
		all_ids = list(all_test_ids.values())
		for i in range(len(all_ids)):
			for j in range(i+1, len(all_ids)):
				if set(all_ids[i]) & set(all_ids[j]):
					is_unique = False
	if all_test_ids:
		print(f"Data between hospitals is {'UNIQUE' if is_unique else 'NOT UNIQUE'} (based on test_ids).\n")

	# choose k-fold vs single pipeline
	k_folds = int(config.get("data_split", {}).get("k_folds", 1))
	if k_folds > 1:
		from src.simulator.k_fold_runner import run_k_fold_experiment
		print(f"Running k-fold cross-validation: {k_folds} folds")
		run_k_fold_experiment(config)
	else:
		print("Starting federated training...")
		train_system(config, hospitals)
		print("Training complete.")
		validation_system(hospitals)
		print("Running evaluation on test data...")
		results = test_system(hospitals)
		show_results(results)
		show_log_location(config)
		

if __name__ == "__main__":
	main()
