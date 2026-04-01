import argparse

from networkx import config
from configs.config_loader import load_config
from src.simulator.controller import (
	initialize_system,
	train_system,
	test_system,
	show_results,
	show_log_location,
	validation_system
)

def main():
	parser = argparse.ArgumentParser(description="Federated Agentic AI CLI")
	parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
	subparsers = parser.add_subparsers(dest='command', required=True)

	# Train command
	train_parser = subparsers.add_parser('train', help='Run federated training')

	# Test command
	test_parser = subparsers.add_parser('test', help='Run evaluation on test data')

	args = parser.parse_args()

	# Step 1: Load config.yaml
	config = load_config(args.config)


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
		print("Starting federated training...")
		train_system(config, hospitals)
		print("Training complete.")
		validation_system(hospitals)
		#print("Running evaluation on test data...")
		#results = test_system(hospitals)
		#show_results(results)
		#show_log_location(config)
		

if __name__ == "__main__":
	main()
