import argparse
from configs.config_loader import load_config
from src.simulator.controller import (
	initialize_system,
	train_system,
	test_system,
	show_results,
	show_log_location
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
	for hid, hospital in hospitals.items():
		split_sizes = hospital.metrics_store.get("split_sizes", {})
		print(f"Hospital {hid}: {split_sizes}")
	print("====================================\n")

	if args.command == 'train':
		print("Starting federated training...")
		train_system(config, hospitals)
		print("Training complete.")
		show_log_location(config)
	elif args.command == 'test':
		print("Running evaluation on test data...")
		results = test_system(hospitals)
		show_results(results)
		show_log_location(config)

if __name__ == "__main__":
	main()
