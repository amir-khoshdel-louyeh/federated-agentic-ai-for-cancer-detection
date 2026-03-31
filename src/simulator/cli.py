import argparse
from configs.config_loader import load_config

def main():
	parser = argparse.ArgumentParser(description="Federated Agentic AI CLI")
	parser.add_argument('--config', type=str, default='configs/config.yaml', help='Path to config.yaml')
	args = parser.parse_args()

	# Step 1: Load config.yaml
	config = load_config(args.config)
	print("Loaded config:")
	print(config)

if __name__ == "__main__":
	main()
