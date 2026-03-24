"""Train tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_train_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Train tab frame."""
	import os
	import yaml
	frame, content = build_scrollable_page(parent)

	# Load config
	config_path = os.path.join("configs", "config.yaml")
	if not os.path.exists(config_path):
		config = {}
	else:
		with open(config_path, "r") as f:
			config = yaml.safe_load(f)

	# Header
	add_header(
		content,
		title="Training Control",
		subtitle="Configure and launch federated training. Review the current setup below.",
		badge="Train"
	)

	# Dataset info
	dataset_name = "HAM10000" if config.get("ham_csv") else ("ISIC2019" if config.get("isic_csv") else "N/A")
	out_dir = config.get("out_dir", "outputs")
	add_card(
		content,
		title="Dataset",
		lines=[
			f"Name: {dataset_name}",
			f"Output Dir: {out_dir}"
		]
	)

	# Algorithm info
	fed = config.get("federation", {})
	algo = fed.get("aggregation_algorithm", "N/A")
	methods = ["fedavg"]
	if "fedprox" in fed:
		methods.append("fedprox")
	if "adaptive" in fed:
		methods.append("adaptive")
	methods_str = ", ".join(methods)
	add_card(
		content,
		title="Federated Algorithm",
		lines=[
			f"Selected: {algo}",
			f"Available: {methods_str}"
		]
	)

	# Agents info
	agents = config.get("agents", {})
	agent_types = agents.get("types", [])
	patterns = agents.get("patterns", {}).get("available", [])
	default_mapping = agents.get("patterns", {}).get("default_mapping", {})
	add_card(
		content,
		title="Agents",
		lines=[
			f"Types: {', '.join(agent_types)}",
			f"Patterns: {', '.join(patterns)}",
			f"Default Mapping: {default_mapping}"
		]
	)

	# Control buttons
	controls = ttk.Frame(content)
	controls.pack(fill="x", padx=20, pady=(16, 10))

	start_btn = ttk.Button(controls, text="Start Training", style="Colored.TButton")
	start_btn.pack(side="left", padx=8)
	stop_btn = ttk.Button(controls, text="Stop Training", style="Danger.TButton")
	stop_btn.pack(side="left", padx=8)
	# Optionally: add more controls (pause, resume, etc.)

	return frame
