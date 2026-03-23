"""Information tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_info_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Information tab frame."""
	frame, content = build_scrollable_page(parent)

	add_header(
		content,
		title="Welcome To Federated Agentic AI Simulator",
		subtitle=(
			"This workspace helps you configure multi-hospital experiments, run local training, "
			"evaluate models, and compare federated aggregation outcomes for cancer detection."
		),
		badge="Welcome",
	)

	add_card(
		content,
		title="What This Project Does",
		lines=[
			"Builds one hospital node per site and trains local cancer agents without sharing raw data.",
			"Exports standardized local updates and validates schema consistency between hospitals.",
			"Runs federated aggregation with fedavg, fedprox, or adaptive weighting.",
		],
	)

	add_card(
		content,
		title="Quick Start",
		lines=[
			"Step 1: Open Configuration and set HAM metadata CSV + ISIC labels CSV paths.",
			"Step 2: Define hospital IDs and choose aggregation mode.",
			"Step 3: Run training/evaluation and inspect Test + Results tabs.",
			"Step 4: Use Logs for lifecycle tracing and debugging warnings.",
		],
	)

	add_card(
		content,
		title="Expected Inputs",
		lines=[
			"HAM10000 metadata CSV (for example: HAM10000_metadata.csv).",
			"ISIC 2019 ground-truth CSV (for example: ISIC_2019_Training_GroundTruth.csv).",
			"Hospital IDs (single-hospital or multi-hospital simulation).",
		],
	)

	add_card(
		content,
		title="Supported Patterns And Algorithms",
		lines=[
			"Thinking patterns: rule_based, rule_based_strict, bayesian, deep_learning.",
			"Federation algorithms: fedavg, fedprox, adaptive.",
			"Cancer categories used across portfolio: BCC, SCC, MELANOMA, AKIEC.",
		],
	)

	add_card(
		content,
		title="Key Metrics You Will See",
		lines=[
			"Accuracy and F1-score for local model quality.",
			"AUC for ranking quality and cross-hospital comparisons.",
			"Sensitivity and specificity for clinical balance.",
		],
	)

	add_card(
		content,
		title="Tab Guide",
		lines=[
			"Information: welcome page and project documentation.",
			"Configuration: paths, hospitals, patterns, seed, and aggregation controls.",
			"Train: dataset readiness, split health, and training lifecycle overview.",
			"Test: per-agent performance and metric interpretation.",
			"Results: hospital-to-hospital and global federation outcomes.",
			"Logs: warnings, validation checks, and execution details.",
		],
	)

	add_card(
		content,
		title="Generated Artifacts",
		lines=[
			"<hospital_id>_output.json and <hospital_id>_summary.json per hospital.",
			"federation_comparison.json and federation_run_metadata.json.",
			"multi_hospital_report.json with combined run report.",
		],
	)

	add_card(
		content,
		title="Important Notes",
		lines=[
			"Multi-hospital mode requires at least 2 hospitals.",
			"The system aggregates updates/metrics, not raw patient data.",
			"Start with fedavg baseline, then compare fedprox/adaptive for robustness.",
		],
	)

	return frame
