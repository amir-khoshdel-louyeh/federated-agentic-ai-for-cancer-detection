"""Configuration tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_configuration_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Configuration tab frame."""
	frame, content = build_scrollable_page(parent)

	add_header(
		content,
		title="Configuration Workspace",
		subtitle=(
			"Central place for datasets, hospital setup, aggregation strategy, and pattern policy. "
			"All tabs consume these settings."
		),
		badge="Setup",
	)

	add_card(
		content,
		title="Dataset Paths",
		lines=[
			"HAM metadata CSV path.",
			"ISIC labels CSV path.",
			"Output directory for generated artifacts.",
		],
	)

	add_card(
		content,
		title="Hospital Topology",
		lines=[
			"Hospital IDs list (for example: hospital_1,hospital_2,hospital_3).",
			"Multi-hospital mode requires at least 2 hospital IDs.",
			"Seed controls reproducibility across hospital runs.",
		],
	)

	add_card(
		content,
		title="Thinking Pattern Policy",
		lines=[
			"Assign pattern per cancer type: BCC, SCC, MELANOMA, AKIEC.",
			"Supported patterns: rule_based, rule_based_strict, bayesian, deep_learning.",
			"Use static mapping now, adaptive policy later from validation leaderboard.",
		],
	)

	add_card(
		content,
		title="Federation Controls",
		lines=[
			"Aggregation algorithm: fedavg | fedprox | adaptive.",
			"Enable compare-all to execute all algorithms on identical local updates.",
			"Track selected algorithm and exported run metadata.",
		],
	)

	return frame
