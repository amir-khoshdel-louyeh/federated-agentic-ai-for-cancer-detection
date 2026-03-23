"""Results tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_results_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Results tab frame."""
	frame, content = build_scrollable_page(parent)

	add_header(
		content,
		title="Federated Results",
		subtitle=(
			"Compare local hospital outputs and global aggregation outcomes in one place."
		),
		badge="Results",
	)

	add_card(
		content,
		title="Hospital-Level Comparison",
		lines=[
			"Review local summary metrics per hospital (average accuracy, average f1, average auc).",
			"Inspect best agent by AUC for each hospital.",
			"Compare selected thinking patterns across centers.",
		],
	)

	add_card(
		content,
		title="Global Federation Outcomes",
		lines=[
			"Global metrics are produced per algorithm round.",
			"Hospital weights explain each center's contribution to aggregation.",
			"Dropped updates are explicitly listed with reason codes.",
		],
	)

	add_card(
		content,
		title="Algorithm Comparison",
		lines=[
			"fedavg: sample-size weighted baseline.",
			"fedprox: stabilization with proximal blending and mu coefficient.",
			"adaptive: combines sample size, quality, and reliability signals.",
		],
	)

	return frame
