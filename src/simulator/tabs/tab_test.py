"""Test tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_test_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Test tab frame."""
	frame, content = build_scrollable_page(parent)

	add_header(
		content,
		title="Testing And Evaluation",
		subtitle=(
			"Review per-agent and per-pattern test performance for each hospital."
		),
		badge="Evaluate",
	)

	add_card(
		content,
		title="Primary Metrics",
		lines=[
			"AUC (ROC area): ranking quality across thresholds.",
			"F1-score: balance between precision and recall.",
			"Accuracy: overall classification correctness.",
			"Sensitivity: true positive rate (recall for positives).",
			"Specificity: true negative rate.",
		],
	)

	add_card(
		content,
		title="What To Compare",
		lines=[
			"Per-agent metrics inside each hospital update.",
			"Selected pattern performance for each cancer type.",
			"Candidate pattern leaderboard ranked by validation AUC.",
		],
	)

	add_card(
		content,
		title="Interpretation Hints",
		lines=[
			"High AUC with low F1 can indicate threshold calibration issues.",
			"Sensitivity and specificity trade-offs should be monitored per cancer subtype.",
			"Track consistency across hospitals to detect domain heterogeneity.",
		],
	)

	return frame
