"""Train tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_train_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Train tab frame."""
	frame, content = build_scrollable_page(parent)

	add_header(
		content,
		title="Training Overview",
		subtitle=(
			"Inspect dataset readiness, split health, and local training lifecycle before evaluation."
		),
		badge="Train",
	)

	add_card(
		content,
		title="Dataset Readiness",
		lines=[
			"Validate HAM and ISIC CSV availability and schema before training.",
			"Confirm train/val/test split sizes for every hospital.",
			"Review random seed used by VirtualHospital for reproducibility.",
		],
	)

	add_card(
		content,
		title="Local Training Lifecycle",
		lines=[
			"initialize: load local data and apply selected pattern mapping.",
			"train: fit all cancer agents and generate val/test probability outputs.",
			"evaluate: compute metrics and prepare standardized local update export.",
		],
	)

	add_card(
		content,
		title="Training Quality Signals",
		lines=[
			"Training warnings appear when one-vs-rest labels are not diverse enough in a split.",
			"Fallback behavior uses malignant binary labels to keep training stable.",
			"Use these warnings to diagnose class imbalance per hospital.",
		],
	)

	return frame
