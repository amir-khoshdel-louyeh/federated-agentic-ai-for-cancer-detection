"""Logs tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_logs_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Logs tab frame."""
	frame, content = build_scrollable_page(parent)

	add_header(
		content,
		title="Execution Logs",
		subtitle=(
			"Track lifecycle events, warnings, validation checks, and federation round details for debugging."
		),
		badge="Observability",
	)

	add_card(
		content,
		title="Lifecycle Trace",
		lines=[
			"initialized -> trained -> evaluated for each hospital.",
			"Local update export and global update application flags.",
			"Round index and algorithm selection for each federation run.",
		],
	)

	add_card(
		content,
		title="Warning Channels",
		lines=[
			"Training warnings from insufficient one-vs-rest label diversity.",
			"Contract or schema mismatch errors across local updates.",
			"Dropped hospital updates and recorded reason codes.",
		],
	)

	add_card(
		content,
		title="Recommended Log Blocks",
		lines=[
			"Config snapshot: data paths, hospital IDs, seed, algorithm.",
			"Per-hospital summary: selected patterns and key metrics.",
			"Federation summary: global metrics, weights, and validation report.",
		],
	)

	return frame
