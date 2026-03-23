"""Results tab module."""

from tkinter import ttk


def build_results_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Results tab frame."""
	frame = ttk.Frame(parent)
	label = ttk.Label(frame, text="content for Results")
	label.pack(padx=20, pady=20)
	return frame
