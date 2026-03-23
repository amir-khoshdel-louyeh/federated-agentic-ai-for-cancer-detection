"""Train tab module."""

from tkinter import ttk


def build_train_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Train tab frame."""
	frame = ttk.Frame(parent)
	label = ttk.Label(frame, text="content for Train")
	label.pack(padx=20, pady=20)
	return frame
