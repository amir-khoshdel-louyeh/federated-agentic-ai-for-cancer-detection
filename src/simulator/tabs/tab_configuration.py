"""Configuration tab module."""

from tkinter import ttk


def build_configuration_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Configuration tab frame."""
	frame = ttk.Frame(parent)
	label = ttk.Label(frame, text="content for Configuration")
	label.pack(padx=20, pady=20)
	return frame
