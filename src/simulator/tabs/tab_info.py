"""Information tab module."""

from tkinter import ttk


def build_info_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Information tab frame."""
	frame = ttk.Frame(parent)
	label = ttk.Label(frame, text="content for Information")
	label.pack(padx=20, pady=20)
	return frame
