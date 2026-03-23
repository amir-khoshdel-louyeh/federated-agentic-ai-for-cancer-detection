"""Test tab module."""

from tkinter import ttk


def build_test_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Test tab frame."""
	frame = ttk.Frame(parent)
	label = ttk.Label(frame, text="content for Test")
	label.pack(padx=20, pady=20)
	return frame
