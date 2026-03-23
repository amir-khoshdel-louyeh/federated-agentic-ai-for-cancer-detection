"""Logs tab module."""

from tkinter import ttk


def build_logs_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Logs tab frame."""
	frame = ttk.Frame(parent)
	label = ttk.Label(frame, text="content for Logs")
	label.pack(padx=20, pady=20)
	return frame
