"""Configuration tab module."""

from tkinter import ttk

from ..ui_kit import add_card, add_header, build_scrollable_page


def build_configuration_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Configuration tab frame."""
	frame, content = build_scrollable_page(parent)

	
	return frame
