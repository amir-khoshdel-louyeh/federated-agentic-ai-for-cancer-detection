"""Professional Tkinter splash screen."""

import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk

from .tabs import (
	build_configuration_tab,
	build_info_tab,
	build_logs_tab,
	build_results_tab,
	build_test_tab,
	build_train_tab,
)
from .ui_kit import configure_app_style



import os
import logging
import yaml

def main() -> None:
	# Load config for logging control
	config_path = "config.yaml"
	config = {}
	if os.path.exists(config_path):
		with open(config_path, "r") as f:
			config = yaml.safe_load(f) or {}
	tracking = config.get("tracking", {})
	save_logs = tracking.get("save_logs", True)
	log_dir = tracking.get("log_dir", "outputs/logs")
	log_file = os.path.join(log_dir, "simulation.log")

	# Configure logging
	if save_logs:
		os.makedirs(log_dir, exist_ok=True)
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s | %(levelname)s | %(message)s",
			handlers=[
				logging.FileHandler(log_file, encoding="utf-8"),
				logging.StreamHandler()
			]
		)
	else:
		logging.basicConfig(
			level=logging.INFO,
			format="%(asctime)s | %(levelname)s | %(message)s",
			handlers=[logging.StreamHandler()]
		)

	root = tk.Tk()
	root.title("Federated Agentic AI")

	# Maximize across platforms with a Linux-safe fallback.
	try:
		root.attributes("-zoomed", True)
	except tk.TclError:
		pass
	root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
	style, style_name = configure_app_style(root)
	style.configure(f"{style_name}.Tab", anchor="center")

	# Create notebook (tab container)
	notebook = ttk.Notebook(root, style=style_name)
	notebook.pack(fill="both", expand=True, padx=10, pady=10)

	# Create tabs from dedicated modules.
	tab_information = build_info_tab(notebook)
	tab_configuration = build_configuration_tab(notebook)
	tab_train = build_train_tab(notebook)
	tab_test = build_test_tab(notebook)
	tab_results = build_results_tab(notebook)
	tab_logs = build_logs_tab(notebook)

	notebook.add(tab_information, text="Information")
	notebook.add(tab_configuration, text="Configuration")
	notebook.add(tab_train, text="Train")
	notebook.add(tab_test, text="Test")
	notebook.add(tab_results, text="Results")
	notebook.add(tab_logs, text="Logs")

	def _refresh_tab_widths(_: tk.Event | None = None) -> None:
		"""Distribute tab widths so tabs always span the full notebook width."""
		tab_count = len(notebook.tabs())
		if tab_count == 0:
			return

		notebook_width = notebook.winfo_width()
		if notebook_width <= 1:
			return

		# ttk tab width is in characters, so convert pixels to character units.
		font = tkfont.nametofont("TkDefaultFont")
		char_width = max(font.measure("0"), 1)
		usable_width = max(notebook_width - 12, 1)
		width_chars = max(1, usable_width // (tab_count * char_width))
		style.configure(f"{style_name}.Tab", width=width_chars)

	notebook.bind("<Configure>", _refresh_tab_widths)
	notebook.bind("<<NotebookTabChanged>>", _refresh_tab_widths)
	root.after(50, _refresh_tab_widths)

	root.mainloop()


if __name__ == "__main__":
	main()
