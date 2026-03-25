"""Train tab module."""

import threading
from tkinter import ttk
import tkinter as tk

from ..ui_kit import add_card, add_header, build_scrollable_page

# Import controller functions
from ..controller import load_config, initialize_system, train_system

def build_train_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Train tab frame."""
	frame, content = build_scrollable_page(parent)
	add_header(
		content,
		title="Federated Training",
		subtitle="Start federated training for all hospital agents using the current configuration.",
		badge="TRAIN"
	)

	status_var = tk.StringVar(value="Idle.")

	def run_training():
		status_var.set("Loading config...")
		try:
			config = load_config()
			status_var.set("Initializing system...")
			hospitals = initialize_system(config)
			status_var.set("Training in progress...")
			train_system(config, hospitals)
			status_var.set("Training completed!")
		except Exception as e:
			status_var.set(f"Error: {e}")

	def on_train_click():
		threading.Thread(target=run_training, daemon=True).start()

	train_btn = ttk.Button(content, text="Start Training", command=on_train_click)
	train_btn.pack(pady=16)

	status_label = ttk.Label(content, textvariable=status_var, style="Subheading.TLabel")
	status_label.pack(anchor="w", padx=20, pady=(8, 0))

	return frame
