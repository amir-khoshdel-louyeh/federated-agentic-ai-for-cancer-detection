"""Logs tab module."""



import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import queue
import logging
from ..ui_kit import add_header, build_scrollable_page
from ..tkinter_queue_handler import TkinterQueueHandler



def build_logs_tab(parent: ttk.Notebook, log_queue: queue.Queue = None) -> ttk.Frame:
	"""Create and return the Logs tab frame."""
	frame, content = build_scrollable_page(parent)

	add_header(
		content,
		title="Simulation Logs",
		subtitle="View, filter, search, and download logs from the simulation, agents, and federated learning process.",
		badge="Logs"
	)

	# Controls frame
	controls = ttk.Frame(content)
	controls.pack(fill="x", padx=20, pady=(0, 10))

	# Log level filter
	log_levels = ["ALL", "INFO", "WARNING", "ERROR"]
	log_level_var = tk.StringVar(value="ALL")
	ttk.Label(controls, text="Level:").pack(side="left", padx=(0, 4))
	log_level_menu = ttk.Combobox(controls, textvariable=log_level_var, values=log_levels, state="readonly", width=8)
	log_level_menu.pack(side="left", padx=(0, 12))

	# Search entry
	search_var = tk.StringVar()
	ttk.Label(controls, text="Search:").pack(side="left", padx=(0, 4))
	search_entry = ttk.Entry(controls, textvariable=search_var, width=20)
	search_entry.pack(side="left", padx=(0, 12))

	# Download button
	def download_logs():
		logs = log_text.get("1.0", tk.END)
		if not logs.strip():
			messagebox.showinfo("Download Logs", "No logs to download.")
			return
		file_path = filedialog.asksaveasfilename(
			title="Save Logs As",
			defaultextension=".log",
			filetypes=[("Log Files", "*.log"), ("Text Files", "*.txt"), ("All Files", "*")],
		)
		if file_path:
			try:
				with open(file_path, "w", encoding="utf-8") as f:
					f.write(logs)
				messagebox.showinfo("Download Logs", f"Logs saved to {file_path}")
			except Exception as e:
				messagebox.showerror("Download Logs", f"Failed to save logs: {e}")

	download_btn = ttk.Button(controls, text="Download", command=download_logs)
	download_btn.pack(side="right")

	# Log display (scrollable)
	log_frame = ttk.Frame(content, style="Card.TFrame", padding=8)
	log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 16))

	log_text = tk.Text(log_frame, wrap="none", height=24, font=("DejaVu Sans Mono", 9), bg="#f8fafc", fg="#16202a", borderwidth=0)
	log_text.pack(side="left", fill="both", expand=True)
	log_text.config(state="disabled")

	yscroll = ttk.Scrollbar(log_frame, orient="vertical", command=log_text.yview)
	yscroll.pack(side="right", fill="y")
	log_text.config(yscrollcommand=yscroll.set)

	# Setup queue and handler for real-time logs
	if log_queue is None:
		log_queue = queue.Queue()
	queue_handler = TkinterQueueHandler(log_queue)
	queue_handler.setLevel(logging.INFO)
	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
	queue_handler.setFormatter(formatter)
	logging.getLogger().addHandler(queue_handler)

	def poll_log_queue():
		while True:
			try:
				msg = log_queue.get_nowait()
			except queue.Empty:
				break
			log_text.config(state="normal")
			log_text.insert(tk.END, msg + "\n")
			log_text.see(tk.END)
			log_text.config(state="disabled")
		log_text.after(200, poll_log_queue)

	poll_log_queue()

	# Load log directory and file name from configs/config.yaml
	import os
	import yaml
	config_path = os.path.join("configs", "config.yaml")
	if not os.path.exists(config_path):
		log_dir = "outputs/logs"
		log_file_name = "simulation.log"
	else:
		with open(config_path, "r") as f:
			config = yaml.safe_load(f)
		tracking = config.get("tracking", {})
		log_dir = tracking.get("log_dir", "outputs/logs")
		log_file_name = tracking.get("log_file_name", "simulation.log")

	log_file = os.path.join(log_dir, log_file_name)

	def read_logs():
		if not os.path.exists(log_file):
			return []
		with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
			return f.readlines()

	# Filtering logic
	def filter_logs():
		lines = read_logs()
		level = log_level_var.get()
		search = search_var.get().strip().lower()
		filtered = []
		for line in lines:
			l = line.lower()
			if level != "ALL":
				if f"{level.lower()}" not in l:
					continue
			if search and search not in l:
				continue
			filtered.append(line)
		log_text.config(state="normal")
		log_text.delete("1.0", tk.END)
		log_text.insert("1.0", "".join(filtered))
		log_text.config(state="disabled")

	# Bind filter/search
	log_level_menu.bind("<<ComboboxSelected>>", lambda e: filter_logs())
	search_var.trace_add("write", lambda *_: filter_logs())

	# Initial load
	filter_logs()

	# Optionally: add a refresh button or auto-refresh (not implemented for simplicity)

	return frame
