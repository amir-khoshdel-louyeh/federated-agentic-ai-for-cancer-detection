"""Train tab module."""


import threading
from tkinter import ttk
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from ..ui_kit import add_card, add_header, build_scrollable_page

# Import controller functions
from ..controller import load_config, initialize_system, train_system


def build_train_tab(parent: ttk.Notebook) -> ttk.Frame:

	# Event for pausing/resuming training
	train_pause_event = threading.Event()
	train_pause_event.set()  # Start as 'not paused'
	train_stop_flag = {"stop": False}

	def stop_training():
		train_stop_flag["stop"] = True
		train_pause_event.set()  # Unpause if waiting

	def continue_training():
		train_pause_event.set()


	def train_all_rounds():
		def run_all():
			train_stop_flag["stop"] = False
			while train_state["current_round"] < train_state["num_rounds"]:
				if train_stop_flag["stop"]:
					break
				train_pause_event.wait()  # Wait if paused
				train_pause_event.clear()  # Pause after each round unless continued
				run_one_round()
				# Optionally, add a small delay for UI responsiveness
				# import time; time.sleep(0.1)
		threading.Thread(target=run_all, daemon=True).start()

	"""Create and return the Train tab frame."""
	frame, content = build_scrollable_page(parent)
	add_header(
		content,
		title="Federated Training",
		subtitle="Start federated training for all hospital agents using the current configuration.",
		badge="TRAIN"
	)

	status_var = tk.StringVar(value="Idle.")

	import logging

	# State for one-round-at-a-time training
	train_state = {
		"config": None,
		"hospitals": None,
		"orchestrator": None,
		"current_round": 0,
		"num_rounds": 0,
		"last_image_id": None,
		"last_features": None,
		"last_agent_metrics": None,
		"metrics_history": {},  # {agent: {metric: [values]}}
		"global_metrics_history": {},
		"round_times": [],
	}

	# --- Metrics/Charts UI ---
	charts_frame = ttk.Frame(content)
	charts_frame.pack(fill="x", padx=20, pady=(8, 0))

	# Placeholders for matplotlib figures (per metric)
	metric_figures = {}
	metric_canvases = {}
	metrics_to_plot = ["accuracy", "loss", "f1", "auc"]

	for i, metric in enumerate(metrics_to_plot):
		fig, ax = plt.subplots(figsize=(3, 2), dpi=100)
		ax.set_title(metric.capitalize())
		ax.set_xlabel("Round")
		ax.set_ylabel(metric.capitalize())
		canvas = FigureCanvasTkAgg(fig, master=charts_frame)
		canvas.get_tk_widget().grid(row=0, column=i, padx=8)
		metric_figures[metric] = (fig, ax)
		metric_canvases[metric] = canvas

	# --- End Metrics/Charts UI ---

	# UI for image and metrics display
	image_frame = ttk.Frame(content)
	image_frame.pack(fill="x", padx=20, pady=(8, 0))

	orig_img_label = ttk.Label(image_frame, text="Original Image:")
	orig_img_label.grid(row=0, column=0, sticky="w")

	preproc_img_label = ttk.Label(image_frame, text="Preprocessed (features):")
	preproc_img_label.grid(row=0, column=1, sticky="w")

	orig_img_canvas = tk.Label(image_frame)
	orig_img_canvas.grid(row=1, column=0, padx=8, pady=4)

	preproc_img_canvas = tk.Label(image_frame)
	preproc_img_canvas.grid(row=1, column=1, padx=8, pady=4)

	metrics_frame = ttk.Frame(content)
	metrics_frame.pack(fill="x", padx=20, pady=(8, 0))

	metrics_label = ttk.Label(metrics_frame, text="Agent Metrics (last round):")
	metrics_label.pack(anchor="w")

	metrics_text = tk.Text(metrics_frame, height=8, width=80, font=("DejaVu Sans Mono", 9))
	metrics_text.pack(fill="x")
	metrics_text.config(state="disabled")

	def initialize_training():
		status_var.set("Loading config...")
		logging.info("Loading config...")
		config = load_config()
		status_var.set("Initializing system...")
		logging.info("Initializing system...")
		hospitals = initialize_system(config)
		train_state["config"] = config
		train_state["hospitals"] = hospitals
		from src.client_side.hospital.orchestrator import FederatedRoundOrchestrator
		aggregation_name = config["federation"]["aggregation_algorithm"]
		train_state["orchestrator"] = FederatedRoundOrchestrator.from_algorithm(name=aggregation_name)
		train_state["current_round"] = 0
		train_state["num_rounds"] = config["simulation"]["num_rounds"]
		status_var.set("Ready. Click 'Next Round' to train.")

	def run_one_round():
		# Only call controller logic, handle UI updates here
		if not train_state["hospitals"] or not train_state["orchestrator"]:
			status_var.set("Please initialize training first.")
			return
		if train_state["current_round"] >= train_state["num_rounds"]:
			status_var.set("All rounds completed.")
			return
		try:
			import time
			from ..controller import run_one_training_round
			start_time = time.time()
			train_state["current_round"] += 1
			round_idx = train_state["current_round"]
			status_var.set(f"Training round {round_idx}...")
			logging.info(f"Training round {round_idx}...")
			# Call controller to run one round
			round_output, agent_metrics = run_one_training_round(
				train_state["orchestrator"], train_state["hospitals"], round_idx, for_training=True
			)
			# --- UI updates (metrics, charts, etc.) ---
			# Example: update metrics_text with agent_metrics
			metrics_text.config(state="normal")
			metrics_text.delete("1.0", tk.END)
			for hid, metrics in agent_metrics.items():
				metrics_text.insert(tk.END, f"Hospital {hid}:\n")
				for cancer_type, pattern in metrics.items():
					metrics_text.insert(tk.END, f"  {cancer_type}: pattern={pattern}\n")
			metrics_text.config(state="disabled")
			# (Optional) update charts, round times, etc. as needed
			end_time = time.time()
			train_state["round_times"].append(end_time - start_time)
			status_var.set(f"Completed round {round_idx}.")
			logging.info(f"Completed round {round_idx}.")
		except Exception as e:
			import traceback
			status_var.set(f"Error: {e}")
			logging.error(f"Error: {e}\nTraceback:\n{traceback.format_exc()}" )

	def on_initialize_click():
		threading.Thread(target=initialize_training, daemon=True).start()

	def on_next_round_click():
		threading.Thread(target=run_one_round, daemon=True).start()

	init_btn = ttk.Button(content, text="Initialize Training", command=on_initialize_click)
	init_btn.pack(pady=8)

	next_round_btn = ttk.Button(content, text="Next Round", command=on_next_round_click)
	next_round_btn.pack(pady=8)


	train_all_btn = ttk.Button(content, text="Train All Rounds", command=train_all_rounds)
	train_all_btn.pack(pady=8)

	stop_btn = ttk.Button(content, text="Stop Training", command=stop_training)
	stop_btn.pack(pady=8)

	continue_btn = ttk.Button(content, text="Continue Training", command=continue_training)
	continue_btn.pack(pady=8)

	status_label = ttk.Label(content, textvariable=status_var, style="Subheading.TLabel")
	status_label.pack(anchor="w", padx=20, pady=(8, 0))

	return frame
