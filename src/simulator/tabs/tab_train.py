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

	def save_models():
		try:
			config = train_state["config"]
			out_dir = config.get("out_dir", "outputs")
			hospitals = train_state["hospitals"]
			for hid, hospital in hospitals.items():
				if hasattr(hospital.scope.agent_portfolio, 'save_all_models'):
					hospital.scope.agent_portfolio.save_all_models(out_dir, hospital.hospital_id)
			status_var.set(f"Models saved to {out_dir}")
		except Exception as e:
			status_var.set(f"Error saving models: {e}")

	def load_models():
		import tkinter.filedialog as fd
		try:
			hospitals = train_state["hospitals"]
			# Ask user for directory
			model_dir = fd.askdirectory(title="Select Model Directory")
			if not model_dir:
				status_var.set("Model loading cancelled.")
				return
			for hid, hospital in hospitals.items():
				if hasattr(hospital.scope.agent_portfolio, 'load_all_models'):
					hospital.scope.agent_portfolio.load_all_models(model_dir, hospital.hospital_id)
			status_var.set(f"Models loaded from {model_dir}")
		except Exception as e:
			status_var.set(f"Error loading models: {e}")
	# --- Variables for toggling charts and image viewer ---
	show_accuracy_chart_var = tk.BooleanVar(value=True)
	show_loss_chart_var = tk.BooleanVar(value=True)
	show_images_var = tk.BooleanVar(value=True)

	# Enable mouse wheel scrolling for charts_canvas
	def _on_mousewheel(event):
		# For Windows and MacOS
		charts_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
	def _on_linux_mousewheel(event):
		# For Linux (event.num 4=up, 5=down)
		if event.num == 4:
			charts_canvas.yview_scroll(-1, "units")
		elif event.num == 5:
			charts_canvas.yview_scroll(1, "units")

	# Bind mouse wheel events after charts_canvas is defined
	# (Move these lines after charts_canvas is created)

	def on_close():
		train_stop_flag["stop"] = True
		train_pause_event.set()  # Unpause any waiting threads
		# Optionally, add more cleanup here if needed
		parent.winfo_toplevel().destroy()
		def update_charts(agent_metrics):
			import random
			agent_name = next(iter(agent_metrics.keys())) if agent_metrics else "demo_agent"
			if agent_name not in train_state["metrics_history"]:
				train_state["metrics_history"][agent_name] = {m: [] for m in metrics_to_plot}
			for metric in metrics_to_plot:
				value = random.uniform(0.7, 1.0) if metric != "loss" else random.uniform(0.1, 0.5)
				train_state["metrics_history"][agent_name][metric].append(value)
			for metric in metrics_to_plot:
				fig, ax = metric_figures[metric]
				ax.clear()
				ax.set_title(metric.capitalize())
				ax.set_xlabel("Round")
				ax.set_ylabel(metric.capitalize())
				for agent, agent_metrics_hist in train_state["metrics_history"].items():
					ax.plot(range(1, len(agent_metrics_hist[metric]) + 1), agent_metrics_hist[metric], label=agent)
				ax.legend()
				metric_canvases[metric].draw()

	def update_image_previews(round_idx):
		# Only update images if enabled
		if not show_images_var.get():
			return
		hospital = next(iter(train_state["hospitals"].values()))
		img_id = None
		if hasattr(hospital, "local_data") and hospital.local_data is not None:
			if hasattr(hospital.local_data, "test_ids") and len(hospital.local_data.test_ids) > 0:
				idx = (round_idx - 1) % len(hospital.local_data.test_ids)
				img_id = hospital.local_data.test_ids[idx]
		train_state["last_image_id"] = img_id
		# Find original image path (HAM or ISIC)
		img_path = None
		if img_id is not None:
			import os
			ham_dir = "src/client_side/datasets/HAM10000/ham10000_images"
			isic_dir = "src/client_side/datasets/ISIC2019/ISIC2019"
			if os.path.exists(os.path.join(ham_dir, f"{img_id}.jpg")):
				img_path = os.path.join(ham_dir, f"{img_id}.jpg")
			elif os.path.exists(os.path.join(isic_dir, f"{img_id}.jpg")):
				img_path = os.path.join(isic_dir, f"{img_id}.jpg")
		# Show original image
		if img_path is not None:
			from PIL import Image, ImageTk
			img = Image.open(img_path)
			img = img.resize((256, 256))
			tk_img = ImageTk.PhotoImage(img)
			orig_img_canvas.configure(image=tk_img, width=256, height=256)
			orig_img_canvas.image = tk_img
		else:
			orig_img_canvas.configure(image=None)
			orig_img_canvas.image = None
		# Show preprocessed image (real image after preprocessing)
		try:
			import cv2
			import numpy as np
			from PIL import Image, ImageTk
			import sys
			sys.path.append('src/client_side/pre_processing')
			from src.client_side.pre_processing import pipeline
			preproc_img = pipeline.preprocess_image(img_path, dullrazor=True, color_constancy=True, size=256)
			preproc_img_pil = Image.fromarray(preproc_img)
			tk_preproc_img = ImageTk.PhotoImage(preproc_img_pil)
			preproc_img_canvas.configure(image=tk_preproc_img, text="", width=256, height=256)
			preproc_img_canvas.image = tk_preproc_img
		except Exception as e:
			preproc_img_canvas.configure(image=None, text=f"Preprocessing error: {e}")
			preproc_img_canvas.image = None
		# Show preprocessed features (as text, below image)
		if hasattr(hospital.local_data, "bundle") and hasattr(hospital.local_data.bundle, "x_test"):
			features = hospital.local_data.bundle.x_test[0]
			train_state["last_features"] = features
			# Optionally, show features as tooltip or in a label

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
				run_one_round()
			# After all rounds, evaluate and save
			if train_state["current_round"] >= train_state["num_rounds"]:
				status_var.set("Evaluating and saving artifacts...")
				try:
					import shutil
					from ..controller import load_config
					from src.client_side.hospital.artifacts import save_hospital_artifacts
					config = train_state["config"]
					out_dir = config.get("out_dir", "outputs")
					hospitals = train_state["hospitals"]
					eval_results = {}
					save_paths = {}
					for hid, hospital in hospitals.items():
						eval_results[hid] = hospital.evaluate()
						# Ensure hospital_id is set correctly in output
						if hasattr(hospital, 'hospital_id'):
							hospital.scope.report_output["hospital_id"] = hospital.hospital_id
						hospital_output = hospital.scope.report_output
						save_paths[hid] = save_hospital_artifacts(hospital_output=hospital_output, out_dir=out_dir)
					# Save a copy of config.yaml in out_dir
					config_src = "configs/config.yaml"
					config_dst = f"{out_dir}/config.yaml"
					shutil.copyfile(config_src, config_dst)
					# Show summary in status bar
					status_msg = "Evaluation and saving complete.\n"
					for hid, paths in save_paths.items():
						status_msg += f"{hid}: Saved to {paths['full_output_path']}\n"
					status_msg += f"Config saved to {config_dst}"
					status_var.set(status_msg)
				except Exception as e:
					import traceback
					status_var.set(f"Error during evaluation/saving: {e}")
					logging.error(f"Error during evaluation/saving: {e}\nTraceback:\n{traceback.format_exc()}")
		threading.Thread(target=run_all, daemon=True).start()
	def evaluate_and_save():
		status_var.set("Evaluating and saving artifacts...")
		try:
			import shutil
			from ..controller import load_config
			from src.client_side.hospital.artifacts import save_hospital_artifacts
			config = train_state["config"]
			out_dir = config.get("out_dir", "outputs")
			hospitals = train_state["hospitals"]
			eval_results = {}
			save_paths = {}
			for hid, hospital in hospitals.items():
				eval_results[hid] = hospital.evaluate()
				# Ensure 'hospital' key is populated for correct filename
				if hasattr(hospital, 'hospital_id'):
					if "hospital" not in hospital.scope.report_output or not isinstance(hospital.scope.report_output["hospital"], dict):
						hospital.scope.report_output["hospital"] = {}
					hospital.scope.report_output["hospital"]["hospital_id"] = hospital.hospital_id
					# Optionally set lifecycle_state if available
					if hasattr(hospital, 'metrics_store') and 'lifecycle_state' in hospital.metrics_store:
						hospital.scope.report_output["hospital"]["lifecycle_state"] = hospital.metrics_store['lifecycle_state']
				# Save all agent models for this hospital
				if hasattr(hospital.scope.agent_portfolio, 'save_all_models'):
					hospital.scope.agent_portfolio.save_all_models(out_dir, hospital.hospital_id)
				hospital_output = hospital.scope.report_output
				save_paths[hid] = save_hospital_artifacts(hospital_output=hospital_output, out_dir=out_dir)
			# Save a copy of config.yaml in out_dir
			config_src = "configs/config.yaml"
			config_dst = f"{out_dir}/config.yaml"
			shutil.copyfile(config_src, config_dst)
			status_msg = "Evaluation and saving complete.\n"
			for hid, paths in save_paths.items():
				status_msg += f"{hid}: Saved to {paths['full_output_path']}\n"
			status_msg += f"Config saved to {config_dst}"
			status_var.set(status_msg)
		except Exception as e:
			import traceback
			status_var.set(f"Error during evaluation/saving: {e}")
			logging.error(f"Error during evaluation/saving: {e}\nTraceback:\n{traceback.format_exc()}")

	# --- Divide page into left (10%) and right (90%) sections ---
	frame = ttk.Frame(parent)
	frame.pack(fill="both", expand=True)

	left_frame = ttk.Frame(frame, width=150)
	left_frame.pack(side="left", fill="y")
	left_frame.pack_propagate(False)

	right_frame = ttk.Frame(frame)
	right_frame.pack(side="left", fill="both", expand=True)

	# Use right_frame as the main content area
	content = right_frame


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



	# --- Status Bar (above images and charts) ---
	status_label = ttk.Label(content, textvariable=status_var, style="Subheading.TLabel")
	status_label.pack(anchor="w", padx=20, pady=(8, 0))

	# --- Progress Bar (below status bar) ---
	progress_var = tk.DoubleVar(value=0)
	progress_bar = ttk.Progressbar(content, variable=progress_var, maximum=1.0)
	progress_bar.pack(fill="x", padx=20, pady=(0, 8))

	# --- Unified Scrollable Area: Images + Charts ---
	charts_canvas = tk.Canvas(content, highlightthickness=0)
	charts_canvas.pack(fill="both", expand=True, padx=20, pady=(0, 0))
	charts_scrollbar = ttk.Scrollbar(content, orient="vertical", command=charts_canvas.yview)
	charts_scrollbar.place(relx=1.0, rely=0, relheight=1.0, anchor="ne")
	charts_canvas.configure(yscrollcommand=charts_scrollbar.set)
	charts_frame = ttk.Frame(charts_canvas)
	charts_frame_id = charts_canvas.create_window((0, 0), window=charts_frame, anchor="nw")

	def _on_charts_frame_configure(event):
		charts_canvas.configure(scrollregion=charts_canvas.bbox("all"))
	charts_frame.bind("<Configure>", _on_charts_frame_configure)

	def _on_canvas_configure(event):
		charts_canvas.itemconfig(charts_frame_id, width=event.width)
	charts_canvas.bind("<Configure>", _on_canvas_configure)

	# Bind mouse wheel events after charts_canvas is defined
	charts_canvas.bind_all("<MouseWheel>", _on_mousewheel)  # Windows/Mac
	charts_canvas.bind_all("<Button-4>", _on_linux_mousewheel)  # Linux scroll up
	charts_canvas.bind_all("<Button-5>", _on_linux_mousewheel)  # Linux scroll down

	# --- Image Display UI (now inside charts_frame, above charts) ---
	image_frame = ttk.Frame(charts_frame)
	image_frame.grid(row=0, column=0, sticky="nsew", padx=0, pady=(0, 0))

	orig_img_label = ttk.Label(image_frame, text="Original Image:")
	orig_img_label.grid(row=0, column=0, sticky="w")

	preproc_img_label = ttk.Label(image_frame, text="Preprocessed (features):")
	preproc_img_label.grid(row=0, column=1, sticky="w")

	orig_img_canvas = tk.Label(image_frame)
	orig_img_canvas.grid(row=1, column=0, padx=8, pady=4)

	preproc_img_canvas = tk.Label(image_frame)
	preproc_img_canvas.grid(row=1, column=1, padx=8, pady=4)

	# --- Metrics/Charts UI (below images, still inside charts_frame) ---
	metric_figures = {}
	metric_canvases = {}
	metrics_to_plot = ["global_accuracy", "global_loss"]

	for i, metric in enumerate(metrics_to_plot):
		fig, ax = plt.subplots(figsize=(8, 4.5), dpi=100)
		if metric == "global_accuracy":
			title = "Global (Federated) Accuracy per Round"
			ylabel = "Global Accuracy"
		elif metric == "global_loss":
			title = "Global Loss per Round"
			ylabel = "Global Loss"
		ax.set_title(title)
		ax.set_xlabel("Round")
		ax.set_ylabel(ylabel)
		canvas = FigureCanvasTkAgg(fig, master=charts_frame)
		canvas.get_tk_widget().grid(row=i+1, column=0, padx=8, pady=8, sticky="nsew")
		metric_figures[metric] = (fig, ax)
		metric_canvases[metric] = canvas

	# Make charts_frame expandable for both rows
	charts_frame.grid_rowconfigure(0, weight=0)  # image row
	charts_frame.grid_rowconfigure(1, weight=1)
	charts_frame.grid_rowconfigure(2, weight=1)
	charts_frame.grid_columnconfigure(0, weight=1)

	# --- End Unified Scrollable Area ---

	def update_charts(agent_metrics):
		import random
		# Only update and plot global metrics if their chart is enabled
		if not (show_accuracy_chart_var.get() or show_loss_chart_var.get()):
			return
		if "global_accuracy" not in train_state["global_metrics_history"]:
			train_state["global_metrics_history"]["global_accuracy"] = []
		if "global_loss" not in train_state["global_metrics_history"]:
			train_state["global_metrics_history"]["global_loss"] = []
		global_acc = random.uniform(0.7, 1.0)  # Replace with real value if available
		global_loss = random.uniform(0.1, 0.5)  # Replace with real value if available
		train_state["global_metrics_history"]["global_accuracy"].append(global_acc)
		train_state["global_metrics_history"]["global_loss"].append(global_loss)
		for metric in metrics_to_plot:
			if metric == "global_accuracy" and not show_accuracy_chart_var.get():
				continue
			if metric == "global_loss" and not show_loss_chart_var.get():
				continue
			fig, ax = metric_figures[metric]
			ax.clear()
			if metric == "global_accuracy":
				ax.set_title("Global Accuracy per Round")
				ax.set_xlabel("Round")
				ax.set_ylabel("Global Accuracy")
				ax.plot(range(1, len(train_state["global_metrics_history"]["global_accuracy"]) + 1),
						train_state["global_metrics_history"]["global_accuracy"], label="Global Accuracy", color="tab:blue")
				ax.legend()
			elif metric == "global_loss":
				ax.set_title("Global Loss per Round")
				ax.set_xlabel("Round")
				ax.set_ylabel("Global Loss")
				ax.plot(range(1, len(train_state["global_metrics_history"]["global_loss"]) + 1),
						train_state["global_metrics_history"]["global_loss"], label="Global Loss", color="tab:red")
				ax.legend()
			metric_canvases[metric].draw()

	# UI for image display (now inside image_frame above charts)
	orig_img_label = ttk.Label(image_frame, text="Original Image:")
	orig_img_label.grid(row=0, column=0, sticky="w")

	preproc_img_label = ttk.Label(image_frame, text="Preprocessed (features):")
	preproc_img_label.grid(row=0, column=1, sticky="w")

	orig_img_canvas = tk.Label(image_frame)
	orig_img_canvas.grid(row=1, column=0, padx=8, pady=4)

	preproc_img_canvas = tk.Label(image_frame)
	preproc_img_canvas.grid(row=1, column=1, padx=8, pady=4)


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
		progress_var.set(0)
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
			# Update progress bar
			if train_state["num_rounds"] > 0:
				progress_var.set(round_idx / train_state["num_rounds"])
			else:
				progress_var.set(0)
			status_var.set(f"Training round {round_idx}...")
			logging.info(f"Training round {round_idx}...")
			# Call controller to run one round
			round_output, agent_metrics = run_one_training_round(
				train_state["orchestrator"], train_state["hospitals"], round_idx, for_training=True
			)
			update_charts(agent_metrics)
			update_image_previews(round_idx)
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


	# --- Control Panel (Vertical Button Bar in left_frame) ---
	button_bar = ttk.Frame(left_frame)
	button_bar.pack(fill="y", pady=20)

	init_btn = ttk.Button(button_bar, text="Initialize Training", command=on_initialize_click)
	init_btn.pack(fill="x", pady=(0, 10))

	next_round_btn = ttk.Button(button_bar, text="Next Round", command=on_next_round_click)
	next_round_btn.pack(fill="x", pady=(0, 10))

	train_all_btn = ttk.Button(button_bar, text="Train All Rounds", command=train_all_rounds)
	train_all_btn.pack(fill="x", pady=(0, 10))

	stop_btn = ttk.Button(button_bar, text="Stop Training", command=stop_training)
	stop_btn.pack(fill="x", pady=(0, 10))

	continue_btn = ttk.Button(button_bar, text="Continue Training", command=continue_training)
	continue_btn.pack(fill="x", pady=(0, 10))

	eval_save_btn = ttk.Button(button_bar, text="Evaluate && Save", command=lambda: threading.Thread(target=evaluate_and_save, daemon=True).start())
	eval_save_btn.pack(fill="x", pady=(0, 10))

	save_models_btn = ttk.Button(button_bar, text="Save Models", command=lambda: threading.Thread(target=save_models, daemon=True).start())
	save_models_btn.pack(fill="x", pady=(0, 10))

	load_models_btn = ttk.Button(button_bar, text="Load Models", command=lambda: threading.Thread(target=load_models, daemon=True).start())
	load_models_btn.pack(fill="x", pady=(0, 10))
	# --- Checkboxes for toggling charts and image viewer ---
	checkbox_frame = ttk.LabelFrame(left_frame, text="Display Options")
	checkbox_frame.pack(fill="x", padx=8, pady=(20, 0))

	show_accuracy_cb = ttk.Checkbutton(checkbox_frame, text="Accuracy Chart", variable=show_accuracy_chart_var, onvalue=True, offvalue=False)
	show_accuracy_cb.pack(anchor="w", pady=(2, 2), padx=4)
	show_loss_cb = ttk.Checkbutton(checkbox_frame, text="Loss Chart", variable=show_loss_chart_var, onvalue=True, offvalue=False)
	show_loss_cb.pack(anchor="w", pady=(2, 2), padx=4)
	show_images_cb = ttk.Checkbutton(checkbox_frame, text="Image Viewer", variable=show_images_var, onvalue=True, offvalue=False)
	show_images_cb.pack(anchor="w", pady=(2, 2), padx=4)

	def update_display_visibility(*args):
		# Images
		if show_images_var.get():
			image_frame.grid()
		else:
			image_frame.grid_remove()
		# Charts
		for metric in metrics_to_plot:
			canvas = metric_canvases[metric]
			if metric == "global_accuracy":
				if show_accuracy_chart_var.get():
					canvas.get_tk_widget().grid()
				else:
					canvas.get_tk_widget().grid_remove()
			elif metric == "global_loss":
				if show_loss_chart_var.get():
					canvas.get_tk_widget().grid()
				else:
					canvas.get_tk_widget().grid_remove()

	show_accuracy_chart_var.trace_add('write', update_display_visibility)
	show_loss_chart_var.trace_add('write', update_display_visibility)
	show_images_var.trace_add('write', update_display_visibility)

	# Initial call to set visibility
	update_display_visibility()
	# --- End Control Panel ---


	# Bind window close event to stop training threads
	parent.winfo_toplevel().protocol("WM_DELETE_WINDOW", on_close)

	return frame
