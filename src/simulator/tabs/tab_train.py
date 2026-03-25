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
	}

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
		if not train_state["hospitals"] or not train_state["orchestrator"]:
			status_var.set("Please initialize training first.")
			return
		if train_state["current_round"] >= train_state["num_rounds"]:
			status_var.set("All rounds completed.")
			return
		try:
			train_state["current_round"] += 1
			round_idx = train_state["current_round"]
			status_var.set(f"Training round {round_idx}...")
			logging.info(f"Training round {round_idx}...")
			# For demo: use first hospital
			hospital = next(iter(train_state["hospitals"].values()))
			hospital.train()
			# Get a different image_id for each round (cycle through test_ids)
			if hasattr(hospital, "local_data") and hospital.local_data is not None:
				img_id = None
				if hasattr(hospital.local_data, "test_ids") and len(hospital.local_data.test_ids) > 0:
					idx = (round_idx - 1) % len(hospital.local_data.test_ids)
					img_id = hospital.local_data.test_ids[idx]
				train_state["last_image_id"] = img_id
				# Find original image path (HAM or ISIC)
				img_path = None
				if img_id is not None:
					import os
					# Try HAM first
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
					img = img.resize((128, 128))
					tk_img = ImageTk.PhotoImage(img)
					orig_img_canvas.configure(image=tk_img)
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
					preproc_img = pipeline.preprocess_image(img_path, dullrazor=True, color_constancy=True, size=128)
					# preproc_img is np.ndarray, convert to PIL Image
					preproc_img_pil = Image.fromarray(preproc_img)
					tk_preproc_img = ImageTk.PhotoImage(preproc_img_pil)
					preproc_img_canvas.configure(image=tk_preproc_img, text="")
					preproc_img_canvas.image = tk_preproc_img
				except Exception as e:
					preproc_img_canvas.configure(image=None, text=f"Preprocessing error: {e}")
					preproc_img_canvas.image = None

				# Show preprocessed features (as text, below image)
				if hasattr(hospital.local_data, "bundle") and hasattr(hospital.local_data.bundle, "x_test"):
					features = hospital.local_data.bundle.x_test[0]
					train_state["last_features"] = features
					# Optionally, show features as tooltip or in a label
				# ...existing code...
			# Show agent metrics (after training)
			metrics_text.config(state="normal")
			metrics_text.delete("1.0", tk.END)
			if hasattr(hospital, "scope") and hasattr(hospital.scope, "agent_portfolio"):
				for cancer_type in hospital.scope.agent_portfolio.cancer_types:
					agent = hospital.scope.agent_portfolio.get_agent(cancer_type)
					metrics_text.insert(tk.END, f"{cancer_type}: pattern={agent.thinking_pattern_name}\n")
			metrics_text.config(state="disabled")
			# Continue with federated round
			local_updates = {hid: h.get_local_update(for_training=True) for hid, h in train_state["hospitals"].items()}
			round_output = train_state["orchestrator"].run_round(
				round_index=round_idx,
				local_updates=local_updates,
			)
			train_state["orchestrator"].broadcast_global_state(train_state["hospitals"], round_output.global_state)
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

	status_label = ttk.Label(content, textvariable=status_var, style="Subheading.TLabel")
	status_label.pack(anchor="w", padx=20, pady=(8, 0))

	return frame
