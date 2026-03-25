"""Configuration tab module."""

import tkinter as tk
from tkinter import ttk
import yaml
from pathlib import Path

from ..ui_kit import add_workflow_section


# ========================
# FIELD BUILDER
# ========================
def _add_field(parent, label, var, field_type="entry", options=None):
	row = ttk.Frame(parent)
	row.pack(fill="x", pady=2)

	ttk.Label(row, text=label, width=22, anchor="w").pack(side="left")

	if field_type == "entry":
		widget = ttk.Entry(row, textvariable=var)
		widget.pack(side="right", fill="x", expand=True)

	elif field_type == "combo":
		widget = ttk.Combobox(row, textvariable=var, values=options, state="readonly")
		widget.pack(side="right", fill="x", expand=True)

	elif field_type == "check":
		widget = ttk.Checkbutton(row, variable=var)
		widget.pack(side="right")

	return var


# ========================
# MAIN TAB
# ========================

def build_configuration_tab(parent: ttk.Notebook) -> ttk.Frame:
	page = ttk.Frame(parent, style="App.TFrame")

	# Add custom button styles for color
	style = ttk.Style()
	style.configure("Colored.TButton", background="#0f766e", foreground="#ffffff", font=("DejaVu Sans", 10, "bold"))
	style.map(
		"Colored.TButton",
		background=[("active", "#115e59"), ("pressed", "#0d4d4d")],
		foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
	)
	style.configure("Danger.TButton", background="#be123c", foreground="#ffffff", font=("DejaVu Sans", 10, "bold"))
	style.map(
		"Danger.TButton",
		background=[("active", "#991b1b"), ("pressed", "#7f1d1d")],
		foreground=[("active", "#ffffff"), ("pressed", "#ffffff")],
	)

	# ========================
	# LOAD CONFIG using shared config loader
	# ========================
	from configs.config_loader import load_config
	try:
		config = load_config()
	except Exception:
		import tkinter.messagebox as mb
		mb.showerror("Missing config.yaml", "The configuration file 'configs/config.yaml' was not found or could not be loaded. Please ensure it exists in the configs directory.")
		return page

	# ========================
	# SCROLLABLE LAYOUT
	# ========================
	canvas = tk.Canvas(page, highlightthickness=0, borderwidth=0, bg="#edf2f7")
	scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
	canvas.configure(yscrollcommand=scrollbar.set)

	canvas.grid(row=0, column=0, sticky="nsew")
	scrollbar.grid(row=0, column=1, sticky="ns")
	page.grid_rowconfigure(0, weight=1)
	page.grid_columnconfigure(0, weight=1)

	content = ttk.Frame(canvas, style="App.TFrame")
	window = canvas.create_window((0, 0), window=content, anchor="nw")

	def _on_content_configure(_: tk.Event):
		canvas.configure(scrollregion=canvas.bbox("all"))

	def _on_canvas_configure(event: tk.Event):
		canvas.itemconfigure(window, width=event.width)

	content.bind("<Configure>", _on_content_configure)
	canvas.bind("<Configure>", _on_canvas_configure)

	# Cross-platform mouse wheel scrolling
	def _on_mousewheel(event: tk.Event):
		# Windows and MacOS
		if event.num == 4 or event.delta > 0:
			canvas.yview_scroll(-1, "units")
		elif event.num == 5 or event.delta < 0:
			canvas.yview_scroll(1, "units")

	# Linux uses Button-4 and Button-5, Windows/Mac use MouseWheel
	canvas.bind_all("<MouseWheel>", _on_mousewheel, add="+")  # Windows/Mac
	canvas.bind_all("<Button-4>", _on_mousewheel, add="+")    # Linux scroll up
	canvas.bind_all("<Button-5>", _on_mousewheel, add="+")    # Linux scroll down

	def _set_fixed_height(frame, height=400):
		frame.configure(height=height)
		frame.pack_propagate(False)

	def _center_section_title(section_body):
		section_frame = section_body.master
		for child in section_frame.winfo_children():
			if isinstance(child, ttk.Label):
				child.configure(anchor="center", justify="center")
				child.pack_configure(anchor="center", fill="x")

	section_viz_frames = {}

	def _add_pair_row(row, setting_title, viz_title, key=None):
		# Set grid weights for 2 columns: settings (left) and viz (right) equal width
		content.grid_columnconfigure(0, weight=1)
		content.grid_columnconfigure(1, weight=2)

		setting_outer = ttk.Frame(content, style="App.TFrame")
		setting_outer.grid(row=row, column=0, sticky="nsew", padx=(14, 7), pady=(0, 8))
		setting_section = add_workflow_section(setting_outer, title=setting_title)
		_center_section_title(setting_section)
		_set_fixed_height(setting_section)

		viz_outer = ttk.Frame(content, style="App.TFrame")
		viz_outer.grid(row=row, column=1, sticky="nsew", padx=(7, 14), pady=(0, 8))

		viz_label = ttk.Label(
			viz_outer,
			text=f"{viz_title} - VISUALIZATION",
			font=("DejaVu Sans", 10, "bold"),
			foreground="#0f766e",
			anchor="center",
		)
		viz_label.pack(fill="x", pady=(0, 6))

		viz_body = ttk.Frame(viz_outer, style="App.TFrame")
		viz_body.pack(fill="x", expand=True)

		_set_fixed_height(viz_body)

		if key:
			section_viz_frames[key] = viz_body

		return setting_section, viz_body

	def _add_divider_row(row, text):
		ttk.Label(
			content,
			text=text,
			font=("DejaVu Sans", 8, "bold"),
			foreground="#0f766e",
			anchor="center",
		).grid(row=row, column=0, columnspan=2, sticky="ew", pady=(6, 4))

	# ========================
	# SECTIONS
	# ========================
	row = 0
	from ..ui_kit import add_card
	hospital_section, _ = _add_pair_row(row, "HOSPITAL", "HOSPITAL")
	row += 1

	# --- Professional Card for Hospital Settings ---
	hospital_card = ttk.Frame(hospital_section, style="Card.TFrame", padding=14)
	hospital_card.pack(fill="x", padx=10, pady=(0, 12))
	patient_section, _ = _add_pair_row(row, "PATIENT DATA", "PATIENT DATA")
	row += 1
	agent_section, _ = _add_pair_row(row, "AGENTIC AI LAYER", "AGENTIC AI LAYER")
	row += 1
	local_meta_section, _ = _add_pair_row(row, "LOCAL META-AGENT", "LOCAL META")
	row += 1
	privacy_section, _ = _add_pair_row(row, "PRIVACY", "PRIVACY")
	row += 1

	_add_divider_row(row, "▼ FEDERATION ▼")
	row += 1

	fed_section, _ = _add_pair_row(row, "FEDERATED AGGREGATOR", "FEDERATION")
	row += 1
	global_meta_section, _ = _add_pair_row(row, "GLOBAL META-AGENT", "GLOBAL META")

	# ========================
	# HOSPITAL
	# ========================
	hospital_vars = {}
	# Default to HOSPITAL1 if not present in config
	default_hosp = config.get("hospital_ids", "").strip()
	if not default_hosp:
		default_hosp = "HOSPITAL1"
	hospital_vars["hospital_ids"] = _add_field(hospital_card, "Hospital IDs", tk.StringVar(value=default_hosp))
	hospital_vars["num_agents"] = _add_field(hospital_card, "Agents", tk.IntVar(value=config.get("num_agents_per_hospital", 4)))
	hospital_vars["random_seed"] = _add_field(hospital_card, "random_seed", tk.IntVar(value=config.get("random_seed", 42)))

	# Hospital counter and Add/Remove buttons (vertical layout)
	counter_frame = ttk.Frame(hospital_card)
	counter_frame.pack(fill="x", pady=(8, 0))

	hospital_count_var = tk.StringVar()
	def update_hospital_count(*_):
		ids = [x.strip() for x in hospital_vars["hospital_ids"].get().split(",") if x.strip()]
		hospital_count_var.set(f"Total Hospitals: {len(ids)}")

	hospital_count_label = ttk.Label(counter_frame, textvariable=hospital_count_var, font=("DejaVu Sans", 10, "bold"), foreground="#0f766e")
	hospital_count_label.pack(side="top", pady=(0, 6))

	button_frame = ttk.Frame(counter_frame)
	button_frame.pack(side="top")

	def add_hospital():
		current = hospital_vars["hospital_ids"].get()
		ids = [x.strip() for x in current.split(",") if x.strip()]
		# Find the next available integer suffix for HOSPITALn
		i = 1
		while f"HOSPITAL{i}" in ids:
			i += 1
		new_id = f"HOSPITAL{i}"
		ids.append(new_id)
		hospital_vars["hospital_ids"].set(", ".join(ids))
		update_hospital_count()

	def remove_hospital():
		current = hospital_vars["hospital_ids"].get()
		ids = [x.strip() for x in current.split(",") if x.strip()]
		if len(ids) > 1:
			ids.pop()
			hospital_vars["hospital_ids"].set(", ".join(ids))
		update_hospital_count()

	add_btn = ttk.Button(button_frame, text="Add Hospital", command=add_hospital, style="Colored.TButton")
	add_btn.pack(side="left", padx=6, ipadx=8, ipady=2)
	remove_btn = ttk.Button(button_frame, text="Remove Hospital", command=remove_hospital, style="Danger.TButton")
	remove_btn.pack(side="left", padx=6, ipadx=8, ipady=2)

	# Update count when hospital_ids changes (manual and programmatic)
	hospital_vars["hospital_ids"].trace_add("write", lambda *_: update_hospital_count())
	update_hospital_count()

	# ========================
	# PATIENT
	# ========================
	patient_vars = {}
	# Add dataset_type field with extra vertical padding
	dataset_type_var = tk.StringVar(value="HAM10000")
	patient_vars["dataset_type"] = _add_field(
		patient_section,
		"Dataset",
		dataset_type_var,
		"combo",
		["HAM10000", "ISIC2019"],
	)
	# Add vertical padding below the dataset_type field
	last_row = patient_section.winfo_children()[-1]
	last_row.pack_configure(pady=(0, 16))
	import tkinter.filedialog as fd
	# HAM CSV
	ham_var = tk.StringVar(value=config.get("ham_csv", ""))
	patient_vars["ham_csv"] = _add_field(patient_section, "HAM CSV", ham_var)
	ham_browse = ttk.Button(patient_section, text="Browse", command=lambda: ham_var.set(fd.askopenfilename(title="Select HAM CSV", filetypes=[("CSV Files", "*.csv"), ("All Files", "*")])))
	ham_browse.pack(fill="x", padx=22, pady=(0, 16))

	# ISIC CSV
	isic_var = tk.StringVar(value=config.get("isic_csv", ""))
	patient_vars["isic_csv"] = _add_field(patient_section, "ISIC CSV", isic_var)
	isic_browse = ttk.Button(patient_section, text="Browse", command=lambda: isic_var.set(fd.askopenfilename(title="Select ISIC CSV", filetypes=[("CSV Files", "*.csv"), ("All Files", "*")])))
	isic_browse.pack(fill="x", padx=22, pady=(0, 16))

	# Output Dir
	out_var = tk.StringVar(value=config.get("out_dir", "outputs"))
	patient_vars["out_dir"] = _add_field(patient_section, "Output Dir", out_var)
	out_browse = ttk.Button(patient_section, text="Browse", command=lambda: out_var.set(fd.askdirectory(title="Select Output Directory")))
	out_browse.pack(fill="x", padx=22, pady=(0, 16))
	

	# ========================
	# AGENTS
	# ========================
	agents_cfg = config.get("agents", {})
	patterns = agents_cfg.get("patterns", {}).get("default_mapping", {})

	agent_vars = {}
	agent_vars["switch"] = _add_field(
		agent_section,
		"Dynamic Switch",
		tk.BooleanVar(value=agents_cfg.get("allow_dynamic_switch", True)),
		"check",
	)

	# Agent Types (multi-select listbox)
	agent_types = agents_cfg.get("types", ["akiec_agent", "bcc_agent", "melanoma_agent", "scc_agent"])
	agent_types_var = tk.Variable(value=agent_types)
	ttk.Label(agent_section, text="Agent Types", anchor="w").pack(fill="x", padx=2, pady=(8,2))
	agent_types_listbox = tk.Listbox(agent_section, listvariable=agent_types_var, selectmode="multiple", height=4, exportselection=False)
	for i, t in enumerate(["akiec_agent", "bcc_agent", "melanoma_agent", "scc_agent"]):
		if t in agent_types:
			agent_types_listbox.selection_set(i)
	agent_types_listbox.pack(fill="x", padx=8, pady=(0,8))
	agent_vars["types_listbox"] = agent_types_listbox

	# Patterns Available (multi-select listbox)
	patterns_available = agents_cfg.get("patterns", {}).get("available", ["rule_based", "bayesian", "deep_learning"])
	patterns_available_var = tk.Variable(value=patterns_available)
	ttk.Label(agent_section, text="Patterns Available", anchor="w").pack(fill="x", padx=2, pady=(8,2))
	patterns_available_listbox = tk.Listbox(agent_section, listvariable=patterns_available_var, selectmode="multiple", height=4, exportselection=False)
	for i, t in enumerate(["rule_based", "bayesian", "deep_learning"]):
		if t in patterns_available:
			patterns_available_listbox.selection_set(i)
	patterns_available_listbox.pack(fill="x", padx=8, pady=(0,8))
	agent_vars["patterns_available_listbox"] = patterns_available_listbox

	for k in ["BCC", "SCC", "MELANOMA", "AKIEC"]:
		agent_vars[k] = _add_field(
			agent_section,
			k,
			tk.StringVar(value=patterns.get(k, "")),
			"combo",
			["rule_based", "bayesian", "deep_learning"],
		)

	# ========================
	# LOCAL META
	# ========================
	local_cfg = config.get("meta_agent", {}).get("local", {})
	local_vars = {}
	local_vars["enable"] = _add_field(local_meta_section, "Enable", tk.BooleanVar(value=local_cfg.get("enable", True)), "check")
	local_vars["strategy"] = _add_field(
		local_meta_section,
		"Strategy",
		tk.StringVar(value=local_cfg.get("weighting_strategy", "adaptive")),
		"combo",
		["static", "adaptive"],
	)
	# Score Function checkboxes
	score_cfg = local_cfg.get("score_function", {})
	local_vars["use_auc"] = tk.BooleanVar(value=score_cfg.get("use_auc", True))
	local_vars["use_f1"] = tk.BooleanVar(value=score_cfg.get("use_f1", True))
	local_vars["use_confidence"] = tk.BooleanVar(value=score_cfg.get("use_confidence", True))
	score_frame = ttk.LabelFrame(local_meta_section, text="Score Function", padding=(8,4))
	score_frame.pack(fill="x", padx=4, pady=(8,4))
	ttk.Checkbutton(score_frame, text="Use AUC", variable=local_vars["use_auc"]).pack(anchor="w", padx=6, pady=2)
	ttk.Checkbutton(score_frame, text="Use F1", variable=local_vars["use_f1"]).pack(anchor="w", padx=6, pady=2)
	ttk.Checkbutton(score_frame, text="Use Confidence", variable=local_vars["use_confidence"]).pack(anchor="w", padx=6, pady=2)

	# ========================
	# PRIVACY
	# ========================
	privacy_cfg = config.get("privacy", {})
	dp = privacy_cfg.get("differential_privacy", {})

	privacy_vars = {}
	privacy_vars["dp"] = _add_field(privacy_section, "DP Enabled", tk.BooleanVar(value=dp.get("enabled", True)), "check")
	privacy_vars["epsilon"] = _add_field(privacy_section, "Epsilon", tk.DoubleVar(value=dp.get("epsilon", 1.0)))
	privacy_vars["secure"] = _add_field(privacy_section, "Secure Agg", tk.BooleanVar(value=privacy_cfg.get("secure_aggregation", True)), "check")

	# ========================
	# FEDERATION
	# ========================
	fed_cfg = config.get("federation", {})
	fedprox = fed_cfg.get("fedprox", {})
	adaptive = fed_cfg.get("adaptive", {})

	fed_vars = {}
	fed_vars["algo"] = _add_field(
		fed_section,
		"Algorithm",
		tk.StringVar(value=fed_cfg.get("aggregation_algorithm", "fedavg")),
		"combo",
		["fedavg", "fedprox", "adaptive"],
	)
	# FedProx mu
	fed_vars["mu"] = _add_field(fed_section, "FedProx Mu", tk.DoubleVar(value=fedprox.get("mu", 0.1)))

	fed_vars["alpha"] = _add_field(fed_section, "alpha", tk.DoubleVar(value=adaptive.get("alpha", 0.5)))
	fed_vars["beta"] = _add_field(fed_section, "beta", tk.DoubleVar(value=adaptive.get("beta", 0.3)))
	fed_vars["gamma"] = _add_field(fed_section, "gamma", tk.DoubleVar(value=adaptive.get("gamma", 0.2)))
	fed_vars["auc_weight"] = _add_field(fed_section, "AUC Weight", tk.DoubleVar(value=adaptive.get("auc_weight", 0.75)))
	fed_vars["f1_weight"] = _add_field(fed_section, "F1 Weight", tk.DoubleVar(value=adaptive.get("f1_weight", 0.25)))
	fed_vars["hospital_weighting"] = _add_field(
		fed_section,
		"Hospital Weighting",
		tk.StringVar(value=adaptive.get("hospital_weighting", "dynamic")),
		"combo",
		["static", "dynamic"],
	)

	# ========================
	# GLOBAL META
	# ========================
	global_cfg = config.get("meta_agent", {}).get("global", {})

	global_vars = {}
	global_vars["enable"] = _add_field(global_meta_section, "Enable", tk.BooleanVar(value=global_cfg.get("enable", True)), "check")
	global_vars["anomaly"] = _add_field(global_meta_section, "Anomaly", tk.BooleanVar(value=global_cfg.get("anomaly_detection", True)), "check")
	global_vars["trust"] = _add_field(global_meta_section, "Trust", tk.BooleanVar(value=global_cfg.get("trust_reweighting", True)), "check")

	


	# ========================
	# Missed Settings (Redesigned) with Tracking & Logging
	# ========================
	missed_section = ttk.LabelFrame(content, text="Other Settings", style="App.TLabelframe")
	missed_section.grid(row=row+1, column=0, columnspan=2, sticky="ew", padx=18, pady=(12, 8))

	# Section headers
	sim_header = ttk.Label(missed_section, text="Simulation Settings", font=("DejaVu Sans", 9, "bold"), foreground="#0f766e")
	sim_header.grid(row=0, column=0, columnspan=3, sticky="w", padx=4, pady=(4, 2))

	# Simulation settings and explanations
	simulation_vars = {}
	simulation_cfg = config.get("simulation", {})
	sim_settings = [
		("simulate_multi", "Simulate Multi", tk.BooleanVar(value=simulation_cfg.get("simulate_multi", True)), "check", "Run multiple simulations in parallel (recommended for federated experiments)."),
		("compare_all", "Compare All", tk.BooleanVar(value=simulation_cfg.get("compare_all", True)), "check", "Compare different aggregation algorithms (FedAvg, FedProx, Adaptive) in the same run."),
		("num_rounds", "Num Rounds", tk.IntVar(value=simulation_cfg.get("num_rounds", 20)), "entry", "Number of communication rounds for the simulation.")
	]
	for i, (key, label, var, field_type, explanation) in enumerate(sim_settings):
		ttk.Label(missed_section, text=label, width=18, anchor="w").grid(row=i+1, column=0, sticky="w", padx=(8,2), pady=2)
		if field_type == "check":
			widget = ttk.Checkbutton(missed_section, variable=var)
			widget.grid(row=i+1, column=1, sticky="w", padx=(0,2))
		else:
			widget = ttk.Entry(missed_section, textvariable=var, width=8)
			widget.grid(row=i+1, column=1, sticky="w", padx=(0,2))
		ttk.Label(missed_section, text=explanation, font=("DejaVu Sans", 8), wraplength=320, anchor="w", justify="left").grid(row=i+1, column=2, sticky="w", padx=(6,2))
		simulation_vars[key] = var

	# Output section header
	out_start = len(sim_settings) + 2
	out_header = ttk.Label(missed_section, text="Output Settings", font=("DejaVu Sans", 9, "bold"), foreground="#0f766e")
	out_header.grid(row=out_start, column=0, columnspan=3, sticky="w", padx=4, pady=(10, 2))

	# Output settings and explanations
	output_vars = {}
	output_cfg = config.get("output", {})
	output_settings = [
		("save_global_model", "Save Global Model", tk.BooleanVar(value=output_cfg.get("save_global_model", True)), "check", "Save the final global model after training."),
		("save_local_models", "Save Local Models", tk.BooleanVar(value=output_cfg.get("save_local_models", False)), "check", "Save the final models for each hospital/agent."),
		("save_metrics", "Save Metrics", tk.BooleanVar(value=output_cfg.get("save_metrics", True)), "check", "Save evaluation metrics (accuracy, AUC, etc.) for each run.")
	]
	for j, (key, label, var, field_type, explanation) in enumerate(output_settings):
		ttk.Label(missed_section, text=label, width=18, anchor="w").grid(row=out_start+1+j, column=0, sticky="w", padx=(8,2), pady=2)
		widget = ttk.Checkbutton(missed_section, variable=var)
		widget.grid(row=out_start+1+j, column=1, sticky="w", padx=(0,2))
		ttk.Label(missed_section, text=explanation, font=("DejaVu Sans", 8), wraplength=320, anchor="w", justify="left").grid(row=out_start+1+j, column=2, sticky="w", padx=(6,2))
		output_vars[key] = var

	# Tracking & Logging section header
	track_start = out_start + len(output_settings) + 2
	track_header = ttk.Label(missed_section, text="Tracking & Logging", font=("DejaVu Sans", 9, "bold"), foreground="#0f766e")
	track_header.grid(row=track_start, column=0, columnspan=3, sticky="w", padx=4, pady=(10, 2))

	# Tracking & Logging settings and explanations
	tracking_vars = {}
	tracking_cfg = config.get("tracking", {})
	tracking_settings = [
		("track_per_agent", "Track Per Agent", tk.BooleanVar(value=tracking_cfg.get("track_per_agent", True)), "check", "Track metrics for each agent individually."),
		("track_per_hospital", "Track Per Hospital", tk.BooleanVar(value=tracking_cfg.get("track_per_hospital", True)), "check", "Track metrics for each hospital as a whole."),
		("track_confidence", "Track Confidence", tk.BooleanVar(value=tracking_cfg.get("track_confidence", True)), "check", "Track model confidence scores during evaluation."),
		("track_stability", "Track Stability", tk.BooleanVar(value=tracking_cfg.get("track_stability", True)), "check", "Track stability of model predictions over rounds."),
		("save_logs", "Save Logs", tk.BooleanVar(value=tracking_cfg.get("save_logs", True)), "check", "Save detailed logs of the simulation run."),
		("log_dir", "Log Directory", tk.StringVar(value=tracking_cfg.get("log_dir", "outputs/logs")), "entry", "Directory where logs will be saved.")
	]
	for k, (key, label, var, field_type, explanation) in enumerate(tracking_settings):
		ttk.Label(missed_section, text=label, width=18, anchor="w").grid(row=track_start+1+k, column=0, sticky="w", padx=(8,2), pady=2)
		if field_type == "check":
			widget = ttk.Checkbutton(missed_section, variable=var)
			widget.grid(row=track_start+1+k, column=1, sticky="w", padx=(0,2))
		else:
			widget = ttk.Entry(missed_section, textvariable=var, width=18)
			widget.grid(row=track_start+1+k, column=1, sticky="w", padx=(0,2))
		ttk.Label(missed_section, text=explanation, font=("DejaVu Sans", 8), wraplength=320, anchor="w", justify="left").grid(row=track_start+1+k, column=2, sticky="w", padx=(6,2))
		tracking_vars[key] = var

	# ========================
	# STORE ALL
	# ========================
	page.config_vars = {
		"hospital": hospital_vars,
		"patient": patient_vars,
		"agent": agent_vars,
		"local_meta": local_vars,
		"privacy": privacy_vars,
		"federation": fed_vars,
		"global_meta": global_vars,
		"simulation": simulation_vars,
		"output": output_vars,
		"tracking": tracking_vars,
	}
	# ========================
	# save / load configuration
	# ========================

	def _gather_vars(var_dict):
		result = {}
		for k, v in var_dict.items():
			if isinstance(v, dict):
				result[k] = _gather_vars(v)
			else:
				val = v.get()
				# Convert string numbers to int/float if possible
				if isinstance(v, tk.IntVar):
					try:
						val = int(val)
					except Exception:
						pass
				elif isinstance(v, tk.DoubleVar):
					try:
						val = float(val)
					except Exception:
						pass
				elif isinstance(v, tk.BooleanVar):
					val = bool(val)
				result[k] = val
		return result

	import tkinter.filedialog as fd
	import tkinter.messagebox as mb
	try:
		from ruamel.yaml import YAML
	except ImportError:
		mb.showerror("Missing Dependency", "ruamel.yaml is required for commented YAML export. Please install it via pip.")
		return

	def save_config():
		# Ask user for file name and destination
		file_path = fd.asksaveasfilename(
			title="Save Configuration As",
			defaultextension=".yaml",
			filetypes=[("YAML Files", "*.yaml"), ("All Files", "*")],
			initialfile="config.yaml"
		)
		if not file_path:
			return

		# Build config with comments using ruamel.yaml
		yaml = YAML()
		yaml.indent(mapping=2, sequence=4, offset=2)
		from ruamel.yaml.comments import CommentedMap
		d = CommentedMap()

		# DATA SOURCES
		d.yaml_set_comment_before_after_key('ham_csv', before="# ========================\n# DATA SOURCES\n# ========================")
		d['ham_csv'] = patient_vars['ham_csv'].get()
		d['isic_csv'] = patient_vars['isic_csv'].get()
		d['out_dir'] = patient_vars['out_dir'].get()

		# SYSTEM SETUP
		d.yaml_set_comment_before_after_key('hospital_ids', before="\n# ========================\n# SYSTEM SETUP\n# ========================")
		d['hospital_ids'] = hospital_vars['hospital_ids'].get()
		d['num_agents_per_hospital'] = hospital_vars['num_agents'].get()
		d['random_seed'] = hospital_vars['random_seed'].get()

		# FEDERATED LEARNING
		d.yaml_set_comment_before_after_key('federation', before="\n# ========================\n# FEDERATED LEARNING\n# ========================")
		federation = CommentedMap()
		federation['aggregation_algorithm'] = fed_vars['algo'].get()
		fedprox = CommentedMap()
		fedprox['mu'] = fed_vars['mu'].get()
		federation['fedprox'] = fedprox
		adaptive = CommentedMap()
		adaptive['alpha'] = fed_vars['alpha'].get()
		adaptive['beta'] = fed_vars['beta'].get()
		adaptive['gamma'] = fed_vars['gamma'].get()
		adaptive['auc_weight'] = fed_vars['auc_weight'].get()
		adaptive['f1_weight'] = fed_vars['f1_weight'].get()
		adaptive['hospital_weighting'] = fed_vars['hospital_weighting'].get()
		federation['adaptive'] = adaptive
		d['federation'] = federation

		# AGENT CONFIGURATION
		d.yaml_set_comment_before_after_key('agents', before="\n# ========================\n# AGENT CONFIGURATION\n# ========================")
		agents = CommentedMap()
		# Save agent types from listbox
		selected_types = ["akiec_agent", "bcc_agent", "melanoma_agent", "scc_agent"]
		types_selected = [selected_types[i] for i in agent_vars["types_listbox"].curselection()]
		agents['types'] = types_selected
		patterns = CommentedMap()
		# Save patterns available from listbox
		all_patterns = ["rule_based", "bayesian", "deep_learning"]
		patterns_selected = [all_patterns[i] for i in agent_vars["patterns_available_listbox"].curselection()]
		patterns['available'] = patterns_selected
		patterns['default_mapping'] = CommentedMap({k: agent_vars[k].get() for k in ["BCC", "SCC", "MELANOMA", "AKIEC"]})
		patterns['allow_dynamic_switch'] = agent_vars['switch'].get()
		agents['patterns'] = patterns
		d['agents'] = agents

		# META-AGENT CONTROL
		d.yaml_set_comment_before_after_key('meta_agent', before="\n# ========================\n# META-AGENT CONTROL\n# ========================")
		meta_agent = CommentedMap()
		local = CommentedMap()
		local['enable'] = local_vars['enable'].get()
		local['weighting_strategy'] = local_vars['strategy'].get()
		score_function = CommentedMap()
		score_function['use_auc'] = local_vars['use_auc'].get()
		score_function['use_f1'] = local_vars['use_f1'].get()
		score_function['use_confidence'] = local_vars['use_confidence'].get()
		local['score_function'] = score_function
		meta_agent['local'] = local
		global_m = CommentedMap()
		global_m['enable'] = global_vars['enable'].get()
		global_m['anomaly_detection'] = global_vars['anomaly'].get()
		global_m['trust_reweighting'] = global_vars['trust'].get()
		meta_agent['global'] = global_m
		d['meta_agent'] = meta_agent

		# PRIVACY & SECURITY
		d.yaml_set_comment_before_after_key('privacy', before="\n# ========================\n# PRIVACY & SECURITY\n# ========================")
		privacy = CommentedMap()
		dp = CommentedMap()
		dp['enabled'] = privacy_vars['dp'].get()
		dp['epsilon'] = privacy_vars['epsilon'].get()
		privacy['differential_privacy'] = dp
		privacy['secure_aggregation'] = privacy_vars['secure'].get()
		d['privacy'] = privacy

		# TRACKING & LOGGING
		d.yaml_set_comment_before_after_key('tracking', before="\n# ========================\n# TRACKING & LOGGING\n# ========================")
		tracking = CommentedMap()
		tracking['track_per_agent'] = tracking_vars['track_per_agent'].get()
		tracking['track_per_hospital'] = tracking_vars['track_per_hospital'].get()
		tracking['track_confidence'] = tracking_vars['track_confidence'].get()
		tracking['track_stability'] = tracking_vars['track_stability'].get()
		tracking['save_logs'] = tracking_vars['save_logs'].get()
		tracking['log_dir'] = tracking_vars['log_dir'].get()
		d['tracking'] = tracking

		# SIMULATION (Missed Settings)
		d.yaml_set_comment_before_after_key('simulation', before="\n# ========================\n# EXPERIMENT CONTROL\n# ========================")
		simulation = CommentedMap()
		simulation['simulate_multi'] = simulation_vars['simulate_multi'].get()
		simulation['compare_all'] = simulation_vars['compare_all'].get()
		simulation['num_rounds'] = simulation_vars['num_rounds'].get()
		d['simulation'] = simulation

		# OUTPUT (Missed Settings)
		d.yaml_set_comment_before_after_key('output', before="\n# ========================\n# OUTPUT CONTROL\n# ========================")
		output = CommentedMap()
		output['save_global_model'] = output_vars['save_global_model'].get()
		output['save_local_models'] = output_vars['save_local_models'].get()
		output['save_metrics'] = output_vars['save_metrics'].get()
		d['output'] = output

		# Write to file
		try:
			with open(file_path, "w") as f:
				yaml.dump(d, f)
			mb.showinfo("Save Successful", f"Configuration saved to:\n{file_path}")
		except Exception as e:
			mb.showerror("Save Failed", f"Failed to save configuration:\n{e}")

	def load_config():
		# Set tracking
		tracking = loaded.get("tracking", {})
		tracking_vars["track_per_agent"].set(tracking.get("track_per_agent", True))
		tracking_vars["track_per_hospital"].set(tracking.get("track_per_hospital", True))
		tracking_vars["track_confidence"].set(tracking.get("track_confidence", True))
		tracking_vars["track_stability"].set(tracking.get("track_stability", True))
		tracking_vars["save_logs"].set(tracking.get("save_logs", True))
		tracking_vars["log_dir"].set(tracking.get("log_dir", "outputs/logs"))
		config_path = Path("configs/config.yaml")
		if not config_path.exists():
			return
		with open(config_path, "r") as f:
			loaded = yaml.safe_load(f)
		# Set hospital
		hospital_vars["hospital_ids"].set(loaded.get("hospital_ids", ""))
		hospital_vars["num_agents"].set(loaded.get("num_agents_per_hospital", 4))
		hospital_vars["random_seed"].set(loaded.get("random_seed", 42))
		# Set patient
		patient_vars["ham_csv"].set(loaded.get("ham_csv", ""))
		patient_vars["isic_csv"].set(loaded.get("isic_csv", ""))
		patient_vars["out_dir"].set(loaded.get("out_dir", ""))
		# Set agents
		agents = loaded.get("agents", {})
		# Load agent types into listbox
		types_loaded = agents.get("types", ["akiec_agent", "bcc_agent", "melanoma_agent", "scc_agent"])
		agent_vars["types_listbox"].selection_clear(0, "end")
		for i, t in enumerate(["akiec_agent", "bcc_agent", "melanoma_agent", "scc_agent"]):
			if t in types_loaded:
				agent_vars["types_listbox"].selection_set(i)
		agent_vars["switch"].set(agents.get("allow_dynamic_switch", True))
		patterns_dict = agents.get("patterns", {})
		# Load patterns available into listbox
		available_loaded = patterns_dict.get("available", ["rule_based", "bayesian", "deep_learning"])
		agent_vars["patterns_available_listbox"].selection_clear(0, "end")
		for i, t in enumerate(["rule_based", "bayesian", "deep_learning"]):
			if t in available_loaded:
				agent_vars["patterns_available_listbox"].selection_set(i)
		patterns = patterns_dict.get("default_mapping", {})
		for k in ["BCC", "SCC", "MELANOMA", "AKIEC"]:
			agent_vars[k].set(patterns.get(k, ""))
		# Set local meta
		local = loaded.get("meta_agent", {}).get("local", {})
		local_vars["enable"].set(local.get("enable", True))
		local_vars["strategy"].set(local.get("weighting_strategy", "adaptive"))
		score = local.get("score_function", {})
		local_vars["use_auc"].set(score.get("use_auc", True))
		local_vars["use_f1"].set(score.get("use_f1", True))
		local_vars["use_confidence"].set(score.get("use_confidence", True))
		# Set privacy
		privacy = loaded.get("privacy", {})
		dp = privacy.get("differential_privacy", {})
		privacy_vars["dp"].set(dp.get("enabled", True))
		privacy_vars["epsilon"].set(dp.get("epsilon", 1.0))
		privacy_vars["secure"].set(privacy.get("secure_aggregation", True))
		# Set federation
		federation = loaded.get("federation", {})
		fed_vars["algo"].set(federation.get("aggregation_algorithm", "adaptive"))
		fedprox = federation.get("fedprox", {})
		fed_vars["mu"].set(fedprox.get("mu", 0.1))
		adaptive = federation.get("adaptive", {})
		for k in ["alpha", "beta", "gamma", "auc_weight", "f1_weight"]:
			fed_vars[k].set(adaptive.get(k, 0))
		fed_vars["hospital_weighting"].set(adaptive.get("hospital_weighting", "dynamic"))
		# Set global meta
		global_meta = loaded.get("meta_agent", {}).get("global", {})
		global_vars["enable"].set(global_meta.get("enable", True))
		global_vars["anomaly"].set(global_meta.get("anomaly_detection", True))
		global_vars["trust"].set(global_meta.get("trust_reweighting", True))
		# Set simulation (Missed Settings)
		simulation = loaded.get("simulation", {})
		simulation_vars["simulate_multi"].set(simulation.get("simulate_multi", True))
		simulation_vars["compare_all"].set(simulation.get("compare_all", True))
		simulation_vars["num_rounds"].set(simulation.get("num_rounds", 20))
		# Set output (Missed Settings)
		output = loaded.get("output", {})
		output_vars["save_global_model"].set(output.get("save_global_model", True))
		output_vars["save_local_models"].set(output.get("save_local_models", False))
		output_vars["save_metrics"].set(output.get("save_metrics", True))

	# Add Save/Load buttons at the very bottom (after Missed Settings)
	button_frame = ttk.Frame(content)
	button_frame.grid(row=row+2, column=0, columnspan=2, pady=(16, 8))
	save_btn = ttk.Button(button_frame, text="Save Configuration", command=save_config, style="Colored.TButton")
	save_btn.pack(side="left", padx=8)
	load_btn = ttk.Button(button_frame, text="Load Configuration", command=load_config, style="Colored.TButton")
	load_btn.pack(side="left", padx=8)

	return page