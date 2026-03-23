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

	# ========================
	# LOAD CONFIG
	# ========================
	config_path = Path("config.yaml")
	if config_path.exists():
		with open(config_path, "r") as f:
			config = yaml.safe_load(f)
	else:
		config = {}

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

	def _on_mousewheel(event: tk.Event):
		if getattr(event, "delta", 0):
			canvas.yview_scroll(int(-event.delta / 120), "units")

	for widget in (page, canvas, content):
		widget.bind_all("<MouseWheel>", _on_mousewheel, add="+")

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
	hospital_section, _ = _add_pair_row(row, "HOSPITAL", "HOSPITAL")
	row += 1
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
	hospital_vars["ham_csv"] = _add_field(hospital_section, "HAM CSV", tk.StringVar(value=config.get("ham_csv", "")))
	hospital_vars["isic_csv"] = _add_field(hospital_section, "ISIC CSV", tk.StringVar(value=config.get("isic_csv", "")))
	hospital_vars["out_dir"] = _add_field(hospital_section, "Output Dir", tk.StringVar(value=config.get("out_dir", "")))
	hospital_vars["hospital_ids"] = _add_field(hospital_section, "Hospital IDs", tk.StringVar(value=config.get("hospital_ids", "")))
	hospital_vars["num_agents"] = _add_field(hospital_section, "Agents", tk.IntVar(value=config.get("num_agents_per_hospital", 4)))
	hospital_vars["seed"] = _add_field(hospital_section, "Seed", tk.IntVar(value=config.get("seed", 42)))

	# ========================
	# PATIENT
	# ========================
	patient_vars = {}
	patient_vars["dataset_type"] = _add_field(
		patient_section,
		"Dataset",
		tk.StringVar(value="HAM10000"),
		"combo",
		["HAM10000", "ISIC2019"],
	)

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

	for k in ["BCC", "SCC", "MELANOMA", "AKIEC"]:
		agent_vars[k] = _add_field(
			agent_section,
			k,
			tk.StringVar(value=patterns.get(k, "")),
			"combo",
			["rule_based", "bayesian", "deep_learning", "hybrid"],
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
	adaptive = fed_cfg.get("adaptive", {})

	fed_vars = {}
	fed_vars["algo"] = _add_field(
		fed_section,
		"Algorithm",
		tk.StringVar(value=fed_cfg.get("aggregation_algorithm", "adaptive")),
		"combo",
		["fedavg", "fedprox", "adaptive"],
	)

	for k in ["alpha", "beta", "gamma"]:
		fed_vars[k] = _add_field(fed_section, k, tk.DoubleVar(value=adaptive.get(k, 0)))

	# ========================
	# GLOBAL META
	# ========================
	global_cfg = config.get("meta_agent", {}).get("global", {})

	global_vars = {}
	global_vars["enable"] = _add_field(global_meta_section, "Enable", tk.BooleanVar(value=global_cfg.get("enable", True)), "check")
	global_vars["anomaly"] = _add_field(global_meta_section, "Anomaly", tk.BooleanVar(value=global_cfg.get("anomaly_detection", True)), "check")
	global_vars["trust"] = _add_field(global_meta_section, "Trust", tk.BooleanVar(value=global_cfg.get("trust_reweighting", True)), "check")

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
	}

	return page