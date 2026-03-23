"""Configuration tab module."""

import tkinter as tk
from tkinter import ttk

from ..ui_kit import add_workflow_section, build_scrollable_page


def build_configuration_tab(parent: ttk.Notebook) -> ttk.Frame:
	"""Create and return the Configuration tab frame.

	Layout:
	- Whole page split into 2 vertical panels
	- Left panel: 30% for workflow-based settings sections (scrollable)
	- Right panel: 70% for future visualizations / graphics
	
	Workflow sections organized as in workflow.txt:
	1. HOSPITAL (data sources, IDs)
	2. PATIENT DATA (dataset paths, splits)
	3. AGENTIC AI LAYER (agent patterns per cancer type)
	4. LOCAL META-AGENT CONTROLLER (local ensemble settings)
	5. PRIVACY & SECURITY LAYER (DP, aggregation prep)
	6. Model Update Exchange marker
	7. FEDERATED AGGREGATOR (algorithm choice, parameters)
	8. GLOBAL MODEL (federation outcomes)
	9. META-AGENT (GLOBAL CONTROLLER) (global monitoring)
	10. Model Broadcast marker
	"""
	page = ttk.Frame(parent, style="App.TFrame")
	page.grid_columnconfigure(0, weight=3, uniform="config_split")
	page.grid_columnconfigure(1, weight=7, uniform="config_split")
	page.grid_rowconfigure(0, weight=1)

	# Left panel: scrollable workflow section list
	left_outer = ttk.Frame(page, style="App.TFrame")
	left_outer.grid(row=0, column=0, sticky="nsew", padx=(14, 7), pady=14)
	left_outer.grid_columnconfigure(0, weight=1)
	left_outer.grid_rowconfigure(0, weight=1)

	left_canvas = tk.Canvas(left_outer, highlightthickness=0, borderwidth=0, bg="#edf2f7")
	left_scrollbar = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
	left_content = ttk.Frame(left_canvas, style="App.TFrame")

	left_canvas.configure(yscrollcommand=left_scrollbar.set)
	left_window = left_canvas.create_window((0, 0), window=left_content, anchor="nw")

	def _on_left_content_configure(_: tk.Event) -> None:
		left_canvas.configure(scrollregion=left_canvas.bbox("all"))

	def _on_left_canvas_configure(event: tk.Event) -> None:
		left_canvas.itemconfigure(left_window, width=event.width)

	left_content.bind("<Configure>", _on_left_content_configure)
	left_canvas.bind("<Configure>", _on_left_canvas_configure)

	left_canvas.pack(side="left", fill="both", expand=True)
	left_scrollbar.pack(side="right", fill="y")

	# Bind mouse wheel to left content scrollbar
	def _on_left_mousewheel(event: tk.Event) -> None:
		if getattr(event, "delta", 0):
			left_canvas.yview_scroll(int(-event.delta / 120), "units")
		elif getattr(event, "num", 0) == 4:
			left_canvas.yview_scroll(-1, "units")
		elif getattr(event, "num", 0) == 5:
			left_canvas.yview_scroll(1, "units")

	for widget in (left_outer, left_canvas, left_content):
		widget.bind("<MouseWheel>", _on_left_mousewheel)
		widget.bind("<Button-4>", _on_left_mousewheel)
		widget.bind("<Button-5>", _on_left_mousewheel)

	# Add workflow sections to left content
	hospital_section = add_workflow_section(left_content, title="HOSPITAL")
	patient_data_section = add_workflow_section(left_content, title="PATIENT DATA")
	agentic_ai_section = add_workflow_section(left_content, title="AGENTIC AI LAYER")
	local_meta_section = add_workflow_section(left_content, title="LOCAL META-AGENT CONTROLLER")
	privacy_section = add_workflow_section(left_content, title="PRIVACY & SECURITY LAYER")

	# Model update divider
	divider_1 = ttk.Label(
		left_content,
		text="▼ (ONLY MODEL UPDATES SHARED) ▼",
		font=("DejaVu Sans", 8, "bold"),
		foreground="#0f766e",
	)
	divider_1.pack(fill="x", padx=8, pady=(6, 4))

	federated_agg_section = add_workflow_section(left_content, title="FEDERATED AGGREGATOR (SERVER)")
	global_model_section = add_workflow_section(left_content, title="GLOBAL MODEL")
	global_meta_section = add_workflow_section(left_content, title="META-AGENT (GLOBAL CONTROLLER)")

	# Model broadcast divider
	divider_2 = ttk.Label(
		left_content,
		text="▲ (MODEL SENT BACK TO HOSPITALS) ▲",
		font=("DejaVu Sans", 8, "bold"),
		foreground="#0f766e",
	)
	divider_2.pack(fill="x", padx=8, pady=(6, 4))

	# Right panel: visualization space
	right_panel = ttk.Frame(page, style="Card.TFrame", padding=12)
	right_panel.grid(row=0, column=1, sticky="nsew", padx=(7, 14), pady=14)
	right_panel.grid_columnconfigure(0, weight=1)
	right_panel.grid_rowconfigure(0, weight=1)

	# Placeholder for visualizations
	viz_placeholder = ttk.Label(
		right_panel,
		text="Visualization & Live Updates Area\n(Right-click sections to edit)",
		font=("DejaVu Sans", 11),
		foreground="#4a5a67",
	)
	viz_placeholder.place(relx=0.5, rely=0.5, anchor="center")

	# Expose all section frames for next implementation steps
	page.left_panel = left_content  # type: ignore[attr-defined]
	page.right_panel = right_panel  # type: ignore[attr-defined]
	page.hospital_section = hospital_section  # type: ignore[attr-defined]
	page.patient_data_section = patient_data_section  # type: ignore[attr-defined]
	page.agentic_ai_section = agentic_ai_section  # type: ignore[attr-defined]
	page.local_meta_section = local_meta_section  # type: ignore[attr-defined]
	page.privacy_section = privacy_section  # type: ignore[attr-defined]
	page.federated_agg_section = federated_agg_section  # type: ignore[attr-defined]
	page.global_model_section = global_model_section  # type: ignore[attr-defined]
	page.global_meta_section = global_meta_section  # type: ignore[attr-defined]

	return page
