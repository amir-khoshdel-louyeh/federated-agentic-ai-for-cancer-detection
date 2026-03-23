"""Shared UI styling and layout helpers for the simulator."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

APP_BG = "#edf2f7"
SURFACE_BG = "#ffffff"
BORDER = "#d5dde5"
ACCENT = "#0f766e"
TEXT_MAIN = "#16202a"
TEXT_MUTED = "#4a5a67"
TAB_IDLE = "#dce5ee"
TAB_ACTIVE = "#ffffff"
FONT_FAMILY = "DejaVu Sans"


def configure_app_style(root: tk.Tk) -> tuple[ttk.Style, str]:
	"""Configure and return app-wide ttk style objects."""
	style = ttk.Style(root)
	try:
		style.theme_use("clam")
	except tk.TclError:
		pass

	root.configure(bg=APP_BG)

	style.configure("App.TFrame", background=APP_BG)
	style.configure("Card.TFrame", background=SURFACE_BG, relief="solid", borderwidth=1)

	style.configure(
		"Heading.TLabel",
		background=APP_BG,
		foreground=TEXT_MAIN,
		font=(FONT_FAMILY, 15, "bold"),
	)
	style.configure(
		"Subheading.TLabel",
		background=APP_BG,
		foreground=TEXT_MUTED,
		font=(FONT_FAMILY, 10),
	)
	style.configure(
		"CardTitle.TLabel",
		background=SURFACE_BG,
		foreground=TEXT_MAIN,
		font=(FONT_FAMILY, 11, "bold"),
	)
	style.configure(
		"CardText.TLabel",
		background=SURFACE_BG,
		foreground=TEXT_MUTED,
		font=(FONT_FAMILY, 10),
	)
	style.configure(
		"Badge.TLabel",
		background="#d9f6ef",
		foreground=ACCENT,
		font=(FONT_FAMILY, 9, "bold"),
		padding=(8, 3),
	)

	notebook_style = "App.TNotebook"
	style.configure(notebook_style, background=APP_BG, borderwidth=0, tabmargins=(10, 8, 10, 0))
	style.configure(
		f"{notebook_style}.Tab",
		background=TAB_IDLE,
		foreground="#24404f",
		font=(FONT_FAMILY, 10, "bold"),
		padding=(16, 10),
	)
	style.map(
		f"{notebook_style}.Tab",
		background=[("selected", TAB_ACTIVE), ("active", "#e7eef5")],
		foreground=[("selected", TEXT_MAIN), ("active", TEXT_MAIN)],
	)

	return style, notebook_style


def build_scrollable_page(parent: ttk.Notebook) -> tuple[ttk.Frame, ttk.Frame]:
	"""Create a scrollable page and return (outer_frame, content_frame)."""
	outer = ttk.Frame(parent, style="App.TFrame")
	canvas = tk.Canvas(outer, highlightthickness=0, borderwidth=0, bg=APP_BG)
	scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
	content = ttk.Frame(canvas, style="App.TFrame")

	canvas.configure(yscrollcommand=scrollbar.set)
	window_id = canvas.create_window((0, 0), window=content, anchor="nw")

	def _on_content_configure(_: tk.Event) -> None:
		canvas.configure(scrollregion=canvas.bbox("all"))

	def _on_canvas_configure(event: tk.Event) -> None:
		canvas.itemconfigure(window_id, width=event.width)

	content.bind("<Configure>", _on_content_configure)
	canvas.bind("<Configure>", _on_canvas_configure)

	canvas.pack(side="left", fill="both", expand=True)
	scrollbar.pack(side="right", fill="y")

	def _is_descendant(widget: tk.Misc | None, ancestor: tk.Misc) -> bool:
		current = widget
		while current is not None:
			if current == ancestor:
				return True
			parent_name = current.winfo_parent()
			if not parent_name:
				return False
			try:
				current = current.nametowidget(parent_name)
			except tk.TclError:
				return False
		return False

	def _on_mousewheel(event: tk.Event) -> None:
		widget = getattr(event, "widget", None)
		if not isinstance(widget, tk.Misc):
			return
		if not _is_descendant(widget, outer):
			return

		bbox = canvas.bbox("all")
		if not bbox:
			return
		if (bbox[3] - bbox[1]) <= canvas.winfo_height():
			return

		if getattr(event, "delta", 0):
			canvas.yview_scroll(int(-event.delta / 120), "units")
		elif getattr(event, "num", 0) == 4:
			canvas.yview_scroll(-1, "units")
		elif getattr(event, "num", 0) == 5:
			canvas.yview_scroll(1, "units")

	outer.bind_all("<MouseWheel>", _on_mousewheel, add="+")
	outer.bind_all("<Button-4>", _on_mousewheel, add="+")
	outer.bind_all("<Button-5>", _on_mousewheel, add="+")

	return outer, content


def add_header(container: ttk.Frame, *, title: str, subtitle: str, badge: str | None = None) -> None:
	"""Add a section header with optional badge text."""
	header = ttk.Frame(container, style="App.TFrame")
	header.pack(fill="x", padx=20, pady=(18, 10))

	if badge:
		ttk.Label(header, text=badge, style="Badge.TLabel").pack(anchor="w", pady=(0, 6))

	ttk.Label(header, text=title, style="Heading.TLabel").pack(anchor="w")
	ttk.Label(header, text=subtitle, style="Subheading.TLabel", wraplength=980, justify="left").pack(
		anchor="w", pady=(4, 0)
	)


def add_card(container: ttk.Frame, *, title: str, lines: list[str]) -> None:
	"""Add a styled content card with bullet lines."""
	card = ttk.Frame(container, style="Card.TFrame", padding=14)
	card.pack(fill="x", padx=20, pady=(0, 12))

	ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 8))
	for line in lines:
		ttk.Label(
			card,
			text=f"- {line}",
			style="CardText.TLabel",
			wraplength=980,
			justify="left",
		).pack(anchor="w", pady=1)


def add_workflow_section(container: ttk.Frame, *, title: str) -> ttk.Frame:
	"""Add a labeled workflow section container and return its body frame for future inputs.
	
	Used to organize settings into workflow pipeline stages.
	"""
	section = ttk.Frame(container, style="Card.TFrame", padding=10)
	section.pack(fill="x", padx=0, pady=(0, 8))

	ttk.Label(
		section,
		text=title,
		style="CardTitle.TLabel",
		foreground=ACCENT,
	).pack(anchor="w", pady=(0, 6))

	body = ttk.Frame(section, style="App.TFrame")
	body.pack(fill="both", expand=True, padx=4)

	return body
