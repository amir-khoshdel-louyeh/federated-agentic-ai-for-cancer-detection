"""Professional Tkinter splash screen."""

import runpy
import tkinter as tk
from tkinter import ttk


def main() -> None:
	root = tk.Tk()
	root.title("Federated Agentic AI - Splash")

	# Maximize across platforms with a Linux-safe fallback.
	try:
		root.attributes("-zoomed", True)
	except tk.TclError:
		pass
	root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
	root.configure(bg="#0B1220")

	transitioned = False

	def transition_to_simulator(_event: tk.Event | None = None) -> None:
		nonlocal transitioned
		if transitioned:
			return
		transitioned = True
		root.destroy()
		runpy.run_module("src.simulator.simulator", run_name="simulator")

	canvas = tk.Canvas(root, highlightthickness=0, bg="#0B1220")
	canvas.pack(fill="both", expand=True)

	# Decorative background blocks to give a polished visual style.
	canvas.create_rectangle(0, 0, 10_000, 10_000, fill="#0B1220", outline="")
	canvas.create_rectangle(0, 0, 10_000, 220, fill="#111C33", outline="")
	canvas.create_oval(-180, -220, 560, 520, fill="#17345B", outline="")
	canvas.create_oval(1020, 260, 1700, 980, fill="#122643", outline="")

	content = tk.Frame(root, bg="#0B1220")
	canvas.create_window(
		root.winfo_screenwidth() // 2,
		root.winfo_screenheight() // 2,
		anchor="center",
		window=content,
	)

	title = tk.Label(
		content,
		text="Federated Agentic AI",
		font=("Helvetica", 44, "bold"),
		fg="#F4F7FF",
		bg="#0B1220",
	)
	title.pack(pady=(0, 12))

	subtitle = tk.Label(
		content,
		text="Cancer Detection Platform",
		font=("Helvetica", 20),
		fg="#9FB6D9",
		bg="#0B1220",
	)
	subtitle.pack(pady=(0, 24))

	status = tk.Label(
		content,
		text="Initializing simulator...",
		font=("Helvetica", 14),
		fg="#CED9EE",
		bg="#0B1220",
	)
	status.pack(pady=(0, 18))

	style = ttk.Style(root)
	style.theme_use("default")
	style.configure(
		"Splash.Horizontal.TProgressbar",
		troughcolor="#223250",
		background="#4DA3FF",
		bordercolor="#223250",
		lightcolor="#4DA3FF",
		darkcolor="#4DA3FF",
	)
	progress = ttk.Progressbar(
		content,
		style="Splash.Horizontal.TProgressbar",
		mode="determinate",
		length=420,
		maximum=100,
		value=100,
	)
	progress.pack(pady=(0, 24))

	hint = tk.Label(
		content,
		text="Press any key to continue",
		font=("Helvetica", 12),
		fg="#7F97BE",
		bg="#0B1220",
	)
	hint.pack()

	root.bind_all("<Key>", transition_to_simulator)
	root.focus_force()
	root.after(1000, transition_to_simulator)
	root.mainloop()


if __name__ == "__main__":
	main()
