"""Professional Tkinter splash screen."""

import tkinter as tk
from tkinter import ttk


def main() -> None:
	root = tk.Tk()
	root.title("Federated Agentic AI")

	# Maximize across platforms with a Linux-safe fallback.
	try:
		root.attributes("-zoomed", True)
	except tk.TclError:
		pass
	root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")
	root.configure(bg="#525252")

	# Create notebook (tab container)
	notebook = ttk.Notebook(root)
	notebook.pack(fill="both", expand=True, padx=10, pady=10)

	# Create tabs
	tab1 = ttk.Frame(notebook)
	tab2 = ttk.Frame(notebook)
	tab3 = ttk.Frame(notebook)

	notebook.add(tab1, text="Overview")
	notebook.add(tab2, text="Settings")
	notebook.add(tab3, text="Results")

	# Populate Tab 1 (Overview)
	label1 = ttk.Label(tab1, text="Federated Learning Simulation")
	label1.pack(padx=20, pady=20)

	# Populate Tab 2 (Settings)
	label2 = ttk.Label(tab2, text="Configuration Settings")
	label2.pack(padx=20, pady=20)

	# Populate Tab 3 (Results)
	label3 = ttk.Label(tab3, text="Simulation Results")
	label3.pack(padx=20, pady=20)

	root.mainloop()


if __name__ == "__main__":
	main()
