"""Professional Tkinter splash screen."""

import tkinter as tk
import tkinter.font as tkfont
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
	style = ttk.Style(root)
	style_name = "Expanded.TNotebook"
	style.configure(style_name)
	style.configure(f"{style_name}.Tab", anchor="center")

	# Create notebook (tab container)
	notebook = ttk.Notebook(root, style=style_name)
	notebook.pack(fill="both", expand=True, padx=10, pady=10)

	# Create tabs
	tab1 = ttk.Frame(notebook)
	tab2 = ttk.Frame(notebook)
	tab3 = ttk.Frame(notebook)
	tab4 = ttk.Frame(notebook)
	tab5 = ttk.Frame(notebook)

	notebook.add(tab1, text="tab1")
	notebook.add(tab2, text="tab2")
	notebook.add(tab3, text="tab3")
	notebook.add(tab4, text="tab4")
	notebook.add(tab5, text="tab5")

	def _refresh_tab_widths(_: tk.Event | None = None) -> None:
		"""Distribute tab widths so tabs always span the full notebook width."""
		tab_count = len(notebook.tabs())
		if tab_count == 0:
			return

		notebook_width = notebook.winfo_width()
		if notebook_width <= 1:
			return

		# ttk tab width is in characters, so convert pixels to character units.
		font = tkfont.nametofont("TkDefaultFont")
		char_width = max(font.measure("0"), 1)
		usable_width = max(notebook_width - 12, 1)
		width_chars = max(1, usable_width // (tab_count * char_width))
		style.configure(f"{style_name}.Tab", width=width_chars)

	notebook.bind("<Configure>", _refresh_tab_widths)
	notebook.bind("<<NotebookTabChanged>>", _refresh_tab_widths)
	root.after(50, _refresh_tab_widths)

	# Populate Tab 1
	label1 = ttk.Label(tab1, text="content for tab 1")
	label1.pack(padx=20, pady=20)

	# Populate Tab 2
	label2 = ttk.Label(tab2, text="content for tab 2")
	label2.pack(padx=20, pady=20)

	# Populate Tab 3
	label3 = ttk.Label(tab3, text="content for tab 3")
	label3.pack(padx=20, pady=20)

	# Populate Tab 4
	label4 = ttk.Label(tab4, text="content for tab 4")
	label4.pack(padx=20, pady=20)

	# Populate Tab 5
	label5 = ttk.Label(tab5, text="content for tab 5")
	label5.pack(padx=20, pady=20)

	root.mainloop()


if __name__ == "__main__":
	main()
