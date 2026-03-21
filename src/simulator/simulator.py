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

	

	root.mainloop()


if __name__ == "__main__":
	main()
