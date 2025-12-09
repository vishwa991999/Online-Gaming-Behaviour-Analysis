"""
main.py

Entrypoint: load the model using `data.load_and_train_model`, start the Tkinter UI from `ui.GamingBehaviourApp`.
"""

import tkinter as tk
from tkinter import messagebox

from data import load_and_train_model
from ui import GamingBehaviourApp


def main():
    try:
        print("[main] Loading and training model with PySpark...")
        model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df = load_and_train_model()
        print(f"[main] Model trained. Accuracy={accuracy:.4f}")
    except Exception as e:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Failed to load/train model:\n{e}")
        root.destroy()
        return

    root = tk.Tk()
    print("[main] Building UI...")
    app = GamingBehaviourApp(root, model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df)
    print("[main] Entering mainloop")
    root.mainloop()


if __name__ == "__main__":
    main()

