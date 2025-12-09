"""
main.py

Entrypoint: load the model using `data.load_and_train_model`.
If display available: start Tkinter GUI from `ui.GamingBehaviourApp`.
If headless: save results to JSON file.
"""

import os
import json
from data import load_and_train_model


def has_display():
    """Check if a display is available (for GUI support)."""
    return os.environ.get('DISPLAY') or os.name == 'nt'


def run_headless(model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df):
    """Run in headless mode: save results to file instead of GUI."""
    results = {
        "model_accuracy": round(accuracy, 4),
        "categorical_features": cat_cols,
        "numeric_features": num_cols,
        "categorical_options": {k: sorted(v) if isinstance(v, list) else v for k, v in cat_options.items()},
        "numeric_defaults": {k: float(v) if isinstance(v, (int, float)) else v for k, v in num_defaults.items()},
        "dataset_stats": {
            "total_rows": len(df),
            "engagement_distribution": df['EngagementLevel'].value_counts().to_dict() if 'EngagementLevel' in df.columns else {}
        }
    }
    
    output_file = "model_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n[headless] Model trained successfully!")
    print(f"[headless] Accuracy: {accuracy:.4f}")
    print(f"[headless] Results saved to: {output_file}")
    print(f"[headless] Dataset rows: {len(df)}")


def run_gui(model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df):
    """Run GUI mode: launch Tkinter window."""
    import tkinter as tk
    from ui import GamingBehaviourApp
    
    root = tk.Tk()
    print("[main] Building UI...")
    app = GamingBehaviourApp(root, model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df)
    print("[main] Entering mainloop")
    root.mainloop()


def main():
    try:
        print("[main] Loading and training model with PySpark...")
        model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df = load_and_train_model()
        print(f"[main] Model trained. Accuracy={accuracy:.4f}")
    except Exception as e:
        print(f"[ERROR] Failed to load/train model: {e}")
        import traceback
        traceback.print_exc()
        return

    # Choose GUI or headless mode
    if has_display():
        print("[main] Display detected. Running GUI mode...")
        try:
            run_gui(model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df)
        except Exception as e:
            print(f"[WARNING] GUI failed: {e}. Falling back to headless mode...")
            run_headless(model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df)
    else:
        print("[main] No display available. Running headless mode...")
        run_headless(model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df)


if __name__ == "__main__":
    main()


