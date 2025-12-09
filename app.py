"""
Project: Online Gaming Behaviour Analysis
File   : online_gaming_behaviour_analysis.py

Requirements (install once):
    pip install pandas scikit-learn

Keep 'online_gaming_behavior_dataset.csv' in the SAME folder as this file.
Then run:
    python online_gaming_behaviour_analysis.py
"""

import tkinter as tk
from tkinter import ttk, messagebox

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# ===================== 1. LOAD DATA & TRAIN MODEL =====================

def load_and_train_model(csv_path=r"online_gaming_behavior_dataset.csv"):
    # Load dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"CSV file '{csv_path}' not found. "
            "Make sure it is in the same folder as this script."
        )

    # Columns in your dataset
    # ['PlayerID', 'Age', 'Gender', 'Location', 'GameGenre', 'PlayTimeHours',
    #  'InGamePurchases', 'GameDifficulty', 'SessionsPerWeek',
    #  'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked',
    #  'EngagementLevel']

    target_col = "EngagementLevel"
    drop_cols = ["PlayerID", target_col]

    # Define categorical & numeric columns
    categorical_cols = ["Gender", "Location", "GameGenre", "GameDifficulty"]
    numeric_cols = [c for c in df.columns if c not in categorical_cols + drop_cols]

    X = df[categorical_cols + numeric_cols]
    y = df[target_col]

    # Preprocessor: OneHot for categorial, passthrough for numeric
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    # Model: Random Forest (good default for tabular data)
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train
    clf.fit(X_train, y_train)

    # Accuracy
    accuracy = clf.score(X_test, y_test)

    # For building UI choices (dropdown values)
    cat_options = {
        col: sorted(df[col].dropna().unique().tolist())
        for col in categorical_cols
    }

    # Simple numeric defaults (median values)
    num_defaults = {
        col: float(df[col].median())
        for col in numeric_cols
    }

    return clf, categorical_cols, numeric_cols, cat_options, num_defaults, accuracy, df


# ===================== 2. BUILD TKINTER UI =====================

class GamingBehaviourApp:
    def __init__(self, root, model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df):
        self.root = root
        self.model = model
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.cat_options = cat_options
        self.num_defaults = num_defaults
        self.accuracy = accuracy
        self.df = df

        self.root.title("Online Gaming Behaviour Analysis")
        self.root.geometry("1400x800")
        self.root.resizable(True, True)

        # Use ttk style for a cleaner UI
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure("TLabel", padding=4, font=("Segoe UI", 10))
        style.configure("TButton", padding=6, font=("Segoe UI", 10, "bold"))
        style.configure("Header.TLabel", font=("Segoe UI", 16, "bold"))
        style.configure("SubHeader.TLabel", font=("Segoe UI", 11, "italic"))

        # ---- Layout: Three columns ----
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill="both", expand=True)

        left_frame = ttk.LabelFrame(main_frame, text="Player Information", padding=10)
        left_frame.pack(side="left", fill="both", expand=False, padx=(0, 5))

        middle_frame = ttk.LabelFrame(main_frame, text="Prediction & Analysis", padding=10)
        middle_frame.pack(side="left", fill="both", expand=True, padx=(5, 5))

        right_frame = ttk.LabelFrame(main_frame, text="Data Visualizations", padding=10)
        right_frame.pack(side="left", fill="both", expand=True, padx=(5, 0))

        # ---- Title & subtitle ----
        title_label = ttk.Label(left_frame, text="Online Gaming Behaviour Analysis",
                                style="Header.TLabel")
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

        subtitle_label = ttk.Label(
            left_frame,
            text="Enter player details to predict Engagement Level",
            style="SubHeader.TLabel"
        )
        subtitle_label.grid(row=1, column=0, columnspan=2, pady=(0, 15))

        # ---- Input widgets for categorical & numeric features ----
        self.inputs_cat = {}
        self.inputs_num = {}

        row_idx = 2

        # Categorical fields (dropdowns)
        for col in self.cat_cols:
            ttk.Label(left_frame, text=col + " :").grid(row=row_idx, column=0, sticky="e", pady=4, padx=(0, 5))
            var = tk.StringVar(value=self.cat_options[col][0] if self.cat_options[col] else "")
            combo = ttk.Combobox(left_frame, textvariable=var, values=self.cat_options[col], state="readonly", width=25)
            combo.grid(row=row_idx, column=1, sticky="w", pady=4)
            self.inputs_cat[col] = var
            row_idx += 1

        # Numeric fields (entry boxes)
        for col in self.num_cols:
            ttk.Label(left_frame, text=col + " :").grid(row=row_idx, column=0, sticky="e", pady=4, padx=(0, 5))
            var = tk.StringVar(value=str(self.num_defaults.get(col, "")))
            entry = ttk.Entry(left_frame, textvariable=var, width=28)
            entry.grid(row=row_idx, column=1, sticky="w", pady=4)
            self.inputs_num[col] = var
            row_idx += 1

        # ---- Buttons ----
        button_frame = ttk.Frame(left_frame)
        button_frame.grid(row=row_idx, column=0, columnspan=2, pady=(20, 10), sticky="ew")

        predict_btn = ttk.Button(button_frame, text="Predict Engagement", command=self.predict_engagement, width=20)
        predict_btn.pack(side="left", padx=10, pady=5)

        clear_btn = ttk.Button(button_frame, text="Clear", command=self.clear_inputs, width=15)
        clear_btn.pack(side="left", padx=10, pady=5)

        # ---- Middle panel: Output & model info ----
        self.prediction_label = ttk.Label(
            middle_frame,
            text="Prediction will appear here.",
            style="Header.TLabel",
            foreground="blue",
            wraplength=300,
            justify="center"
        )
        self.prediction_label.pack(pady=(10, 20))

        accuracy_text = f"Model Test Accuracy: {self.accuracy * 100:.2f}%"
        self.accuracy_label = ttk.Label(middle_frame, text=accuracy_text, font=("Segoe UI", 11, "bold"))
        self.accuracy_label.pack(pady=(0, 10))

        # Dataset Statistics
        stats_frame = ttk.LabelFrame(middle_frame, text="Dataset Statistics", padding=10)
        stats_frame.pack(pady=10, fill="x")
        
        total_players = len(self.df)
        engagement_dist = self.df['EngagementLevel'].value_counts()
        
        stats_text = f"Total Players: {total_players}\n"
        for level, count in engagement_dist.items():
            pct = (count/total_players)*100
            stats_text += f"{level}: {count} ({pct:.1f}%)\n"
        
        stats_label = ttk.Label(stats_frame, text=stats_text, justify="left")
        stats_label.pack()

        info_text = (
            "This model uses player demographics, game preferences, and "
            "gameplay statistics to predict the Engagement Level "
            "as High, Medium, or Low."
        )
        info_label = ttk.Label(middle_frame, text=info_text, wraplength=300, justify="left")
        info_label.pack(pady=(10, 20))

        # ---- Right panel: Visualizations ----
        viz_buttons_frame = ttk.Frame(right_frame)
        viz_buttons_frame.pack(pady=10, fill="x")
        
        ttk.Button(viz_buttons_frame, text="Age Distribution", command=self.plot_age_dist).pack(side="left", padx=5)
        ttk.Button(viz_buttons_frame, text="Play Time Analysis", command=self.plot_playtime).pack(side="left", padx=5)
        ttk.Button(viz_buttons_frame, text="Engagement by Genre", command=self.plot_engagement_genre).pack(side="left", padx=5)
        ttk.Button(viz_buttons_frame, text="Session Analysis", command=self.plot_session_analysis).pack(side="left", padx=5)
        
        # Canvas for plots
        self.plot_frame = ttk.Frame(right_frame)
        self.plot_frame.pack(fill="both", expand=True, pady=10)

    # ----------------- Helper Methods -----------------

    def get_input_data(self):
        """Collect input values from UI and build a one-row DataFrame."""
        data = {}

        # Categorical
        for col, var in self.inputs_cat.items():
            val = var.get()
            if val == "":
                raise ValueError(f"Please select a value for '{col}'.")
            data[col] = [val]

        # Numeric
        for col, var in self.inputs_num.items():
            txt = var.get().strip()
            if txt == "":
                raise ValueError(f"Please enter a value for '{col}'.")
            try:
                # Most columns are integers, but float is safe
                val = float(txt)
            except ValueError:
                raise ValueError(f"'{col}' must be a number.")
            data[col] = [val]

        return pd.DataFrame(data)

    def predict_engagement(self):
        """Handle Predict button click."""
        try:
            input_df = self.get_input_data()
            prediction = self.model.predict(input_df)[0]

            msg = f"Predicted Engagement Level:\n{prediction}"
            self.prediction_label.config(text=msg, foreground="green")

        except Exception as e:
            messagebox.showerror("Error", str(e))

    def clear_inputs(self):
        """Reset all inputs to defaults."""
        # Reset categorical
        for col, var in self.inputs_cat.items():
            if self.cat_options.get(col):
                var.set(self.cat_options[col][0])
            else:
                var.set("")

        # Reset numeric
        for col, var in self.inputs_num.items():
            if col in self.num_defaults:
                var.set(str(self.num_defaults[col]))
            else:
                var.set("")

        self.prediction_label.config(text="Prediction will appear here.", foreground="blue")

    def clear_plot(self):
        """Clear current plot."""
        for widget in self.plot_frame.winfo_children():
            widget.destroy()

    def plot_age_dist(self):
        """Plot age distribution."""
        self.clear_plot()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        ax.hist(self.df['Age'], bins=20, color='skyblue', edgecolor='black')
        ax.set_xlabel('Age', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Player Age Distribution', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_playtime(self):
        """Plot play time analysis."""
        self.clear_plot()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        engagement_playtime = self.df.groupby('EngagementLevel')['PlayTimeHours'].mean().sort_values()
        colors = ['#ff9999', '#ffcc99', '#99ff99']
        engagement_playtime.plot(kind='barh', ax=ax, color=colors)
        
        ax.set_xlabel('Average Play Time (Hours)', fontsize=10)
        ax.set_ylabel('Engagement Level', fontsize=10)
        ax.set_title('Average Play Time by Engagement Level', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_engagement_genre(self):
        """Plot engagement by game genre."""
        self.clear_plot()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Count engagement levels by genre
        genre_engagement = pd.crosstab(self.df['GameGenre'], self.df['EngagementLevel'], normalize='index') * 100
        
        genre_engagement.plot(kind='bar', ax=ax, stacked=False, 
                             color=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        
        ax.set_xlabel('Game Genre', fontsize=10)
        ax.set_ylabel('Percentage (%)', fontsize=10)
        ax.set_title('Engagement Distribution by Game Genre', fontsize=12, fontweight='bold')
        ax.legend(title='Engagement', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        fig.tight_layout()
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def plot_session_analysis(self):
        """Plot session analysis."""
        self.clear_plot()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        # Scatter plot: Sessions per week vs Avg session duration, colored by engagement
        engagement_colors = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        
        for level in self.df['EngagementLevel'].unique():
            data = self.df[self.df['EngagementLevel'] == level]
            ax.scatter(data['SessionsPerWeek'], data['AvgSessionDurationMinutes'], 
                      label=level, alpha=0.6, s=50, color=engagement_colors.get(level, 'gray'))
        
        ax.set_xlabel('Sessions Per Week', fontsize=10)
        ax.set_ylabel('Avg Session Duration (Minutes)', fontsize=10)
        ax.set_title('Session Patterns by Engagement Level', fontsize=12, fontweight='bold')
        ax.legend(title='Engagement', fontsize=8)
        ax.grid(alpha=0.3)
        
        canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)


# ===================== 3. MAIN =====================

def main():
    # Train model
    try:
        model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df = load_and_train_model()
    except Exception as e:
        # If training fails, show simple messagebox window
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror("Error", f"Failed to load/train model:\n{e}")
        root.destroy()
        return

    # Build GUI
    root = tk.Tk()
    app = GamingBehaviourApp(root, model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df)
    root.mainloop()


from main import main


if __name__ == "__main__":
    main()