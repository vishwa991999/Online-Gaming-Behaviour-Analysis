"""
Web-based GUI for Online Gaming Behaviour Analysis
Run with: python web_app.py
Then open http://localhost:5000 in your browser
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

app = Flask(__name__)

# Global variables
model = None
cat_cols = None
num_cols = None
cat_options = None
num_defaults = None
accuracy = None
df = None

def load_and_train_model(csv_path=r"online_gaming_behavior_dataset.csv"):
    """Load dataset and train model"""
    try:
        df_local = pd.read_csv(csv_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"CSV file '{csv_path}' not found. "
            "Make sure it is in the same folder as this script."
        )

    target_col = "EngagementLevel"
    drop_cols = ["PlayerID", target_col]

    categorical_cols = ["Gender", "Location", "GameGenre", "GameDifficulty"]
    numeric_cols = [c for c in df_local.columns if c not in categorical_cols + drop_cols]

    X = df_local[categorical_cols + numeric_cols]
    y = df_local[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model_rf = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    clf = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model_rf)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)
    accuracy_val = clf.score(X_test, y_test)

    cat_options_dict = {
        col: sorted(df_local[col].dropna().unique().tolist())
        for col in categorical_cols
    }

    num_defaults_dict = {
        col: float(df_local[col].median())
        for col in numeric_cols
    }

    return clf, categorical_cols, numeric_cols, cat_options_dict, num_defaults_dict, accuracy_val, df_local


def generate_plot(plot_type):
    """Generate matplotlib plots as base64"""
    if df is None:
        return None
    
    plt.figure(figsize=(8, 5))
    
    if plot_type == "age_dist":
        plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
        plt.xlabel('Age', fontsize=11)
        plt.ylabel('Frequency', fontsize=11)
        plt.title('Player Age Distribution', fontsize=13, fontweight='bold')
        plt.grid(axis='y', alpha=0.3)
    
    elif plot_type == "playtime":
        engagement_playtime = df.groupby('EngagementLevel')['PlayTimeHours'].mean().sort_values()
        colors = ['#ff9999', '#ffcc99', '#99ff99']
        engagement_playtime.plot(kind='barh', color=colors)
        plt.xlabel('Average Play Time (Hours)', fontsize=11)
        plt.ylabel('Engagement Level', fontsize=11)
        plt.title('Average Play Time by Engagement Level', fontsize=13, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
    
    elif plot_type == "genre":
        genre_engagement = pd.crosstab(df['GameGenre'], df['EngagementLevel'], normalize='index') * 100
        genre_engagement.plot(kind='bar', stacked=False, color=['#ff6b6b', '#ffd93d', '#6bcf7f'])
        plt.xlabel('Game Genre', fontsize=11)
        plt.ylabel('Percentage (%)', fontsize=11)
        plt.title('Engagement Distribution by Game Genre', fontsize=13, fontweight='bold')
        plt.legend(title='Engagement', fontsize=9)
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=45, ha='right')
    
    elif plot_type == "sessions":
        engagement_colors = {'High': '#2ecc71', 'Medium': '#f39c12', 'Low': '#e74c3c'}
        for level in df['EngagementLevel'].unique():
            data = df[df['EngagementLevel'] == level]
            plt.scatter(data['SessionsPerWeek'], data['AvgSessionDurationMinutes'], 
                       label=level, alpha=0.6, s=50, color=engagement_colors.get(level, 'gray'))
        plt.xlabel('Sessions Per Week', fontsize=11)
        plt.ylabel('Avg Session Duration (Minutes)', fontsize=11)
        plt.title('Session Patterns by Engagement Level', fontsize=13, fontweight='bold')
        plt.legend(title='Engagement', fontsize=9)
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url


@app.route('/')
def index():
    """Main page"""
    if model is None:
        return "Model not loaded. Please restart the app.", 500
    
    return render_template('index.html',
                         accuracy=f"{accuracy*100:.2f}",
                         cat_cols=cat_cols,
                         num_cols=num_cols,
                         cat_options=cat_options,
                         num_defaults=num_defaults)


@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.json
        
        # Build DataFrame from input
        input_dict = {}
        for col in cat_cols:
            input_dict[col] = [data.get(col)]
        for col in num_cols:
            input_dict[col] = [float(data.get(col, 0))]
        
        input_df = pd.DataFrame(input_dict)
        prediction = model.predict(input_df)[0]
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/plot/<plot_type>')
def get_plot(plot_type):
    """API endpoint for plots"""
    try:
        plot_url = generate_plot(plot_type)
        if plot_url:
            return jsonify({
                'success': True,
                'plot': f"data:image/png;base64,{plot_url}"
            })
        return jsonify({'success': False, 'error': 'Plot generation failed'}), 400
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/stats')
def get_stats():
    """API endpoint for dataset statistics"""
    try:
        total_players = len(df)
        engagement_dist = df['EngagementLevel'].value_counts().to_dict()
        
        stats = {
            'total_players': total_players,
            'engagement_dist': engagement_dist,
            'avg_age': float(df['Age'].mean()),
            'avg_playtime': float(df['PlayTimeHours'].mean()),
            'avg_sessions': float(df['SessionsPerWeek'].mean())
        }
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("Loading and training model...")
    try:
        model, cat_cols, num_cols, cat_options, num_defaults, accuracy, df = load_and_train_model()
        print(f"✓ Model trained successfully! Accuracy: {accuracy:.2%}")
        print("\nStarting web server...")
        print("🌐 Open http://localhost:5000 in your browser")
        app.run(debug=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error: {e}")
