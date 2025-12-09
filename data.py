"""
data.py

Contains the data loading and model training function moved from `app.py`.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from pyspark.sql import SparkSession


def load_and_train_model(csv_path=r"online_gaming_behavior_dataset.csv"):
    """Load dataset from `csv_path` using PySpark, train a RandomForest pipeline and
    return (clf, categorical_cols, numeric_cols, cat_options, num_defaults, accuracy, df).
    """
    try:
        spark = SparkSession.builder \
            .appName("OnlineGamingBehaviourAnalysis") \
            .getOrCreate()

        spark_df = spark.read.csv(csv_path, header=True, inferSchema=True)

        # convert to pandas for sklearn model training
        df = spark_df.toPandas()

    except Exception as e:
        raise Exception(f"Spark failed to load dataset: {e}")

    target_col = "EngagementLevel"
    drop_cols = ["PlayerID", target_col]

    categorical_cols = ["Gender", "Location", "GameGenre", "GameDifficulty"]
    numeric_cols = [c for c in df.columns if c not in categorical_cols + drop_cols]

    X = df[categorical_cols + numeric_cols]
    y = df[target_col]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf.fit(X_train, y_train)

    accuracy = clf.score(X_test, y_test)

    cat_options = {col: sorted(df[col].dropna().unique().tolist()) for col in categorical_cols}

    num_defaults = {col: float(df[col].median()) for col in numeric_cols}

    return clf, categorical_cols, numeric_cols, cat_options, num_defaults, accuracy, df
