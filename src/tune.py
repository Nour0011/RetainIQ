"""
Purpose:
    This script performs hyperparameter tuning for the churn prediction model
    using Optuna. Every tuning trial is tracked in MLflow.

Requirements covered:
    1. Hyperparameter tuning
    2. MLflow tracking for every trial
    3. Parameter logging
    4. Metric logging
    5. Best tuned model saving
    6. Production candidate model creation
    7. MLflow model signature logging
"""

from pathlib import Path
import json
import warnings

import joblib
import mlflow
import mlflow.sklearn
import optuna
import pandas as pd

from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")


# =========================
# Project paths
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "tuning"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "RetainIQ-Pro-Hyperparameter-Tuning"


def load_data():
    """
    Load the cleaned dataset and split it into training and testing sets.

    Returns:
        x_train, x_test, y_train, y_test
    """
    df = pd.read_csv(DATA_PATH)

    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    return train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def build_preprocessor(x_train):
    """
    Build preprocessing pipeline.

    Numeric columns:
        - StandardScaler

    Categorical columns:
        - OneHotEncoder

    The preprocessor is included inside the final model pipeline so the same
    transformations are used during both training and deployment.
    """
    numeric_features = x_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = x_train.select_dtypes(
        include=["object"]
    ).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor


def calculate_metrics(model, x_test, y_test):
    """
    Evaluate the model using classification metrics.

    Recall is important because the company wants to identify customers who may
    churn before they leave.
    """
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    return metrics


def create_model_params(trial):
    """
    Ask Optuna to suggest Random Forest hyperparameters.

    These parameters control:
        - number of trees
        - tree depth
        - split behavior
        - leaf size
    """
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "class_weight": "balanced",
        "random_state": 42,
    }

    return params


def main():
    """
    Run Optuna tuning and save the best tuned model.

    Final output:
        - MLflow trial runs
        - best tuning results JSON
        - best_tuned_churn_model.pkl
        - MLflow model with input/output signature
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, x_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor(x_train)

    def objective(trial):
        """
        Objective function used by Optuna.

        Optuna tries many parameter combinations.
        MLflow logs each trial as a separate run.
        """
        params = create_model_params(trial)

        with mlflow.start_run(run_name=f"Optuna Trial {trial.number}"):
            model = RandomForestClassifier(**params)

            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            # Train model for this trial.
            pipeline.fit(x_train, y_train)

            # Evaluate trial model.
            metrics = calculate_metrics(pipeline, x_test, y_test)

            # Log trial details to MLflow.
            mlflow.log_param("model_type", "Tuned Random Forest")
            mlflow.log_param("trial_number", trial.number)
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)

            return metrics["f1_score"]

    # Create and run the tuning study.
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=25)

    # Retrieve best parameters found by Optuna.
    best_params = study.best_params
    best_params["class_weight"] = "balanced"
    best_params["random_state"] = 42

    best_model = RandomForestClassifier(**best_params)

    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_model),
        ]
    )

    # Train final tuned model using the best parameters.
    best_pipeline.fit(x_train, y_train)

    # Evaluate final tuned model.
    best_metrics = calculate_metrics(best_pipeline, x_test, y_test)

    # Generate predictions for MLflow schema inference.
    y_pred = best_pipeline.predict(x_test)

    # Infer input/output schema for MLflow Model Registry.
    # This makes the Registry page show input and output schema instead of empty fields.
    signature = infer_signature(x_test, y_pred)

    # Log final tuned model as a production candidate.
    with mlflow.start_run(run_name="Best Tuned Random Forest"):
        mlflow.log_param("model_type", "Best Tuned Random Forest")
        mlflow.log_param("model_stage", "production_candidate")
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)

        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=x_test.head(5),
        )

    # Save model locally for deployment and registry steps.
    model_path = MODEL_DIR / "best_tuned_churn_model.pkl"
    joblib.dump(best_pipeline, model_path)

    # Save tuning results for report and documentation.
    results = {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_model_path": str(model_path),
    }

    results_path = REPORT_DIR / "best_tuning_results.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    print("\nBest tuned model saved successfully.")
    print("\nBest parameters:")
    print(best_params)

    print("\nBest metrics:")
    print(best_metrics)

    print(f"\nSaved tuned model to: {model_path}")
    print(f"Saved tuning report to: {results_path}")


if __name__ == "__main__":
    main()