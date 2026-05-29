"""
Gradient Boosting hyperparameter tuning with Optuna and MLflow.

This script is used to tune the Gradient Boosting model because the
baseline training comparison showed that Gradient Boosting achieved the
highest ROC-AUC score among the compared models.

Main responsibilities:
    1. Load the cleaned Telco churn dataset.
    2. Split the data into training and testing sets.
    3. Build preprocessing steps for numeric and categorical features.
    4. Use Optuna to search for the best Gradient Boosting parameters.
    5. Log every Optuna trial into MLflow.
    6. Log parameters, metrics, and model artifacts.
    7. Train the final best tuned Gradient Boosting model.
    8. Save the final tuned model locally for deployment.
    9. Save tuning results as a JSON report.
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
from sklearn.ensemble import GradientBoostingClassifier
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


# Ignore warnings to keep terminal and MLflow output clean.
warnings.filterwarnings("ignore")


# ============================================================
# Project paths
# ============================================================

# BASE_DIR points to the main RetainIQ project folder.
# Example:
# C:/Users/ASUS/Downloads/RetainIQ
BASE_DIR = Path(__file__).resolve().parents[1]

# Cleaned dataset used for model training and tuning.
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"

# Folder where trained model files are stored.
MODEL_DIR = BASE_DIR / "models"

# Folder where tuning results are saved for the report.
REPORT_DIR = BASE_DIR / "reports" / "tuning_gradient_boosting"

# Create folders if they do not already exist.
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# MLflow experiment name for Gradient Boosting tuning.
EXPERIMENT_NAME = "RetainIQ-Pro-GradientBoosting-Tuning"


def load_data():
    """
    Load the cleaned dataset and create train/test split.

    The target column is Churn.
    Features are all columns except Churn.

    Stratification is used to preserve the churn/non-churn distribution
    in both training and testing sets.

    Returns:
        x_train: Training features.
        x_test: Testing features.
        y_train: Training labels.
        y_test: Testing labels.
    """
    df = pd.read_csv(DATA_PATH)

    # Separate input features and target variable.
    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    # Use fixed random_state for reproducibility.
    return train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )


def build_preprocessor(x_train):
    """
    Build preprocessing pipeline for mixed data types.

    Numeric features:
        - StandardScaler is used to normalize numeric values.

    Categorical features:
        - OneHotEncoder converts text categories into numeric columns.
        - handle_unknown='ignore' prevents errors if new categories appear
          during prediction.

    Args:
        x_train: Training feature dataframe.

    Returns:
        ColumnTransformer object.
    """
    # Select numeric columns such as tenure, MonthlyCharges, TotalCharges.
    numeric_features = x_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Select categorical columns such as gender, Contract, PaymentMethod.
    categorical_features = x_train.select_dtypes(
        include=["object"]
    ).columns

    # Combine numeric and categorical transformations into one preprocessor.
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    return preprocessor


def calculate_metrics(model, x_test, y_test):
    """
    Evaluate the trained model using classification metrics.

    Metrics:
        accuracy:
            Overall correct predictions.

        precision:
            How many predicted churn customers were actually churn.

        recall:
            How many actual churn customers were detected.

        f1_score:
            Balance between precision and recall.

        roc_auc:
            Measures how well the model ranks churn risk probability.

    Args:
        model: Trained sklearn pipeline.
        x_test: Test features.
        y_test: Test labels.

    Returns:
        Dictionary containing evaluation metrics.
    """
    # Predicted churn class labels.
    y_pred = model.predict(x_test)

    # Predicted churn probability for ROC-AUC calculation.
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
    Define the Gradient Boosting hyperparameter search space.

    Optuna will try different values for these parameters and select
    the best combination based on ROC-AUC.

    Args:
        trial: Optuna trial object.

    Returns:
        Dictionary of Gradient Boosting parameters.
    """
    params = {
        # Number of boosting stages.
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),

        # Controls how much each tree contributes.
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),

        # Maximum depth of individual decision trees.
        "max_depth": trial.suggest_int("max_depth", 2, 6),

        # Minimum samples required to split an internal node.
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),

        # Minimum samples required at a leaf node.
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),

        # Fraction of samples used for fitting each tree.
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),

        # Fixed random state for reproducibility.
        "random_state": 42,
    }

    return params


def main():
    """
    Run the full Gradient Boosting tuning workflow.

    Workflow:
        1. Connect to MLflow tracking server.
        2. Load data.
        3. Build preprocessing pipeline.
        4. Run Optuna tuning trials.
        5. Log each trial to MLflow.
        6. Train final best tuned model.
        7. Log best model to MLflow.
        8. Save model locally.
        9. Save tuning results to JSON.
    """
    # Connect to the local MLflow tracking server running on port 5000.
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # Create or select the MLflow experiment for this tuning workflow.
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load dataset and build preprocessing pipeline.
    x_train, x_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor(x_train)

    def objective(trial):
        """
        Optuna objective function.

        Each time Optuna calls this function:
            - A new set of hyperparameters is suggested.
            - A Gradient Boosting model is trained.
            - The model is evaluated.
            - Metrics and parameters are logged to MLflow.

        The returned value is ROC-AUC because churn ranking is important
        for identifying high-risk customers.
        """
        params = create_model_params(trial)

        # Each Optuna trial is logged as a separate MLflow run.
        with mlflow.start_run(run_name=f"Gradient Boosting Trial {trial.number}"):

            # Create Gradient Boosting model with Optuna parameters.
            model = GradientBoostingClassifier(**params)

            # Combine preprocessing and model into one pipeline.
            # This ensures the same preprocessing is used during training
            # and future deployment.
            pipeline = Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("classifier", model),
                ]
            )

            # Train model for this trial.
            pipeline.fit(x_train, y_train)

            # Evaluate model performance.
            metrics = calculate_metrics(pipeline, x_test, y_test)

            # Log model type and trial number.
            mlflow.log_param("model_type", "Tuned Gradient Boosting")
            mlflow.log_param("trial_number", trial.number)

            # Log all hyperparameters suggested by Optuna.
            mlflow.log_params(params)

            # Log all evaluation metrics.
            mlflow.log_metrics(metrics)

            # Optimize ROC-AUC because the project focuses on ranking
            # customers by churn risk.
            return metrics["roc_auc"]

    # Create Optuna study.
    # direction='maximize' means Optuna tries to maximize ROC-AUC.
    study = optuna.create_study(direction="maximize")

    # Run 25 tuning trials.
    # Increase this number later only if you have more time.
    study.optimize(objective, n_trials=25)

    # Get best hyperparameters from Optuna.
    best_params = study.best_params
    best_params["random_state"] = 42

    # Build the final Gradient Boosting model using best parameters.
    best_model = GradientBoostingClassifier(**best_params)

    # Create final pipeline with preprocessing + tuned model.
    best_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", best_model),
        ]
    )

    # Train final tuned model.
    best_pipeline.fit(x_train, y_train)

    # Evaluate final tuned model.
    best_metrics = calculate_metrics(best_pipeline, x_test, y_test)

    # Generate predictions for MLflow model signature.
    y_pred = best_pipeline.predict(x_test)

    # Infer model input/output schema for MLflow.
    signature = infer_signature(x_test, y_pred)

    # Log final tuned Gradient Boosting model to MLflow.
    with mlflow.start_run(run_name="Best Tuned Gradient Boosting"):
        mlflow.log_param("model_type", "Best Tuned Gradient Boosting")
        mlflow.log_param("model_stage", "production_candidate")
        mlflow.log_params(best_params)
        mlflow.log_metrics(best_metrics)

        # Save the trained model pipeline as an MLflow artifact.
        mlflow.sklearn.log_model(
            sk_model=best_pipeline,
            artifact_path="model",
            signature=signature,
            input_example=x_test.head(5),
        )

    # Save model locally for deployment.
    model_path = MODEL_DIR / "best_tuned_gradient_boosting_model.pkl"
    joblib.dump(best_pipeline, model_path)

    # Save best parameters and metrics for report documentation.
    results = {
        "best_params": best_params,
        "best_metrics": best_metrics,
        "best_model_path": str(model_path),
    }

    results_path = REPORT_DIR / "best_gradient_boosting_tuning_results.json"

    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Print final results in terminal.
    print("\nBest tuned Gradient Boosting model saved successfully.")
    print("\nBest parameters:")
    print(best_params)

    print("\nBest metrics:")
    print(best_metrics)

    print(f"\nSaved model to: {model_path}")
    print(f"Saved tuning report to: {results_path}")


if __name__ == "__main__":
    main()