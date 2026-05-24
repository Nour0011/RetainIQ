"""
Purpose:
    This script trains multiple machine learning models for customer churn
    prediction and logs all experiments to MLflow.

Requirements covered:
    1. Experiment tracking
    2. Model training
    3. MLflow parameter logging
    4. MLflow metric logging
    5. MLflow artifact logging
    6. Best model selection
    7. Model saving for deployment
    8. MLflow model signature logging
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd

from lightgbm import LGBMClassifier
from mlflow.models.signature import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


# =========================
# Project paths
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
ARTIFACT_DIR = BASE_DIR / "reports" / "training_artifacts"
MODEL_DIR = BASE_DIR / "models"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "RetainIQ-Pro-Churn-Experiments"


def load_data():
    """
    Load the cleaned dataset and split it into training and testing sets.

    The target column is:
        Churn

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
    Build preprocessing steps for numeric and categorical features.

    Numeric features:
        - StandardScaler

    Categorical features:
        - OneHotEncoder

    This creates a reusable production pipeline that can be saved and deployed
    together with the trained model.
    """
    numeric_features = x_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = x_train.select_dtypes(
        include=["object"]
    ).columns

    numeric_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def get_models():
    """
    Define all baseline models used for comparison.

    Contribution:
        Multiple algorithms are compared instead of training only one model.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            class_weight="balanced",
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            eval_metric="logloss",
            random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            random_state=42,
            verbose=-1,
        ),
    }

    return models


def save_confusion_matrix(y_test, y_pred, model_name):
    """
    Save the confusion matrix as an image artifact.

    This artifact helps explain:
        - true churn predictions
        - false churn predictions
        - missed churn customers
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix(y_test, y_pred),
        display_labels=["No Churn", "Churn"],
    )

    display.plot(ax=ax)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()

    path = ARTIFACT_DIR / f"{model_name}_confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_roc_curve(model, x_test, y_test, model_name):
    """
    Save ROC curve as an image artifact.

    ROC-AUC shows how well the model separates churn and non-churn customers.
    """
    fig, ax = plt.subplots(figsize=(5, 4))

    RocCurveDisplay.from_estimator(model, x_test, y_test, ax=ax)

    plt.title(f"ROC Curve - {model_name}")
    plt.tight_layout()

    path = ARTIFACT_DIR / f"{model_name}_roc_curve.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_classification_report(y_test, y_pred, model_name):
    """
    Save precision, recall, F1-score, and support as a JSON artifact.
    """
    report = classification_report(y_test, y_pred, output_dict=True)

    path = ARTIFACT_DIR / f"{model_name}_classification_report.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return path


def train_and_log_model(
    model_name,
    model,
    preprocessor,
    x_train,
    x_test,
    y_train,
    y_test,
):
    """
    Train one model and log the full experiment to MLflow.

    Logged to MLflow:
        - model parameters
        - accuracy
        - precision
        - recall
        - F1-score
        - ROC-AUC
        - confusion matrix
        - ROC curve
        - classification report
        - trained preprocessing + model pipeline
        - MLflow model signature
        - MLflow input example
    """
    with mlflow.start_run(run_name=model_name):
        # Build a complete pipeline:
        # preprocessing + classifier.
        # This is important because the same transformation steps are used
        # during both training and deployment.
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        # Train the model.
        pipeline.fit(x_train, y_train)

        # Generate predictions.
        y_pred = pipeline.predict(x_test)
        y_proba = pipeline.predict_proba(x_test)[:, 1]

        # Infer MLflow model signature.
        # This helps MLflow understand the expected input and output schema.
        # It also prevents the Model Registry schema section from appearing empty.
        signature = infer_signature(x_test, y_pred)

        # Calculate evaluation metrics.
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        }

        # Log model information to MLflow.
        mlflow.log_param("model_name", model_name)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)

        # Save and log artifacts.
        confusion_matrix_path = save_confusion_matrix(
            y_test,
            y_pred,
            model_name,
        )
        roc_curve_path = save_roc_curve(
            pipeline,
            x_test,
            y_test,
            model_name,
        )
        report_path = save_classification_report(
            y_test,
            y_pred,
            model_name,
        )

        mlflow.log_artifact(str(confusion_matrix_path))
        mlflow.log_artifact(str(roc_curve_path))
        mlflow.log_artifact(str(report_path))

        # Log the complete preprocessing + model pipeline.
        #
        # signature:
        #     Stores the input/output schema in MLflow.
        #
        # input_example:
        #     Stores sample input rows so MLflow can display expected input
        #     format inside the model artifact and registry.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=x_test.head(5),
        )

        print(f"\nModel: {model_name}")
        print(metrics)

        return {
            "model_name": model_name,
            "pipeline": pipeline,
            **metrics,
        }


def main():
    """
    Execute the full baseline training pipeline.

    Final output:
        - MLflow experiment runs
        - training artifacts
        - model comparison CSV
        - best model saved as best_churn_model.pkl
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, x_test, y_train, y_test = load_data()

    preprocessor = build_preprocessor(x_train)
    models = get_models()

    results = []

    for model_name, model in models.items():
        result = train_and_log_model(
            model_name,
            model,
            preprocessor,
            x_train,
            x_test,
            y_train,
            y_test,
        )
        results.append(result)

    # Save comparison table.
    results_df = pd.DataFrame(results).drop(columns=["pipeline"])
    results_path = ARTIFACT_DIR / "model_comparison_results.csv"
    results_df.to_csv(results_path, index=False)

    # Select the best model based on F1-score.
    best_result = max(results, key=lambda item: item["f1_score"])
    best_model_path = MODEL_DIR / "best_churn_model.pkl"

    joblib.dump(best_result["pipeline"], best_model_path)

    print("\nBest Model Selected:")
    print(best_result["model_name"])
    print(f"F1 Score: {best_result['f1_score']:.4f}")
    print(f"ROC-AUC: {best_result['roc_auc']:.4f}")
    print(f"Saved best model to: {best_model_path}")


if __name__ == "__main__":
    main()