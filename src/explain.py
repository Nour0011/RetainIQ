"""

Purpose:
    Explain the trained churn prediction model using SHAP.

"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "best_tuned_gradient_boosting_model.pkl"

REPORT_DIR = BASE_DIR / "reports" / "explainability"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "RetainIQ-Pro-Explainability"


def load_resources():
    """Load dataset and trained model."""
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    _, x_test, _, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return model, x_test.reset_index(drop=True), y_test.reset_index(drop=True)


def get_onehot_encoder(preprocessor):
    """
    Safely extract OneHotEncoder from the preprocessor.

    This handles both cases:
        1. cat transformer is directly OneHotEncoder
        2. cat transformer is a Pipeline containing OneHotEncoder
    """
    cat_transformer = preprocessor.named_transformers_["cat"]

    if hasattr(cat_transformer, "named_steps"):
        return cat_transformer.named_steps["onehot"]

    return cat_transformer


def transform_features(model, x_test):
    """
    Transform test features using the same preprocessing pipeline used
    during training.
    """
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    transformed_data = preprocessor.transform(x_test)

    numeric_features = x_test.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = x_test.select_dtypes(
        include=["object"]
    ).columns.tolist()

    onehot_encoder = get_onehot_encoder(preprocessor)

    categorical_names = onehot_encoder.get_feature_names_out(
        categorical_features
    ).tolist()

    feature_names = numeric_features + categorical_names

    transformed_df = pd.DataFrame(
        transformed_data,
        columns=feature_names,
    )

    return classifier, transformed_df


def compute_shap_values(classifier, transformed_df):
    """Compute SHAP values for the churn class."""
    print("Computing SHAP values...")

    sample_size = min(300, len(transformed_df))
    x_sample = transformed_df.iloc[:sample_size].copy()

    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(x_sample)

    if isinstance(shap_values, list):
        shap_churn = shap_values[1]
    else:
        shap_churn = shap_values

    shap_churn = np.array(shap_churn)

    if len(shap_churn.shape) == 3:
        shap_churn = shap_churn[:, :, 1]

    return explainer, shap_churn, x_sample


def save_global_importance(shap_values, x_sample):
    """Save global SHAP feature importance chart."""
    mean_abs_values = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "feature": x_sample.columns,
            "mean_abs_shap": mean_abs_values,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    top_features = importance_df.head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(
        top_features["feature"][::-1],
        top_features["mean_abs_shap"][::-1],
    )
    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title("Top 15 Global Churn Drivers")

    plt.tight_layout()

    path = FIGURE_DIR / "shap_global_importance.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return path, importance_df


def save_summary_plot(shap_values, x_sample):
    """Save SHAP summary plot."""
    plt.figure(figsize=(9, 7))

    shap.summary_plot(
        shap_values,
        x_sample,
        max_display=15,
        show=False,
    )

    plt.title("SHAP Summary Plot - Churn Prediction")
    plt.tight_layout()

    path = FIGURE_DIR / "shap_summary_plot.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close("all")

    return path


def save_high_risk_customer_explanations(
    explainer,
    shap_values,
    x_sample,
    model,
    original_x_test,
):
    """Save waterfall plots for top 3 high-risk customers."""
    probabilities = model.predict_proba(
        original_x_test.iloc[: len(x_sample)]
    )[:, 1]

    top_indices = np.argsort(probabilities)[-3:][::-1]
    saved_paths = []

    for rank, index in enumerate(top_indices, start=1):
        probability = probabilities[index]

        expected_value = explainer.expected_value

        if isinstance(expected_value, list):
            base_value = expected_value[1]
        elif isinstance(expected_value, np.ndarray):
            base_value = expected_value[1] if len(expected_value) > 1 else expected_value[0]
        else:
            base_value = expected_value

        explanation = shap.Explanation(
            values=shap_values[index],
            base_values=base_value,
            data=x_sample.iloc[index].values,
            feature_names=x_sample.columns.tolist(),
        )

        plt.figure(figsize=(9, 6))

        shap.waterfall_plot(
            explanation,
            max_display=12,
            show=False,
        )

        plt.title(
            f"High-Risk Customer {rank} - Churn Probability: {probability:.1%}"
        )

        plt.tight_layout()

        path = FIGURE_DIR / f"shap_waterfall_customer_{rank}.png"
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close("all")

        saved_paths.append(path)

        print(
            f"Saved high-risk customer {rank} explanation "
            f"(probability={probability:.1%})"
        )

    return saved_paths


def save_explanation_report(importance_df):
    """Save SHAP explanation report as JSON."""
    top_features = importance_df.head(10).copy()
    top_features["mean_abs_shap"] = top_features["mean_abs_shap"].round(5)

    report = {
        "method": "SHAP TreeExplainer",
        "model": "Best Tuned Random Forest",
        "purpose": "Explain churn prediction drivers",
        "top_10_features": top_features.to_dict(orient="records"),
    }

    path = REPORT_DIR / "explanation_report.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return path


def log_to_mlflow(figure_paths, report_path, importance_df):
    """Log SHAP artifacts to MLflow."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="SHAP Explainability Run"):
        mlflow.log_param("explainability_method", "SHAP TreeExplainer")
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("sample_size", 300)

        for _, row in importance_df.head(5).iterrows():
            metric_name = (
                "shap_importance_"
                + row["feature"]
                .replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")[:40]
            )

            mlflow.log_metric(
                metric_name,
                round(float(row["mean_abs_shap"]), 5),
            )

        for figure_path in figure_paths:
            mlflow.log_artifact(str(figure_path))

        mlflow.log_artifact(str(report_path))


def main():
    """Run full SHAP explainability pipeline."""
    print("\nRetainIQ Pro - SHAP Explainability\n")

    model, x_test, _ = load_resources()

    classifier, transformed_df = transform_features(
        model,
        x_test,
    )

    explainer, shap_values, x_sample = compute_shap_values(
        classifier,
        transformed_df,
    )

    global_path, importance_df = save_global_importance(
        shap_values,
        x_sample,
    )

    summary_path = save_summary_plot(
        shap_values,
        x_sample,
    )

    waterfall_paths = save_high_risk_customer_explanations(
        explainer,
        shap_values,
        x_sample,
        model,
        x_test,
    )

    report_path = save_explanation_report(
        importance_df,
    )

    figure_paths = [
        global_path,
        summary_path,
        *waterfall_paths,
    ]

    log_to_mlflow(
        figure_paths,
        report_path,
        importance_df,
    )

    print("\nExplainability completed successfully.")
    print(f"Reports saved to: {REPORT_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print("Results logged to MLflow.")


if __name__ == "__main__":
    main()