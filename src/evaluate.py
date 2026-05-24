"""

Purpose:
    This script evaluates the best tuned churn model beyond basic accuracy.
    It includes standard ML metrics, business cost analysis, threshold
    optimization, calibration analysis, and MLflow artifact logging.

Professional contributions:
    1. ROC-AUC evaluation
    2. Precision-recall analysis for imbalanced churn data
    3. Calibration curve
    4. Business cost matrix
    5. Optimal decision threshold
    6. Cross-validation
    7. MLflow evaluation artifacts

"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)

warnings.filterwarnings("ignore")


# Project paths

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "best_tuned_churn_model.pkl"

REPORT_DIR = BASE_DIR / "reports" / "evaluation"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "RetainIQ-Pro-Deep-Evaluation"

# Business assumptions in USD.
# False negative = missed churner, usually expensive.
# False positive = retention offer given to a customer who would stay.
COST_FALSE_NEGATIVE = 200
COST_FALSE_POSITIVE = 25


def load_resources():
    """
    Load cleaned dataset and best tuned model.

    Returns:
        model:
            Trained production model.

        x_train, x_test, y_train, y_test:
            Train/test split for evaluation.
    """
    df = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    return model, x_train, x_test, y_train, y_test


def compute_standard_metrics(model, x_test, y_test):
    """
    Compute standard classification metrics.

    Metrics:
        accuracy:
            Overall correctness.

        precision:
            How many predicted churners were actual churners.

        recall:
            How many actual churners were detected.

        f1_score:
            Balance between precision and recall.

        roc_auc:
            Ability to separate churn and non-churn classes.

        average_precision:
            Precision-recall performance, useful for imbalanced data.

        brier_score:
            Probability calibration error. Lower is better.
    """
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "average_precision": round(
            float(average_precision_score(y_test, y_proba)),
            4,
        ),
        "brier_score": round(float(brier_score_loss(y_test, y_proba)), 4),
    }

    return metrics, y_pred, y_proba


def find_optimal_threshold(y_test, y_proba):
    """
    Find the probability threshold that minimizes business cost.

    Default classification threshold is normally 0.50.

    In churn prediction, missing a real churn customer is more costly than
    offering a discount to a customer who would not churn.

    Cost formula:
        total_cost = false_negatives * COST_FALSE_NEGATIVE
                   + false_positives * COST_FALSE_POSITIVE
    """
    thresholds = np.arange(0.05, 0.95, 0.01)

    best_threshold = 0.50
    best_cost = float("inf")
    cost_results = []

    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test,
            y_pred_threshold,
        ).ravel()

        total_cost = (
            fn * COST_FALSE_NEGATIVE
            + fp * COST_FALSE_POSITIVE
        )

        result = {
            "threshold": round(float(threshold), 2),
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_cost_usd": int(total_cost),
        }

        cost_results.append(result)

        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = threshold

    return float(best_threshold), int(best_cost), cost_results


def compute_cross_validation_score(model, x_train, y_train):
    """
    Estimate generalization using 5-fold stratified cross-validation.

    Cross-validation helps prove that model performance is stable and not
    caused by one lucky train/test split.
    """
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    cv_scores = cross_val_score(
        model,
        x_train,
        y_train,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
    )

    return {
        "cv_roc_auc_mean": round(float(np.mean(cv_scores)), 4),
        "cv_roc_auc_std": round(float(np.std(cv_scores)), 4),
        "cv_scores": [round(float(score), 4) for score in cv_scores],
    }


def save_confusion_matrix(y_test, y_pred):
    """
    Save confusion matrix plot.
    """
    matrix = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix)

    ax.set_title("Confusion Matrix - Best Tuned Model")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                matrix[i, j],
                ha="center",
                va="center",
            )

    plt.tight_layout()

    path = FIGURE_DIR / "confusion_matrix.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_roc_curve(y_test, y_proba):
    """
    Save ROC curve.
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_proba)
    auc_score = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"ROC-AUC = {auc_score:.4f}",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Random Classifier",
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Best Tuned Model")
    ax.legend(loc="lower right")

    plt.tight_layout()

    path = FIGURE_DIR / "roc_curve.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_precision_recall_curve(y_test, y_proba):
    """
    Save precision-recall curve.

    Precision-recall curves are very useful for imbalanced datasets where the
    minority class, churners, is the most important class.
    """
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    average_precision = average_precision_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        recall,
        precision,
        label=f"Average Precision = {average_precision:.4f}",
    )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Best Tuned Model")
    ax.legend(loc="upper right")

    plt.tight_layout()

    path = FIGURE_DIR / "precision_recall_curve.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_calibration_curve(y_test, y_proba):
    """
    Save calibration curve.

    A calibrated model gives probabilities that match real event frequencies.
    This matters because RetainIQ uses churn probability for risk levels.
    """
    fraction_positive, mean_predicted = calibration_curve(
        y_test,
        y_proba,
        n_bins=10,
    )

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.plot(
        mean_predicted,
        fraction_positive,
        marker="o",
        label="Model Calibration",
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration",
    )

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Actual Churn")
    ax.set_title("Calibration Curve - Best Tuned Model")
    ax.legend()

    plt.tight_layout()

    path = FIGURE_DIR / "calibration_curve.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_cost_threshold_curve(cost_results):
    """
    Save business cost versus decision threshold plot.

    This plot shows which probability threshold minimizes business cost.
    """
    thresholds = [item["threshold"] for item in cost_results]
    costs = [item["total_cost_usd"] for item in cost_results]

    best_index = costs.index(min(costs))
    best_threshold = thresholds[best_index]
    best_cost = costs[best_index]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(
        thresholds,
        costs,
        linewidth=2,
    )

    ax.axvline(
        best_threshold,
        linestyle="--",
        label=f"Best threshold = {best_threshold:.2f}",
    )

    ax.scatter(
        best_threshold,
        best_cost,
        s=80,
    )

    ax.set_xlabel("Decision Threshold")
    ax.set_ylabel("Total Business Cost (USD)")
    ax.set_title("Business Cost vs Decision Threshold")
    ax.legend()

    plt.tight_layout()

    path = FIGURE_DIR / "business_cost_threshold_curve.png"
    plt.savefig(path, dpi=300)
    plt.close(fig)

    return path


def save_classification_report_json(y_test, y_pred):
    """
    Save classification report as JSON.
    """
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
    )

    path = REPORT_DIR / "classification_report.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return path


def save_evaluation_report(
    metrics,
    cv_results,
    best_threshold,
    best_cost,
    cost_results,
):
    """
    Save complete evaluation report as JSON.
    """
    report = {
        "standard_metrics": metrics,
        "cross_validation": cv_results,
        "business_cost_analysis": {
            "cost_false_negative_usd": COST_FALSE_NEGATIVE,
            "cost_false_positive_usd": COST_FALSE_POSITIVE,
            "optimal_threshold": round(float(best_threshold), 2),
            "minimum_total_cost_usd": int(best_cost),
        },
        "threshold_cost_table": cost_results,
    }

    path = REPORT_DIR / "evaluation_report.json"

    with open(path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return path


def log_to_mlflow(
    metrics,
    cv_results,
    best_threshold,
    best_cost,
    figure_paths,
    report_paths,
):
    """
    Log evaluation metrics and artifacts to MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name="Deep Evaluation Run"):
        # Standard metrics.
        mlflow.log_metrics(metrics)

        # Cross-validation metrics.
        mlflow.log_metric(
            "cv_roc_auc_mean",
            cv_results["cv_roc_auc_mean"],
        )

        mlflow.log_metric(
            "cv_roc_auc_std",
            cv_results["cv_roc_auc_std"],
        )

        # Business metrics.
        mlflow.log_metric(
            "optimal_threshold",
            round(float(best_threshold), 4),
        )

        mlflow.log_metric(
            "minimum_business_cost_usd",
            int(best_cost),
        )

        mlflow.log_param(
            "cost_false_negative_usd",
            COST_FALSE_NEGATIVE,
        )

        mlflow.log_param(
            "cost_false_positive_usd",
            COST_FALSE_POSITIVE,
        )

        # Log all plots.
        for figure_path in figure_paths:
            mlflow.log_artifact(str(figure_path))

        # Log JSON reports.
        for report_path in report_paths:
            mlflow.log_artifact(str(report_path))


def main():
    """
    Run the full deep evaluation pipeline.
    """
    print("\nRetainIQ Pro - Deep Model Evaluation\n")

    model, x_train, x_test, y_train, y_test = load_resources()

    metrics, y_pred, y_proba = compute_standard_metrics(
        model,
        x_test,
        y_test,
    )

    best_threshold, best_cost, cost_results = find_optimal_threshold(
        y_test,
        y_proba,
    )

    cv_results = compute_cross_validation_score(
        model,
        x_train,
        y_train,
    )

    confusion_path = save_confusion_matrix(
        y_test,
        y_pred,
    )

    roc_path = save_roc_curve(
        y_test,
        y_proba,
    )

    pr_path = save_precision_recall_curve(
        y_test,
        y_proba,
    )

    calibration_path = save_calibration_curve(
        y_test,
        y_proba,
    )

    cost_curve_path = save_cost_threshold_curve(
        cost_results,
    )

    classification_report_path = save_classification_report_json(
        y_test,
        y_pred,
    )

    evaluation_report_path = save_evaluation_report(
        metrics,
        cv_results,
        best_threshold,
        best_cost,
        cost_results,
    )

    figure_paths = [
        confusion_path,
        roc_path,
        pr_path,
        calibration_path,
        cost_curve_path,
    ]

    report_paths = [
        classification_report_path,
        evaluation_report_path,
    ]

    log_to_mlflow(
        metrics,
        cv_results,
        best_threshold,
        best_cost,
        figure_paths,
        report_paths,
    )

    print("Evaluation completed successfully.")
    print("\nStandard metrics:")
    print(metrics)

    print("\nCross-validation:")
    print(cv_results)

    print("\nBusiness threshold analysis:")
    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Minimum business cost: ${best_cost:,}")

    print(f"\nReports saved to: {REPORT_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")
    print("\nResults logged to MLflow.")


if __name__ == "__main__":
    main()