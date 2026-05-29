"""

Purpose:
    This script simulates production data, monitors model performance,
    detects data drift, generates alerts, and logs monitoring results
    to MLflow.

Professional contributions:
    1. Production data simulation over multiple time windows
    2. PSI drift detection
    3. Kolmogorov-Smirnov statistical drift test
    4. Alert levels: LOW, MEDIUM, HIGH, CRITICAL
    5. Monitoring reports saved as JSON artifacts
    6. Monitoring metrics logged to MLflow

"""

from pathlib import Path
import json
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")


# Project paths

BASE_DIR = Path(__file__).resolve().parents[1]

DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "best_tuned_gradient_boosting_model.pkl"

REPORT_DIR = BASE_DIR / "reports" / "monitoring"
PRODUCTION_DIR = BASE_DIR / "data" / "production_simulation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENT_NAME = "RetainIQ-Pro-Monitoring"

# PSI thresholds used for drift interpretation.
PSI_LOW = 0.10
PSI_MEDIUM = 0.20
PSI_HIGH = 0.25


def load_resources():
    """
    Load the cleaned reference dataset and the trained production model.

    Returns:
        data:
            Cleaned reference dataset.

        model:
            Best tuned churn prediction model.
    """
    data = pd.read_csv(DATA_PATH)
    model = joblib.load(MODEL_PATH)

    return data, model


def create_production_data(data):
    """
    Simulate three production windows with increasing drift.

    Window 1:
        Mild drift through monthly charge increase.

    Window 2:
        Moderate drift through shorter tenure and more month-to-month
        contracts.

    Window 3:
        Stronger drift through combined price, tenure, contract, and
        senior citizen distribution changes.

    Returns:
        Dictionary containing simulated production datasets and paths.
    """
    windows = {}

    simulation_settings = [
        (1, 0.25, 42),
        (2, 0.30, 99),
        (3, 0.35, 7),
    ]

    for window_id, fraction, seed in simulation_settings:
        production_data = data.sample(
            frac=fraction,
            random_state=seed,
        ).copy()

        # Simulate price drift.
        production_data["MonthlyCharges"] = (
            production_data["MonthlyCharges"] * (1 + 0.08 * window_id)
        )

        production_data["TotalCharges"] = (
            production_data["TotalCharges"] * (1 + 0.05 * window_id)
        )

        # Simulate newer customers with shorter tenure.
        production_data["tenure"] = np.maximum(
            production_data["tenure"] - (3 * window_id),
            1,
        )

        # Simulate more risky month-to-month contracts.
        if "Contract" in production_data.columns and window_id >= 2:
            shift_index = production_data.sample(
                frac=0.20 * window_id,
                random_state=seed,
            ).index

            production_data.loc[
                shift_index,
                "Contract",
            ] = "Month-to-month"

        # Simulate senior citizen distribution shift in final window.
        if window_id == 3 and "SeniorCitizen" in production_data.columns:
            senior_index = production_data.sample(
                frac=0.15,
                random_state=seed,
            ).index

            production_data.loc[
                senior_index,
                "SeniorCitizen",
            ] = 1

        output_path = PRODUCTION_DIR / f"production_window_{window_id}.csv"
        production_data.to_csv(output_path, index=False)

        windows[window_id] = {
            "data": production_data,
            "path": output_path,
        }

    return windows


def calculate_psi(reference_series, production_series, bins=10):
    """
    Calculate Population Stability Index.

    PSI interpretation:
        < 0.10  : Stable
        < 0.20  : Monitor
        < 0.25  : Significant drift
        >= 0.25 : Critical drift

    Args:
        reference_series:
            Feature values from training/reference data.

        production_series:
            Feature values from simulated production data.

        bins:
            Number of bins used to compare distributions.

    Returns:
        PSI score as float.
    """
    reference_series = reference_series.dropna()
    production_series = production_series.dropna()

    reference_min = min(reference_series.min(), production_series.min())
    reference_max = max(reference_series.max(), production_series.max())

    breakpoints = np.linspace(reference_min, reference_max, bins + 1)

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    reference_counts = np.histogram(
        reference_series,
        bins=breakpoints,
    )[0]

    production_counts = np.histogram(
        production_series,
        bins=breakpoints,
    )[0]

    reference_percent = np.where(
        reference_counts == 0,
        0.0001,
        reference_counts / len(reference_series),
    )

    production_percent = np.where(
        production_counts == 0,
        0.0001,
        production_counts / len(production_series),
    )

    psi = np.sum(
        (production_percent - reference_percent)
        * np.log(production_percent / reference_percent)
    )

    return float(psi)


def classify_psi(psi_value):
    """
    Convert PSI score into readable drift severity.
    """
    if psi_value < PSI_LOW:
        return "Stable"

    if psi_value < PSI_MEDIUM:
        return "Monitor"

    if psi_value < PSI_HIGH:
        return "Significant"

    return "Critical"


def run_ks_test(reference_series, production_series):
    """
    Run Kolmogorov-Smirnov test.

    The KS test checks whether two distributions are statistically different.

    If p-value < 0.05:
        distribution shift is statistically significant.
    """
    statistic, p_value = ks_2samp(
        reference_series.dropna(),
        production_series.dropna(),
    )

    return float(statistic), float(p_value)


def calculate_feature_drift(reference_data, production_data):
    """
    Calculate drift report for each numeric feature.

    Report includes:
        - PSI score
        - PSI severity
        - KS statistic
        - KS p-value
        - distribution shift flag
        - mean shift percentage
    """
    numeric_columns = reference_data.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if "Churn" in numeric_columns:
        numeric_columns.remove("Churn")

    drift_report = {}

    for column in numeric_columns:
        reference_series = reference_data[column]
        production_series = production_data[column]

        psi_score = calculate_psi(
            reference_series,
            production_series,
        )

        psi_severity = classify_psi(psi_score)

        ks_statistic, ks_p_value = run_ks_test(
            reference_series,
            production_series,
        )

        reference_mean = reference_series.mean()
        production_mean = production_series.mean()

        mean_shift_percent = (
            abs(production_mean - reference_mean)
            / (reference_mean + 1e-9)
            * 100
        )

        drift_report[column] = {
            "psi_score": round(float(psi_score), 4),
            "psi_severity": psi_severity,
            "ks_statistic": round(float(ks_statistic), 4),
            "ks_p_value": round(float(ks_p_value), 4),
            "distribution_shifted": bool(ks_p_value < 0.05),
            "reference_mean": round(float(reference_mean), 4),
            "production_mean": round(float(production_mean), 4),
            "mean_shift_percent": round(float(mean_shift_percent), 2),
        }

    return drift_report


def calculate_production_metrics(model, production_data):
    """
    Calculate model performance metrics on simulated production data.

    In a real company, labels may arrive later.
    For this academic project, the existing Churn column is used as simulated
    delayed ground truth.
    """
    x_production = production_data.drop(columns=["Churn"])
    y_true = production_data["Churn"]

    y_pred = model.predict(x_production)
    y_proba = model.predict_proba(x_production)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "recall": round(float(recall_score(y_true, y_pred)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "average_churn_probability": round(float(np.mean(y_proba)), 4),
        "predicted_churn_rate": round(float(np.mean(y_pred)), 4),
    }

    return metrics


def generate_alert(drift_report, metrics):
    """
    Generate monitoring alert based on drift severity and performance.

    Alert logic:
        CRITICAL:
            multiple critical features or ROC-AUC below 0.70

        HIGH:
            multiple significant features or ROC-AUC below 0.75

        MEDIUM:
            at least one significant feature or ROC-AUC below 0.80

        LOW:
            model is stable
    """
    critical_features = [
        feature
        for feature, info in drift_report.items()
        if info["psi_severity"] == "Critical"
    ]

    significant_features = [
        feature
        for feature, info in drift_report.items()
        if info["psi_severity"] in ("Significant", "Critical")
    ]

    if len(critical_features) >= 2 or metrics["roc_auc"] < 0.70:
        level = "CRITICAL"
        action = (
            "Immediate model retraining required. "
            "Production performance or data distribution changed strongly."
        )

    elif len(significant_features) >= 2 or metrics["roc_auc"] < 0.75:
        level = "HIGH"
        action = (
            "Schedule retraining. Significant production drift detected."
        )

    elif len(significant_features) >= 1 or metrics["roc_auc"] < 0.80:
        level = "MEDIUM"
        action = (
            "Monitor closely. Some features show distribution shift."
        )

    else:
        level = "LOW"
        action = "Model stable. Continue standard monitoring."

    return {
        "alert_level": level,
        "recommended_action": action,
        "critical_features": critical_features,
        "significant_features": significant_features,
    }


def save_monitoring_report(
    window_id,
    metrics,
    drift_report,
    alert,
    production_path,
):
    """
    Save monitoring report as JSON.

    This file is also logged as an MLflow artifact.
    """
    report = {
        "window_id": window_id,
        "production_data_path": str(production_path),
        "alert": alert,
        "performance_metrics": metrics,
        "feature_drift": drift_report,
    }

    report_path = REPORT_DIR / f"monitoring_window_{window_id}.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    return report_path


def log_to_mlflow(window_id, metrics, drift_report, alert, report_path):
    """
    Log monitoring metrics, alert level, and report artifact to MLflow.
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run(run_name=f"Production Window {window_id}"):
        # Log model performance metrics.
        mlflow.log_metrics(metrics)

        # Log drift metrics.
        psi_values = [
            feature_info["psi_score"]
            for feature_info in drift_report.values()
        ]

        mlflow.log_metric(
            "average_psi_score",
            round(float(np.mean(psi_values)), 4),
        )

        mlflow.log_metric(
            "max_psi_score",
            round(float(np.max(psi_values)), 4),
        )

        mlflow.log_metric(
            "critical_features_count",
            len(alert["critical_features"]),
        )

        mlflow.log_metric(
            "significant_features_count",
            len(alert["significant_features"]),
        )

        # Log alert information.
        mlflow.set_tag("alert_level", alert["alert_level"])
        mlflow.set_tag("window_id", str(window_id))

        mlflow.log_param(
            "recommended_action",
            alert["recommended_action"],
        )

        # Log JSON report as artifact.
        mlflow.log_artifact(str(report_path))


def main():
    """
    Run the full multi-window monitoring pipeline.

    Final outputs:
        - simulated production CSV files
        - monitoring JSON reports
        - MLflow monitoring runs
        - drift alert levels
    """
    print("\nRetainIQ Pro - Production Monitoring and PSI Drift Detection\n")

    reference_data, model = load_resources()

    production_windows = create_production_data(reference_data)

    for window_id, window_info in production_windows.items():
        print(f"\n--- Production Window {window_id} ---")

        production_data = window_info["data"]
        production_path = window_info["path"]

        drift_report = calculate_feature_drift(
            reference_data,
            production_data,
        )

        metrics = calculate_production_metrics(
            model,
            production_data,
        )

        alert = generate_alert(
            drift_report,
            metrics,
        )

        report_path = save_monitoring_report(
            window_id,
            metrics,
            drift_report,
            alert,
            production_path,
        )

        log_to_mlflow(
            window_id,
            metrics,
            drift_report,
            alert,
            report_path,
        )

        print(f"Alert level: {alert['alert_level']}")
        print(f"Action: {alert['recommended_action']}")
        print(f"ROC-AUC: {metrics['roc_auc']}")
        print(f"F1-score: {metrics['f1_score']}")
        print(f"Predicted churn rate: {metrics['predicted_churn_rate']:.1%}")
        print(f"Report saved to: {report_path}")

    print("\nMonitoring complete. All windows logged to MLflow.")


if __name__ == "__main__":
    main()