"""
Purpose:
    Detect whether production/customer data has changed compared
    to the original training data.

Why this matters:
    In real companies, customer behavior changes over time.
    A model that worked well during training may become less accurate
    if production data starts drifting.

Outputs:
    - Drift summary CSV
    - Drift report JSON
    - Drift bar chart
"""

from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


# Project paths


BASE_DIR = Path(__file__).resolve().parent.parent

REFERENCE_DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_clean.csv"
PRODUCTION_DATA_PATH = BASE_DIR / "data" / "production_simulation" / "production_customers.csv"

REPORT_DIR = BASE_DIR / "reports" / "drift"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)



# Configuration


NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
]

DRIFT_THRESHOLD = 0.10



# Helper functions


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load reference training data and simulated production data.
    """

    if not REFERENCE_DATA_PATH.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_DATA_PATH}")

    reference_df = pd.read_csv(REFERENCE_DATA_PATH)

    if PRODUCTION_DATA_PATH.exists():
        production_df = pd.read_csv(PRODUCTION_DATA_PATH)
    else:
        # If no production simulation exists, create a realistic sample
        # from the reference dataset and slightly modify values.
        production_df = reference_df.sample(
            n=min(500, len(reference_df)),
            random_state=42
        ).copy()

        production_df["MonthlyCharges"] = production_df["MonthlyCharges"] * 1.08
        production_df["tenure"] = production_df["tenure"] * 0.92

        PRODUCTION_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        production_df.to_csv(PRODUCTION_DATA_PATH, index=False)

    return reference_df, production_df


def calculate_numeric_drift(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
    column: str,
) -> dict:
    """
    Calculate simple drift score for one numeric column.

    Drift score is calculated as relative mean difference:
        abs(reference_mean - production_mean) / reference_mean
    """

    reference_mean = reference_df[column].mean()
    production_mean = production_df[column].mean()

    if reference_mean == 0:
        drift_score = 0
    else:
        drift_score = abs(production_mean - reference_mean) / abs(reference_mean)

    return {
        "feature": column,
        "reference_mean": round(reference_mean, 4),
        "production_mean": round(production_mean, 4),
        "drift_score": round(drift_score, 4),
        "drift_detected": drift_score >= DRIFT_THRESHOLD,
    }


def generate_drift_summary(
    reference_df: pd.DataFrame,
    production_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate drift summary for selected numeric columns.
    """

    results = []

    for column in NUMERIC_COLUMNS:
        if column in reference_df.columns and column in production_df.columns:
            results.append(
                calculate_numeric_drift(reference_df, production_df, column)
            )

    return pd.DataFrame(results)


def save_drift_outputs(drift_df: pd.DataFrame) -> None:
    """
    Save drift results as CSV, JSON, and bar chart.
    """

    csv_path = REPORT_DIR / "drift_summary.csv"
    json_path = REPORT_DIR / "drift_report.json"
    figure_path = FIGURE_DIR / "drift_scores.png"

    drift_df.to_csv(csv_path, index=False)

    report = {
        "method": "Relative mean difference",
        "drift_threshold": DRIFT_THRESHOLD,
        "total_features_checked": int(len(drift_df)),
        "features_with_drift": int(drift_df["drift_detected"].sum()),
        "drift_detected": bool(drift_df["drift_detected"].any()),
        "features": drift_df.to_dict(orient="records"),
    }

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    plt.figure(figsize=(8, 5))
    plt.bar(drift_df["feature"], drift_df["drift_score"])
    plt.axhline(
        DRIFT_THRESHOLD,
        linestyle="--",
        label=f"Threshold = {DRIFT_THRESHOLD}"
    )
    plt.title("RetainIQ Pro - Data Drift Scores")
    plt.xlabel("Feature")
    plt.ylabel("Drift Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    print("\nDrift monitoring completed successfully.")
    print(f"CSV saved to: {csv_path}")
    print(f"JSON report saved to: {json_path}")
    print(f"Figure saved to: {figure_path}")


def main() -> None:
    """
    Main execution function.
    """

    print("\nRetainIQ Pro - Data Drift Monitoring\n")

    reference_df, production_df = load_data()

    drift_df = generate_drift_summary(reference_df, production_df)

    if drift_df.empty:
        raise ValueError("No valid numeric columns found for drift monitoring.")

    print("Drift Summary:")
    print(drift_df)

    save_drift_outputs(drift_df)


if __name__ == "__main__":
    main()