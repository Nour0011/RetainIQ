"""

Purpose:

This module generates business-focused retention
recommendations for high-risk churn customers.

Features:

1. Customer risk segmentation
2. Retention recommendation engine
3. Estimated revenue protection
4. ROI-style business analysis
5. MLflow experiment tracking
6. Export reports for dashboards


"""

# IMPORT LIBRARIES


import os
import json
import warnings
from datetime import datetime

import joblib
import mlflow
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")



# PROJECT PATHS


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "telco_churn_clean.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "best_tuned_gradient_boosting_model.pkl"
)

PREPROCESSOR_PATH = os.path.join(
    BASE_DIR,
    "models",
    "preprocessor.pkl"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports",
    "recommendations"
)

os.makedirs(REPORT_DIR, exist_ok=True)



# LOAD DATA


def load_dataset():
    """
    Load and clean customer dataset.

    Returns:
        pd.DataFrame
    """

    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    return df



# LOAD MODEL


def load_model_objects():
    """
    Load trained model pipeline.

    The saved model already includes preprocessing and classifier.
    """

    model = joblib.load(MODEL_PATH)

    return model



# PREPARE FEATURES

def prepare_features(df):
    """
    Prepare features for prediction.

    Parameters:
        df (pd.DataFrame)

    Returns:
        X, y
    """

    target_column = "Churn"

    X = df.drop(
        columns=[
            target_column,
            "customerID",
        ],
        errors="ignore",
    )

    # The processed dataset already contains numeric labels:
    # 0 = No Churn
    # 1 = Churn
    y = df[target_column]

    return X, y



# GENERATE PREDICTIONS


def generate_predictions(model, X):
    """
    Generate churn probabilities using the saved model pipeline.
    """

    probabilities = model.predict_proba(X)[:, 1]

    return probabilities



# ASSIGN RISK LEVEL


def assign_risk_level(probability):
    """
    Assign customer risk category.

    Parameters:
        probability (float)

    Returns:
        str
    """

    if probability >= 0.80:
        return "Critical"

    if probability >= 0.60:
        return "High"

    if probability >= 0.40:
        return "Medium"

    return "Low"



# GENERATE BUSINESS ACTIONS


def generate_recommendation(row):
    """
    Generate personalized retention strategy.

    Parameters:
        row (pd.Series)

    Returns:
        str
    """

    recommendations = []

    if row["Contract"] == "Month-to-month":
        recommendations.append(
            "Offer long-term contract discount"
        )

    if row["OnlineSecurity"] == "No":
        recommendations.append(
            "Provide free online security trial"
        )

    if row["TechSupport"] == "No":
        recommendations.append(
            "Offer premium technical support"
        )

    if row["InternetService"] == "Fiber optic":
        recommendations.append(
            "Investigate fiber service satisfaction"
        )

    if row["PaymentMethod"] == "Electronic check":
        recommendations.append(
            "Encourage automatic payment setup"
        )

    if row["tenure"] < 12:
        recommendations.append(
            "Launch early customer engagement campaign"
        )

    if len(recommendations) == 0:
        recommendations.append(
            "Standard customer retention follow-up"
        )

    return " | ".join(recommendations)



# ESTIMATE REVENUE AT RISK


def estimate_revenue_risk(row):
    """
    Estimate annual revenue at risk.

    Parameters:
        row (pd.Series)

    Returns:
        float
    """

    return round(
        row["MonthlyCharges"] * 12,
        2
    )



# MAIN BUSINESS ANALYSIS


def run_business_recommendation_engine():
    """
    Main business recommendation workflow.
    """

    print("\nRetainIQ Pro - Recommendation Engine\n")


    # Load dataset


    df = load_dataset()


    # Load model
   

    model = load_model_objects()

 
    # Prepare features
  

    X, y = prepare_features(df)


    # Generate predictions


    probabilities = generate_predictions(
        model,
        X
    )


    # Create business dataframe


    results_df = df.copy()

    results_df["churn_probability"] = probabilities

    results_df["risk_level"] = results_df[
        "churn_probability"
    ].apply(assign_risk_level)

    results_df["retention_action"] = results_df.apply(
        generate_recommendation,
        axis=1
    )

    results_df["estimated_annual_value"] = (
        results_df.apply(
            estimate_revenue_risk,
            axis=1
        )
    )


    # Sort high-risk customers
   

    results_df = results_df.sort_values(
        by="churn_probability",
        ascending=False
    )

 
    # Save top customers
   

    top_customers = results_df.head(100)

    csv_path = os.path.join(
        REPORT_DIR,
        "top_100_high_risk_customers.csv"
    )

    top_customers.to_csv(
        csv_path,
        index=False
    )

   
    # Summary statistics
   

    total_customers = len(results_df)

    critical_customers = len(
        results_df[
            results_df["risk_level"] == "Critical"
        ]
    )

    high_customers = len(
        results_df[
            results_df["risk_level"] == "High"
        ]
    )

    total_revenue_risk = round(
        results_df[
            results_df["risk_level"].isin(
                ["Critical", "High"]
            )
        ]["estimated_annual_value"].sum(),
        2
    )

  
    # Build summary report
   

    summary_report = {
        "generated_at": str(datetime.now()),
        "total_customers": int(total_customers),
        "critical_risk_customers": int(
            critical_customers
        ),
        "high_risk_customers": int(
            high_customers
        ),
        "estimated_revenue_at_risk":
            total_revenue_risk
    }

    json_path = os.path.join(
        REPORT_DIR,
        "business_summary.json"
    )

    with open(json_path, "w") as file:
        json.dump(
            summary_report,
            file,
            indent=4
        )


    # MLFLOW LOGGING


    mlflow.set_experiment(
        "RetainIQ-Pro-Recommendation-Engine"
    )

    with mlflow.start_run(
        run_name="Business Recommendation Run"
    ):

        mlflow.log_metric(
            "total_customers",
            total_customers
        )

        mlflow.log_metric(
            "critical_customers",
            critical_customers
        )

        mlflow.log_metric(
            "high_risk_customers",
            high_customers
        )

        mlflow.log_metric(
            "estimated_revenue_at_risk",
            total_revenue_risk
        )

        mlflow.log_artifact(csv_path)

        mlflow.log_artifact(json_path)

   
    # TERMINAL OUTPUT
 

    print("Recommendation engine completed.\n")

    print("Summary:")
    print(summary_report)

    print("\nReports saved to:")
    print(REPORT_DIR)

    print("\nTop customers file:")
    print(csv_path)


# SCRIPT ENTRY POINT


if __name__ == "__main__":
    run_business_recommendation_engine()