"""
RetainIQ Pro - FastAPI Backend

Purpose:
    Product-style API for an AI customer retention platform.

Features:
    - Health check
    - Single customer churn prediction
    - Batch CSV churn prediction
    - Model-based what-if explanations
    - Retention simulation endpoint
    - ROI calculator
    - Monitoring counters
    - MLflow Registry production model loading with local fallback
"""

from pathlib import Path
from io import StringIO
from datetime import datetime
from typing import Dict, Any

import joblib
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow

mlflow.set_tracking_uri("file:///app/mlruns")


# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="RetainIQ Pro API",
    version="4.0",
    description="AI-powered customer retention intelligence platform",
)


# ============================================================
# CORS configuration
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Paths and production model loading
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_tuned_churn_model.pkl"

MODEL_NAME = "RetainIQ-Churn-Predictor"
MODEL_STAGE = "Production"

try:
    model = mlflow.pyfunc.load_model(
        model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
    )
    MODEL_SOURCE = "MLflow Model Registry"

except Exception:
    model = joblib.load(MODEL_PATH)
    MODEL_SOURCE = "Local PKL Fallback"


# ============================================================
# Monitoring counters
# ============================================================

monitoring_metrics = {
    "total_predictions": 0,
    "high_risk_predictions": 0,
    "medium_risk_predictions": 0,
    "low_risk_predictions": 0,
    "simulations_run": 0,
    "last_batch_size": 0,
    "last_prediction_time": None,
}


# ============================================================
# Input schema
# ============================================================

class CustomerData(BaseModel):
    """Input schema for one customer."""

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class SimulationRequest(BaseModel):
    """Request schema for what-if simulation."""

    customer: CustomerData
    intervention_cost: float = 100.0
    annual_customer_value: float = 1000.0


# ============================================================
# Utility functions
# ============================================================

def predict_probability(input_df: pd.DataFrame) -> float:
    """Return churn probability for a single-row DataFrame."""

    probability = model.predict_proba(input_df)[0][1]
    return float(probability)


def get_risk_level(probability: float) -> str:
    """Convert probability into business risk category."""

    if probability >= 0.80:
        return "High"

    if probability >= 0.50:
        return "Medium"

    return "Low"


def update_metrics(risk_level: str) -> None:
    """Update API monitoring counters."""

    monitoring_metrics["total_predictions"] += 1
    monitoring_metrics["last_prediction_time"] = datetime.now().isoformat()

    if risk_level == "High":
        monitoring_metrics["high_risk_predictions"] += 1

    elif risk_level == "Medium":
        monitoring_metrics["medium_risk_predictions"] += 1

    else:
        monitoring_metrics["low_risk_predictions"] += 1


def roi_calculator(
    original_probability: float,
    new_probability: float,
    annual_customer_value: float,
    intervention_cost: float,
) -> Dict[str, Any]:
    """Calculate estimated business ROI."""

    risk_reduction = max(
        original_probability - new_probability,
        0,
    )

    expected_revenue_protected = (
        risk_reduction * annual_customer_value
    )

    net_value = (
        expected_revenue_protected - intervention_cost
    )

    if intervention_cost > 0:
        roi = net_value / intervention_cost
    else:
        roi = 0

    decision = (
        "Worth Retaining"
        if net_value > 0
        else "Not Cost Effective"
    )

    return {
        "risk_reduction": round(risk_reduction, 4),
        "expected_revenue_protected": round(
            expected_revenue_protected,
            2,
        ),
        "intervention_cost": round(intervention_cost, 2),
        "net_value": round(net_value, 2),
        "roi": round(roi, 2),
        "decision": decision,
    }


# ============================================================
# Model-based explanation engine
# ============================================================

def generate_counterfactual_candidates(
    row: pd.Series,
) -> Dict[str, Any]:
    """Generate safer alternative values for what-if testing."""

    candidates = {}

    if row.get("Contract") == "Month-to-month":
        candidates["Contract"] = "One year"

    if row.get("TechSupport") == "No":
        candidates["TechSupport"] = "Yes"

    if row.get("OnlineSecurity") == "No":
        candidates["OnlineSecurity"] = "Yes"

    if row.get("PaymentMethod") == "Electronic check":
        candidates["PaymentMethod"] = "Credit card (automatic)"

    monthly_charge = float(row.get("MonthlyCharges", 0))

    if monthly_charge > 0:
        candidates["MonthlyCharges"] = round(
            monthly_charge * 0.80,
            2,
        )

    return candidates


def model_based_explanation(
    input_df: pd.DataFrame,
) -> list[Dict[str, Any]]:
    """Explain churn risk using model-based what-if tests."""

    input_df = input_df.reset_index(drop=True)

    row = input_df.iloc[0]

    original_probability = predict_probability(input_df)

    candidates = generate_counterfactual_candidates(row)

    explanations = []

    for feature, new_value in candidates.items():
        modified_df = input_df.copy()

        old_value = modified_df.iloc[0][feature]

        modified_df.iloc[
            0,
            modified_df.columns.get_loc(feature),
        ] = new_value

        new_probability = predict_probability(modified_df)

        risk_drop = original_probability - new_probability

        if risk_drop > 0.01:
            explanations.append(
                {
                    "feature": feature,
                    "current_value": str(old_value),
                    "simulated_value": str(new_value),
                    "risk_before": round(original_probability, 4),
                    "risk_after": round(new_probability, 4),
                    "risk_drop": round(risk_drop, 4),
                    "explanation": (
                        f"Changing {feature} from "
                        f"'{old_value}' to '{new_value}' "
                        f"reduces predicted churn risk by "
                        f"{round(risk_drop * 100, 2)}%."
                    ),
                }
            )

    explanations.sort(
        key=lambda item: item["risk_drop"],
        reverse=True,
    )

    if not explanations:
        return [
            {
                "feature": "Model signal",
                "current_value": "Current profile",
                "simulated_value": "No strong intervention found",
                "risk_before": round(original_probability, 4),
                "risk_after": round(original_probability, 4),
                "risk_drop": 0,
                "explanation": (
                    "The model did not find a single tested "
                    "intervention that significantly reduces churn risk."
                ),
            }
        ]

    return explanations[:3]


def recommend_action(
    explanations: list[Dict[str, Any]],
    risk_level: str,
) -> str:
    """Generate recommendation from strongest explanation."""

    if risk_level == "Low":
        return "Customer appears stable. Continue regular engagement."

    top_feature = explanations[0]["feature"]

    if top_feature == "MonthlyCharges":
        return "Test targeted discount or value-based plan adjustment."

    if top_feature == "Contract":
        return "Offer annual contract incentive or loyalty commitment benefit."

    if top_feature == "TechSupport":
        return "Offer priority support or proactive technical assistance."

    if top_feature == "OnlineSecurity":
        return "Bundle online security as a retention benefit."

    if top_feature == "PaymentMethod":
        return "Encourage automatic payment method with a small incentive."

    return "Review customer profile and trigger retention outreach."


def predict_dataframe(input_df: pd.DataFrame) -> pd.DataFrame:
    """Run batch predictions and add product-level insights."""

    probabilities = model.predict_proba(input_df)[:, 1]

    results = input_df.copy()

    results["churn_probability"] = probabilities

    results["prediction"] = results[
        "churn_probability"
    ].apply(
        lambda p: "Churn" if p >= 0.50 else "No Churn"
    )

    results["risk_level"] = results[
        "churn_probability"
    ].apply(get_risk_level)

    explanations = []
    recommendations = []

    for _, row in results.iterrows():
        single_df = pd.DataFrame([row[input_df.columns]])

        explanation_items = model_based_explanation(single_df)

        risk_level = get_risk_level(
            float(row["churn_probability"])
        )

        explanations.append(
            explanation_items[0]["explanation"]
        )

        recommendations.append(
            recommend_action(
                explanation_items,
                risk_level,
            )
        )

    results["explanation"] = explanations
    results["recommended_action"] = recommendations

    return results


# ============================================================
# API endpoints
# ============================================================

@app.get("/")
def home():
    """Root endpoint."""

    return {
        "message": "RetainIQ Pro API is running",
        "version": "4.0",
        "product": "AI Customer Retention Intelligence Platform",
        "model_source": MODEL_SOURCE,
    }


@app.get("/health")
def health():
    """Health check endpoint."""

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_source": MODEL_SOURCE,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/model_info")
def model_info():
    """Return model/product information."""

    return {
        "model_name": MODEL_NAME,
        "stage": MODEL_STAGE,
        "version": "4.0",
        "model_source": MODEL_SOURCE,
        "model_file": str(MODEL_PATH.name),
        "explanation_method": (
            "Model-based counterfactual what-if analysis"
        ),
        "mlops": [
            "MLflow experiment tracking",
            "MLflow model registry",
            "Docker deployment",
            "Monitoring",
            "Batch inference",
        ],
    }


@app.get("/metrics")
def metrics():
    """Return API monitoring counters."""

    return monitoring_metrics


@app.post("/predict")
def predict(customer: CustomerData):
    """Predict churn for one customer with explanation."""

    input_df = pd.DataFrame([customer.dict()])

    probability = predict_probability(input_df)

    risk_level = get_risk_level(probability)

    explanations = model_based_explanation(input_df)

    recommendation = recommend_action(
        explanations,
        risk_level,
    )

    update_metrics(risk_level)

    return {
        "prediction": (
            "Churn"
            if probability >= 0.50
            else "No Churn"
        ),
        "churn_probability": round(probability, 4),
        "risk_level": risk_level,
        "top_model_based_explanations": explanations,
        "recommended_action": recommendation,
    }


@app.post("/batch_predict")
async def batch_predict(file: UploadFile = File(...)):
    """Analyze many customers from uploaded CSV."""

    if not file.filename.endswith(".csv"):
        return {
            "error": "Only CSV files are supported."
        }

    content = await file.read()

    input_df = pd.read_csv(
        StringIO(content.decode("utf-8"))
    )

    result_df = predict_dataframe(input_df)

    for risk_level in result_df["risk_level"]:
        update_metrics(risk_level)

    monitoring_metrics["last_batch_size"] = int(
        len(result_df)
    )

    high_risk = int(
        (result_df["risk_level"] == "High").sum()
    )

    medium_risk = int(
        (result_df["risk_level"] == "Medium").sum()
    )

    low_risk = int(
        (result_df["risk_level"] == "Low").sum()
    )

    avg_monthly_charge = float(
        result_df["MonthlyCharges"].mean()
    )

    estimated_revenue_at_risk = (
        high_risk * avg_monthly_charge * 12
    )

    top_risky = (
        result_df.sort_values(
            by="churn_probability",
            ascending=False,
        )
        .head(10)[
            [
                "churn_probability",
                "prediction",
                "risk_level",
                "explanation",
                "recommended_action",
            ]
        ]
        .round(4)
        .to_dict(orient="records")
    )

    return {
        "filename": file.filename,
        "customers_analyzed": int(len(result_df)),
        "high_risk_customers": high_risk,
        "medium_risk_customers": medium_risk,
        "low_risk_customers": low_risk,
        "estimated_revenue_at_risk": round(
            estimated_revenue_at_risk,
            2,
        ),
        "top_risky_customers": top_risky,
    }


@app.post("/simulate")
def simulate(request: SimulationRequest):
    """Run what-if retention simulation."""

    input_df = pd.DataFrame([request.customer.dict()])

    original_probability = predict_probability(input_df)

    explanations = model_based_explanation(input_df)

    best_intervention = explanations[0]

    simulated_probability = float(
        best_intervention["risk_after"]
    )

    risk_level_before = get_risk_level(
        original_probability
    )

    risk_level_after = get_risk_level(
        simulated_probability
    )

    roi = roi_calculator(
        original_probability=original_probability,
        new_probability=simulated_probability,
        annual_customer_value=request.annual_customer_value,
        intervention_cost=request.intervention_cost,
    )

    monitoring_metrics["simulations_run"] += 1

    return {
        "current_churn_probability": round(
            original_probability,
            4,
        ),
        "current_risk_level": risk_level_before,
        "best_intervention": best_intervention,
        "simulated_churn_probability": round(
            simulated_probability,
            4,
        ),
        "simulated_risk_level": risk_level_after,
        "roi_analysis": roi,
        "business_summary": (
            f"The strongest tested intervention is changing "
            f"{best_intervention['feature']} from "
            f"'{best_intervention['current_value']}' to "
            f"'{best_intervention['simulated_value']}'. "
            f"This reduces churn risk from "
            f"{round(original_probability * 100, 2)}% to "
            f"{round(simulated_probability * 100, 2)}%. "
            f"Decision: {roi['decision']}."
        ),
    }