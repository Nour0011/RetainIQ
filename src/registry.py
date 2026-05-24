"""

Purpose:
    This script registers the best tuned churn prediction model into
    the MLflow Model Registry and demonstrates lifecycle transitions.

Requirements covered:
    1. MLflow Model Registry
    2. Model version management
    3. Stage transitions
    4. Staging -> Production workflow
    5. Enterprise lifecycle management

"""

from pathlib import Path
import time

import mlflow
from mlflow import MlflowClient


# Project paths

BASE_DIR = Path(__file__).resolve().parents[1]

# Path to the tuned model saved from tune.py
MODEL_PATH = BASE_DIR / "models" / "best_tuned_churn_model.pkl"

# MLflow experiment name
EXPERIMENT_NAME = "RetainIQ-Pro-Hyperparameter-Tuning"

# Registered model name inside MLflow Registry
REGISTERED_MODEL_NAME = "RetainIQ-Churn-Predictor"


def get_latest_run_id(client, experiment_name):
    """
    Retrieve the latest MLflow run ID from the tuning experiment.

    This allows us to register the most recent tuned model automatically.

    Args:
        client:
            MLflow client object.

        experiment_name:
            Name of MLflow experiment.

    Returns:
        Latest MLflow run ID.
    """
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise ValueError(
            f"Experiment '{experiment_name}' does not exist."
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No runs found in experiment.")

    latest_run = runs[0]

    print("\nLatest MLflow Run Found:")
    print(f"Run ID: {latest_run.info.run_id}")

    return latest_run.info.run_id


def register_model(client, run_id):
    """
    Register the tuned model into MLflow Model Registry.

    The model artifact path inside MLflow is:
        model

    Returns:
        Registered model version object.
    """
    model_uri = f"runs:/{run_id}/model"

    print("\nRegistering model to MLflow Registry...")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print(
        f"Model registered successfully. "
        f"Version: {registered_model.version}"
    )

    return registered_model


def transition_model_stage(client, version):
    """
    Transition model through lifecycle stages.

    Lifecycle:
        None -> Staging -> Production

    This simulates a real enterprise deployment workflow.
    """
    print("\nTransitioning model to STAGING...")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
    )

    print("Model moved to Staging.")

    time.sleep(2)

    print("\nTransitioning model to PRODUCTION...")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
    )

    print("Model moved to Production.")


def archive_old_versions(client, current_version):
    """
    Archive older model versions.

    This keeps the registry clean and simulates proper enterprise
    lifecycle management.
    """
    print("\nChecking for old model versions...")

    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    for version in versions:
        version_number = int(version.version)

        if version_number != int(current_version):
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=version.version,
                stage="Archived",
            )

            print(
                f"Archived old model version: {version.version}"
            )


def main():
    """
    Execute the full MLflow registry workflow.

    Workflow:
        1. Connect to MLflow
        2. Retrieve latest tuned model
        3. Register model
        4. Move model to Staging
        5. Move model to Production
        6. Archive old versions
    """
    print("\nRetainIQ Pro - Model Registry Management\n")

    # Connect to MLflow tracking server.
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    client = MlflowClient()

    # Get latest tuning run.
    run_id = get_latest_run_id(
        client,
        EXPERIMENT_NAME,
    )

    # Register model.
    registered_model = register_model(
        client,
        run_id,
    )

    # Move through lifecycle stages.
    transition_model_stage(
        client,
        registered_model.version,
    )

    # Archive old versions.
    archive_old_versions(
        client,
        registered_model.version,
    )

    print("\nRegistry workflow completed successfully.")

    print("\nOpen MLflow UI:")
    print("http://127.0.0.1:5000")

    print("\nCheck:")
    print("- Models tab")
    print("- Model versions")
    print("- Production stage")
    print("- Archived versions")


if __name__ == "__main__":
    main()