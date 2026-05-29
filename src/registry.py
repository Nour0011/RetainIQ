"""
Register the final tuned Gradient Boosting model in MLflow Model Registry.

This script searches all MLflow experiments for the run named
"Best Tuned Gradient Boosting", registers its model artifact, moves it to
Staging, then Production, and archives older versions.
"""

import time
import mlflow
from mlflow import MlflowClient


MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
FINAL_RUN_NAME = "Best Tuned Gradient Boosting"
REGISTERED_MODEL_NAME = "RetainIQ-Churn-Predictor"


def find_best_tuned_gradient_boosting_run(client):
    """
    Search all MLflow experiments for the final tuned Gradient Boosting run.
    """
    experiments = client.search_experiments()

    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string=f"tags.mlflow.runName = '{FINAL_RUN_NAME}'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs:
            run = runs[0]
            print("\nFound final tuned Gradient Boosting run.")
            print(f"Experiment: {experiment.name}")
            print(f"Run ID: {run.info.run_id}")
            return run.info.run_id

    print("\nAvailable experiments:")
    for experiment in experiments:
        print(f"- {experiment.name}")

    raise ValueError(
        f"No run named '{FINAL_RUN_NAME}' was found in any MLflow experiment."
    )


def register_model(run_id):
    """
    Register the model artifact from the selected MLflow run.
    """
    model_uri = f"runs:/{run_id}/model"

    print("\nRegistering model...")
    print(f"Model URI: {model_uri}")

    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    print(
        f"\nRegistered successfully: "
        f"{REGISTERED_MODEL_NAME} version {registered_model.version}"
    )

    return registered_model.version


def move_to_production(client, version):
    """
    Move the new model version to Staging, then Production.
    """
    print("\nMoving model to Staging...")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Staging",
    )

    time.sleep(2)

    print("Moving model to Production...")

    client.transition_model_version_stage(
        name=REGISTERED_MODEL_NAME,
        version=version,
        stage="Production",
    )

    print("Model is now in Production.")


def archive_old_versions(client, current_version):
    """
    Archive all older versions of the registered model.
    """
    versions = client.search_model_versions(
        f"name='{REGISTERED_MODEL_NAME}'"
    )

    for model_version in versions:
        if int(model_version.version) != int(current_version):
            client.transition_model_version_stage(
                name=REGISTERED_MODEL_NAME,
                version=model_version.version,
                stage="Archived",
            )
            print(f"Archived old version: {model_version.version}")


def main():
    """
    Execute the full registry workflow.
    """
    print("\nRetainIQ - Gradient Boosting Registry Workflow")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    run_id = find_best_tuned_gradient_boosting_run(client)
    version = register_model(run_id)

    move_to_production(client, version)
    archive_old_versions(client, version)

    print("\nRegistry completed successfully.")
    print("Open: http://127.0.0.1:5000/#/models")
    print("Check that the latest Production version comes from:")
    print(FINAL_RUN_NAME)


if __name__ == "__main__":
    main()