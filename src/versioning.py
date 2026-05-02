"""
NYC TLC — Artifact Versioning
==============================

Three helpers that log data, model, and feature artifacts to W&B via
ExperimentTracker.  Each call creates a new artifact version automatically.
W&B builds a lineage graph from these calls that shows:

  data artifact  →  feature artifact  →  model artifact  →  mitigated model artifact

Usage
-----
    from src.versioning import log_data_artifact, log_model_artifact, log_feature_artifact

    log_data_artifact(tracker, "data/processed/dec_eval.parquet",
                      "dec-eval-set", metadata={"n_rows": 5000})

    log_feature_artifact(tracker, "models/engineered/scaler.pkl",
                         active_feature_steps=["_add_pickup_hour", ...])

    log_model_artifact(tracker, "models/tuned/random_forest.pkl",
                       "tuned-champion", metadata={"mae": 3.21})
"""


def log_data_artifact(tracker, path, name: str, metadata: dict = None):
    """
    Log a parquet or CSV file as a versioned W&B dataset artifact.

    Args:
        tracker  : ExperimentTracker instance
        path     : path to the data file
        name     : artifact name shown in W&B (e.g. "dec-eval-set")
        metadata : optional dict logged alongside the artifact
    """
    tracker.log_artifact(
        file_path     = path,
        artifact_name = name,
        artifact_type = "dataset",
        metadata      = metadata or {},
    )


def log_model_artifact(tracker, model_path, name: str, metadata: dict = None):
    """
    Log a model .pkl file as a versioned W&B model artifact.

    Args:
        tracker     : ExperimentTracker instance
        model_path  : path to the saved model .pkl file
        name        : artifact name shown in W&B (e.g. "tuned-champion")
        metadata    : optional dict (e.g. {"mae": 3.21, "source": "grid-search"})
    """
    tracker.log_artifact(
        file_path     = model_path,
        artifact_name = name,
        artifact_type = "model",
        metadata      = metadata or {},
    )


def log_feature_artifact(tracker, scaler_path, active_feature_steps: list,
                         metadata: dict = None):
    """
    Log the fitted scaler as a versioned W&B feature artifact.

    The list of active feature step names is stored as artifact metadata so
    W&B records exactly which engineering steps were active for this run.
    Comparing two versions of this artifact in W&B shows exactly which steps
    were added or removed between runs.

    Args:
        tracker               : ExperimentTracker instance
        scaler_path           : path to the saved scaler .pkl file
        active_feature_steps  : list of step function names that were active
                                (e.g. ["_add_pickup_hour", "_add_day_of_week", ...])
        metadata              : optional extra metadata dict
    """
    meta = {"active_steps": active_feature_steps, **(metadata or {})}
    tracker.log_artifact(
        file_path     = scaler_path,
        artifact_name = "feature-pipeline",
        artifact_type = "feature",
        metadata      = meta,
    )
