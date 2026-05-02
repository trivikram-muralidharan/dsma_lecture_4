"""
NYC TLC — Drift Detection (Evidently AI)
=========================================

A parallel drift detection module that uses Evidently AI alongside the existing
PSI / KS approach in drift_detection.py.

Three public functions form a clean pipeline:

  run_evidently_drift_report   →  Evidently Report object (save as HTML, inspect)
  parse_drift_results          →  plain dict of drift statistics
  select_mitigation_strategy   →  strategy string consumed by drift_mitigation.mitigate()

The strategy selection is driven entirely by the Evidently report output.
All thresholds are configurable at the top of this file — adjust them to change
how aggressively drift is flagged and which strategy is triggered.

Both reference and current DataFrames must contain ENGINEERED features (i.e. the
output of run_feature_pipeline), not raw columns.  This lets Evidently reason about
the exact representation the model sees at prediction time.
"""

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently import ColumnMapping

from src.features import TARGET_COL


# ── Configurable thresholds ───────────────────────────────────────────────────
#
# Tune these to match the tolerance of your business context.
# All "share" values are fractions of feature columns (0.0 – 1.0).

DRIFT_THRESHOLDS = {
    # Fraction of feature columns drifted at or below which only scaler
    # recalibration is needed (scale-level shift, model still valid).
    "mild_drift_share": 0.30,

    # Individual feature drift score above which a feature is "severely drifted".
    # Score is Wasserstein distance (normed) for continuous features.
    "severe_feature_score": 0.30,

    # Minimum number of severely drifted features needed to prefer
    # "drop_features" over "reweight_retrain".
    "min_severe_features": 2,
}


# ── Drift report ──────────────────────────────────────────────────────────────

def run_evidently_drift_report(reference_df: pd.DataFrame,
                                current_df:   pd.DataFrame) -> Report:
    """
    Run an Evidently drift report comparing reference vs. current engineered features.

    The report covers:
      • DataDriftPreset   — per-feature distribution shift + dataset-level summary
      • TargetDriftPreset — distribution shift in the target variable (trip_duration_minutes)

    Args:
        reference_df : engineered DataFrame from training data (Jan 2024)
                       Must include TARGET_COL alongside feature columns.
        current_df   : engineered DataFrame from recent data (Dec first 3 weeks)
                       Must have the same columns as reference_df.

    Returns:
        Evidently Report object.
        Call .save_html(path) to get the interactive HTML dashboard.
        Call .as_dict() to access raw metrics programmatically.
    """
    feature_cols = [c for c in reference_df.columns if c != TARGET_COL]

    col_map = ColumnMapping(
        target             = TARGET_COL,
        numerical_features = feature_cols,
    )

    report = Report(metrics=[DataDriftPreset(), TargetDriftPreset()])
    report.run(
        reference_data = reference_df,
        current_data   = current_df,
        column_mapping = col_map,
    )
    return report


# ── Parse results ─────────────────────────────────────────────────────────────

def parse_drift_results(report: Report) -> dict:
    """
    Extract key drift statistics from an Evidently Report into a plain dict.

    Returns a dict with the following keys:
      overall_drift    (bool)  — True if dataset-level drift is detected
      share_drifted    (float) — fraction of feature columns that drifted (0–1)
      n_drifted        (int)   — number of drifted feature columns
      drifted_features (list)  — names of drifted feature columns
      drift_scores     (dict)  — {column_name: drift_score} for all features
    """
    result  = report.as_dict()
    metrics = result["metrics"]

    # ── Dataset-level summary ─────────────────────────────────────────────────
    dataset_metric = next(
        m for m in metrics if m["metric"] == "DatasetDriftMetric"
    )
    overall = dataset_metric["result"]

    overall_drift = bool(overall.get("dataset_drift", False))
    # Evidently uses different key names across minor versions — handle both
    share_drifted = float(
        overall.get("share_of_drifted_columns", overall.get("drift_share", 0.0))
    )
    n_drifted = int(overall.get("number_of_drifted_columns", 0))

    # ── Per-column drift ──────────────────────────────────────────────────────
    drift_table = next(
        (m for m in metrics if m["metric"] == "DataDriftTable"), None
    )
    drift_by_col = {}
    if drift_table:
        drift_by_col = drift_table["result"].get("drift_by_columns", {})

    # Exclude the target column from feature-level results
    drifted_features = [
        col for col, stats in drift_by_col.items()
        if stats.get("drift_detected", False) and col != TARGET_COL
    ]
    drift_scores = {
        col: float(stats.get("drift_score", 0.0))
        for col, stats in drift_by_col.items()
        if col != TARGET_COL
    }

    return {
        "overall_drift":    overall_drift,
        "share_drifted":    share_drifted,
        "n_drifted":        n_drifted,
        "drifted_features": drifted_features,
        "drift_scores":     drift_scores,
    }


# ── Strategy selection ────────────────────────────────────────────────────────

def select_mitigation_strategy(drift_results: dict) -> str:
    """
    Choose a mitigation strategy based on Evidently drift results.

    Decision logic (thresholds are set in DRIFT_THRESHOLDS above):

      1. No overall drift detected
             → "none"        nothing to do

      2. Overall drift, but few columns affected (share ≤ mild_drift_share)
             → "recalibrate" scale shift only; refit the scaler, keep the model

      3. Overall drift AND several features have high individual drift scores
         (≥ min_severe_features features exceed severe_feature_score)
             → "drop_features"  specific features are unreliable; remove their
                                 creation steps and retrain

      4. Otherwise (widespread drift, no single dominant bad feature)
             → "reweight_retrain"  combine old + recent data with recency bias

    Returns:
        one of: "none" | "recalibrate" | "reweight_retrain" | "drop_features"
    """
    if not drift_results["overall_drift"]:
        print("  No overall drift detected.")
        return "none"

    share  = drift_results["share_drifted"]
    scores = drift_results["drift_scores"]

    if share <= DRIFT_THRESHOLDS["mild_drift_share"]:
        print(f"  Mild drift ({share:.1%} of features drifted) — strategy: recalibrate")
        return "recalibrate"

    severe_features = [
        feat for feat, score in scores.items()
        if score > DRIFT_THRESHOLDS["severe_feature_score"]
    ]
    if len(severe_features) >= DRIFT_THRESHOLDS["min_severe_features"]:
        print(f"  {len(severe_features)} severely drifted features "
              f"({severe_features}) — strategy: drop_features")
        return "drop_features"

    print(f"  Widespread drift ({share:.1%} of features drifted) — strategy: reweight_retrain")
    return "reweight_retrain"
