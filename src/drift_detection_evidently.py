"""
NYC TLC — Drift Detection (Evidently AI)
=========================================

A parallel drift detection module that uses Evidently AI alongside the existing
PSI / KS approach in drift_detection.py.

Five public functions form a clean pipeline:

  run_evidently_drift_report          →  Evidently Report (DataDrift + TargetDrift)
  parse_drift_results                 →  plain dict of dataset + label drift statistics
  run_evidently_concept_drift_report  →  Evidently Report (RegressionPreset)
  parse_concept_drift_results         →  plain dict of concept drift / performance statistics
  select_mitigation_strategy          →  strategy string consumed by drift_mitigation.mitigate()

Dataset drift (DataDriftPreset) measures P(X) shift — feature distributions.
Target drift  (TargetDriftPreset) measures P(Y) shift — label distribution.
Concept drift (RegressionPreset) measures P(Y|X) shift — model performance degradation.

The strategy selection is driven by both dataset and concept drift signals.
All thresholds are configurable at the top of this file.

DataFrames for run_evidently_drift_report must contain ENGINEERED features + TARGET_COL.
DataFrames for run_evidently_concept_drift_report must also contain a 'prediction' column.
"""

import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, RegressionPreset
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

    # MAE percentage increase (0.0–1.0) above which concept drift is flagged.
    # 0.20 = model MAE has grown by more than 20% vs. the reference period.
    "concept_drift_mae_pct": 0.20,
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

    # ── Target / label drift (from TargetDriftPreset) ────────────────────────
    # TargetDriftPreset emits a ColumnDriftMetric entry for TARGET_COL.
    # This measures P(Y) shift — useful signal, but not the same as concept drift.
    target_col_metric = next(
        (m for m in metrics
         if m["metric"] == "ColumnDriftMetric"
         and m.get("result", {}).get("column_name") == TARGET_COL),
        None,
    )
    target_drift_detected = False
    target_drift_score    = 0.0
    if target_col_metric:
        target_drift_detected = bool(target_col_metric["result"].get("drift_detected", False))
        target_drift_score    = float(target_col_metric["result"].get("drift_score", 0.0))

    return {
        "overall_drift":      overall_drift,
        "share_drifted":      share_drifted,
        "n_drifted":          n_drifted,
        "drifted_features":   drifted_features,
        "drift_scores":       drift_scores,
        "target_drift":       target_drift_detected,
        "target_drift_score": target_drift_score,
    }


# ── Concept drift report ──────────────────────────────────────────────────────

def run_evidently_concept_drift_report(reference_df: pd.DataFrame,
                                        current_df:   pd.DataFrame) -> Report:
    """
    Run a RegressionPreset Evidently report to detect concept drift (P(Y|X) shift).

    Concept drift is not detectable from feature distributions alone — it requires
    comparing actual model performance on reference vs. current data.  This report
    uses ground-truth labels and model predictions to compute MAE, RMSE, and error
    distributions for both periods.

    Args:
        reference_df : Jan test set — must contain feature columns, TARGET_COL,
                       and a 'prediction' column (model predictions on that set).
        current_df   : Dec eval set — same column schema as reference_df.

    Returns:
        Evidently Report object (save as HTML, or call .as_dict()).
    """
    feature_cols = [c for c in reference_df.columns
                    if c not in (TARGET_COL, "prediction")]

    col_map = ColumnMapping(
        target             = TARGET_COL,
        prediction         = "prediction",
        numerical_features = feature_cols,
    )

    report = Report(metrics=[RegressionPreset()])
    report.run(
        reference_data = reference_df,
        current_data   = current_df,
        column_mapping = col_map,
    )
    return report


def parse_concept_drift_results(report: Report) -> dict:
    """
    Extract concept drift statistics from a RegressionPreset Evidently report.

    Returns a dict with:
      concept_drift_detected (bool)  — True if current MAE exceeds reference by
                                       more than DRIFT_THRESHOLDS['concept_drift_mae_pct']
      ref_mae                (float) — reference period MAE (Jan test set)
      cur_mae                (float) — current period MAE  (Dec eval set)
      mae_pct_increase       (float) — fractional MAE increase (e.g. 0.25 = 25%)
    """
    metrics = report.as_dict()["metrics"]

    quality_metric = next(
        (m for m in metrics if m["metric"] == "RegressionQualityMetric"),
        None,
    )

    ref_mae = 0.0
    cur_mae = 0.0
    if quality_metric:
        ref_mae = float(quality_metric["result"].get("reference", {}).get("mean_abs_error", 0.0))
        cur_mae = float(quality_metric["result"].get("current",   {}).get("mean_abs_error", 0.0))

    mae_pct_increase       = (cur_mae - ref_mae) / ref_mae if ref_mae > 0 else 0.0
    concept_drift_detected = mae_pct_increase > DRIFT_THRESHOLDS["concept_drift_mae_pct"]

    return {
        "concept_drift_detected": concept_drift_detected,
        "ref_mae":                ref_mae,
        "cur_mae":                cur_mae,
        "mae_pct_increase":       mae_pct_increase,
    }


# ── Strategy selection ────────────────────────────────────────────────────────

def select_mitigation_strategy(drift_results: dict,
                                concept_drift_results: dict = None) -> str:
    """
    Choose a mitigation strategy based on Evidently dataset and concept drift results.

    Decision logic (thresholds are set in DRIFT_THRESHOLDS above):

      0. Concept drift detected (MAE degraded > concept_drift_mae_pct threshold)
             → "reweight_retrain"  P(Y|X) has changed; the model itself needs updating
                                   regardless of whether feature distributions shifted.

      1. No overall dataset drift detected (and no concept drift)
             → "none"        nothing to do

      2. Overall drift, but few columns affected (share ≤ mild_drift_share)
             → "recalibrate" scale shift only; refit the scaler, keep the model

      3. Overall drift AND several features have high individual drift scores
         (≥ min_severe_features features exceed severe_feature_score)
             → "drop_features"  specific features are unreliable; remove their
                                 creation steps and retrain

      4. Otherwise (widespread drift, no single dominant bad feature)
             → "reweight_retrain"  combine old + recent data with recency bias

    Args:
        drift_results         : output of parse_drift_results()
        concept_drift_results : output of parse_concept_drift_results(), or None

    Returns:
        one of: "none" | "recalibrate" | "reweight_retrain" | "drop_features"
    """
    # Concept drift takes priority — a degrading model must be retrained even
    # if the input feature distributions look superficially unchanged.
    if concept_drift_results and concept_drift_results["concept_drift_detected"]:
        pct = concept_drift_results["mae_pct_increase"]
        print(f"  Concept drift detected: MAE increased by {pct:.1%} — strategy: reweight_retrain")
        return "reweight_retrain"

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
