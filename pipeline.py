"""
NYC TLC ETA Prediction — Full Pipeline
========================================

Run this script to execute the complete end-to-end pipeline:

  Step 1   — Data Cleaning
  Step 2   — Data Splitting + Subsampling
  Step 3   — Experiment A: Baseline         (raw features only, no engineering)
  Step 4   — Experiment B: Full Engineering  (all feature creation, transformation, store)
  Step 5   — Head-to-head comparison         (same models, same data, features differ)
  Step 6   — Champion + Feature Importance   (logged to W&B via ExperimentTracker)
  Step 7   — Hyperparameter Tuning           (random search sweep → grid search sweep)
  Step 8   — Error Analysis                  (per-sample errors, segment breakdowns)
  Step 9   — Drift Detection                 (monthly MAE curve, PSI + KS report)
  Step 9.1 — Drift Detection (Evidently AI)  (feature-level report on DRIFTED MONTH data)
  Step 10  — Drift Mitigation                (Evidently-driven strategy, before/after)

Modularity note
---------------
Each concern lives in its own src/ module.  pipeline.py is pure orchestration:
it calls modules in order, passes outputs between them, and logs results to W&B
through ExperimentTracker.  No business logic lives here.
"""

import argparse
import numpy as np
np.float_ = np.float64
import pandas as pd
import joblib
from pathlib import Path

from src.cleaning                  import clean_parquet, clean_dataframe
from src.splitting                 import split_train_test, subsample_splits
from src.features                  import run_feature_pipeline, run_baseline_pipeline, TARGET_COL
from src.models                    import train_all_models, load_model, CANDIDATE_MODELS
from src.evaluation                import evaluate_all_models, select_champion, plot_feature_importance
from src.experiment_tracking       import ExperimentTracker, log_monthly_drift_run
from src.tuning                    import (RANDOM_SEARCH_CONFIG, GRID_SEARCH_CONFIGS,
                                           run_wandb_sweep, retrain_best_model)
from src.error_analysis            import run_error_analysis
from src.drift_detection           import (load_monthly_eval, run_monthly_drift_analysis,
                                           plot_monthly_mae_curve,
                                           plot_label_drift_distribution)
from src.drift_detection_evidently import (run_evidently_drift_report, parse_drift_results,
                                           run_evidently_concept_drift_report,
                                           parse_concept_drift_results,
                                           select_mitigation_strategy)
from src.drift_mitigation          import mitigate, plot_mitigation_comparison
from src.versioning                import log_data_artifact, log_model_artifact, log_feature_artifact


# ── Path configuration ────────────────────────────────────────────────────────

RAW_CLEAN_PARQUET    = "data/processed/yellow_tripdata_2024-01_clean.parquet"
CLEANED_PARQUET      = "data/processed/yellow_tripdata_2024-01_cleaned.parquet"
PROCESSED_DIR        = "data/processed"
MODEL_DIR_BASELINE   = "models/baseline"
MODEL_DIR_ENGINEERED = "models/engineered"
MODEL_DIR_TUNED      = "models/tuned"
MODEL_DIR_MITIGATED  = "models/mitigated"
PLOTS_DIR            = "outputs/plots"
MONTHLY_EVAL_PARQUET = "data/raw/last_weeks_all_months_2024.parquet"

# ── W&B configuration ─────────────────────────────────────────────────────────

WANDB_PROJECT = "dsma-lecture4_tests"

# Number of sweep trials — keep small for classroom runtime.
# Random search: 15 trials across both model families.
# Grid search:   runs ALL combinations (≤ 8 per family — see tuning.py).



# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)



def _comparison_table(baseline_results, engineered_results):
    merged = baseline_results[["model", "mae", "rmse"]].merge(
        engineered_results[["model", "mae", "rmse"]],
        on="model", suffixes=("_baseline", "_engineered"),
    )
    merged["mae_improvement_%"] = (
        (merged["mae_baseline"] - merged["mae_engineered"]) / merged["mae_baseline"] * 100
    ).round(1)

    print(f"\n  {'Model':<25} {'Baseline MAE':>13} {'Engineered MAE':>15} {'Improvement':>12}")
    print("  " + "-" * 68)
    for _, row in merged.iterrows():
        print(
            f"  {row['model']:<25} "
            f"{row['mae_baseline']:>13.2f} "
            f"{row['mae_engineered']:>15.2f} "
            f"{row['mae_improvement_%']:>11.1f}%"
        )
    return merged


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(wandb_project=WANDB_PROJECT):

    # ── Step 1: Data Cleaning ──────────────────────────────────────────────────
    _print_header("STEP 1 — Data Cleaning")
    df_clean = clean_parquet(RAW_CLEAN_PARQUET, CLEANED_PARQUET)

    # ── Step 2: Data Splitting + Subsampling ───────────────────────────────────
    _print_header("STEP 2 — Data Splitting + Subsampling")
    train_path, test_path = split_train_test(CLEANED_PARQUET, PROCESSED_DIR)
    print("\n  Subsampling splits ...")
    train_raw, test_raw = subsample_splits(train_path, test_path)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A — Baseline (raw features only)
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 3 — Experiment A: Baseline (Raw Features Only)")

    baseline_train, baseline_scaler = run_baseline_pipeline(train_raw, is_training=True)
    X_train_base = baseline_train.drop(columns=[TARGET_COL])
    y_train_base = baseline_train[TARGET_COL]
    print(f"  Baseline feature columns ({len(X_train_base.columns)}): "
          f"{X_train_base.columns.tolist()}")

    train_all_models(X_train_base, y_train_base, MODEL_DIR_BASELINE)

    baseline_test, _ = run_baseline_pipeline(
        test_raw, scaler=baseline_scaler, is_training=False
    )
    X_test_base = baseline_test.drop(columns=[TARGET_COL])
    y_test_base = baseline_test[TARGET_COL]

    print("\n  Baseline model results:")
    baseline_results = evaluate_all_models(X_test_base, y_test_base, MODEL_DIR_BASELINE)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B — Full Feature Engineering
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 4 — Experiment B: Full Feature Engineering")

    eng_train, eng_scaler = run_feature_pipeline(train_raw, is_training=True)
    X_train_eng = eng_train.drop(columns=[TARGET_COL])
    y_train_eng = eng_train[TARGET_COL]
    print(f"  Engineered feature columns ({len(X_train_eng.columns)}): "
          f"{X_train_eng.columns.tolist()}")

    train_all_models(X_train_eng, y_train_eng, MODEL_DIR_ENGINEERED)

    

    eng_test, _ = run_feature_pipeline(test_raw, scaler=eng_scaler, is_training=False)
    X_test_eng  = eng_test.drop(columns=[TARGET_COL])
    y_test_eng  = eng_test[TARGET_COL]

    print("\n  Engineered model results:")
    engineered_results = evaluate_all_models(X_test_eng, y_test_eng, MODEL_DIR_ENGINEERED)
    
    # Save the fitted scaler so the Streamlit app can load it without rerunning the pipeline
    Path(MODEL_DIR_ENGINEERED).mkdir(parents=True, exist_ok=True)
    joblib.dump(eng_scaler, Path(MODEL_DIR_ENGINEERED) / "scaler.pkl")
    print(f"  Scaler saved → {MODEL_DIR_ENGINEERED}/scaler.pkl")
    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Head-to-head comparison
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 5 — Feature Engineering Impact: Head-to-Head Comparison")
    print("  Metric: MAE (minutes).  Lower is better.")
    print("  Improvement = % reduction in MAE achieved by feature engineering.\n")
    comparison = _comparison_table(baseline_results, engineered_results)

    best_model_row = comparison.loc[comparison["mae_improvement_%"].idxmax()]
    print(
        f"\n  Largest gain : {best_model_row['model']} "
        f"improved by {best_model_row['mae_improvement_%']:.1f}% with feature engineering"
    )

    # ── Step 6: Champion + Feature Importance + W&B (via ExperimentTracker) ───
    _print_header("STEP 6 — Champion Model + Feature Importance")
    champion_name  = select_champion(engineered_results, metric="mae")
    champion_model = load_model(champion_name, MODEL_DIR_ENGINEERED)
    champion_row   = engineered_results.loc[
        engineered_results["model"] == champion_name
    ].iloc[0]

    plot_feature_importance(
        model         = champion_model,
        feature_names = X_test_eng.columns.tolist(),
        model_name    = champion_name,
        output_dir    = PLOTS_DIR,
    )

    tracker = ExperimentTracker(
        project  = wandb_project,
        run_name = "champion-eval",
        tags     = ["champion", "engineered-features"],
        config   = {"champion_model": champion_name},
    )
    tracker.log_summary({
        "champion_model": champion_name,
        "mae":            float(champion_row["mae"]),
        "rmse":           float(champion_row["rmse"]),
        "mape":           float(champion_row["mape"]),
    })
    # Log feature importance plot saved to disk
    fi_path = Path(PLOTS_DIR) / f"feature_importance_{champion_name}.png"
    if fi_path.exists():
        tracker.log_image_file(fi_path, "feature_importance")
    tracker.log_code()
    url = tracker.finish()
    print(f"\n  W&B run logged → {url}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7 — Hyperparameter Tuning
    #
    # Phase 1: Random search across Random Forest + Gradient Boosting.
    #          Purpose: identify which model family fits this dataset.
    # Phase 2: Grid search on the winning family over a narrow parameter grid.
    #          Purpose: rigorously finalise the best hyperparameters.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 7 — Hyperparameter Tuning")

    # ── Phase 1: Random search ────────────────────────────────────────────────
    print("\n  Phase 1 — Random Search (both model families, wide grid)")
    print(f"  Running {15} trials via W&B sweep ...")

    random_sweep_id, best_random_config, best_random_mae = run_wandb_sweep(
        X_train     = X_train_eng,
        y_train     = y_train_eng,
        sweep_config = RANDOM_SEARCH_CONFIG,
        project     = wandb_project,
        n_runs      = 15,
    )
    winning_family = best_random_config.get("model_type", "random_forest")
    print(f"\n  Random search complete.")
    print(f"  Best model family : {winning_family}")
    print(f"  Best CV MAE       : {best_random_mae:.4f}")
    print(f"  Best config       : {best_random_config}")
    print(f"  W&B sweep         : {random_sweep_id}")

    # ── Phase 2: Grid search on the winning family ────────────────────────────
    print(f"\n  Phase 2 — Grid Search ({winning_family}, narrow grid, all combinations)")

    grid_config = GRID_SEARCH_CONFIGS[winning_family]
    _, best_grid_config, best_grid_mae = run_wandb_sweep(
        X_train      = X_train_eng,
        y_train      = y_train_eng,
        sweep_config = grid_config,
        project      = wandb_project,
        n_runs       = 50,   # grid sweeps run all combinations regardless of count
    )
    print(f"\n  Grid search complete.")
    print(f"  Best CV MAE : {best_grid_mae:.4f}  (random search was {best_random_mae:.4f})")
    print(f"  Best config : {best_grid_config}")

    # ── Retrain on full training set with best config ─────────────────────────
    print("\n  Retraining tuned champion on full training set ...")
    tuned_champion_model = retrain_best_model(
        best_config = best_grid_config,
        X_train     = X_train_eng,
        y_train     = y_train_eng,
        model_dir   = MODEL_DIR_TUNED,
    )

    # Evaluate tuned model and log to W&B
    y_pred_tuned = tuned_champion_model.predict(X_test_eng)
    tuned_mae  = float(np.mean(np.abs(y_test_eng.values - y_pred_tuned)))
    tuned_rmse = float(np.sqrt(np.mean((y_test_eng.values - y_pred_tuned) ** 2)))

    print(f"\n  Tuned model evaluation:")
    print(f"    MAE  : {tuned_mae:.4f}  (champion baseline: {float(champion_row['mae']):.4f})")
    print(f"    RMSE : {tuned_rmse:.4f}  (champion baseline: {float(champion_row['rmse']):.4f})")

    tuning_tracker = ExperimentTracker(
        project  = wandb_project,
        run_name = f"tuned-{winning_family}",
        tags     = ["tuned", "grid-search"],
        config   = best_grid_config,
    )
    tuning_tracker.log_summary({"mae": tuned_mae, "rmse": tuned_rmse})

    # Version the tuned model and feature pipeline as W&B artifacts.
    # Each call creates a new version (v0, v1, v2 …) — the lineage graph in
    # W&B will later show:  feature-pipeline → tuned-champion → mitigated-model
    tuned_pkl = Path(MODEL_DIR_TUNED) / f"tuned_{winning_family}.pkl"
    log_model_artifact(
        tuning_tracker, tuned_pkl, "tuned-champion",
        metadata={"source": "grid-search", "mae": tuned_mae},
    )
    log_feature_artifact(
        tuning_tracker,
        Path(MODEL_DIR_ENGINEERED) / "scaler.pkl",
        active_feature_steps=X_train_eng.columns.tolist(),
        metadata={"n_features": len(X_train_eng.columns)},
    )
    url = tuning_tracker.finish()
    print(f"  W&B run logged → {url}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8 — Error Analysis
    #
    # Per-sample breakdown of where the tuned champion model fails.
    # Segments: rush hour, distance quartile, airport trips, day of week.
    # The error DataFrame is logged to W&B as an interactive Table so
    # you can filter and sort in the browser.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 8 — Error Analysis")

    error_df, error_figs = run_error_analysis(
        X_test     = X_test_eng,
        y_test     = y_test_eng,
        model      = tuned_champion_model,
        output_dir = PLOTS_DIR,
    )

    error_tracker = ExperimentTracker(
        project  = wandb_project,
        run_name = "error-analysis",
        tags     = ["error-analysis", "tuned"],
        config   = {"model": winning_family, "n_test_samples": len(error_df)},
    )
    # W&B Table — you can filter by rush_hour_label, trip_type, etc.
    error_tracker.log_table(
        error_df[["actual", "predicted", "abs_error", "pct_error",
                  "rush_hour_label", "trip_type", "distance_bucket",
                  "day_name", "time_of_day"]].dropna(how="all"),
        table_name = "per_sample_errors",
    )
    error_tracker.log_summary({"mae": float(error_df["abs_error"].mean()),
                                "p90_abs_error":  float(error_df["abs_error"].quantile(0.9))})
    for col, fig in error_figs.items():
        error_tracker.log_plot(fig, f"error_by_{col}")
    url = error_tracker.finish()
    print(f"\n  W&B run logged → {url}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9 — Drift Detection
    #
    # Loads the 2024 multi-month evaluation parquet (last week of every month).
    # For each month:
    #     Data drift  : PSI + KS test on raw feature distributions
    #     Concept drift: model MAE delta vs. January reference
    # Each month is logged as its own W&B run → build a MAE-over-time
    # line chart directly in the W&B dashboard.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 9 — Drift Detection")

    monthly_eval_path = Path(MONTHLY_EVAL_PARQUET)
    if not monthly_eval_path.exists():
        print(f"  Monthly eval parquet not found at {MONTHLY_EVAL_PARQUET}")
        print("  Skipping Steps 9 and 10.  Place the file there to enable drift analysis.")
        return

    monthly_eval_df = load_monthly_eval(MONTHLY_EVAL_PARQUET)

    monthly_summary, drift_reports = run_monthly_drift_analysis(
        monthly_eval_df  = monthly_eval_df,
        reference_raw_df = train_raw,
        model            = tuned_champion_model,
        scaler           = eng_scaler,
        output_dir       = PLOTS_DIR,
    )

    # MAE-over-time curve to see if the performance degrades over the months.
    # Take into consideration that you won't have access to so much future data
    # in the real world. This is just to showcase what you need to be looking at.
    mae_curve_fig = plot_monthly_mae_curve(monthly_summary, output_dir=PLOTS_DIR)

    # Label drift plot — pick the month with the highest label PSI as the
    # representative "future" month to contrast against the reference.
    worst_label_month = monthly_summary.loc[monthly_summary["label_psi"].idxmax(), "month"]
    worst_mask        = monthly_eval_df["tpep_pickup_datetime"].dt.strftime("%b") == worst_label_month
    worst_month_data  = monthly_eval_df[worst_mask].reset_index(drop=True)
    plot_label_drift_distribution(
        reference_raw_df = train_raw,
        current_raw_df   = worst_month_data,
        ref_label        = "Jan (reference)",
        cur_label        = f"{worst_label_month} (eval)",
        output_dir       = PLOTS_DIR,
    )

    # Log one W&B run per month — select all runs in dashboard → custom
    # line chart with month_num on x-axis and mae on y-axis
    print("\n  Logging per-month drift runs to W&B ...")
    for _, row in monthly_summary.iterrows():
        month_label  = row["month"]
        drift_report = drift_reports.get(month_label)
        if drift_report is None:
            continue
        log_monthly_drift_run(
            month_label  = month_label,
            month_num    = int(row["month_num"]),
            mae          = float(row["mae"]),
            drift_report = drift_report,
            project      = wandb_project,
            mae_delta    = float(row["mae_delta"]),
            n_trips      = int(row["n_trips"]),
            label_drift  = {
                "psi":        float(row["label_psi"]),
                "ks_pvalue":  float(row["label_ks_pvalue"]),
                "drifted":    bool(row["label_drifted"]),
                "ref_mean":   float(row["label_ref_mean"]),
                "cur_mean":   float(row["label_cur_mean"]),
            },
        )

    # Alert if worst month exceeds 20 % MAE increase
    worst = monthly_summary.loc[monthly_summary["mae_delta"].idxmax()]
    if worst["mae_pct_increase"] > 20:
        drift_tracker = ExperimentTracker(
            project  = wandb_project,
            run_name = "drift-summary",
            tags     = ["drift-detection"],
        )
        drift_tracker.log_summary({
            "worst_month":        worst["month"],
            "worst_mae":          float(worst["mae"]),
            "worst_mae_pct_increase": float(worst["mae_pct_increase"]),
            "months_with_drift":  int((monthly_summary["n_drifted_features"] > 0).sum()),
        })
        drift_tracker.log_plot(mae_curve_fig, "monthly_mae_curve")
        drift_tracker.alert(
            title = "Concept Drift Detected",
            text  = (
                f"Model MAE increased by {worst['mae_pct_increase']:.1f}% "
                f"in {worst['month']} vs. January baseline."
            ),
        )
        url = drift_tracker.finish()
        print(f"\n  Drift summary logged → {url}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 9.1 — Drift Detection using Evidently AI — evaluated over Drifted Months in 2024
    #
    # Loads Different months in 2024 data and splits it by date:
    #     First 3 weeks (1–21) → "current" distribution for Evidently
    #     Last week     (22–31) → held-out evaluation set (no leakage)
    #
    # The Evidently report compares January 2024 training features vs. A Drifted month in 2024
    # first-3-weeks features and produces:
    #     An interactive HTML report saved to outputs/
    #     A structured drift_results dict used by Step 10
    #     A selected mitigation strategy string
    # ══════════════════════════════════════════════════════════════════════════
    
    DRIFT_MONTH = "Dec"

    _print_header(f"STEP 9.1 — Drift Detection (Evidently AI) — {DRIFT_MONTH} 2024")
    
    DRIFT_RAW_PARQUET      = {
                            "Dec":"data/raw/yellow_tripdata_2024-12.parquet",
                            "Sept":"data/raw/yellow_tripdata_2024-09.parquet",
                            "July":"data/raw/yellow_tripdata_2024-07.parquet",
                              }.get(DRIFT_MONTH)

    # Drifted Month data sampling — first 3 weeks = mitigation train, last week = eval.
    # Fixed seeds ensure reproducible splits every pipeline run.
    DRIFT_TRAIN_SAMPLE = 20_000
    DRIFT_EVAL_SAMPLE  = 5_000
    DRIFT_SEED         = 42

    # Load + clean Data of the month which we are checking drift for.
    drift_df = pd.read_parquet(DRIFT_RAW_PARQUET)
    drift_df = clean_dataframe(drift_df)
    drift_df["tpep_pickup_datetime"] = pd.to_datetime(drift_df["tpep_pickup_datetime"])

    # Split by calendar date — first 3 weeks for mitigation, last week for eval
    drift_train_raw = (
        drift_df[drift_df["tpep_pickup_datetime"].dt.day <= 21]
        .sample(DRIFT_TRAIN_SAMPLE, random_state=DRIFT_SEED)
        .reset_index(drop=True)
    )
    drift_eval_raw = (
        drift_df[drift_df["tpep_pickup_datetime"].dt.day >= 22]
        .sample(DRIFT_EVAL_SAMPLE, random_state=DRIFT_SEED)
        .reset_index(drop=True)
    )
    print(f"  Drift train set : {len(drift_train_raw):,} rows  ({DRIFT_MONTH} 1–21,  seed={DRIFT_SEED})")
    print(f"  Drift eval set  : {len(drift_eval_raw):,}  rows  ({DRIFT_MONTH} 22–31, seed={DRIFT_SEED})")

    # Engineer features with the January scaler — no refit.
    # We deliberately keep the old scaler so that any scale shift shows up
    # as drift in the Evidently report rather than being silently corrected.
    drift_train_eng, _ = run_feature_pipeline(drift_train_raw, scaler=eng_scaler, is_training=False)
    drift_eval_eng,  _ = run_feature_pipeline(drift_eval_raw,  scaler=eng_scaler, is_training=False)

    # Reconstruct reference DataFrame: January engineered features + target
    ref_eng_df = X_train_eng.copy()
    ref_eng_df[TARGET_COL] = y_train_eng.values

    # Run Evidently report: reference = Jan train, current = first 3 weeks of DRIFTED_MONTH
    # Covers dataset drift (DataDriftPreset) and label drift (TargetDriftPreset).
    print("\n  Running Evidently dataset + label drift report ...")
    evidently_report = run_evidently_drift_report(ref_eng_df, drift_train_eng)
    drift_results    = parse_drift_results(evidently_report)

    print(f"\n  Overall drift detected : {drift_results['overall_drift']}")
    print(f"  Features drifted       : {drift_results['n_drifted']} "
          f"({drift_results['share_drifted']:.1%} of feature columns)")
    if drift_results["drifted_features"]:
        print(f"  Drifted feature names  : {drift_results['drifted_features']}")
    print(f"  Target (label) drift   : {drift_results['target_drift']}  "
          f"(score={drift_results['target_drift_score']:.4f})")

    Path("outputs").mkdir(exist_ok=True)
    evidently_html = Path("outputs") / "evidently_drift_report.html"
    evidently_report.save_html(str(evidently_html))
    print(f"\n  Evidently dataset drift HTML  → {evidently_html}")

    # Concept drift: compare Jan test-set performance vs Drift month eval performance.
    # Uses RegressionPreset — needs actual labels + model predictions for both periods.
    print("\n  Running Evidently concept drift report ...")
    ref_perf_df = X_test_eng.copy()
    ref_perf_df[TARGET_COL]    = y_test_eng.values
    ref_perf_df["prediction"]  = tuned_champion_model.predict(X_test_eng)

    cur_perf_df = drift_eval_eng.copy()
    cur_perf_df["prediction"]  = tuned_champion_model.predict(drift_eval_eng.drop(columns=[TARGET_COL]))

    concept_drift_report   = run_evidently_concept_drift_report(ref_perf_df, cur_perf_df)
    concept_drift_results  = parse_concept_drift_results(concept_drift_report)

    print(f"\n  Concept drift detected : {concept_drift_results['concept_drift_detected']}")
    print(f"  Reference MAE (Jan)    : {concept_drift_results['ref_mae']:.4f} min")
    print(f"  Current MAE ({DRIFT_MONTH} eval) : {concept_drift_results['cur_mae']:.4f} min")
    print(f"  MAE increase           : {concept_drift_results['mae_pct_increase']:.1%}")

    concept_drift_html = Path("outputs") / "evidently_concept_drift_report.html"
    concept_drift_report.save_html(str(concept_drift_html))
    print(f"\n  Evidently concept drift HTML  → {concept_drift_html}")

    selected_strategy = select_mitigation_strategy(drift_results, concept_drift_results)
    print(f"\n  Selected strategy      : {selected_strategy}")

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 10 — Drift Mitigation + Before / After Comparison
    #
    # Uses the strategy selected by Evidently in Step 9.1.
    # Evaluation uses the SAME held-out DRIFT MONTH last-week set for both
    # the tuned champion and the mitigated model — fair, leakage-free.
    #
    # Three artifacts are versioned in W&B:
    #   1. drift-eval-set      — the held-out evaluation parquet (data artifact)
    #   2. mitigated-model   — the retrained model .pkl       (model artifact)
    #   (feature-pipeline artifact was already logged in Step 7)
    #
    # The comparison plot shows three error distributions side-by-side:
    #     Tuned champion on January   (in-distribution baseline)
    #     Tuned champion on Drift Month  (pre-mitigation, drifted)
    #     Mitigated model on Drift Month (post-mitigation)
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 10 — Drift Mitigation + Before / After Comparison")

    # Save the evaluation set once — both models are scored against this file
    drift_eval_parquet = Path(PROCESSED_DIR) / "drift_eval.parquet"
    drift_eval_raw.to_parquet(drift_eval_parquet, index=False)

    # Baseline: tuned champion evaluated on Drift Month last week (pre-mitigation)
    y_drift_eval      = drift_eval_eng[TARGET_COL].values
    y_pred_drift_base = tuned_champion_model.predict(drift_eval_eng.drop(columns=[TARGET_COL]))
    baseline_drift_mae = float(np.mean(np.abs(y_drift_eval - y_pred_drift_base)))
    print(f"\n  Tuned champion MAE — {DRIFT_MONTH} eval (pre-mitigation) : {baseline_drift_mae:.4f} min")

    if selected_strategy == "none":
        print("  No mitigation required — skipping Step 10.")
        return

    # Apply the Evidently-selected mitigation strategy
    print(f"\n  Applying strategy: {selected_strategy}")
    mitigated_model, mitigated_scaler, eval_steps = mitigate(
        strategy         = selected_strategy,
        train_df         = train_raw,
        recent_df        = drift_train_raw,
        model_name       = winning_family,
        model_dir        = MODEL_DIR_MITIGATED,
        base_model       = tuned_champion_model,
        drifted_features = drift_results["drifted_features"],
    )

    # recalibrate returns model=None — fall back to the original tuned champion
    if mitigated_model is None:
        mitigated_model = tuned_champion_model

    # Re-engineer the DRIFT MONTH eval set with the updated scaler (and filtered
    # feature steps if drop_features was used) so the eval is fair
    active_scaler = mitigated_scaler if mitigated_scaler is not None else eng_scaler
    drift_eval_eng_mit, _ = run_feature_pipeline(
        drift_eval_raw, scaler=active_scaler,
        is_training=False, custom_creation_steps=eval_steps,
    )

    # Evaluate mitigated model on the same held-out DRIFT MONTH eval set
    y_pred_drift_mit   = mitigated_model.predict(drift_eval_eng_mit.drop(columns=[TARGET_COL]))
    mitigated_drift_mae = float(np.mean(np.abs(y_drift_eval - y_pred_drift_mit)))
    improvement_pct   = (baseline_drift_mae - mitigated_drift_mae) / baseline_drift_mae * 100
    print(f"  Mitigated model MAE — Drift eval (post-mitigation) : {mitigated_drift_mae:.4f} min")
    print(f"  Improvement                                       : {improvement_pct:+.1f}%")

    # Build the three-way comparison plot
    comparison_fig = plot_mitigation_comparison(
        {
            "Champion — Jan (in-dist)":             np.abs(y_test_eng.values - y_pred_tuned),
            f"Champion — {DRIFT_MONTH} (drifted)":             np.abs(y_drift_eval - y_pred_drift_base),
            f"Mitigated ({selected_strategy}) — {DRIFT_MONTH}": np.abs(y_drift_eval - y_pred_drift_mit),
        },
        output_dir = PLOTS_DIR,
    )

    # Log everything to a dedicated W&B run
    mitigation_tracker = ExperimentTracker(
        project  = wandb_project,
        run_name = f"mitigation-{selected_strategy}-{DRIFT_MONTH}",
        tags     = ["drift-mitigation", f"{DRIFT_MONTH}", selected_strategy],
        config   = {
            "strategy":          selected_strategy,
            "drifted_features":  drift_results["drifted_features"],
            "n_drift_train":       len(drift_train_raw),
            "n_drift_eval":        len(drift_eval_raw),
            "drift_seed":          DRIFT_SEED,
        },
    )
    mitigation_tracker.log_summary({
        "jan_mae":             float(np.mean(np.abs(y_test_eng.values - y_pred_tuned))),
        "baseline_drift_mae":    baseline_drift_mae,
        "mitigated_drift_mae":   mitigated_drift_mae,
        "mae_improvement_pct": improvement_pct,
    })
    mitigation_tracker.log_plot(comparison_fig, "mitigation_comparison")

    # Version the evaluation dataset artifact (shared reference for both models)
    log_data_artifact(
        mitigation_tracker, drift_eval_parquet, f"{DRIFT_MONTH}-eval-set",
        metadata={"month": f"{DRIFT_MONTH} 2024", "n_rows": len(drift_eval_raw), "seed": DRIFT_SEED},
    )

    # Save the mitigated scaler — needed to serve the mitigated model in production
    mitigated_scaler_path = Path(MODEL_DIR_MITIGATED) / "scaler_mitigated.pkl"
    joblib.dump(active_scaler, mitigated_scaler_path)

    # Feature columns the mitigated model actually uses (ground truth, not step names).
    # For drop_features this differs from the original X_train_eng columns.
    mitigated_feature_cols = [
        c for c in drift_eval_eng_mit.columns if c != TARGET_COL
    ]

    # Version the mitigated feature pipeline artifact
    log_feature_artifact(
        mitigation_tracker,
        mitigated_scaler_path,
        active_feature_steps=mitigated_feature_cols,
        metadata={"strategy": selected_strategy, "n_features": len(mitigated_feature_cols)},
    )

    # Version the mitigated model artifact (recalibrate has no new model file)
    mitigated_pkl_name = {
        "reweight_retrain": f"{winning_family}_reweighted.pkl",
        "drop_features":    f"{winning_family}_drop_features.pkl",
    }.get(selected_strategy)

    if mitigated_pkl_name:
        log_model_artifact(
            mitigation_tracker,
            Path(MODEL_DIR_MITIGATED) / mitigated_pkl_name,
            "mitigated-model",
            metadata={
                "strategy":          selected_strategy,
                "mae":               mitigated_drift_mae,
                "improvement_pct":   improvement_pct,
                "drifted_features":  drift_results["drifted_features"],
            },
        )

    # Attach the Evidently HTML report as a versioned artifact for reference
    mitigation_tracker.log_artifact(
        evidently_html, artifact_name="evidently-drift-report", artifact_type="report",
    )

    url = mitigation_tracker.finish()
    print(f"\n  W&B run logged → {url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", default=WANDB_PROJECT,
                        help="W&B project name to log runs into")
    args = parser.parse_args()
    run_pipeline(wandb_project=args.wandb_project)
