"""
NYC TLC — Drift Mitigation
============================

One public entry point (mitigate) routes to one of three strategies based on
the string returned by drift_detection_evidently.select_mitigation_strategy().

Strategies
----------
  "recalibrate"      Refit the StandardScaler on recent data. Model unchanged.
                     Use when drift is mild and scale-level only.

  "reweight_retrain" Retrain on combined old + recent data, giving recent rows
                     RECENCY_WEIGHT times the sample weight of old rows.
                     Use when drift is widespread and the model needs updating.

  "drop_features"    Identify feature creation steps that produce drifted
                     engineered features, remove them, retrain on combined data.
                     Use when specific features are the root cause of drift.

Usage
-----
    model, scaler, eval_steps = mitigate(
        strategy         = selected_strategy,
        train_df         = train_raw,
        recent_df        = dec_train_raw,
        model_name       = "random_forest",
        model_dir        = "models/mitigated",
        drifted_features = drift_results["drifted_features"],
    )

    # eval_steps is None for recalibrate/reweight_retrain.
    # For drop_features it is the filtered FEATURE_CREATION_STEPS list —
    # pass it to run_feature_pipeline when engineering the evaluation set
    # so its feature columns match what the mitigated model was trained on.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

from src.features import (run_feature_pipeline, TARGET_COL,
                           SCALE_FEATURES, FEATURE_CREATION_STEPS)
from src.models   import save_model, CANDIDATE_MODELS


# ── Configuration ─────────────────────────────────────────────────────────────

RECENCY_WEIGHT = 3.0  # sample weight multiplier for recent rows in reweight_retrain

# Engineered feature name → the feature creation step function that produces it.
# Used by drop_features to identify which pipeline steps to remove when a
# particular engineered feature is flagged as drifted by Evidently.
FEATURE_SOURCE_MAP = {
    "pickup_hour":            "_add_pickup_hour",
    "day_of_week":            "_add_day_of_week",
    "is_weekend":             "_add_is_weekend",
    "time_of_day_bucket":     "_add_time_of_day_bucket",
    "is_rush_hour":           "_add_time_of_day_bucket",
    "distance_x_time_of_day": "_add_distance_x_time_of_day",
    "pickup_zone_x_hour":     "_add_pickup_zone_x_hour",
    "distance_x_rush_hour":   "_add_distance_x_rush_hour",
    "pickup_hour_sin":        "_add_cyclical_hour_encoding",
    "pickup_hour_cos":        "_add_cyclical_hour_encoding",
}


# ── Public entry point ────────────────────────────────────────────────────────

def mitigate(strategy, train_df, recent_df, model_name, model_dir,
             base_model=None, drifted_features=None):
    """
    Apply the chosen drift mitigation strategy.

    Args:
        strategy         : "none" | "recalibrate" | "reweight_retrain" | "drop_features"
        train_df         : original training data (raw, Jan 2024)
        recent_df        : recent data for retraining (raw, Dec first 3 weeks)
        model_name       : key into CANDIDATE_MODELS — used for the save filename
        model_dir        : directory to save the retrained model .pkl
        base_model       : fitted model whose hyperparameters to clone when
                           retraining (e.g. tuned_champion_model). 
        drifted_features : list of drifted engineered feature names from
                           parse_drift_results() — only used by "drop_features"

    Returns:
        (model, scaler, eval_steps)
        model      — new fitted model, or None when strategy is "none"/"recalibrate"
                     (caller should fall back to the original tuned champion)
        scaler     — new fitted StandardScaler, or None when strategy is "none"
        eval_steps — filtered FEATURE_CREATION_STEPS list for "drop_features",
                     None for all other strategies
    """
    model_template = base_model 

    if strategy == "none":
        print("  No mitigation needed.")
        return None, None, None

    if strategy == "recalibrate":
        scaler = _recalibrate(recent_df)
        return None, scaler, None

    if strategy == "reweight_retrain":
        model, scaler = _reweight_retrain(train_df, recent_df, model_name, model_dir, model_template)
        return model, scaler, None

    if strategy == "drop_features":
        model, scaler, eval_steps = _drop_and_retrain(
            train_df, recent_df, model_name, model_dir, drifted_features or [], model_template
        )
        return model, scaler, eval_steps

    raise ValueError(f"Unknown strategy: {strategy!r}")


# ── Private strategy implementations ─────────────────────────────────────────

def _recalibrate(recent_df: pd.DataFrame) -> StandardScaler:
    """Refit the StandardScaler on recent data. Model weights are unchanged."""
    available = [f for f in SCALE_FEATURES if f in recent_df.columns]
    scaler = StandardScaler()
    scaler.fit(recent_df[available].dropna())
    print(f"  Scaler recalibrated on {len(recent_df):,} recent rows "
          f"(columns: {available})")
    return scaler


def _reweight_retrain(train_df, recent_df, model_name, model_dir, model_template):
    """Retrain on old + recent data, giving recent rows RECENCY_WEIGHT× sample weight."""
    print(f"  Combining {len(train_df):,} old rows (w=1.0) + "
          f"{len(recent_df):,} recent rows (w={RECENCY_WEIGHT})")

    combined = pd.concat([
        train_df.assign(_w=1.0),
        recent_df.assign(_w=float(RECENCY_WEIGHT)),
    ], ignore_index=True)

    features, scaler = run_feature_pipeline(combined, is_training=True)
    weights = features.pop("_w").values
    X = features.drop(columns=[TARGET_COL])
    y = features[TARGET_COL]

    model = clone(model_template)
    model.fit(X, y, sample_weight=weights)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_model(model, f"{model_name}_reweighted", model_dir)
    print(f"  Saved → {model_dir}/{model_name}_reweighted.pkl")
    return model, scaler


def _drop_and_retrain(train_df, recent_df, model_name, model_dir, drifted_features, model_template):
    """Remove feature steps that produce drifted features, retrain on combined data."""
    steps_to_drop = {
        FEATURE_SOURCE_MAP[feat]
        for feat in drifted_features
        if feat in FEATURE_SOURCE_MAP
    }
    filtered_steps = [s for s in FEATURE_CREATION_STEPS
                      if s.__name__ not in steps_to_drop]
    removed_names  = [s.__name__ for s in FEATURE_CREATION_STEPS
                      if s.__name__ in steps_to_drop]
    print(f"  Dropping feature steps: {removed_names}")

    combined = pd.concat([train_df, recent_df], ignore_index=True)
    features, scaler = run_feature_pipeline(
        combined, is_training=True, custom_creation_steps=filtered_steps
    )
    X = features.drop(columns=[TARGET_COL])
    y = features[TARGET_COL]

    model = clone(model_template)
    model.fit(X, y)

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    save_model(model, f"{model_name}_drop_features", model_dir)
    print(f"  Saved → {model_dir}/{model_name}_drop_features.pkl")
    return model, scaler, filtered_steps


# ── Comparison plot ────────────────────────────────────────────────────────────

def plot_mitigation_comparison(error_arrays: dict, output_dir=None):
    """
    Overlaid error-distribution plot comparing mitigation strategies.

    Args:
        error_arrays : dict mapping label → np.array of absolute errors.
                       Recommended keys:
                         "Champion — Jan (in-dist)"
                         "Champion — Dec (drifted)"
                         "Mitigated (<strategy>) — Dec"
        output_dir   : optional directory to save the plot as a PNG

    Returns:
        matplotlib Figure
    """
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#F44336", "#9C27B0"]
    maes   = {label: arr.mean() for label, arr in error_arrays.items()}

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (label, errors) in enumerate(error_arrays.items()):
        ax.hist(
            np.clip(errors, 0, 30), bins=60, alpha=0.5, density=True,
            color=colors[i % len(colors)],
            label=f"{label}  (MAE = {maes[label]:.2f} min)",
        )

    ax.set_xlabel("Absolute Error (minutes, clipped at 30)")
    ax.set_ylabel("Density")
    ax.set_title("Drift Mitigation — Error Distribution Before & After",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        path = Path(output_dir) / "mitigation_comparison.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved → {path}")

    orig_label = next(iter(maes))
    orig_mae   = maes[orig_label]
    print(f"\n  {'Strategy':<40} {'MAE':>8}  {'vs first':>12}")
    print("  " + "-" * 63)
    for label, mae in maes.items():
        if label == orig_label:
            print(f"  {label:<40} {mae:>8.2f}  {'(reference)':>12}")
        else:
            pct = (orig_mae - mae) / orig_mae * 100 if orig_mae > 0 else 0
            print(f"  {label:<40} {mae:>8.2f}  {pct:>+11.1f}%")

    return fig
