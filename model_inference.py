"""
model_inference.py — production inference layer for app.py.

Two inference paths:

1. get_demo_profile(proc_type)
   Returns real model output for a real OULAD student (see
   build_full_demo_profiles.py for how these were generated — actual trained
   models run on actual held-out feature rows, not hand-picked numbers).

2. score_checkin_features(history_df)
   Takes a student's accumulated check-in history (≥3 rows) and derives
   behavioural proxy features structurally analogous to the OULAD feature
   space, then runs those through the trained Random Forest type-classifier.

   IMPORTANT: these are PROXY features, not the same click-stream features
   the RF was trained on. This is disclosed in the return dict and must be
   surfaced in the UI. See 'source' and 'proxy_disclosure' keys.
"""
import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


# ── Mood → proxy signal mappings ─────────────────────────────────────────────
# Each mood maps to the signal it most strongly represents (0–1 scale).
_MOOD_DISTRACTION = {
    "distracted": 1.0, "overwhelmed": 0.6, "anxious": 0.3,
    "stressed": 0.2,   "tired": 0.4,       "okay": 0.0, "motivated": 0.0,
}
_MOOD_DEADLINE_PANIC = {
    "anxious": 1.0, "stressed": 0.8, "overwhelmed": 0.4,
    "tired": 0.2,   "distracted": 0.1, "okay": 0.0, "motivated": 0.0,
}

# Minimum check-ins before we trust the proxy features enough to run inference.
MIN_CHECKINS = 3


def _build_proxy_features(hdf: pd.DataFrame) -> dict:
    """
    Derive OULAD-analogous features from accumulated check-in history.

    hdf columns: ts (datetime-parseable), mood (str), energy (int 1–10),
                 sleep_hours (float), tasks_on_time (str), task_name (str).

    Returns a flat dict of floats in [0, 1] matching the RF's expected
    feature names where possible.
    """
    n = len(hdf)
    hdf = hdf.copy()
    hdf["ts"] = pd.to_datetime(hdf["ts"])
    hdf = hdf.sort_values("ts").reset_index(drop=True)

    # ── delay_ratio proxy ─────────────────────────────────────────────────────
    # "Rarely/Sometimes submitting on time" ≈ high late-submission rate.
    on_time_map = {"Always": 0.95, "Often": 0.75, "Sometimes": 0.5, "Rarely": 0.2}
    on_time_vals = hdf["tasks_on_time"].map(on_time_map).fillna(0.5)
    delay_ratio_mean = float(1.0 - on_time_vals.mean())       # inverted: high = late
    delay_ratio_std  = float(on_time_vals.std() if n > 1 else 0.0)

    # ── is_last_minute proxy ──────────────────────────────────────────────────
    # Fraction of check-ins with a deadline-panic mood.
    is_last_minute_mean = float(hdf["mood"].map(_MOOD_DEADLINE_PANIC).fillna(0).mean())

    # ── non_submit proxy ──────────────────────────────────────────────────────
    # Same task appearing across ≥3 consecutive check-ins without a break
    # ≈ task has never been submitted.
    same_task_repeat = hdf["task_name"].duplicated(keep=False).mean() if n > 1 else 0.0
    non_submit_mean = float(same_task_repeat)

    # ── delay_deviation proxy ─────────────────────────────────────────────────
    # Sleep variability is the best available proxy for irregular schedule
    # (irregular schedule → erratic submission timing in OULAD).
    sleep_std = float(hdf["sleep_hours"].std()) if n > 1 else 0.0
    sleep_std = 0.0 if np.isnan(sleep_std) else sleep_std
    delay_deviation_pos_mean = min(sleep_std / 3.0, 1.0)   # 3h std → fully irregular
    delay_deviation_pos_std  = 0.0

    # ── avoidance_score proxy ─────────────────────────────────────────────────
    # Fraction of check-ins showing distraction signals.
    avoidance_score_pos_mean = float(hdf["mood"].map(_MOOD_DISTRACTION).fillna(0).mean())

    # ── click_drop_ratio / focus_trend proxy ──────────────────────────────────
    # Drop in energy from first to last check-in (declining engagement).
    energies = hdf["energy"].values.astype(float)
    if n > 1:
        trend = float(energies[0] - energies[-1]) / 10.0
        click_drop_ratio_pos_mean = max(0.0, trend)
        focus_trend_pos_mean      = max(0.0, trend)
    else:
        click_drop_ratio_pos_mean = 0.0
        focus_trend_pos_mean      = 0.0

    # ── screen_ratio proxy ────────────────────────────────────────────────────
    screen_ratio_mean = avoidance_score_pos_mean   # same signal, different name in OULAD

    # ── consec_low proxy ──────────────────────────────────────────────────────
    # Longest streak of energy ≤ 3 (normalised by n).
    streak = cur = 0
    for e in energies:
        if e <= 3:
            cur += 1
            streak = max(streak, cur)
        else:
            cur = 0
    consec_low_mean = streak / max(n, 1)

    # ── distraction_freq proxy ────────────────────────────────────────────────
    distraction_freq_mean = float((hdf["mood"] == "distracted").mean())

    return {
        "delay_ratio_mean":           delay_ratio_mean,
        "delay_ratio_std":            delay_ratio_std,
        "is_last_minute_mean":        is_last_minute_mean,
        "non_submit_mean":            non_submit_mean,
        "delay_deviation_pos_mean":   delay_deviation_pos_mean,
        "delay_deviation_pos_std":    delay_deviation_pos_std,
        "avoidance_score_pos_mean":   avoidance_score_pos_mean,
        "click_drop_ratio_pos_mean":  click_drop_ratio_pos_mean,
        "screen_ratio_mean":          screen_ratio_mean,
        "focus_trend_pos_mean":       focus_trend_pos_mean,
        "consec_low_mean":            consec_low_mean,
        "distraction_freq_mean":      distraction_freq_mean,
    }


class ModelInference:
    def __init__(self, models_dir="stage3_models_v2", demo_json_path="demo_profiles_v2.json"):
        self.dir = Path(models_dir)
        self.scaler = pickle.load(open(self.dir / "scaler.pkl", "rb"))
        self.iso    = pickle.load(open(self.dir / "isolation_forest.pkl", "rb"))
        self.lr     = pickle.load(open(self.dir / "logistic_regression_supervised.pkl", "rb"))
        self.ifnorm = pickle.load(open(self.dir / "if_score_normalization.pkl", "rb"))
        self.meta   = pickle.load(open(self.dir / "stage3a_metadata.pkl", "rb"))
        self.rf_pkg = pickle.load(open(self.dir / "random_forest_type.pkl", "rb"))

        demo_path = self.dir / demo_json_path
        if not demo_path.exists():
            demo_path = Path(demo_json_path)
        self.demo_profiles = json.load(open(demo_path))

    # ── Path 1: Demo profiles ─────────────────────────────────────────────────

    def get_demo_profile(self, proc_type: str) -> dict:
        """
        Returns real model output for a real OULAD student of that type.

        proc_type: one of 'deadline_panic', 'distraction_escape',
                   'perfectionism_paralysis'.
        Keys returned: id_student, is_true_holdout_2014, blended_score,
                       rf_predicted_type, rf_confidence, shap_explanation,
                       disclosure.
        """
        if proc_type not in self.demo_profiles:
            raise KeyError(
                f"No demo profile for '{proc_type}'. Available: "
                f"{list(self.demo_profiles.keys())}"
            )
        profile = dict(self.demo_profiles[proc_type])
        # Ensure 'disclosure' key always present (guards against older JSON files)
        profile.setdefault(
            "disclosure",
            "Anomaly score and type confidence computed by running this student's "
            "real OULAD feature vector through the trained Stage 3A/3B models. "
            "Student selected from the 2014 holdout set (not seen during training)."
        )
        return profile

    # ── Path 2: Live check-in proxy inference ─────────────────────────────────

    def score_checkin_features(self, history_df: pd.DataFrame) -> dict:
        """
        Infer procrastination type from a student's check-in history using
        behavioral proxy features.

        Parameters
        ----------
        history_df : pd.DataFrame
            The student's full check-in history including the just-submitted
            check-in. Required columns:
              ts (datetime-parseable), mood (str), energy (int 1–10),
              sleep_hours (float), tasks_on_time (str), task_name (str).

        Returns
        -------
        dict with keys:
          sufficient_history  : bool — False if n < MIN_CHECKINS
          n_checkins          : int
          proc_type           : str | None
          type_confidence     : float | None — RF probability for top class
          is_low_confidence   : bool — True if confidence < 0.55
          all_type_proba      : dict | None — {type: probability}
          proxy_features      : dict — the derived feature values
          source              : str — always "checkin_proxy_rf"
          proxy_disclosure    : str — must be surfaced in the UI
        """
        n = len(history_df)

        _DISCLOSURE = (
            "Type detected from your check-in history using proxy features "
            f"(mood trend, energy trend, task-repeat rate, sleep variability) "
            f"derived from {n} check-in(s). These are structurally analogous to — "
            "but NOT the same as — the OULAD submission/click-stream features the "
            "model was trained on. Treat this as a personalised heuristic, not a "
            "validated classification."
        )

        if n < MIN_CHECKINS:
            return {
                "sufficient_history": False,
                "n_checkins": n,
                "proc_type": None,
                "type_confidence": None,
                "is_low_confidence": True,
                "all_type_proba": None,
                "proxy_features": {},
                "source": "checkin_proxy_rf",
                "proxy_disclosure": (
                    f"Need at least {MIN_CHECKINS} check-ins to detect a behavioural "
                    f"pattern — you have {n} so far. Complete your check-in each day "
                    "and the model will start scoring after the third one."
                ),
            }

        proxy = _build_proxy_features(history_df)

        # ── Run through the RF type-classifier ───────────────────────────────
        rf      = self.rf_pkg["model"]
        le      = self.rf_pkg["encoder"]
        rf_cols = self.rf_pkg["features"]

        x = pd.DataFrame([{col: proxy.get(col, 0.0) for col in rf_cols}])
        proba    = rf.predict_proba(x)[0]
        top_idx  = int(np.argmax(proba))
        top_type = le.inverse_transform([top_idx])[0]
        top_conf = float(proba[top_idx])

        return {
            "sufficient_history": True,
            "n_checkins": n,
            "proc_type": top_type,
            "type_confidence": round(top_conf, 3),
            "is_low_confidence": top_conf < 0.55,
            "all_type_proba": {
                le.inverse_transform([i])[0]: round(float(p), 3)
                for i, p in enumerate(proba)
            },
            "proxy_features": proxy,
            "source": "checkin_proxy_rf",
            "proxy_disclosure": _DISCLOSURE,
        }

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def feature_columns(self):
        return self.meta["features"]

    @property
    def stage3a_metrics(self):
        return self.meta.get("metrics_holdout_2013_2014", {})

    @property
    def stage3b_validity(self):
        return self.rf_pkg.get("validity", {})
