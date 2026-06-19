"""
model_inference.py — production inference layer for app.py.

ModelInference.get_demo_profile(proc_type) returns real model output for a
real OULAD student (see build_full_demo_profiles.py for how these were
generated — actual trained models run on actual held-out feature rows at
JSON-build time, then cached to demo_profiles_v2.json for fast page loads.
This is a precomputed snapshot of real inference, not a live call per
request — stated explicitly here because the distinction matters).

ModelInference.score_checkin_proxy(history) is the live-inference path:
it builds real behavioral proxy features from a student's own accumulated
check-in history and runs them through the SAME trained Random Forest used
for Stage 3B, at request time. This is a genuine model call, not a lookup —
but the features it scores are check-in-derived proxies (see
_build_proxy_features docstring below), structurally analogous to the OULAD
training features, not the same data. That distinction is preserved in
every value this method returns (see `is_proxy_features` in the output) so
the caller can't accidentally present it as equivalent to the Demo Profiles
page.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

MIN_CHECKINS_FOR_PROXY_SCORING = 3


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
            demo_path = Path(demo_json_path)  # fall back to repo root
        self.demo_profiles = json.load(open(demo_path))

    def get_demo_profile(self, proc_type):
        """
        proc_type: one of 'deadline_panic', 'distraction_escape',
                   'perfectionism_paralysis'.
        Returns a PRECOMPUTED snapshot (see module docstring) of real model
        output for a real OULAD student of that type:
            id_student, is_true_holdout_2014, blended_score,
            rf_predicted_type, rf_confidence, shap_explanation
        """
        if proc_type not in self.demo_profiles:
            raise KeyError(
                f"No demo profile for '{proc_type}'. Available: "
                f"{list(self.demo_profiles.keys())}"
            )
        return self.demo_profiles[proc_type]

    @property
    def feature_columns(self):
        return self.meta["features"]

    @property
    def stage3a_metrics(self):
        return self.meta.get("metrics_holdout_2013_2014", {})

    @property
    def stage3b_validity(self):
        return self.rf_pkg.get("validity", {})

    # ──────────────────────────────────────────────────────────────────
    # Live proxy-feature scoring (Option 1: actually run the trained RF
    # on check-in-derived features, instead of only a rule-based stand-in)
    # ──────────────────────────────────────────────────────────────────
    MOOD_DISTRACTION = {"distracted": 1.0, "overwhelmed": 0.6, "anxious": 0.3,
                         "stressed": 0.2, "tired": 0.4, "okay": 0.0, "motivated": 0.0}
    MOOD_DEADLINE_PANIC = {"anxious": 1.0, "stressed": 0.8, "overwhelmed": 0.4,
                            "tired": 0.2, "distracted": 0.1, "okay": 0.0, "motivated": 0.0}

    def _build_proxy_features(self, history_df):
        """
        history_df columns required: ts, mood, energy, task_name, blockers
        (blockers: comma/semicolon-joined string of blocker keys for that
        check-in, may be empty).

        Builds features structurally analogous to the OULAD Stage 3B
        TYPE_FEATURES this RF was trained on (delay_ratio_mean,
        is_last_minute_mean, non_submit_mean, delay_deviation_pos_mean,
        avoidance_score_pos_mean, click_drop_ratio_pos_mean,
        screen_ratio_mean, focus_trend_pos_mean, consec_low_mean,
        distraction_freq_mean) — NOT the same data, a documented proxy.
        """
        n = len(history_df)
        same_task_rate = history_df["task_name"].duplicated(keep=False).mean() if n > 1 else 0.0
        last_minute_rate = history_df["mood"].map(self.MOOD_DEADLINE_PANIC).fillna(0).mean()
        non_submit_proxy = same_task_rate

        sleep_col = history_df["sleep_hours"] if "sleep_hours" in history_df else pd.Series([7.0]*n)
        delay_deviation_pos = float(sleep_col.std()) if n > 1 else 0.0
        delay_deviation_pos = 0.0 if np.isnan(delay_deviation_pos) else delay_deviation_pos

        avoidance_score_pos = history_df["mood"].map(self.MOOD_DISTRACTION).fillna(0).mean()

        if n > 1 and "energy" in history_df:
            energies = history_df.sort_values("ts")["energy"].values
            click_drop_ratio_pos = max(0.0, float(energies[0] - energies[-1]) / 10.0)
            focus_trend_pos = click_drop_ratio_pos
        else:
            click_drop_ratio_pos = 0.0
            focus_trend_pos = 0.0

        screen_ratio = avoidance_score_pos

        energies = history_df.sort_values("ts")["energy"].values if "energy" in history_df else np.array([5]*n)
        streak = cur = 0
        for e in energies:
            if e <= 3:
                cur += 1
                streak = max(streak, cur)
            else:
                cur = 0
        consec_low = streak / max(n, 1)

        distraction_freq = (history_df["mood"] == "distracted").mean()

        return {
            "delay_ratio_mean": same_task_rate, "delay_ratio_std": 0.0,
            "is_last_minute_mean": last_minute_rate,
            "non_submit_mean": non_submit_proxy,
            "delay_deviation_pos_mean": delay_deviation_pos, "delay_deviation_pos_std": 0.0,
            "avoidance_score_pos_mean": avoidance_score_pos,
            "click_drop_ratio_pos_mean": click_drop_ratio_pos,
            "screen_ratio_mean": screen_ratio,
            "focus_trend_pos_mean": focus_trend_pos,
            "consec_low_mean": consec_low,
            "distraction_freq_mean": distraction_freq,
        }

    def score_checkin_proxy(self, history_df):
        """
        Real inference call: builds proxy features from check-in history and
        runs them through the actual trained Stage 3B Random Forest.

        Returns None if there's not enough history yet (< MIN_CHECKINS_FOR_PROXY_SCORING)
        — caller should fall back to the disclosed rule-based diagnostic in
        that case, not present a guess as a model output.

        Return dict includes `is_proxy_features: True` and
        `n_checkins_used` so the UI can never silently drop the disclosure.
        """
        n = len(history_df)
        if n < MIN_CHECKINS_FOR_PROXY_SCORING:
            return None

        rf, le, rf_features = self.rf_pkg["model"], self.rf_pkg["encoder"], self.rf_pkg["features"]
        proxy = self._build_proxy_features(history_df)
        x = pd.DataFrame([{f: proxy.get(f, 0.0) for f in rf_features}])
        proba = rf.predict_proba(x)[0]
        pred_idx = int(np.argmax(proba))
        proc_type = le.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        return {
            "proc_type": proc_type,
            "type_confidence": round(confidence, 3),
            "is_proxy_features": True,
            "is_low_confidence": confidence < 0.5,
            "n_checkins_used": n,
            "proxy_features_used": proxy,
            "model": "Stage 3B Random Forest (real model call, proxy features — see docstring)",
        }