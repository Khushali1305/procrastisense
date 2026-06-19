"""
model_inference.py — ProcrastiSense v2

Loads the v2 retrained models (isolation_forest_v2.pkl, scaler_v2.pkl,
logistic_regression_v2.pkl, random_forest_type_v2.pkl) and runs REAL
inference against the 3 precomputed demo-profile feature vectors
(demo_profiles_v2.json — real OULAD students from the 2014 holdout,
selected because the v2 models genuinely flag them and RF/K-Means agree).

This module does NOT run inference for live check-in users. The v2
models were trained on OULAD submission-history features (rolling delay
averages, click-engagement baselines, etc.) that a brand-new check-in
user has no history for — running them on a fabricated/guessed feature
vector would be exactly the kind of dishonest "fake model output" this
rewrite exists to remove. Live check-in personalization comes from
cohort_matching.py instead (real cohort statistics + self-reported type
+ LLM nudge) — see app.py.

Usage:
    inference = ModelInference(models_dir="...", demo_json_path="...")
    profile = inference.get_demo_profile("deadline_panic")
    # -> dict with if_score, blended_score, rf_predicted_type, rf_confidence,
    #    top3_shap_features, all computed for real by the v2 models.
"""
import json
import pickle
from pathlib import Path

FEATURE_COLS_V2 = [
    'delay_days_pos', 'delay_ratio', 'delay_deviation_pos',
    'rolling_avg_delay_ratio', 'rolling_last_min_rate', 'rolling_nonsub_rate',
    'delay_zscore_pos', 'non_submit', 'is_last_minute',
    'click_baseline', 'click_drop_ratio_pos', 'click_trend_pos',
    'is_dropout_week', 'consec_low',
    'day_of_week', 'days_to_next_deadline', 'exam_season',
]

FEATURE_LABELS = {
    'delay_days_pos': 'days late (0 if early/on-time)',
    'delay_ratio': 'delay relative to time allowed',
    'delay_deviation_pos': 'delay above personal baseline',
    'rolling_avg_delay_ratio': 'rolling avg delay ratio (last 3 assessments)',
    'rolling_last_min_rate': 'rolling last-minute submission rate',
    'rolling_nonsub_rate': 'rolling non-submission rate',
    'delay_zscore_pos': 'delay z-score above personal baseline',
    'non_submit': 'did not submit this assessment',
    'is_last_minute': 'submitted in the last 24h before deadline',
    'click_baseline': 'VLE click baseline (first 4 weeks)',
    'click_drop_ratio_pos': 'drop in VLE engagement vs baseline',
    'click_trend_pos': 'declining click trend',
    'is_dropout_week': 'in a week with course-wide engagement drop',
    'consec_low': 'consecutive low-engagement weeks',
    'day_of_week': 'day of week (0=Mon)',
    'days_to_next_deadline': 'days until next deadline',
    'exam_season': 'in exam season',
}


class ModelInference:
    def __init__(self, models_dir=".", demo_json_path="demo_profiles_v2.json"):
        models_dir = Path(models_dir)
        self.scaler = pickle.load(open(models_dir / "scaler_v2.pkl", "rb"))
        self.if_model = pickle.load(open(models_dir / "isolation_forest_v2.pkl", "rb"))
        self.log_reg = pickle.load(open(models_dir / "logistic_regression_v2.pkl", "rb"))
        rf_dict = pickle.load(open(models_dir / "random_forest_type_v2.pkl", "rb"))
        self.rf_model = rf_dict["model"]
        self.rf_encoder = rf_dict["encoder"]
        self.rf_features = rf_dict["features"]
        self.demo_profiles = json.load(open(demo_json_path))

    def get_demo_profile(self, proc_type: str) -> dict:
        """
        Returns the precomputed real-inference result for one of the 3
        demo archetypes: 'deadline_panic', 'distraction_escape',
        'perfectionism_paralysis'. All numbers (if_score, blended_score,
        rf_predicted_type, rf_confidence, SHAP top-3) came from actually
        running the v2 models on a real OULAD student's feature row —
        see retrain_v2.py for how these were selected and computed.
        """
        if proc_type not in self.demo_profiles:
            raise ValueError(f"Unknown type '{proc_type}'. Options: {list(self.demo_profiles.keys())}")

        p = self.demo_profiles[proc_type]
        shap_explained = [
            {
                "feature": feat,
                "label": FEATURE_LABELS.get(feat, feat),
                "value": val,
                "shap_contribution": shap_val,
                "direction": "pushes toward anomalous" if shap_val < 0 else "pushes toward normal",
            }
            for feat, val, shap_val in p["top3_shap_features"]
        ]

        return {
            "id_student": p["id_student"],
            "is_real_oulad_student": True,
            "is_true_holdout_2014": True,
            "if_score": round(p["if_score"], 3),
            "blended_score": round(p["blended_score"], 3),
            "rf_predicted_type": p["rf_predicted_type"],
            "rf_confidence": round(p["rf_confidence"], 3),
            "rf_agrees_with_archetype": p["rf_matches_kmeans_label"],
            "shap_explanation": shap_explained,
            "disclosure": (
                "Computed by running the actual v2 Isolation Forest, Logistic "
                "Regression, Random Forest, and SHAP TreeExplainer on this real "
                "OULAD student's true feature vector from the 2014 holdout set "
                "(never seen during model fitting)."
            ),
        }

    def model_metadata(self) -> dict:
        return {
            "n_features_anomaly_models": len(FEATURE_COLS_V2),
            "feature_cols": FEATURE_COLS_V2,
            "if_contamination": self.if_model.contamination,
            "rf_classes": list(self.rf_encoder.classes_),
        }


if __name__ == "__main__":
    inf = ModelInference(
        models_dir="/home/claude/v2_output/stage3_models_v2",
        demo_json_path="/home/claude/v2_output/demo_profiles_v2.json",
    )
    for t in ["deadline_panic", "distraction_escape", "perfectionism_paralysis"]:
        prof = inf.get_demo_profile(t)
        print(f"\n=== {t} (student {prof['id_student']}) ===")
        print(f"  blended_score={prof['blended_score']}, RF predicts={prof['rf_predicted_type']} "
              f"(conf={prof['rf_confidence']})")
        for s in prof["shap_explanation"]:
            print(f"    - {s['label']}: value={s['value']:.2f}, {s['direction']}")
