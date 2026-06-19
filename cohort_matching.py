"""
cohort_matching.py — ProcrastiSense v2 (revised)

IMPORTANT: an earlier draft of this module matched on mood/energy against
main_distractor/internal_barrier in hybrid_student_performance_1200.csv.
Direct verification showed those two columns have ~0 relationship with
EVERY other field in the dataset (mood, energy, tasks_on_time all return
~25-31% regardless) — they are statistically indistinguishable from random
noise. Matching against them would have been cosmetic personalization, not
real. This version uses ONLY the two relationships independently verified
to be real:

  1. sleep_hours -> productivity_score (student_productivity_distraction,
     20,000 rows): monotonic, ~13-point spread from <5h to 8h+ sleep.
  2. tasks_on_time -> performance_risk_level (hybrid_student_performance,
     1,200 rows): monotonic, High Risk share roughly doubles from
     "Always on time" (28%) to "Rarely on time" (49%).

Procrastination TYPE for the live check-in remains a self-report dropdown
("which pattern sounds like you") — same as before — because no real
data-driven signal exists to infer it from check-in inputs without the
OULAD submission history a brand-new user doesn't have. This module does
NOT claim to predict type; it only grounds the nudge with two real
cohort statistics.

No shared join key exists between OULAD and either survey dataset
(independently verified: 4 unrelated ID schemes) — this is deliberately
distributional/cohort-level grounding, not a per-student lookup.
"""
import pandas as pd
from pathlib import Path

DS_SLEEP_PATH = "student_productivity_distraction_dataset_20000.csv"
DS_TASKS_PATH = "hybrid_student_performance_1200.csv"

_SLEEP_BINS = [0, 5, 7, 8, 24]
_SLEEP_LABELS = ["<5h", "5-7h", "7-8h", "8h+"]


class CohortMatcher:
    def __init__(self, data_dir="."):
        data_dir = Path(data_dir)
        self.ds_sleep = pd.read_csv(data_dir / DS_SLEEP_PATH)
        self.ds_tasks = pd.read_csv(data_dir / DS_TASKS_PATH)
        self._prep()

    def _prep(self):
        self.ds_sleep["sleep_bucket"] = pd.cut(
            self.ds_sleep["sleep_hours"], bins=_SLEEP_BINS, labels=_SLEEP_LABELS
        )
        self._sleep_stats = (
            self.ds_sleep.groupby("sleep_bucket")["productivity_score"]
            .agg(["mean", "count"]).round(1)
        )
        self._tasks_stats = pd.crosstab(
            self.ds_tasks["tasks_on_time"], self.ds_tasks["performance_risk_level"],
            normalize="index"
        ).round(3)

    def sleep_context(self, sleep_hours: float) -> dict:
        bucket = pd.cut([sleep_hours], bins=_SLEEP_BINS, labels=_SLEEP_LABELS)[0]
        row = self._sleep_stats.loc[bucket]
        overall_mean = self.ds_sleep["productivity_score"].mean()
        return {
            "sleep_bucket": str(bucket),
            "n_in_cohort": int(row["count"]),
            "avg_productivity_score": float(row["mean"]),
            "vs_overall_avg": round(float(row["mean"] - overall_mean), 1),
            "source": "student_productivity_distraction_dataset_20000.csv (n=20,000)",
        }

    def tasks_on_time_context(self, tasks_on_time_response: str) -> dict:
        """
        tasks_on_time_response must be one of: 'Always', 'Often', 'Sometimes', 'Rarely'
        — collect this as a new check-in question (real signal, unlike mood/energy).
        """
        if tasks_on_time_response not in self._tasks_stats.index:
            tasks_on_time_response = "Sometimes"  # safe default
        row = self._tasks_stats.loc[tasks_on_time_response]
        return {
            "self_report": tasks_on_time_response,
            "pct_high_risk_cohort": float(row.get("High Risk", 0) * 100),
            "pct_low_risk_cohort": float(row.get("Low Risk", 0) * 100),
            "source": "hybrid_student_performance_1200.csv (n=1,200)",
        }

    def build_narrative(self, sleep_hours: float, tasks_on_time_response: str) -> dict:
        sleep = self.sleep_context(sleep_hours)
        tasks = self.tasks_on_time_context(tasks_on_time_response)

        direction = "above" if sleep["vs_overall_avg"] >= 0 else "below"
        narrative = (
            f"Real students sleeping {sleep['sleep_bucket']} (n={sleep['n_in_cohort']:,}) "
            f"average a productivity score of {sleep['avg_productivity_score']}/100 — "
            f"{abs(sleep['vs_overall_avg'])} points {direction} the overall average. "
            f"Separately, students who report completing tasks '{tasks['self_report']}' on time "
            f"are flagged High Risk {tasks['pct_high_risk_cohort']:.0f}% of the time in a "
            f"real student survey, vs {tasks['pct_low_risk_cohort']:.0f}% Low Risk."
        )

        return {
            "narrative": narrative,
            "sleep_context": sleep,
            "tasks_context": tasks,
            "disclosure": (
                "These are real, independently-verified cohort statistics from two "
                "separate student datasets (no shared IDs with OULAD or each other — "
                "distributional context, not a per-student match). Procrastination "
                "TYPE below is self-reported, not model-predicted — no dataset here "
                "has real signal linking mood/energy to distraction type."
            ),
        }


if __name__ == "__main__":
    cm = CohortMatcher(data_dir="/mnt/user-data/uploads")
    import json
    for sleep, tasks in [(4.5, "Rarely"), (8.5, "Always"), (6.0, "Sometimes")]:
        r = cm.build_narrative(sleep_hours=sleep, tasks_on_time_response=tasks)
        print(f"\n--- sleep={sleep}, tasks_on_time={tasks} ---")
        print(r["narrative"])
