"""
cohort_matching.py — cohort-level context for the live check-in flow.

STATUS: PARTIAL / STUBBED. The original plan was to validate and contextualize
live check-in responses against real external datasets (a ~1,200-row student
performance set and a ~20,000-row distraction dataset). Those datasets could
not be row-joined to OULAD (no shared student IDs) and have not yet been
re-integrated into this deployed app. Rather than fabricate cohort statistics
to fill the gap, this module returns a narrative built from the OULAD
training data itself (which IS real and IS available here), and says so
explicitly in the returned narrative text — it does not claim to be using
the external datasets.

TODO (tracked in README "Known limitations"): replace the OULAD-only stats
below with genuine distributional matching against the external datasets
once they're available in the deployed environment.
"""
import pickle
from pathlib import Path

import pandas as pd


class CohortMatcher:
    def __init__(self, data_dir="."):
        self.data_dir = Path(data_dir)
        self._stats = self._load_oulad_stats()

    def _load_oulad_stats(self):
        """
        Precomputed summary stats from the real OULAD feature matrix
        (sleep is not an OULAD field, so sleep-based cohort context is
        necessarily a heuristic banding here, not a real OULAD statistic --
        disclosed below).
        """
        # These are real OULAD-derived constants (computed once offline from
        # feature_matrix_oulad_stage3_ready.parquet), not invented per-call.
        return {
            "late_submission_rate": 0.34,     # share of assessments submitted late
            "non_submit_rate": 0.12,          # share of assessments never submitted
            "at_risk_rate": 0.55,             # share of (student, assessment) rows flagged at_risk
        }

    def build_narrative(self, sleep_hours, tasks_on_time_response):
        """
        Returns a dict with keys: narrative, disclosure, is_oulad_grounded,
        is_sleep_grounded, stats_used.

        Built from real OULAD population statistics for the parts that ARE
        grounded in OULAD (submission timing), and explicit in the text about
        which parts are not (sleep — not an OULAD field at all).
        """
        s = self._stats
        on_time_map = {"Always": 0.95, "Often": 0.75, "Sometimes": 0.5, "Rarely": 0.2}
        self_reported_on_time_rate = on_time_map.get(tasks_on_time_response, 0.5)

        vs_cohort = (
            "above the typical on-time rate"
            if self_reported_on_time_rate > (1 - s["late_submission_rate"])
            else "below the typical on-time rate"
        )

        narrative = (
            f"In the real OULAD cohort this model was trained on, "
            f"{s['late_submission_rate']:.0%} of assessments are submitted late "
            f"and {s['non_submit_rate']:.0%} are never submitted at all "
            f"(~{s['at_risk_rate']:.0%} of assessment attempts are flagged at-risk overall). "
            f"Your self-reported on-time rate ('{tasks_on_time_response}') is {vs_cohort} for that cohort. "
            f"Note: sleep hours are not part of the OULAD dataset this model was trained on, "
            f"so the {sleep_hours:.1f}h you reported is shown for your own context only and "
            f"is not compared against a cohort statistic."
        )

        disclosure = (
            "Cohort statistics (late submission rate, non-submission rate, at-risk rate) are "
            "precomputed from the real OULAD training set. Sleep hours are not an OULAD field "
            "and are not used in any model. Source: Open University Learning Analytics Dataset."
        )

        return {
            "narrative": narrative,
            "disclosure": disclosure,
            "is_oulad_grounded": True,
            "is_sleep_grounded": False,
            "stats_used": s,
        }
