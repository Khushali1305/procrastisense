# Model Card — ProcrastiSense Stage 3 Models

## Stage 3A — Risk / Anomaly Detection

**Models**: Isolation Forest (unsupervised) + Logistic Regression
(supervised), blended 0.3 / 0.7.

**Label**: `at_risk` (derived from OULAD `final_result`: Withdrawn/Fail = 1).
Base rate ≈ 52–58% depending on split — i.e. NOT a rare-event/minority
label, which matters (see Known Issues Fixed below).

**Validation**: two independent holdouts —
1. Time-based: train on 2013 presentations, test on 2014 (guards against
   any temporal leakage).
2. Student-level random 80/20 split.

| Split | IF alone (AUC-ROC) | LR alone (AUC-ROC) | Blended (AUC-ROC) | Blended (AUC-PR) |
|---|---|---|---|---|
| Time-based (2013→2014) | 0.73 | 0.89 | 0.894 | 0.934 |
| Student-level random | 0.76 | 0.90 | 0.896 | 0.932 |

### Known issues fixed in this version

1. **Symmetric features made "early/engaged" look as anomalous as
   "late/disengaged."** Fixed by building one-sided ("direction-corrected")
   versions of every directional feature (`delay_days_pos`,
   `click_trend_pos`, etc.) so only the harmful direction can increase the
   anomaly signal.
2. **Anomaly-score sign was inverted for this label.** Isolation Forest's
   usual convention (lower `score_samples()` = more anomalous) assumes
   anomalies are a minority. Here, `at_risk` is the *majority* class
   (~55%), so the standard convention pointed the detector at the wrong
   group. Confirmed by testing both signs against the label directly; the
   non-negated `score_samples()` is correct for this dataset. This took the
   IF-alone AUC-ROC from 0.27 to 0.73.
3. **Original validation was circular** (an earlier `at_risk`-adjacent
   target was partially derived from the same engineered features being
   scored). Fixed by holding out an entire cohort year and an entire random
   student set, neither seen during training.

## Stage 3B — Procrastination Type Classification

**Models**: K-Means (k=3) on aggregated per-student behavioral features →
Random Forest trained to reproduce cluster assignment for new students.

**Important limitation, stated plainly**: there is no ground-truth label
for "procrastination type" anywhere in OULAD or any dataset we had access
to. The three types are a K-Means-defined construct. The Random Forest is
trained to mimic K-Means, not to predict an independently verified type.

| Check | Value | Interpretation |
|---|---|---|
| Silhouette score | 0.25 | Borderline cluster separation — disclosed as the weakest-validated part of the pipeline, not hidden |
| Stability (ARI across 4 reseeds) | 0.998 | Clusters are highly reproducible given the same features, regardless of separation quality |
| RF vs K-Means agreement (test) | 0.97 | This is agreement with the clustering, NOT validated "type accuracy" |
| RF vs K-Means agreement (5-fold CV) | 0.97 ± 0.003 | |

**What would change this**: an external dataset with independently-labeled
procrastination type, joinable to OULAD at the student level. None of the
candidate external datasets evaluated for this project could be joined
(no shared student identifiers); see `docs/PROJECT_DESCRIPTION.md` for what
was tried.

## Live App Inference

- **Demo Profiles page**: uses 3 real OULAD students, selected as the
  highest-confidence example of each type, with anomaly/type scores computed
  by running their actual feature vectors through the models above. No
  numbers on this page are hand-picked or invented.
- **Daily Check-in page**: a 60-second form cannot reproduce OULAD's
  click-stream features. Type/anomaly inference here uses behavioral proxy
  features derived from the user's own check-in history (mood trend, energy
  trend, repeated-task avoidance, sleep variability) run through the same
  trained Random Forest. This is disclosed in-app as a proxy, and requires
  at least 2 check-ins before producing a prediction (below that, the app
  says so rather than guessing).

## Open items for future work

- Cohort-level (non-row-joined) validation of `avoidance_score` against
  external distraction/performance datasets was scoped but not completed in
  this submission.
- `cohort_matching.py`, imported by `app.py`, is a stub pending that work —
  see repository README "Known limitations."
