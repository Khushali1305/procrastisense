# Project Description — ProcrastiSense

## Problem

Generic "you have an assignment due" reminders don't work because
procrastination isn't a single behavior — it has different root causes
(deadline anxiety, distraction/avoidance, perfectionism) that need different
interventions. Most academic nudge tools treat all avoidance the same way.

## Solution

ProcrastiSense uses real student behavioral data (OULAD: assignment
submission timing, VLE engagement/click patterns) to:

1. Score each student's current risk/anomaly level using a blended
   unsupervised + supervised model (Isolation Forest + Logistic Regression).
2. Classify the student's likely procrastination *type* using a Random
   Forest trained on K-Means-derived behavioral clusters.
3. Generate a type-specific, LLM-written nudge (Llama 3.3 70B via Groq) with
   a concrete, sub-10-minute micro-action — not a generic reminder.

## Target users

Students on online/hybrid learning platforms, and the platforms themselves
(as an engagement/retention signal for advisors or learning-support staff).

## Technical approach

- **Data**: OULAD (32,593 students, 22 courses, ~10.6M VLE interaction
  records), engineered into a per-assessment feature matrix (393K rows, 50
  features) covering submission delay, click-trend, and engagement-drop
  signals.
- **Stage 3A (risk/anomaly)**: Isolation Forest + Logistic Regression,
  blended 0.3/0.7. AUC-ROC 0.894 on a time-based holdout (train on 2013
  cohort, test on 2014 cohort — guards against leakage across presentations).
- **Stage 3B (type)**: K-Means (k=3) on per-student aggregated behavioral
  features, with a Random Forest trained to generalize cluster assignment to
  new students without re-running K-Means each time.
- **Stage 5 (nudge generation)**: Groq-hosted Llama 3.3 70B, prompted with
  the student's type, mood, and task, constrained to JSON output.
- **Stage 6 (deployment)**: Streamlit app with two flows — a demo-profile
  explorer (using real OULAD students' real model output) and a live daily
  check-in flow (using behavioral proxy features derived from the user's own
  check-in history, since live click-stream data isn't available from a
  short form).

## What makes this honest, not just functional

Every metric reported in the Model Card was obtained from a held-out split
that the model never saw during training. Where a metric reflects an inherent
limitation (e.g., RF "accuracy" against K-Means-derived labels, rather than
ground-truth type accuracy), that limitation is named explicitly rather than
presented as a clean number.

## Impact / next steps

- Validate procrastination type against an external dataset with genuine
  type labels, if one becomes available (none currently joins to OULAD at
  the student level).
- A/B test nudge effectiveness against a generic-reminder control group.
- Expand beyond OULAD to a live institution's LMS data.
