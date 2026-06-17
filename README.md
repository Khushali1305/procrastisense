# ProcrastiSense

**A behavioral procrastination detection and intervention system, built on real student learning data.**

> Mid-Evaluation Checkpoint Submission — 7-Day AI Buildathon

---

## Problem Statement

Procrastination is one of the most common — and most invisible — drivers of poor academic outcomes. Students don't fail because they don't understand the material; they often fail because they consistently delay engaging with it until it's too late to recover. Existing learning platforms track *what* students do (grades, submissions, logins), but almost none try to detect *how* a student is procrastinating — and even fewer respond to it with anything more than a generic deadline reminder.

ProcrastiSense asks two questions for every student in a course: **(1) is this student showing a procrastination pattern severe enough to be a risk signal, and (2) what *type* of procrastination is it** — deadline panic, distraction escape, or perfectionism paralysis — **so that any intervention can be specific instead of generic.**

The system is built and validated on the **Open University Learning Analytics Dataset (OULAD)**, a real, publicly available dataset of ~32,000 students' submission timing, VLE (virtual learning environment) clickstream behavior, and final outcomes, which makes it possible to train and honestly evaluate against ground-truth pass/fail/withdrawal outcomes rather than synthetic labels.

---

## Current Progress

### Stage 1 — Data Pipeline (Complete)
- Loaded and cleaned all 7 OULAD tables (studentInfo, studentRegistration, studentAssessment, assessments, courses, vle, studentVle).
- Solved the **10.6M-row studentVle memory problem** with chunked loading (200k rows/chunk) rather than loading the full clickstream into memory at once.
- Identified and handled three real data traps in OULAD that are easy to get wrong: **silent non-submission** (a missing row in studentAssessment means a student never submitted — it is *not* a null value to be imputed), **`is_banked` corruption** (banked assessment results must be filtered out, `is_banked == 0`, or they corrupt delay calculations), and **B/J presentation mixing** (OULAD runs two course presentations per year; mixing them without separating cohorts skews "typical" submission timing).
- Ground truth label: `at_risk = 1` if a student's `final_result` is Fail or Withdrawn (52.8% of the dataset) — used throughout for honest, non-circular evaluation.

### Stage 2 — Feature Engineering (Complete)
Built four feature groups from raw behavioral signals:
- **Delay signals** — days-late distributions, rolling 4-week mean/std per student, z-scored relative to a personalized baseline (with cohort fallback for new students).
- **VLE engagement signals** — click-count trends, week-over-week engagement drops, "disengagement window" detection.
- **Distraction/psychological signals** — avoidance score, focus trend (currently simulated pending a real wearable/app behavioral dataset — clearly flagged in the notebook).
- **Temporal context** — day-of-week, days-to-deadline, exam-season flags.

### Stage 3 — Modeling (Complete, including one significant fix)
- **3A — Isolation Forest (unsupervised anomaly detection):** flags rows with unusual behavioral patterns, with a "2 consecutive high-score assessments" rule to reduce false-positive noise down to a usable ~2.6% alert rate.
- **3B — Random Forest (procrastination type classifier):** K-Means first clusters students into three procrastination archetypes (deadline panic / distraction escape / perfectionism paralysis); a Random Forest then learns to predict that cluster from features so it generalizes to new students without re-running clustering.
- **Bug found and fixed during validation:** the initial Isolation Forest scored Withdrawn students *lower* than Distinction students on "anomaly" — an AUC-ROC of 0.237, i.e. a strong signal pointing the *wrong* direction. Root cause: several features (delay, click-trend) were symmetric, so being unusually *early* or unusually *engaged* counted as equally "anomalous" as being late or disengaged. Fixed by (1) direction-correcting those features so only the "bad" direction increases the score, and (2) blending in a supervised Logistic Regression trained directly on `at_risk`. Final blended AUC-ROC clears the 0.60 threshold. This is documented in-notebook as a worked example of catching a real validation failure, not silently passing a metric.

### Stage 4 — Explainability (Complete)
- SHAP values computed for every flagged row, producing a human-readable explanation string plus the top 3 contributing features per student — this is what makes any downstream intervention specific rather than a black-box alert.

### Stage 5 — LLM Nudge Generator (Complete)
- `generate_nudge()` takes `proc_type` + SHAP explanation + a mood input and returns a 2–3 sentence empathetic nudge plus one concrete micro-action (under 10 minutes, specific to the procrastination type — not generic "manage your time better" advice).
- Using **Groq (Llama 3.3 70B, free tier)** rather than a paid API, since this is a one-time batch job over ~500 flagged students and Groq's free tier covers it with no cost or billing setup. Calls are rotated across multiple free-tier keys to stay within per-key daily request limits.
- Mood input is currently **mocked** (`MOCK_DATA`, clearly labeled in-notebook) since no real check-in UI exists yet — the function signature is designed so swapping mock mood for a real check-in input later requires no changes to the nudge logic itself.
- Full batch run complete: **500/500 nudges generated**, $0 cost. Type distribution across the flagged cohort: **perfectionism paralysis 338 (67.6%)**, **deadline panic 126 (25.2%)**, **distraction escape 36 (7.2%)** — consistent with Stage 3B's overall class balance.
- Sample output (perfectionism paralysis, mood: overwhelmed):
  > *Nudge:* "I can tell you're feeling overwhelmed, and it's understandable given the exam season pressure. It's like the weight of wanting everything to be perfect is slowing you down."
  > *Micro-action:* "Take 5 minutes to write down the minimum requirements for your current task, focusing on what's truly necessary to move forward."
- Exported `stage5_nudges.parquet` (500 rows) and `stage5_summary.json`.

### ⬜ Not Started
- Frontend/dashboard for surfacing nudges to students or instructors.
- Real daily check-in mechanism (currently mocked).
- Deployment.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data processing | Python, pandas, NumPy (chunked loading for large clickstream data) |
| Modeling | scikit-learn (Isolation Forest, Random Forest, K-Means, Logistic Regression) |
| Explainability | SHAP |
| LLM nudge generation | Groq API (Llama 3.3 70B) |
| Notebook environment | Kaggle Notebooks |
| Dataset | Open University Learning Analytics Dataset (OULAD) |
| Planned | Frontend dashboard (TBD), deployment platform (TBD) |

---

## Planned Features (Remaining Work)

1. **Real check-in input** — replace mocked mood with an actual lightweight input mechanism (even a simple form is sufficient for demo purposes).
2. **Minimal frontend** — a simple dashboard (likely Streamlit or a lightweight React page) so a non-technical viewer can see a student's flag, type, explanation, and nudge without reading the notebook directly.
3. **Deployment** — host the demo so it's accessible via a public link for final submission.
4. **Blog.md** — write-up of the technical journey (including the AUC-ROC bug and fix) for the final submission deliverable.

---

## Setup Instructions

### Requirements
```bash
pip install pandas numpy scikit-learn shap groq pyarrow
```

### Running the notebook
1. Clone this repo.
2. Download OULAD from [the Open University's data repository](https://analyse.kmi.open.ac.uk/open_dataset) and place the CSVs in the path expected by the notebook (see the data-loading cell at the top of `notebooks/procrastisense`).
3. **Stages 1–4** run with no API key required.
4. **Stage 5** requires free Groq API key(s):
   - Get keys at [console.groq.com/keys](https://console.groq.com/keys) (no credit card required). One key's free tier (1,000 req/day) is usually enough for ~500 nudges; the notebook supports rotating across multiple keys if you need more headroom.
   - Paste your key(s) into the `GROQ_API_KEYS` list in the Stage 5 setup cell.
   - ⚠️ Don't commit a notebook with real keys pasted in — clear them before sharing/pushing.
5. Run all cells top to bottom. Each stage ends with a self-contained completion-check cell that prints ✅/❌ status for every expected output, so a failed step is immediately visible rather than silently skipped.

### Outputs produced
| File | Produced by | Consumed by |
|---|---|---|
| `stage3_anomaly_scores.parquet` | Stage 3A | Stage 5 (alert flag, blended score) |
| `stage3_type_labels.parquet` | Stage 3B | Stage 5 (procrastination type) |
| `stage3_shap_values.parquet` | Stage 4 | Stage 5 (explanation text) |
| `stage3_summary.json` | Stage 3 | Reporting |
| `stage5_nudges.parquet` | Stage 5 | Future frontend |
| `stage5_summary.json` | Stage 5 | Reporting |

---

## Repository Structure

```
procrastisense/
├── README.md
├── docs/                 # Additional documentation, fix write-ups
├── notebooks/            # Main Kaggle notebook (Stages 1–5)
├── models/               # Saved model artifacts (Isolation Forest, scaler, RF, Logistic Regression)
├── assets/               # Plots: anomaly distribution, SHAP summary, confusion matrix
├── outputs/              # Exported parquet/json files from each stage
├── frontend/             # (planned, not yet started)
└── backend/              # (planned, not yet started)
```