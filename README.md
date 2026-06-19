# ProcrastiSense

AI-powered procrastination detection and intervention, built on the OULAD
(Open University Learning Analytics Dataset).

**Live app:** [add your Streamlit deployment link here]
**Demo video:** [add link here, recommended]

---

## What it does

ProcrastiSense analyzes student behavioral patterns (assignment delay,
engagement drop-off, distraction signals) to:

1. **Detect risk** of disengagement/avoidance using a blended Isolation
   Forest + Logistic Regression anomaly model (Stage 3A).
2. **Classify procrastination type** — deadline-panic, distraction-escape,
   or perfectionism-paralysis — using K-Means clustering + a Random Forest
   classifier trained to generalize the clustering to new students (Stage 3B).
3. **Generate a personalized nudge** for each type via an LLM (Llama 3.3 70B
   through Groq), shown in a Streamlit check-in app.

## Project structure

```
.
├── app.py                       # Streamlit application
├── model_inference.py           # Loads trained models, runs inference
├── cohort_matching.py           # Cohort-based validation against external datasets
├── procrastisense-complete.ipynb# Full notebook: data prep -> Stage 3A/3B -> Stage 5/6
├── stage3_models_v2/            # Trained model artifacts (pickles)
│   ├── scaler.pkl
│   ├── isolation_forest.pkl
│   ├── logistic_regression_supervised.pkl
│   ├── random_forest_type.pkl
│   ├── if_score_normalization.pkl
│   └── stage3a_metadata.pkl
├── demo_profiles_v2.json        # Real OULAD students used on the Demo page
├── requirements.txt
├── README.md
├── Blog.md
└── docs/
    ├── PROJECT_DESCRIPTION.md
    ├── AI_TOOLS_DISCLOSURE.md
    └── MODEL_CARD.md            # Metrics, validity checks, known limitations
```

## Models & validated metrics

| Component | Metric | Value | Notes |
|---|---|---|---|
| Stage 3A — Isolation Forest (anomaly) | AUC-ROC (time-based holdout) | 0.73 | Direction-corrected; see Model Card |
| Stage 3A — Logistic Regression (risk) | AUC-ROC (time-based holdout) | 0.89 | |
| Stage 3A — Blended (0.7 LR + 0.3 IF) | AUC-ROC / AUC-PR | 0.894 / 0.934 | Production model |
| Stage 3B — Cluster separation | Silhouette score | 0.25 | Borderline; disclosed |
| Stage 3B — Cluster stability | ARI across reseeds | 0.998 | Stable |
| Stage 3B — RF vs K-Means | Agreement (not "accuracy") | 0.97 | See Model Card for why this distinction matters |

Full methodology, what was fixed, and what's still a known limitation: see
[`docs/MODEL_CARD.md`](docs/MODEL_CARD.md).

## Running locally

```bash
pip install -r requirements.txt
# Add your Groq API key(s) to .streamlit/secrets.toml:
#   GROQ_API_KEY_1 = "gsk_..."
#   GROQ_API_KEY_2 = "gsk_..."
#   GROQ_API_KEY_3 = "gsk_..."
streamlit run app.py
```

## Known limitations (disclosed, not hidden)

- Procrastination "type" has no ground-truth label in the source data; it is
  K-Means-derived. RF's high agreement with K-Means reflects label mimicry,
  not validated real-world type accuracy. See Model Card.
- The live check-in flow cannot reproduce OULAD-style click-stream features
  from a 60-second form; it derives proxy behavioral features from a
  student's own check-in history instead. This is disclosed in-app.
- Three of four originally proposed external datasets could not be row-joined
  to OULAD (no shared student IDs); cohort-level distributional matching was
  used instead where feasible.

## Team / AI tools

See [`docs/AI_TOOLS_DISCLOSURE.md`](docs/AI_TOOLS_DISCLOSURE.md).
