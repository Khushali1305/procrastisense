# AI Tools Disclosure

In the interest of transparency, here is how AI tools were used in building
ProcrastiSense.

## Claude (Anthropic)

Used throughout development for:
- Code review and bug-finding in the modeling pipeline (e.g., identifying
  that the Isolation Forest's anomaly-score direction was inverted relative
  to the `at_risk` label, and that the live Streamlit app contained no model
  inference at all — it used a self-select dropdown).
- Pair-programming the fixes: direction-corrected features, corrected
  anomaly-score sign, held-out time-based and student-level validation
  splits, and the rewired Streamlit inference layer.
- Drafting this documentation (README, Blog.md, Model Card, this disclosure)
  from the project's actual development history.

All metrics reported in this submission were computed by running the actual
code against the actual dataset in this development environment — they are
not AI-generated or estimated numbers.

## Groq-hosted Llama 3.3 70B

Used in the live product itself (not just during development) to generate
the personalized nudge text and micro-action shown to users in the
check-in flow. This is a core, disclosed feature of the product, not a
development tool — see [`docs/PROJECT_DESCRIPTION.md`](PROJECT_DESCRIPTION.md).

## What was NOT AI-generated

- The underlying OULAD dataset and its statistical properties.
- The model architectures (Isolation Forest, Logistic Regression, K-Means,
  Random Forest) and their evaluation metrics — these came from scikit-learn
  trained and scored against held-out data.
- The decision of which validation splits to use and what limitations to
  disclose — these were human judgment calls, informed by AI-assisted code
  review.

## Why we're disclosing the review process itself

An earlier version of this project had AI-generated demo content (synthetic
datasets, hardcoded "demo profile" scores) that was not clearly distinguished
from real model output. Part of fixing that was being explicit, in this
disclosure and in the Model Card, about exactly which numbers in this
submission come from trained models scored on held-out data versus anything
illustrative or AI-assisted.
