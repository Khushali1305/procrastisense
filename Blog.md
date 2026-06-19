# Building ProcrastiSense: what we got wrong, and how we found out

## The pitch

Procrastination isn't one thing. A student who panics at deadlines needs a
different nudge than one who's doom-scrolling instead of starting, or one
who's stuck because the first draft has to be perfect. ProcrastiSense set out
to detect *which kind* of avoidance a student is showing, using real
behavioral data from the OULAD (Open University Learning Analytics Dataset),
and respond with a targeted, AI-generated nudge instead of a generic
reminder.

## Where it actually started

The first version looked complete: a trained Isolation Forest, three
procrastination "types" from K-Means clustering, SHAP explanations, a working
Streamlit app with demo profiles and a check-in flow. It looked like a
finished product.

It wasn't. A review pass caught three separate problems stacked on top of
each other:

1. **Three of four promised datasets were synthetic.** Code that was supposed
   to load and join external behavioral data was generating it with
   `numpy.random` instead.
2. **The anomaly detector was validated against itself.** The "at-risk" label
   it was scored against had been partly derived from the same features it
   was trained on — a circular validation that made the metrics look better
   than the model actually was.
3. **The live app didn't run any model at all.** Despite all the training
   code in the notebook, the deployed Streamlit app asked users to
   self-select their procrastination type from a dropdown. The "AI
   detection" was the user typing in the answer themselves.

That's not a small bug. That's the entire product not doing the thing it
claimed to do.

## Fixing it for real, not just looking fixed

Rebuilding honestly meant going back to first principles on each piece:

**The anomaly detector's AUC-ROC was 0.237** — worse than a coin flip. The
features were symmetric: a student submitting 10 days *early* looked just as
"anomalous" to the model as one submitting 10 days *late*. We built one-sided
versions of every directional feature (only the "bad" direction counts), and
the AUC moved — but only to 0.27. Still broken.

The actual bug turned out to be more subtle and, honestly, more interesting:
the anomaly *score's sign* was inverted for this specific dataset. Isolation
Forest's standard convention treats the rarer pattern as more anomalous — but
in OULAD, being "at risk" is the *majority* outcome (~55% of students), not a
minority. So the standard convention pointed the detector at the wrong group
entirely. Flipping the score direction took the AUC from 0.27 to **0.73** — a
genuinely useful signal, not noise dressed up with a confident-looking
number.

**The "three procrastination types" had no ground truth.** No dataset
anywhere labels a student's procrastination type — the clusters that define
"deadline-panic" vs "distraction-escape" vs "perfectionism-paralysis" were
themselves a modeling choice, not a fact in the data. The classifier trained
to predict those clusters scored ~97% "accuracy" — but that number measures
how well the classifier can mimic K-Means, not whether the typology is
correct. We kept the model, but we changed how we report it: silhouette
score for cluster separation, stability across re-seeds, and an explicit
label correction — "agreement with K-Means," not "type accuracy" — so the
number can't be misread as more validated than it is.

**The live app got rewired to actually use the models.** The demo profiles
are now three real OULAD students, picked by running their real feature
vectors through the real trained models — not three invented characters with
hand-picked scores. The check-in flow still can't reproduce OULAD's
click-stream features from a 60-second form (that's an honest constraint,
not a bug), so it builds real behavioral proxy features from a student's own
check-in history instead, and says so in the interface rather than letting
users assume it's reading their browser activity.

## What we'd still tell a judge, unprompted

- The procrastination-type clustering's silhouette score (0.25) is right at
  the edge of "trustworthy" — we shipped it, but we'd flag it as the
  weakest-validated part of the pipeline if asked.
- Cohort-level validation against external distraction/performance datasets
  was attempted but only partially completed — OULAD's student IDs don't
  join to any external dataset we found, so validation there is
  distributional, not row-level.
- The proxy features driving the live check-in's type prediction are a
  documented approximation of the OULAD feature space, not the same data.

## The actual lesson

A model that looks finished and a model that actually works are different
claims, and the gap between them is usually small, specific, boring bugs —
a negated sign, a majority-class assumption, a label that was never really
ground truth — not big conceptual failures. Finding them takes checking the
metric you don't want to check, and the discipline to leave the honest
caveat in the README instead of cutting it for a cleaner pitch.
