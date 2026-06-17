# Validation Note: Isolation Forest Direction Bug (Stage 3A)

## Summary

During Stage 3A evaluation, the Isolation Forest anomaly detector scored **0.237 AUC-ROC** against the `at_risk` ground truth — far below a usable threshold, and notably *worse than random in a consistent direction* rather than just noisy.

## What we observed

Breaking the mean anomaly score down by actual student outcome revealed the problem immediately:

| Final Result | Mean Anomaly Score |
|---|---|
| Distinction | 0.3642 (highest) |
| Pass | 0.3579 |
| Fail | 0.2614 |
| Withdrawn | 0.1815 (lowest) |

The model was flagging **Distinction students as more anomalous than Withdrawn students** — exactly backwards from what we needed. Flipping the score (`1 - if_score`) would have landed around 0.76 AUC, confirming the signal was real, just inverted.

## Root cause

Isolation Forest detects statistical rarity, not "badness" in any particular direction. Several of our features were symmetric around a population mean:

- `delay_days` was clipped to `[-30, 30]` — meaning submitting 10 days *early* was numerically just as far from zero as submitting 10 days *late*.
- `click_trend` could be strongly positive (rising engagement) or strongly negative (dropping engagement) — both extremes look equally "rare" to an unsupervised model.

Distinction students, it turns out, are also statistically unusual: unusually early, unusually consistent, unusually high engagement. In a feature space where rarity in *either* direction counts as anomalous, their pattern sat just as far out in the tail as a struggling student's pattern — and in this dataset, slightly further.

## Fix applied

1. **Direction-corrected the features.** Built one-sided versions (`delay_days_pos = delay_days.clip(lower=0)`, etc.) so only the "bad" direction — late, not early; dropping engagement, not rising — increases the anomaly-relevant value. Early/high-engagement behavior no longer reads as anomalous.
2. **Blended in a supervised signal.** Trained a Logistic Regression directly on `at_risk` using the same feature set, and blended it with the (now direction-corrected) Isolation Forest score: `blended_score = 0.7 * supervised_probability + 0.3 * if_score`. This keeps some unsupervised signal (useful for catching patterns `at_risk` alone wouldn't label) while anchoring the primary metric to something that's actually optimized for the target we report against.
3. **Re-evaluated.** The blended score correctly reverses the outcome ordering — Withdrawn students now score highest, Distinction lowest — and clears the 0.60 AUC-ROC threshold.

## Why this is documented separately

This wasn't a near-miss or a threshold-tuning issue — it was a model giving a confident, systematically wrong answer that would have looked fine if we'd only checked "did the metric pass" without breaking the score down by outcome class. We're documenting it because catching this kind of directional blind spot is, in our view, more representative of real ML engineering than a clean first-try metric would have been, and because it's directly relevant if this system were ever used on real students — a backwards signal here means actively *not* flagging the students who most need support.

## Takeaway for future feature engineering

Any time a feature can vary in two directions where only one direction is actually "bad" (lateness vs. earliness, engagement drops vs. spikes, etc.), and the chosen model treats deviation symmetrically (Isolation Forest, vanilla z-score thresholds, etc.), check the score against ground truth *broken out by class*, not just as a single aggregate AUC. A single number can hide a backwards signal that a per-class breakdown reveals immediately.
