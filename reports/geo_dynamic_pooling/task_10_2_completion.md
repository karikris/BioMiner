# Task 10.2 completion — calibrated evidence and audited screening thresholds

Status: completed and pushed to `origin/main` through
`4c38a10012e596b7591945da70821732a0602775`.

## Delivered

- A 75-dimension dynamic-pool feature contract retains raw global/local,
  margin, family, disagreement, geography, route, coverage, scale and query
  evidence. Raw nulls remain visible; model-vector zero fill is paired with an
  explicit availability indicator. The human outcome is never a model feature.
- Route-specific standardized logistic evidence models produce component-grouped
  out-of-fold logits on calibration rows. BioMiner's audited sigmoid calibrator
  fits those held-out logits, then a transparent coefficient runtime is refit on
  calibration rows only.
- Reliability is assessed on validation rows only. Threshold selection audits
  every distinct validation probability and uses the minimum of a one-sided
  Kish-effective-n Wilson precision bound and an exact component-level bound.
  Passing thresholds maximize weighted validation coverage.
- The final-test lane emits zero predictions during fitting and selection. Its
  feature values can change without changing the model, calibrator, validation
  predictions or fit fingerprint.

## Fixture evidence

The deterministic 36-row software fixture freezes to 14 calibration, 11
validation and 11 untouched final-test rows. Four grouped OOF folds produce
validation Brier `0.0055986`, log loss `0.0751940` and ECE `0.0722312`.

Under a deliberately permissive demonstration policy (precision LCB 0.50,
minimum two items and components), threshold `0.9124922` selects six independent
items at weighted precision 1.0, conservative lower bound `0.6069622`, and
weighted coverage `0.5096525`. The production defaults remain 0.95/0.95 with
minimum 30 items and 30 components; the small fixture correctly fails that gate.

These are synthetic fixture metrics, not live Flickr performance estimates.

## Gate

- Calibration, reliability and risk-control suite: 76 passed in 5.68 seconds.
- Full regression: 3,005 passed in 109.62 seconds.
- Changed-file Ruff, format and `git diff --check`: passed.
- Remote `origin/main` resolved to `4c38a10…` after the task push.

## Claim boundary

The selected label is exactly `statistically_supported_screening_candidate`.
It is screening-only and cannot authorize occurrence release. No final-test
performance was measured. Live reviewed data, the default evidence floors,
locked final evaluation, decisive human review and every release gate remain
required.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
