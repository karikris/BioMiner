# Task 10.3 completion — explicit release, screening and unresolved outcomes

Status: completed and pushed to `origin/main` through
`7e7ae4d767ca432cab386d5538a01bd15ff31f09`.

## Delivered

Every source-bound outcome-evidence item now enters exactly one deterministic
lane. The partition is complete and mutually exclusive by item ID, and its
bundle fingerprint binds the source evidence and all three lane fingerprints.

- `human_reviewed_release` accepts only a decisive, source-hash-bound human
  inclusion that independently passes conflict, occurrence-claim and existing
  Flickr release gates. Model evidence never grants release authority.
- `statistical_screening` accepts only unreviewed rows passing a selected,
  fingerprinted calibrated threshold and all route/reference/geography/visual,
  domain-negative and OOD gates. Its exact label is
  `statistically_supported_screening_candidate`.
- `review_required_or_abstained` retains every remaining row, including human
  exclusions. It is ranked by review priority and records explicit review,
  threshold, quality and release blocking reasons.

Screening and unresolved rows set occurrence release and final-dataset
eligibility false. Both schemas carry enough source/review provenance to reach
the existing verified-export validator and are rejected by it. The only table
accepted by that validator is the human-reviewed release projection.

## Gate

- Language, release, export, calibration and outcome suite: 68 passed in 4.11
  seconds.
- Full regression: 3,030 passed in 110.65 seconds.
- Changed-file Ruff, format and `git diff --check`: passed.
- Remote `origin/main` resolved to `7e7ae4d…` after the task push.

## Claim boundary

The new lanes enforce software authority boundaries; they do not measure live
quality. A calibrated probability is not a human review, a statistically
supported screening candidate is not release-ready, and unresolved output is
not a negative biological claim. Live source-bound evidence, decisive human
review and the preregistered final evaluation remain required.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
