# Phase 5 completion — adaptive orchestration and defaults

Phase 5 is complete. All four task commits were pushed and the full repository
suite passed 2,378 tests in 85.13 seconds.

| Task | Commit | Result |
|---|---|---|
| 5.1 | `fda9390e8729ff7d633a9ea71f169c532329cf07` | Explicit adaptive and manual-review stages |
| 5.2 | `98f2e9dcddf0fe5dbb08d3d6dff4f2df0ca9df59` | Conditional admission, review, audit and revision dependencies |
| 5.3 | `725e77539da279332d4c8fd256a99eee44de545e` | Adaptive GBIF production and CLI defaults |
| 5.4 | `410e19a0903532123000d9ef7fd6c4290e04b444` | Fail-closed cross-field configuration validation |

The default production contract uses `adaptive_gbif_fast_start`, GBIF reference
source, and `provisional_reference_ranking`. Flickr release still requires human
verification, and species quality approval still requires statistical reference
audit. Strict and flagged-only reference admission remain explicit CLI modes.

Reference review does not block first scoring. Automated admission does. Final
quality approval depends on Flickr review and statistical audit; targeted review
activates only for statistically flagged species, and affected rebuild/rescore
activate only after a reviewed flag produces a reference-bank revision.

Configuration fails closed for provisional references claiming strict readiness,
unreviewed references in calibration/final-test splits, final Flickr export
without human review, calibrated probability without a calibrator, adaptive mode
without audit, and unsupported non-GBIF unreviewed sources.

This is fixture-tested orchestration and policy evidence. It is not a live GBIF
or Flickr production run, a release-quality dataset claim, or measured evidence
of accuracy, speed, cost, or review-work savings.
