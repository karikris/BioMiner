# Task 9.2 completion — dynamic human-review evidence planning

Status: completed and pushed to `origin/main` through
`7ba61be8ac376a0d8ebac7023384be0600187555`.

## Delivered

- `ReviewEvidencePolicy` preregisters target precision, confidence, lower-bound
  objective, represented-strata minimum, maximum budget, information-fraction
  milestones, Bonferroni familywise control, reviewer/owner/duplicate/
  observation grouping, interval method and stopping rule.
- The dynamic planner calculates exact one-sided independent binomial reference
  requirements, then reports Kish weight, observed grouping and external design
  effects separately before nominal inflation and the stratum floor. Infeasible
  objectives and budget shortfalls remain explicit.
- Milestone updates evaluate immutable first-N eligible decisive event prefixes.
  Targeted, uncertain/nondecisive and release-review events remain visible but
  cannot advance representative stopping. Stopping requires target precision,
  the adjusted lower-bound objective and all required strata; it never grants
  occurrence-release authority.

## Reference calculations

For one look, one-sided 95% confidence, a 95% lower-bound objective, independent
equal-weight decisive records and all successes, 58 successes give a lower
bound of 0.9496607 while 59 give 0.9504924. Thus 59 is correct for that narrow
explanatory case only. The default four-look Bonferroni policy needs 86 under
the same all-success assumptions.

In the weighted/grouped fixture, weight design effect 1.36, maximum grouping
effect 1.35 and external effect 1.2 combine to 2.2032, inflating 59 effective
reviews to 130 nominal reviews. This is explicit conservative planning, not an
exact complex-survey confidence interval.

## Gate

- Statistical, weighted and clustered suite: 73 passed in 3.10 seconds.
- Full regression: 2,970 passed in 106.83 seconds.
- Ruff on both Phase 9 modules and tests: passed.
- A preflight command used two stale test filenames and collected no tests; the
  corrected command used `test_evaluation_uncertainty.py` and
  `test_reference_bank_audit.py` and passed. No product test failed.
- Remote `origin/main` resolved to `7ba61be8…` after push.

## Claim boundary

The calculations and milestone results are deterministic software/reference
fixtures, not collected live human evidence, model calibration, scientific
support or occurrence release. Review requirements remain policy-, error-,
weight-, grouping-, stratum-, milestone- and budget-dependent; 59 must never be
used as a universal production count.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
