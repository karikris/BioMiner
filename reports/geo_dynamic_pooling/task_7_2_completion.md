# Task 7.2 completion — versioned raw fusion and candidate rankings

Status: completed and pushed to `origin/main` through
`50b7e84fc2964365f947d9749f03f305ff2c893e`.

## Delivered

- A lossless component set retains the exact family evidence set and every
  candidate's global, local and disagreement/coverage rows. Source versions,
  query and matrix identities, candidate identities and fingerprints fail
  closed on drift.
- Four provisional methods run over the same component set: unweighted
  component mean, validation-fitted linear fusion, maximum scope evidence and
  robust rank aggregation.
- Linear fusion requires a validation artifact fingerprint plus explicit
  coefficients and intercepts for both six-component and global-only cases.
  Missing local evidence is omitted without imputation.
- Robust aggregation uses average ranks for exact ties, normalizes ranks within
  each available component population and takes the median utility. Outputs
  retain both source cosine values and transformed method-input values.
- Each method independently ranks the complete candidate set by descending raw
  score and accepted-taxon-key tie breaker. Zero margins and adjacent ties are
  explicit, and every non-top candidate remains an ordered alternative.
- Cross-method top-one agreement is diagnostic only. The method selection state
  remains `not_selected`, and no consensus score is manufactured.

## Measured gate

- Fusion ablation and exact-semantics suite: 80 passed in 0.58 seconds.
- Full regression: 2,890 passed in 104.98 seconds.
- Repository-wide Ruff: passed.
- Exact raw-score language scan: zero disallowed calibrated-score terms.
- Provenance: 152 valid JSONL records; all four Task 7.2 records state
  `skipped_user_directive`, `solution_id: null`, and no GitHits call.
- Remote verification: `origin/main` resolved to `50b7e84fc...` after push.

The mixed-availability fixture retained one global-only candidate across all
four methods. A second fixture inverted global and local components: maximum
scope evidence selected `gbif:200`, the other three methods ordered
`gbif:100` first, and two methods retained exact score ties. This demonstrates
method disagreement and tie handling, not method quality or superiority.

## Claim boundary

No method or candidate strategy is selected or made default. No live BioCLIP
or accelerator workload ran, and the fixture does not establish scientific
performance, taxonomy, human review, occurrence release or deployment.
Compute and memory optimization begins in Task 8.1.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
