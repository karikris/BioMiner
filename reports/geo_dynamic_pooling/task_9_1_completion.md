# Task 9.1 completion — dynamic-pool review sampling

Status: completed and pushed to `origin/main` through
`c75e51ed39dd49a5667dc32f192dba7eb772a16b`.

## Delivered

- A versioned audit frame preserves candidate family, genus/species,
  geography/no-geo, query tier, raw score/margin bands, pool disagreement,
  route/domain, subject size, owner, duplicate and observation dimensions.
  Candidate taxonomy remains provisional evidence, no-geo remains missing
  source evidence, and raw model scores are explicitly not probabilities.
- The representative audit defines its target population as one deterministic
  representative per connected duplicate/observation component. Its all-unit
  register records both selected and non-selected units, exact stratum `n_h`
  and `N_h`, inclusion probability `n_h / N_h`, inverse weight, cross-stratum
  component state and owner variance cluster.
- Failure discovery is a separate targeted queue. Named heuristic signals set
  priority, but inclusion probability and weight remain null and the rows are
  not eligible for representative estimation.
- Occurrence-release review retains every final-release candidate, including
  duplicate-group members. Rows begin pending and reviewable but fail closed on
  all human-review, source-binding, duplicate, identity, domain, geography/date,
  second-review/adjudication and final-permit gates.

## Gate

- Sampling and release suite: 55 passed in 1.56 seconds.
- Full regression: 2,944 passed in 107.24 seconds.
- Ruff on the changed module and test file: passed.
- The probability fixture reduced six rows to five connected target-population
  units and selected three with exact inclusion probabilities and weights.
- The failure fixture retained two targeted components with null weights.
- The release fixture retained both final candidates despite a shared duplicate
  group and authorized zero releases.
- Remote `origin/main` resolved to `c75e51ed…` after push.

## Claim boundary

These are deterministic software fixtures, not live Flickr sampling, completed
human review, scientific validation or release. Targeted priorities are not
probabilities. The representative design does not establish precision until
decisive human evidence is collected and analysed under the Task 9.2 count and
stopping policy.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
