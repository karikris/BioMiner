# Task 11.1 completion — hierarchical pooling quality audits

Status: completed and pushed to `origin/main` through
`c70721b4f4f18531577f8887b0f064bd8332e378`.

## Delivered

One long-form quality engine now reports the same 13 metrics overall and by
family, genus, species, geographic availability, country, admin1, bioregion and
geographic cluster. Taxonomic keys must retain a single canonical name and
parent chain. Every source item belongs to exactly one group at each geographic
level; unknown fields and `no_geo` remain explicit data states.

Representative estimates use positive design weights. Confidence intervals use
the smaller of row-level Kish effective sample size and effective sample size
after weights are aggregated by independence component. Targeted failure rows
remain visible in source/exclusion counts but never enter representative
denominators.

Configured item, component and effective-sample floors apply both to the group
and to each metric denominator. Underpowered groups are retained with exact
machine-readable reasons and null estimates. The reports are descriptive only
and cannot authorize occurrence release.

## Gate

- Grouped/weighted hierarchy and adjacent statistical suite: 84 passed in 6.23
  seconds.
- Full regression: 3,048 passed in 109.73 seconds.
- Changed-file Ruff, format and `git diff --check`: passed.
- Remote `origin/main` resolved to `c70721b…` after the task push.

## Claim boundary

These contracts establish auditable quality reporting, not live quality. No
underpowered estimate is emitted, missing geography is not biological absence,
and targeted failure discovery is not representative performance evidence.
Live representative reviews and configured evidence floors remain required.

GitHits contributed no code or architecture because the user disabled all
further calls for this goal.
