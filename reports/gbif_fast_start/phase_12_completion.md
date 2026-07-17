# Phase 12 completion — Papilio demoleus pilot

All five pilot tasks and the acceptance repair were pushed to `main`. The
targeted pilot suite passed 61 tests in 9.17 seconds; the final repository suite
passed 2,526 tests in 99.55 seconds.

The fixture-backed adaptive path reaches provisional scoring in a measured
502.480375 ms with zero prior reference reviews. A real 18,041-record outcome-
blind sampling frame produced a weighted 50-record representative review queue.
No human-reviewed Flickr labels exist yet, so the audit is
`insufficient_sample`, all quality metrics are null, no species is legitimately
flagged, and live remediation remains blocked.

Production fixtures demonstrate targeted remediation without overclaiming:
one affected species is reviewed while one unaffected species remains out of
scope; one bad reference is excluded; cached embeddings are reused; and only
affected Flickr evidence is rescored. These are mechanism tests, not live
Papilio outcomes.

Earlier prototype-only manifests record 81 provider-supported references, 81
embeddings, 26 prototypes and 13,496 classified Flickr records. Their ignored
local artifacts are absent, so those values remain historical context.

The first full regression found a shared production validator containing the
pilot species literal. Commit `d53f57e` made target validation generic while
keeping Papilio specificity in configuration and tests. The subsequent full
suite passed cleanly.
