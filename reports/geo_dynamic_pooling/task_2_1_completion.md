# Task 2.1 completion — geographic reference indexes

Task 2.1 is complete. Its four required subtask commits were pushed directly
to `origin/main` through `cd37037a98a9239c2ff4bb5d30c661e9c950ce66`;
that exact remote SHA was verified at 2026-07-18T03:36:53Z.

BioMiner now materializes five versioned outputs: normalized observation-level
geography, the cached-embedding geography index, a diverse global anchor core,
hierarchical local lookup memberships, and a cross-artifact QA manifest. Source
coordinates, uncertainty, source issues, snapshots and unavailable reasons are
retained. Local memberships expand only at the finest supported precision,
then through valid parents and explicit bioregion, country, continent and
selected-global fallbacks. Geography remains evidence rather than identity.

Global selection admits one biological observation and one duplicate group per
anchor, prioritizes unused known photographers, and records marginal diversity,
quality-flag count, quotas and shortfalls. Admission mode remains provenance;
it is not converted into an image-quality score. The neighbour artifact may
retain several valid embedding rows for one observation, but the QA manifest
reports this multiplicity and defines independence using distinct
`reference_observation_id`, never embedding-row count.

The Phase 2 reference/geography suite passed 133 tests. A full-suite gate first
reported one documentation-only failure because the updated agent state had
replaced the immutable Phase 0 design SHA. The state now retains that baseline
separately from the current implementation SHA; its focused contract passed 21
tests. The final full regression passed 2,730 tests in 101.37 seconds and
repository-wide Ruff passed.

A bounded two-observation reproducibility smoke wrote and reloaded all four
Parquet artifacts plus the JSON manifest. Schemas and semantic fingerprints
round-tripped, physical checksum coverage was complete, QA passed, and the
manifest fingerprint was
`sha256:f34cdf26acd156b93cc2a411dd591a1d603e1b1f76af020bbd88eda74306eda1`.
All five required GitHits calls timed out and are recorded as unavailable; no
external solution, code or prose is claimed.

This is software and deterministic fixture evidence. It does not claim that a
live production bank was indexed, provisional GBIF support was human verified,
model accuracy or calibration improved, geography proves identity, missing
local support proves absence, or downstream scoring, review, statistical
support, occurrence release, publication or deployment is complete.
