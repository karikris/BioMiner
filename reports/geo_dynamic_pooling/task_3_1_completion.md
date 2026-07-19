# Task 3.1 completion — canonical Flickr work units

Task 3.1 is complete. Its three required subtask commits were pushed directly
to `origin/main` through `fe03e46bf00c9c064d1f52c5d83320730a5f86fa`;
that exact remote SHA was verified at 2026-07-18T04:15:40Z.

BioMiner now separates four canonical scoring grains: one photo embedding unit,
one routed organism unit, many query or target associations, and many candidate
species. The organism identity reuses the existing target-full-frame scoring
unit. Image bytes and embedding vectors do not appear in any fan-out or
partition artifact; downstream work refers to the photo, visual-input, and
per-content model-input identities.

Flickr geography is now version 1.1.0. Provider accuracy remains distinct from
optional metric coordinate uncertainty, and the pipeline does not convert the
Flickr 1–16 accuracy scale into fabricated metres. Supported cell resolution,
country, admin region, source-provided or explicitly mapped bioregion, source
quality, and typed no-geo/invalid/unassigned states are retained. Coarse or
unknown precision cannot manufacture a local cell.

Partition assignments are one row per organism unit. Reusable partition keys
cover route, supported geographic work scope, complete candidate taxa, family
pool, and a content-independent model-input contract. Exact content identity
remains a cache reference, so grouping does not collapse to one partition per
image. Family signatures affect batching only; they do not delete candidates.
Every partition has an immutable membership summary with photo, visual-input,
model-input, candidate, family, association, and reuse counts.

The canonical-grain gate passed 118 tests. The intentional Flickr geography
schema evolution initially exposed one stale CLI manifest expectation; that
fixture was advanced from v1.0.0 to v1.1.0 and the complete 79-test geography
and CLI slice passed. The final full regression passed 2,745 tests in 104.48
seconds, and repository-wide Ruff passed.

A bounded five-photo round-trip wrote and reloaded the partition and summary
Parquets. It contained five photo units, five organism units, five query
associations, eleven candidate rows, five geography rows, five assignments,
and four partitions. Four unique visual/model inputs served five units, so one
shared input reuse was observed. The aggregate semantic fingerprint was
`sha256:d0a61da81a81eb574d7b66904de2665da12bcdb9dd0c253302e5f76abbb19fad`.

GitHits supplied one MIT task-level typed asset/provenance example that
reinforced stable IDs and reference-based fan-out. The narrower query returned
no qualifying result and two later calls timed out. No external code was
copied. GitHits therefore had low direct implementation impact and modest
architecture-confirmation impact; BioMiner's committed contracts and accepted
ADR determined the production design.

This is deterministic software and fixture evidence. It does not claim a live
Flickr run, completed image encoding or scoring, improved accuracy or
throughput, taxonomic proof from query/family/geography, human verification,
calibration, statistically supported screening, occurrence release,
publication, or deployment.
