# Bounded dynamic-pooling pilot plan

Plan ID: `geo-dynamic-pool-pilot-20260718-v1`
Plan fingerprint:
`sha256:ac63c18f391164e8bcdb474190a70c6200ff28dfdda93c5c3ad11fe7de9b0092`

## Evidence boundary

This is a current fixture-backed execution plan over a durable real-source
inventory. The inventory proves that committed taxonomy, Papilio discovery
metadata, a historical provider-supported reference bank, frozen BioCLIP
embeddings, and historical unreviewed Flickr scores exist. Those historical
outputs are not counted as a new dynamic-pooling run and are not human labels.

The current workspace has zero source-bound human-reviewed labels available
for this pilot. No live Flickr/GBIF network call, media download, YOLOE run, or
BioCLIP image encoding is planned. Fixture expected taxa are test oracles, not
biological verification. Consequently this pilot can verify software behavior,
artifact lineage, target preservation, scoring comparability, and reuse, but it
cannot estimate real biological accuracy or select a production default.

## Frozen scope

Seven cases cover:

- `Papilio demoleus` in two located fixture regions and one no-geography case;
- four registry-resolved Australian butterfly taxa;
- two same-genus Papilio stress competitors;
- two additional Papilionidae competitors; and
- six distinct located fixture regions across Australia and India.

Fixture regions are controlled contexts only. They are not occurrence claims,
distribution records, or evidence of presence or absence.

Each case is evaluated with the same complete candidate union under three
target-preserving candidate schedules, a global-only control and dynamic
global/local pooling, and all four raw fusion methods. This yields 24 comparable
variants. Candidate order may vary, but membership and the target may not be
pruned. Raw scores remain uncalibrated.

## Execution and acceptance

The current run is bounded to seven unique fixture media identities, 512 MiB of
matrix cache, no source bytes in artifacts, and no network access. Each unique
fixture embedding is materialized once; reference embeddings and candidate/pool
matrices must be reused and measured.

A production selection requires real source-bound human review, target recall
of 1.0, a reviewed precision lower bound of at least 0.95, at least 86 effective
independent reviewed records, at least 30 independent records per evaluated
subgroup, no target-pruning regression, and measured embedding/matrix reuse.
Fixture evidence is forced to `insufficient_evidence` regardless of its numeric
results. Occurrence release remains unauthorized.
