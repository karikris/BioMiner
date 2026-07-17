# Human decision: adaptive reference evidence policy

- Status: accepted
- Decision ID: `adaptive-reference-evidence-policy-v1`
- Decision date: 2026-07-17
- Human reviewer: Kris Kari
- Default mode: `adaptive_gbif_fast_start`
- Strict compatibility mode: `human_verified_strict`

## Decision

Qualifying GBIF taxon assertions may enter model support as **GBIF
provider-asserted provisional support** after all automated admission gates pass.
This evidence is supplied by the provider and is not independent human
verification, ground truth or expert confirmation.

Reference human review no longer blocks the first provisional BioCLIP score in
the adaptive default. Species-level statistical audits using source-bound,
human-reviewed Flickr labels determine which species/reference groups require
targeted reference review. Statistical flags prioritize human work; they do not
prove that a reference image is correctly or incorrectly identified.

Every Flickr record eligible for the final occurrence dataset still requires a
decisive human review bound to the source image hash, with duplicate, conflict,
identity, domain, life-stage, coordinate, date and release checks satisfied.
Pre-review Flickr scores remain candidate evidence.

The strict mode remains available without semantic weakening: only
human-verified references enter strict support. Raw similarity, nearest-
reference evidence, prototype margins and uncalibrated classifier outputs are
not probabilities or confidence values. Probability-like outputs require an
independent calibrator trained on eligible human-reviewed labels.

## Scope and non-decisions

This decision changes the reference-admission default and the timing of initial
screening. It does not weaken taxonomic reconciliation, media rights, decoding,
duplicate resolution, observation/photographer independence, YOLOE route QA,
route separation, provenance, immutable readiness, cloud/local resumability,
calibration, Flickr review or final scientific-release gates.

The implementation and evidence are defined by
`docs/architecture/adaptive_gbif_reference_admission.md`; migration and rollback
are defined by `docs/migrations/adaptive-gbif-reference-default.md`.
