# Cascade and crop runtime removal

Date: 2026-07-19

BioMiner now exposes one adaptive, full-frame production graph. The following
retired implementation cluster was deleted:

- classification-v3 overlay construction and staged taxonomy prompt cache;
- hierarchical family/genus/species cascade and its 0.90 genus shortcut;
- detector-crop creation, crop batching, debug retention, and crop scorer;
- bucketed object evidence, object-evidence join, and legacy vision reports;
- direct/cloud detection, BioCLIP, evidence, and rolling-worker dispatch; and
- production CLI switches and runtime profiles that configured those paths.

The orchestrator plans and governs the adaptive graph and accepts explicit
stage handlers. Dry-run skips execution; live stages without an owning handler
fail closed. Concrete reference operations remain under `biominer references`.

`object-detection-v2` keeps nullable historical `crop_*` columns for stable
Parquet consumption. New detection rows set those fields to null and
`crop_storage_policy=not_created`; BioMiner no longer contains code that can
materialize a spatial crop.

There is no compatibility flag or automatic fallback. Historical source,
fixtures, reports, and outputs remain recoverable from Git. Consumers must
migrate to full-frame embeddings, target-preserving candidate unions, and the
adaptive support/review/release contracts.

GitHits was not called for this task under the user's explicit directive.
Provenance records `githits_status: skipped_user_directive` and
`solution_id: null`.
